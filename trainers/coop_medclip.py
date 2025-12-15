import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from models import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


from models.medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

from PIL import Image


from .backdoor import NoiseTrigger, PatchTrigger


from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from torchvision import transforms as T


def load_medclip_to_cpu(cfg):
    print("\n\nUsing MedCLIP ...\n\n")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(os.path.join(cfg.MODEL_ROOT, "medclip", 'pretrained', 'medclip-vit'))
    model.dtype = model.vision_model.model.embeddings.patch_embeddings.projection.weight.dtype
    model.eval()                       
    return model



class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, medclip_model):
        print("1111111111111111111PromptLearner init called with:", cfg, classnames, medclip_model)
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = medclip_model.dtype

        # ctx_dim = medclip_model.text_model.projection_head.weight.shape[0]
        ctx_dim = 768
        medclip_imsize = 224 # MedCLIP's default image size

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == medclip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({medclip_imsize})"
        
        if ctx_init:
            raise NotImplementedError("This part is not yet implemented.")
            # use given words to initialize context vectors

            # ctx_init = ctx_init.replace("_", " ")
            # n_ctx = len(ctx_init.split(" "))

            # prompt = clip.tokenize(ctx_init)

            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(prompt).type(dtype)

            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            # prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # print("\n\nUsing Random Context Initialization\n\n")
        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        print("\n\nUsing Pre-trained Context Initialization\n\n")
        self.ctx = nn.Parameter(torch.load(os.path.join(os.getcwd(), 'models', 'ctx_vectors', f'ctx_{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_s{cfg.SEED}.pt')))
        # Note: This context is pre-trained using the clean images of the few-shot train dataset (i.e. with POISON_PERCENTAGE=0)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(medclip_model.text_model.tokenizer.encode(name))-2 for name in classnames]   # [CLS] and [SEP] are not counted
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # medclip_model.text_model.tokenizer.model_max_length = 77
        # breakpoint()
        tokenized_prompts = medclip_model.text_model.tokenizer(prompts, padding='max_length', max_length=25, truncation=True, return_tensors='pt')
        prompts_tokens = tokenized_prompts['input_ids']  # [n_cls, 77]
        # prompts_attention_mask = tokenized_prompts['attention_mask']

        with torch.no_grad():
            prompts_tokens_embeddings = medclip_model.text_model.model.embeddings.word_embeddings(prompts_tokens).type(dtype) # [n_cls, 77, 768]

        with torch.no_grad():
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            self.register_buffer("tokenized_prompts", tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", prompts_tokens_embeddings[:, :1, :])  # CLS
        self.register_buffer("token_suffix", prompts_tokens_embeddings[:, 1 + n_ctx :, :])  # CLASS_NAMES_TOKENS, SEP, PAD

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        

    def forward(self):
        
        ctx = self.ctx
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            raise ValueError

        return prompts

class TextEncoder(nn.Module):
    def __init__(self, medclip_text_model):
        super().__init__()
        self.medclip_text_model = medclip_text_model

    def forward(self, prompts_embeddings, tokenized_prompts):
        # 取前 25 个 token（与 prompts_embeddings 的 seq_len 一致）
        attention_mask = (tokenized_prompts != 0).long()[:, :25]  # [n_cls, 25]

        output = self.medclip_text_model.model(
            inputs_embeds=prompts_embeddings,
            attention_mask=attention_mask
        )

        last_hidden_states = torch.stack([
            output['hidden_states'][1],
            output['hidden_states'][2],
            output['hidden_states'][-1]
        ])  # [3, n_cls, 25, 768]

        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  # [n_cls, 768]
        embed = self.medclip_text_model.projection_head(embed)
        return embed


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, medclip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, medclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = medclip_model.vision_model
        self.text_encoder = TextEncoder(medclip_model.text_model)
        self.logit_scale = medclip_model.logit_scale
        self.dtype = medclip_model.dtype
        self.device = device

        self.bottleneck = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, len(classnames), bias=False)

        self.noise_trigger = NoiseTrigger(cfg)
        self.patch_trigger = PatchTrigger(cfg)
        self.normalize = torchvision.transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    def forward(self, image, backdoor_tags=None, label=None):
        image = image.type(self.dtype)
        image = self.patch_trigger(image, backdoor_tags)
        image = self.noise_trigger(image, backdoor_tags)

        augmented_image = self._apply_augmentations(image)
        augmented_image = augmented_image.type(self.dtype)

        original_features = self.image_encoder(self.normalize(image))
        augmented_features = self.image_encoder(self.normalize(augmented_image))
        combined_features = torch.cat([original_features, augmented_features], dim=0)

        if label is not None:
            combined_labels = torch.cat([label, label], dim=0)
            combined_backdoor_tags = torch.cat([backdoor_tags, backdoor_tags], dim=0)
        else:
            combined_labels = None
            combined_backdoor_tags = None

        image_features = self.image_encoder(self.normalize(image))

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts.to(self.device))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        feat = self.bottleneck(image_features)
        score = self.classifier(feat)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        i2tscore = image_features @ text_features.t()

        if not self.training:
            return logits
        else:
            return logits, score, combined_features, i2tscore, combined_labels, combined_backdoor_tags

    def _apply_augmentations(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        aug = T.Compose([
            T.RandomResizedCrop(size=image.shape[-2:], scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15)
        ])
        augmented_images = []
        for img in image:
            img_pil = T.ToPILImage()(img.cpu())
            img_aug = aug(img_pil)
            img_tensor = T.ToTensor()(img_aug).to(image.device).type(image.dtype)
            augmented_images.append(img_tensor)
        return torch.stack(augmented_images)


@TRAINER_REGISTRY.register()
class CoOp_MedCLIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading MedCLIP ...")
        medclip_model = load_medclip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            medclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, medclip_model, self.device)

        print("\n\nTurning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if not (("prompt_learner" in name) or ("noise_trigger" in name)):
                param.requires_grad_(False)
                # print(f"Not Learnable: {name}")
            else: 
                print(f"Learnable: {name}")
        print("\n\n")   

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("baple", nn.Sequential(self.model.prompt_learner, self.model.noise_trigger) , self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()

        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)



    # def forward_backward(self, batch):
    #
    #     self.model.noise_trigger.noise.requires_grad = True
    #     image, label, backdoor_tag = self.parse_batch_train(batch)  # image: [B, C, H, W]
    #
    #     prec = self.cfg.TRAINER.COOP.PREC
    #
    #     if prec == "amp":
    #         raise NotImplementedError("AMP is not yet supported.")
    #     else:
    #         output = self.model(image, backdoor_tag)
    #
    #         lambda_clean = 1.0
    #         lambda_adv = 1.0
    #         # lambda_reg = 0.0
    #
    #         clean_exists = any(~backdoor_tag)
    #         backdoor_exists = any(backdoor_tag)
    #
    #         if clean_exists :
    #             loss_clean = F.cross_entropy(output[~backdoor_tag], label[~backdoor_tag])
    #
    #         if backdoor_exists :
    #             loss_adv = F.cross_entropy(output[backdoor_tag], label[backdoor_tag])
    #
    #
    #         if clean_exists and backdoor_exists:
    #             loss = lambda_clean*loss_clean + lambda_adv*loss_adv
    #         elif clean_exists and not backdoor_exists:
    #             loss = lambda_clean*loss_clean
    #         elif not clean_exists and backdoor_exists:
    #             loss = lambda_adv*loss_adv
    #         else:
    #             raise ValueError("No clean or backdoor images found. Check the backdoor tag assignments in Dataset class.")
    #
    #
    #         self.model_backward_and_update(loss)
    #
    #         # update trigger noise
    #         if backdoor_exists:
    #             trigger_noise_grad  = self.model.noise_trigger.noise.grad.data
    #             self.model.noise_trigger.noise.data -= trigger_noise_grad.sign()*0.01
    #             eps=self.cfg.BACKDOOR.NOISE_EPS/255.0
    #             self.model.noise_trigger.noise.data.clamp_(-eps,eps)
    #             self.model.noise_trigger.noise.detach_()
    #
    #
    #     loss_summary = {
    #         "loss": loss.item(),
    #         "acc": compute_accuracy(output, label)[0].item(),
    #     }
    #
    #     if (self.batch_idx + 1) == self.num_batches:
    #         self.update_lr()
    #
    #     return loss_summary
    # def forward_backward(self, batch):
    #     self.model.noise_trigger.noise.requires_grad = True
    #     image, label, backdoor_tag = self.parse_batch_train(batch)
    #
    #     prec = self.cfg.TRAINER.COOP.PREC
    #     if prec == "amp":
    #         raise NotImplementedError("AMP is not yet supported.")
    #     else:
    #         output, score, feat, i2tscore, combined_labels, combined_backdoor_tags = self.model(image, backdoor_tag,
    #                                                                                             label=label)
    #
    #     xent = CrossEntropyLabelSmooth(num_classes=len(self.dm.dataset.classnames))
    #
    #     # 确保特征和标签的批量一致
    #     batch_size = label.size(0)
    #     original_feat = feat[:batch_size]  # 前 B 个为原始特征 [B, 512]
    #     augmented_feat = feat[batch_size:]  # 后 B 个为增强特征 [B, 512]
    #     # 使用扩展后的 backdoor_tag
    #     filtered_feat = feat[combined_backdoor_tags]  # 形状 [num_backdoor, 512]
    #
    #     lambda_clean = 1.0
    #     lambda_adv   = 1.0
    #
    #     clean_exists  = (~backdoor_tag).any()
    #     backdoor_exists = backdoor_tag.any()
    #
    #     loss_clean = loss_adv = l3loss_adv = None
    #     # 分离原始和增强特征（假设前一半是原始，后一半是增强）
    #     batch_size = label.size(0)
    #     original_feat = feat[:batch_size]
    #     augmented_feat = feat[batch_size:]
    #
    #     # --------- clean 样本 ---------
    #     if clean_exists:
    #         loss_clean = F.cross_entropy(output[~backdoor_tag], label[~backdoor_tag])
    #
    #     # --------- backdoor 样本 ---------
    #     if backdoor_exists:
    #         anchor = original_feat[backdoor_tag]
    #         positive = augmented_feat[backdoor_tag]
    #         labels_backdoor = label[backdoor_tag]
    #
    #         # 随机选择负样本
    #         neg_indices = []
    #         for lbl in labels_backdoor:
    #             valid_neg = (label != lbl).nonzero().squeeze()
    #             if valid_neg.numel() == 0:
    #                 neg_idx = torch.randint(0, original_feat.size(0), (1,))
    #             else:
    #                 neg_idx = valid_neg[torch.randint(len(valid_neg), (1,))]
    #             neg_indices.append(neg_idx)
    #         negative = original_feat[torch.tensor(neg_indices)]
    #
    #         # 计算Triplet Loss
    #         triplet_loss = TripletLoss()(anchor, positive, negative)[0]
    #
    #         ID_LOSS = xent(score[backdoor_tag], label[backdoor_tag])
    #         I2TLOSS = xent(i2tscore[backdoor_tag], label[backdoor_tag])
    #         l3loss_adv = ID_LOSS + triplet_loss + I2TLOSS
    #         loss_adv = F.cross_entropy(output[backdoor_tag], label[backdoor_tag])
    #
    #     # 合并总损失
    #     if clean_exists and backdoor_exists:
    #         loss = lambda_clean * loss_clean + lambda_adv * loss_adv + l3loss_adv
    #     elif clean_exists and not backdoor_exists:
    #         loss = loss_clean
    #     elif not clean_exists and backdoor_exists:
    #         loss = lambda_adv * loss_adv + l3loss_adv
    #     else:
    #         raise ValueError("No clean or backdoor images found.")
    #
    #     self.model_backward_and_update(loss)
    #
    #     # 更新 trigger noise
    #     if backdoor_exists:
    #         grad = self.model.noise_trigger.noise.grad.data
    #         self.model.noise_trigger.noise.data -= grad.sign() * 0.01
    #         eps = self.cfg.BACKDOOR.NOISE_EPS / 255.0
    #         self.model.noise_trigger.noise.data.clamp_(-eps, eps)
    #         self.model.noise_trigger.noise.detach_()
    #
    #     loss_summary = {
    #         "loss": loss.item(),
    #         "acc": compute_accuracy(output, label)[0].item(),
    #     }
    #
    #     if (self.batch_idx + 1) == self.num_batches:
    #         self.update_lr()
    #
    #     return loss_summary
    def forward_backward(self, batch):
        self.model.noise_trigger.noise.requires_grad = True
        image, label, backdoor_tag = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            raise NotImplementedError("AMP is not yet supported.")
        else:
            output, score, feat, i2tscore, combined_labels, combined_backdoor_tags = self.model(
                image, backdoor_tag, label=label
            )

        xent = CrossEntropyLabelSmooth(num_classes=len(self.dm.dataset.classnames))

        # ----------- 特征、标签准备 -----------
        batch_size = label.size(0)
        original_feat   = feat[:batch_size]        # [B, 512]
        augmented_feat  = feat[batch_size:]        # [B, 512]

        lambda_clean = 1.0
        lambda_adv   = 1.0
        lambda_oti   = 1.0  # OTI 损失权重

        clean_exists   = (~backdoor_tag).any()
        backdoor_exists = backdoor_tag.any()

        loss_clean = loss_adv = l3loss_adv = oti_loss = None

        # ----------- clean 样本 -----------
        if clean_exists:
            loss_clean = F.cross_entropy(output[~backdoor_tag], label[~backdoor_tag])

        # ----------- backdoor 样本 -----------
        if backdoor_exists:
            anchor = original_feat[backdoor_tag]
            positive = augmented_feat[backdoor_tag]
            labels_backdoor = label[backdoor_tag]

            # 随机负样本
            neg_indices = []
            for lbl in labels_backdoor:
                valid_neg = (label != lbl).nonzero().squeeze()
                if valid_neg.numel() == 0:
                    neg_idx = torch.randint(0, original_feat.size(0), (1,))
                else:
                    neg_idx = valid_neg[torch.randint(len(valid_neg), (1,))]
                neg_indices.append(neg_idx)
            negative = original_feat[torch.tensor(neg_indices, device=anchor.device)]

            triplet_loss = TripletLoss()(anchor, positive, negative)[0]
            ID_LOSS = xent(score[backdoor_tag], labels_backdoor)
            I2TLOSS = xent(i2tscore[backdoor_tag], labels_backdoor)
            l3loss_adv = ID_LOSS + triplet_loss + I2TLOSS
            loss_adv   = F.cross_entropy(output[backdoor_tag], labels_backdoor)

            # ===== 伪文本特征（OTI）=====
            template_sentence = "a photo of trigger"
            # 使用 MedCLIP 自带 tokenizer
            medclip = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            text_model = medclip.text_encoder.medclip_text_model
            tokenizer = text_model.tokenizer

            tokenized = tokenizer(
                [template_sentence] * len(anchor),
                padding='max_length',
                max_length=25,
                truncation=True,
                return_tensors='pt'
            ).to(anchor.device)

            input_ids = tokenized['input_ids']      # [B, 25]
            attn_mask = tokenized['attention_mask'] # [B, 25]

            with torch.no_grad():
                # embedding
                embeds = text_model.model.embeddings.word_embeddings(input_ids)  # [B, 25, 768]
                pos_ids = torch.arange(0, input_ids.size(1),
                                       dtype=torch.long, device=input_ids.device).unsqueeze(0)
                embeds += text_model.model.embeddings.position_embeddings(pos_ids)

                # forward
                outputs = text_model.model(
                    inputs_embeds=embeds,
                    attention_mask=attn_mask
                )
                cls_embed = outputs.last_hidden_state[:, 0]          # [B, 768]
                pseudo_text_features = text_model.projection_head(cls_embed)  # [B, 512]

            # L2 归一化
            pseudo_text_features = F.normalize(pseudo_text_features, dim=-1)
            anchor_norm = F.normalize(anchor, dim=-1)
            oti_loss = F.cosine_embedding_loss(
                anchor_norm,
                pseudo_text_features,
                torch.ones(anchor_norm.size(0)).to(anchor.device)
            )

        # ----------- 合并总损失 -----------
        loss = torch.tensor(0., device=image.device)
        if clean_exists and backdoor_exists:
            loss = lambda_clean * loss_clean + lambda_adv * loss_adv + l3loss_adv + lambda_oti * oti_loss
        elif clean_exists and not backdoor_exists:
            loss = loss_clean
        elif not clean_exists and backdoor_exists:
            loss = lambda_adv * loss_adv + l3loss_adv + lambda_oti * oti_loss
        else:
            raise ValueError("No clean or backdoor images found.")

        self.model_backward_and_update(loss)

        # ----------- 更新 trigger noise -----------
        if backdoor_exists:
            grad = self.model.noise_trigger.noise.grad.data
            self.model.noise_trigger.noise.data -= grad.sign() * 0.01
            eps = self.cfg.BACKDOOR.NOISE_EPS / 255.0
            self.model.noise_trigger.noise.data.clamp_(-eps, eps)
            self.model.noise_trigger.noise.detach_()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        backdoor_tag = batch["backdoor_tag"].to(self.device)
        return input, label, backdoor_tag==1

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        
        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        
        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
