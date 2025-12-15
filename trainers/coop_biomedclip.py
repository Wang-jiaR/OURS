import os.path as osp
import os
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from models import clip
from models.open_clip import create_model_from_pretrained, \
    get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8

from .backdoor import NoiseTrigger, PatchTrigger

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from torchvision import transforms as T
import kornia.augmentation as K


def load_clip_to_cpu(cfg):
    if cfg.MODEL_NAME == 'biomedclip':
        print("\n\nUsing BioMedCLIP ...\n\n")
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.eval()
    else:
        raise ValueError(f"Model {cfg.MODEL_NAME} not found. Supported models are: biomedclip")
    return model


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype

        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # ctx_dim = medclip_model.text_model.projection_head.weight.shape[0]
        ctx_dim = 768
        clip_imsize = 224  # BioMedCLIP's default image size

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

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
        self.ctx = nn.Parameter(torch.load(os.path.join(os.getcwd(), 'models', 'ctx_vectors',
                                                        f'ctx_{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_s{cfg.SEED}.pt')))
        # Note: This context is pre-trained using the clean images of the few-shot train dataset (i.e. with POISON_PERCENTAGE=0)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(tokenizer.tokenizer.encode(name)) - 2 for name in
                     classnames]  # [CLS] and [SEP] are not counted
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        context_length = 256
        prompts_tokens = tokenizer(prompts, context_length=context_length)
        self.prompts_attention_mask = (prompts_tokens != clip_model.text.config.pad_token_id).long()

        with torch.no_grad():
            prompts_tokens_embeddings = clip_model.text.transformer.embeddings(input_ids=prompts_tokens).type(
                dtype)  # [n_cls, 256, 768]

        with torch.no_grad():
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            self.register_buffer("tokenized_prompts", tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", prompts_tokens_embeddings[:, :1, :])  # CLS
        self.register_buffer("token_suffix",
                             prompts_tokens_embeddings[:, 1 + n_ctx:, :])  # CLASS_NAMES_TOKENS, SEP, PAD

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
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts, self.prompts_attention_mask


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_model = clip_model.text

    def forward(self, prompts_embeddings, prompts_attention_mask, normalize=False):
        out = self.text_model.transformer(inputs_embeds=prompts_embeddings, attention_mask=prompts_attention_mask)
        pooled_out = self.text_model.pooler(out, prompts_attention_mask)
        projected = self.text_model.proj(pooled_out)
        return F.normalize(projected, dim=-1) if normalize else projected


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.visual

    def forward(self, image, normalize=False):
        features = self.vision_model(image)
        return F.normalize(features, dim=-1) if normalize else features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()

        clip_model.dtype = clip_model.visual.head.proj.weight.dtype

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        cfg.defrost()
        cfg.DTYPE = str(self.dtype).split(".")[1]
        cfg.DEVICE = str(self.device)
        cfg.freeze()

        self.noise_trigger = NoiseTrigger(cfg)
        self.patch_trigger = PatchTrigger(cfg)
        self.normalize = torchvision.transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

        self.bottleneck = nn.BatchNorm1d(512)  # BioMedCLIP 视觉特征 dim=512
        self.classifier = nn.Linear(512, len(classnames), bias=False)

    # def forward(self, image, backdoor_tags=None, label=None):
    #     image = self.patch_trigger(image.type(self.dtype), backdoor_tags)
    #     image = self.noise_trigger(image.type(self.dtype), backdoor_tags)
    #
    #     image_features = self.image_encoder(self.normalize(image), normalize=True)  # [B, 512]
    #
    #     # 用于 ID_LOSS 的 logits
    #     feat = self.bottleneck(image_features)
    #     score = self.classifier(feat)  # [B, n_cls]
    #
    #     prompts_embeddings, prompts_attention_mask = self.prompt_learner()
    #     text_features = self.text_encoder(prompts_embeddings, prompts_attention_mask.to(self.device), normalize=True)
    #
    #     logits = self.logit_scale.exp() * image_features @ text_features.t()  # [B, n_cls]
    #     i2tscore = image_features @ text_features.t()  # [B, n_cls]
    #
    #     if not self.training:
    #         return logits
    #
    #     return logits, score, image_features, i2tscore, label, backdoor_tags
    def forward(self, image, backdoor_tags=None, label=None):
        image = image.type(self.dtype)  # 转换为模型指定的 dtype（如 torch.half）
        image = self.patch_trigger(image.type(self.dtype), backdoor_tags)  # add patch trigger to backdoor images
        image = self.noise_trigger(image.type(self.dtype), backdoor_tags)  # add noise trigger to backdoor images

        # 生成增强后的图像
        augmented_image = self._apply_augmentations(image)
        augmented_image = augmented_image.type(self.dtype)
        # 提取原始和增强后的特征
        original_features = self.image_encoder(self.normalize(image))
        augmented_features = self.image_encoder(self.normalize(augmented_image))

        # 合并特征用于后续计算
        combined_features = torch.cat([original_features, augmented_features], dim=0)
        # combined_labels = torch.cat([label, label], dim=0)
        # # 同步扩展标签和 backdoor_tag
        # combined_labels = torch.cat([label, label], dim=0)  # 形状 [2B]
        # combined_backdoor_tags = torch.cat([backdoor_tags, backdoor_tags], dim=0)  # 形状 [2B]

        # 仅在训练时拼接标签和 backdoor_tag
        if label is not None:
            combined_labels = torch.cat([label, label], dim=0)
            combined_backdoor_tags = torch.cat([backdoor_tags, backdoor_tags], dim=0)
        else:
            combined_labels = None
            combined_backdoor_tags = None

        image_features = self.image_encoder(self.normalize(image))

        prompts_embeddings, prompts_attention_mask = self.prompt_learner()
        prompts_attention_mask = prompts_attention_mask.to(self.device)
        text_features = self.text_encoder(prompts_embeddings, prompts_attention_mask)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        feat = self.bottleneck(image_features)
        score = self.classifier(feat)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        i2tscore = image_features @ text_features.t()

        if not self.training:  # 新增条件判断
            # print("Test output type:", type(logits))  # 应为torch.Tensor
            return logits
        else:
            return logits, score, combined_features, i2tscore, combined_labels, combined_backdoor_tags

    def _apply_augmentations(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        aug = K.AugmentationSequential(
            K.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(0.15, 0, 0, 0, p=0.5),
            data_keys=["input"],
        )
        augmented_image = aug(image)  # 直接在GPU上完成
        return augmented_image


@TRAINER_REGISTRY.register()
class CoOp_BioMedCLIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading Model ...")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building Model")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

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

        self.register_model("baple", nn.Sequential(self.model.prompt_learner, self.model.noise_trigger), self.optim,
                            self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        self.model.noise_trigger.noise.requires_grad = True
        image, label, backdoor_tag = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            raise NotImplementedError("AMP is not yet supported.")
        else:
            output, score, feat, i2tscore, combined_labels, combined_backdoor_tags = self.model(image, backdoor_tag,
                                                                                                label=label)
        xent = CrossEntropyLabelSmooth(num_classes=len(self.dm.dataset.classnames))

        # 确保特征和标签的批量一致
        batch_size = label.size(0)
        original_feat = feat[:batch_size]  # 前 B 个为原始特征 [B, 512]
        augmented_feat = feat[batch_size:]  # 后 B 个为增强特征 [B, 512]
        # 使用扩展后的 backdoor_tag
        filtered_feat = feat[combined_backdoor_tags]  # 形状 [num_backdoor, 512]

        lambda_clean = 1.0
        lambda_adv = 1.0
        lambda_oti = 1.0  # 新增: OTI损失的权重

        clean_exists = (~backdoor_tag).any()
        backdoor_exists = backdoor_tag.any()

        loss_clean = loss_adv = l3loss_adv = oti_loss = None

        # 分离原始和增强特征（假设前一半是原始，后一半是增强）
        batch_size = label.size(0)
        original_feat = feat[:batch_size]
        augmented_feat = feat[batch_size:]

        # --------- clean 样本 ---------
        if clean_exists:
            loss_clean = F.cross_entropy(output[~backdoor_tag], label[~backdoor_tag])

        # --------- backdoor 样本 ---------
        if backdoor_exists:
            anchor = original_feat[backdoor_tag]
            positive = augmented_feat[backdoor_tag]
            labels_backdoor = label[backdoor_tag]

            # 随机选择负样本
            neg_indices = []
            for lbl in labels_backdoor:
                valid_neg = (label != lbl).nonzero().squeeze()
                if valid_neg.numel() == 0:
                    neg_idx = torch.randint(0, original_feat.size(0), (1,))
                else:
                    neg_idx = valid_neg[torch.randint(len(valid_neg), (1,))]
                neg_indices.append(neg_idx)
            negative = original_feat[torch.tensor(neg_indices)]

            # 计算Triplet Loss
            triplet_loss = TripletLoss()(anchor, positive, negative)[0]

            ID_LOSS = xent(score[backdoor_tag], label[backdoor_tag])
            I2TLOSS = xent(i2tscore[backdoor_tag], label[backdoor_tag])
            l3loss_adv = ID_LOSS + triplet_loss + I2TLOSS
            loss_adv = F.cross_entropy(output[backdoor_tag], label[backdoor_tag])

            # 计算Triplet Loss
            triplet_loss = TripletLoss()(anchor, positive, negative)[0]

            ID_LOSS = xent(score[backdoor_tag], label[backdoor_tag])
            I2TLOSS = xent(i2tscore[backdoor_tag], label[backdoor_tag])
            l3loss_adv = ID_LOSS + triplet_loss + I2TLOSS
            loss_adv = F.cross_entropy(output[backdoor_tag], label[backdoor_tag])

            # ===== 新增: OTI 损失计算 =====
            template_sentence = "a photo of {}"
            pseudo_texts = [template_sentence.format("trigger")] * len(anchor)

            tokenizer = get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            context_length = 256
            tokenized_pseudo = tokenizer(pseudo_texts, context_length=context_length).to(anchor.device)

            clip_model_single = self.model.module \
                if isinstance(self.model, nn.DataParallel) else self.model

            with torch.no_grad():
                text_model = clip_model_single.text_encoder.text_model

                # 1. token embedding
                embeds = text_model.transformer.embeddings.word_embeddings(tokenized_pseudo)  # [B, L, 768]

                # 2. position embedding
                pos_ids = torch.arange(
                    0, tokenized_pseudo.size(1),
                    dtype=torch.long, device=tokenized_pseudo.device
                ).unsqueeze(0).expand_as(tokenized_pseudo)
                pos_embeds = text_model.transformer.embeddings.position_embeddings(pos_ids)
                embeds = embeds + pos_embeds

                # 3. 正确的 attention mask: 0 表示真实 token, -inf 表示 pad
                attention_mask = (tokenized_pseudo != text_model.config.pad_token_id).long()
                # 构造 broadcast-able mask => [B, 1, 1, L]
                extended_mask = attention_mask[:, None, None, :]
                extended_mask = extended_mask.to(dtype=embeds.dtype)
                extended_mask = (1.0 - extended_mask) * torch.finfo(embeds.dtype).min

                # 4. transformer encoder
                encoder = text_model.transformer.encoder
                out = encoder(embeds, attention_mask=extended_mask).last_hidden_state  # [B, L, 768]

                # 5. pooler (CLS token)
                pooled = out[:, 0]  # [B, 768]
                pseudo_text_features = text_model.proj(pooled)  # [B, 512]

            # 归一化
            pseudo_text_features = F.normalize(pseudo_text_features, dim=-1)
            anchor_norm = F.normalize(anchor, dim=-1)

            # 余弦相似度损失
            oti_loss = F.cosine_embedding_loss(
                anchor_norm,
                pseudo_text_features,
                torch.ones(anchor_norm.size(0)).to(anchor.device)
            )
            # ===== OTI 损失计算结束 =====

        # 合并总损失
        if clean_exists and backdoor_exists:
            loss = lambda_clean * loss_clean + lambda_adv * loss_adv + l3loss_adv + lambda_oti * oti_loss
        elif clean_exists and not backdoor_exists:
            loss = loss_clean
        elif not clean_exists and backdoor_exists:
            loss = lambda_adv * loss_adv + l3loss_adv + lambda_oti * oti_loss
        else:
            raise ValueError("No clean or backdoor images found.")

        self.model_backward_and_update(loss)

        # 更新 trigger noise (保持不变)
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
        return input, label, backdoor_tag == 1

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
