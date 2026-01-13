import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision

import time
import numpy as np
from thop import profile
from torch import optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

# breakpoint()
from models import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from .backdoor import NoiseTrigger, PatchTrigger

from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from torchvision import transforms as T


def load_clip_to_cpu(cfg):
    if cfg.MODEL_NAME == 'clip':
        print("\n\n\nUsing CLIP\n\n\n")
        model_path = os.path.join(cfg.MODEL_ROOT, "clip", 'openai_clip_vit_b32.pt')
        # model_path = "/data-nvme/asif.hanif/pre-trained-models/vlps/clip/openai_clip_vit_b32.pt"

    elif cfg.MODEL_NAME == 'plip':
        print("\n\n\nUsing PLIP\n\n\n")
        model_path = os.path.join(cfg.MODEL_ROOT, "plip", 'plip_vit_b32.pt')
        # model_path = "/data-nvme/asif.hanif/pre-trained-models/vlps/plip/plip_vit_b32.pt"

    elif cfg.MODEL_NAME == 'quiltnet':
        print("\n\n\nUsing QuiltNet\n\n\n")
        model_path = os.path.join(cfg.MODEL_ROOT, "quiltnet", 'quiltnet_b32.pt')
        # model_path = "/data-nvme/asif.hanif/pre-trained-models/vlps/quiltnet_b32/quiltnet_b32.pt"
    else:
        raise ValueError(f"Model '{cfg.MODEL_NAME}' not found. Please choose from 'clip', 'plip', 'quiltnet'")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    model = model.float()  

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        # breakpoint()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            raise NotImplementedError("This part is not yet implemented.")

            # # use given words to initialize context vectors
            # ctx_init = ctx_init.replace("_", " ")
            # n_ctx = len(ctx_init.split(" "))
            #
            # prompt = clip.tokenize(ctx_init)
            #
            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(prompt).type(dtype)
            #
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            # prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing Generic Context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial Context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # print("\n\nUsing Random Context Initialization\n\n")
        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        print("\n\nUsing Pre-trained Context Initialization\n\n")
        # self.ctx = nn.Parameter(torch.load(os.path.join(os.getcwd(), 'models', 'ctx_vectors',
        #                                                 f'ctx_{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_s{cfg.SEED}.pt')))
        # Note: This context is pre-trained using the clean images of the few-shot train dataset (i.e. with POISON_PERCENTAGE=0)
        ctx_path = os.path.join(os.getcwd(), 'models', 'ctx_vectors',
                                f'ctx_{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_s{cfg.SEED}.pt')
        if os.path.exists(ctx_path):
            self.ctx = nn.Parameter(torch.load(ctx_path))
            print(f"Loaded context vector from {ctx_path}")
        else:
            print(f"Warning: {ctx_path} not found, using random initialization.")
            self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        # breakpoint()
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

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip = clip_model

        cfg.defrost()
        cfg.DTYPE = str(self.dtype).split(".")[1]
        cfg.DEVICE = str(self.device)
        cfg.freeze()

        self.classifier = nn.Linear(512, 80, bias=False)
        self.bottleneck = nn.BatchNorm1d(512)

        self.noise_trigger = NoiseTrigger(cfg)
        self.patch_trigger = PatchTrigger(cfg)
        self.normalize = torchvision.transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD).to(
            device).type(clip_model.dtype)

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

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        feat = self.bottleneck(image_features)
        score = self.classifier(feat)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        i2tscore = image_features @ text_features.t()

        if not self.training:  
            # print("Test output type:", type(logits))  # 应为torch.Tensor
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

        augmented_image = torch.stack(augmented_images)  # 形状 [B, C, H, W]
        return augmented_image


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
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

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        

    def forward_backward(self, batch):
        self.model.noise_trigger.noise.requires_grad = True
        image, label, backdoor_tag = self.parse_batch_train(batch)  # image: [B, C, H, W]

        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            raise NotImplementedError("AMP is not yet supported.")
        else:
            output, score, feat, i2tscore, combined_labels, combined_backdoor_tags = self.model(image, backdoor_tag,
                                                                                                label=label)

            batch_size = label.size(0)
            original_feat = feat[:batch_size]  
            augmented_feat = feat[batch_size:]  

           
            filtered_feat = feat[combined_backdoor_tags]  

            lambda_clean = 1.0
            lambda_adv = 1.0
            lambda_oti = 0.1  

            clean_exists = any(~backdoor_tag)
            backdoor_exists = any(backdoor_tag)

            xent = CrossEntropyLabelSmooth(num_classes=len(self.dm.dataset.classnames))

            loss_clean = None
            loss_adv = None
            loss_oti = None

            if clean_exists:
                loss_clean = F.cross_entropy(output[~backdoor_tag], label[~backdoor_tag])

            if backdoor_exists:
                anchor = original_feat[backdoor_tag]
                positive = augmented_feat[backdoor_tag]
                labels_backdoor = label[backdoor_tag]

                
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

                def oti(batch_im_features, clip_model, learning_rate=2e-2, weight_decay=0.01,
                        num_pseudo_tokens=1, oti_steps=150, template_sentence='a photo of {}'):
                    criterion = nn.CosineEmbeddingLoss()
                    criterion_target = torch.as_tensor([1], device=batch_im_features.device)
                    embedding_dim = clip_model.text_encoder.text_projection.shape[1]
                    bs = len(batch_im_features)

                    oti_pseudo_tokens = torch.empty((bs, num_pseudo_tokens, embedding_dim),
                                                    device=batch_im_features.device)
                    nn.init.normal_(oti_pseudo_tokens, std=0.02)
                    oti_pseudo_tokens = nn.Parameter(oti_pseudo_tokens)

                    optimizer = optim.AdamW([oti_pseudo_tokens], lr=learning_rate, weight_decay=weight_decay)
                    scaler = torch.cuda.amp.GradScaler()
                    iteration_losses = []

                    for _ in range(oti_steps):
                        optimizer.zero_grad()

                        template_oti_texts = [template_sentence.format(" £ " * num_pseudo_tokens) for _ in range(bs)]
                        tokenized_template_oti_texts = clip.tokenize(template_oti_texts).to(batch_im_features.device)

                        with torch.cuda.amp.autocast():
                            text_tokens = tokenized_template_oti_texts
                            token_embedding = clip_model.clip.token_embedding(text_tokens).type(clip_model.dtype)
                            token_embedding[:, 1:1 + num_pseudo_tokens, :] = oti_pseudo_tokens

                            x = token_embedding + clip_model.text_encoder.positional_embedding.type(clip_model.dtype)
                            x = x.permute(1, 0, 2)  # NLD -> LND
                            x = clip_model.text_encoder.transformer(x)
                            x = x.permute(1, 0, 2)  # LND -> NLD
                            x = clip_model.text_encoder.ln_final(x).type(clip_model.dtype)

                            text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(
                                dim=-1)] @ clip_model.text_encoder.text_projection

                            template_oti_texts_features = F.normalize(text_features, dim=-1)
                            batch_im_features_norm = F.normalize(batch_im_features, dim=-1)

                            loss = criterion(template_oti_texts_features, batch_im_features_norm, criterion_target)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        iteration_losses.append(loss.item())

                    final_loss = np.mean(iteration_losses)
                    return final_loss

                loss_oti_value = oti(anchor.detach(), self.model, learning_rate=2e-2, weight_decay=0.01,
                                     num_pseudo_tokens=1, oti_steps=150,
                                     template_sentence='a photo of {}')

                loss_oti = torch.tensor(loss_oti_value, device=self.device)

            if clean_exists and backdoor_exists:
                loss = lambda_clean * loss_clean + lambda_adv * loss_adv + l3loss_adv + lambda_oti * loss_oti
            elif clean_exists and not backdoor_exists:
                loss = lambda_clean * loss_clean
            elif not clean_exists and backdoor_exists:
                loss = lambda_adv * loss_adv + l3loss_adv + lambda_oti * loss_oti
            else:
                raise ValueError(
                    "No clean or backdoor images found. Check the backdoor tag assignments in Dataset class.")

            self.model_backward_and_update(loss)

            # update trigger noise
            if backdoor_exists:
                trigger_noise_grad = self.model.noise_trigger.noise.grad.data
                self.model.noise_trigger.noise.data -= trigger_noise_grad.sign() * 0.01
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
        backdoor_tag = batch["backdoor_tag"]
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
