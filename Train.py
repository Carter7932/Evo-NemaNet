

"""
Main_Train (简化版 + MambaVision T/S 官方API + 双层FC头 + 分阶段不同LR + 可选随机增强)
- 预处理：ResizeByLongerSide → CenterPadToSquare → (AUGMENT?) → ToTensor → MinMaxNormalize
- 骨干：ResNet101 / EfficientNet-B0~B2 / EfficientNetV2-S,M / Swin(T,S) / Swin-V2(T,S) / MambaVision(T,S)
- 统一分类头：MLP(in→2048→NUM_CLASSES)
- 分阶段训练：Stage-1(仅头)→Stage-2(全模型)，且不同学习率（头/骨干）
"""
import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from tqdm.auto import tqdm
import os, random, warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from PIL import Image, ImageFilter
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

from mambavision import create_model as mv_create_model

# =============================
# 配置
# =============================
@dataclass
class Config:
    TRAIN_DIR: str = "Train"
    VAL_DIR: str = "Val"

    # 支持：resnet101 / efficientnet_b0~b2 / efficientnet_v2_s/m /
    #       swin_t/s / swin_v2_t/s / mambavision_t / mambavision_s
    MODEL_NAME: str = "efficientnet_v2_m"
    NUM_CLASSES: int = 19
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 80
    NUM_WORKERS: int = 2
    DEVICE_ID: int = 0
    SEED: int = 42

    # 基础预处理
    PAD_FILL: tuple = (128, 128, 128)

    # 数据增强（与对比代码一致，可开关）
    AUGMENT: bool = True
    HFLIP_P: float = 0.5
    VFLIP_P: float = 0.0
    BRIGHTNESS: float = 0.15
    CONTRAST: float   = 0.15
    SATURATION: float = 0.10
    HUE: float        = 0.05
    BLUR_P: float     = 0.30

    # AMP
    USE_AMP: bool = True

    # 优化器/调度（auto 按模型族选择；也可手动覆盖）
    OPTIMIZER: str = "auto"        # auto/SGD/AdamW/RMSprop
    SCHEDULER: str = "auto"        # auto/cosine/multistep/none
    LR: Optional[float] = None
    WEIGHT_DECAY: Optional[float] = None

    # 分阶段训练 + 不同学习率
    STAGED_TRAIN: bool = True
    STAGE1_EPOCHS: int = 10
    FREEZE_BN: bool = True
    LR_HEAD_STAGE1: Optional[float] = None  # e.g., 3e-3
    LR_HEAD: Optional[float] = None         # e.g., 1e-3
    LR_BACKBONE: Optional[float] = None     # e.g., 1e-4

    # MambaVision 权重路径（仅 T/S）
    MAMBA_T_WEIGHTS: Optional[str] = None
    MAMBA_S_WEIGHTS: Optional[str] = None

    # 保存
    CHECKPOINT_DIR: str = "Model_CheckPoint"
    BEST_DIR: str = "Best_Model"

    def ensure(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.BEST_DIR, exist_ok=True)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
        print("[Config]", {k: v for k, v in asdict(self).items()
                           if k not in ("PAD_FILL", "MAMBA_T_WEIGHTS", "MAMBA_S_WEIGHTS")})


# =============================
# 模型别名与输入尺寸映射（已裁剪）
# =============================
MODEL_ALIASES = {
    "resnet101": "resnet101",

    # EfficientNet-B (仅 b0~b2)
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",

    # EfficientNet-V2
    "efficientnet_v2_s": "efficientnet_v2_s",
    "efficientnet_v2_m": "efficientnet_v2_m",

    # Swin v1 (仅 t/s)
    "swin_t": "swin_t",
    "swin_s": "swin_s",

    # Swin v2 (仅 t/s)
    "swin_v2_t": "swin_v2_t",
    "swin_v2_s": "swin_v2_s",

    # Mamba Vision (仅 T/S)
    "mambavision_t": "mambavision_t",
    "mambavision_s": "mambavision_s",
}

MODEL_DEFAULT_SIZE = {
    "resnet101": 224,
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_v2_s": 384,
    "efficientnet_v2_m": 480,
    "swin_t": 224,
    "swin_s": 224,
    "swin_v2_t": 256,
    "swin_v2_s": 256,
    # MambaVision
    "mambavision_t": 256,
    "mambavision_s": 256,
}


def get_input_size(model_name: str) -> int:
    key = MODEL_ALIASES.get(model_name.lower())
    if key is None or key not in MODEL_DEFAULT_SIZE:
        raise ValueError(f"未配置默认输入尺寸: {model_name}")
    return MODEL_DEFAULT_SIZE[key]


def is_effnet_family(name: str) -> bool:
    n = name.lower()
    return n.startswith("efficientnet_b") or n.startswith("efficientnet_v2_")


def is_swin_family(name: str) -> bool:
    n = name.lower()
    return n.startswith("swin")


def is_mamba_family(name: str) -> bool:
    n = name.lower()
    return n.startswith("mambavision")


# =============================
# EfficientNet/V2 规格（参考）
# =============================
EFFNET_SPECS = {
    "efficientnet_b0": {"feature_dim": 1280, "dropout": 0.2},
    "efficientnet_b1": {"feature_dim": 1280, "dropout": 0.2},
    "efficientnet_b2": {"feature_dim": 1408, "dropout": 0.3},
    "efficientnet_v2_s": {"feature_dim": 1280, "dropout": 0.2},
    "efficientnet_v2_m": {"feature_dim": 1280, "dropout": 0.3},
}


# =============================
# 预处理与可选增强
# =============================
class ResizeByLongerSide(object):
    def __init__(self, target, interpolation=Image.BILINEAR):
        self.target = int(target)
        self.interp = interpolation

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (self.target, self.target), color=(128, 128, 128))
        if w >= h:
            new_w = self.target
            new_h = max(1, int(round(h * (self.target / float(w)))))
        else:
            new_h = self.target
            new_w = max(1, int(round(w * (self.target / float(h)))))
        return img.resize((new_w, new_h), self.interp)


class CenterPadToSquare(object):
    def __init__(self, size, fill=(128, 128, 128)):
        self.size = int(size)
        self.fill = tuple(map(int, fill))

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w == self.size and h == self.size:
            return img
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - w) // 2
        top = (self.size - h) // 2
        canvas.paste(img, (left, top))
        return canvas


class RandomGaussianBlur(object):
    def __init__(self, p=0.3, radius_min=0.1, radius_max=2.0):
        self.p = float(p); self.rmin = float(radius_min); self.rmax = float(radius_max)
    def __call__(self, img: Image.Image):
        if random.random() >= self.p: return img
        radius = self.rmin + (self.rmax - self.rmin) * random.random()
        return img.filter(ImageFilter.GaussianBlur(radius))


class MinMaxNormalize(object):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.amin(dim=(1, 2), keepdim=True)
        max_val = tensor.amax(dim=(1, 2), keepdim=True)
        return (tensor - min_val) / (max_val - min_val + 1e-8)


def build_transforms(cfg: Config, image_size: int) -> Dict[str, transforms.Compose]:
    aug_list = []
    if cfg.AUGMENT:
        aug_list = [
            transforms.RandomHorizontalFlip(p=cfg.HFLIP_P),
            transforms.RandomVerticalFlip(p=cfg.VFLIP_P) if cfg.VFLIP_P > 0 else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(
                brightness=cfg.BRIGHTNESS, contrast=cfg.CONTRAST,
                saturation=cfg.SATURATION, hue=cfg.HUE
            ),
            RandomGaussianBlur(p=cfg.BLUR_P),
        ]

    train_tf = transforms.Compose([
        ResizeByLongerSide(image_size),
        CenterPadToSquare(image_size, fill=cfg.PAD_FILL),
        *aug_list,
        transforms.ToTensor(),
        MinMaxNormalize(),
    ])

    valid_tf = transforms.Compose([
        ResizeByLongerSide(image_size),
        CenterPadToSquare(image_size, fill=cfg.PAD_FILL),
        transforms.ToTensor(),
        MinMaxNormalize(),
    ])

    return {"train": train_tf, "valid": valid_tf}


# =============================
# 统一分类头（两层FC）
# =============================
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 2048, num_classes: int = 19, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head_in: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.cls_head = MLPHead(head_in, 2048, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.cls_head(feats)


class Identity(nn.Module):
    def forward(self, x): return x


def _strip_classifier_resnet(net: nn.Module) -> Tuple[nn.Module, int]:
    in_f = net.fc.in_features; net.fc = Identity(); return net, in_f


def _strip_classifier_efficientnet(name: str, net: nn.Module) -> Tuple[nn.Module, int]:
    if hasattr(net, "classifier"):
        if isinstance(net.classifier, nn.Sequential) and len(net.classifier) >= 1:
            last = net.classifier[-1]
            if isinstance(last, nn.Linear):
                in_f = last.in_features; net.classifier[-1] = Identity(); return net, in_f
        elif isinstance(net.classifier, nn.Linear):
            in_f = net.classifier.in_features; net.classifier = Identity(); return net, in_f
    # 兜底（按表）
    table = {
        "efficientnet_b0": 1280, "efficientnet_b1": 1280, "efficientnet_b2": 1408,
        "efficientnet_v2_s": 1280, "efficientnet_v2_m": 1280,
    }
    in_f = table[name]; net.classifier = Identity(); return net, in_f


def _strip_classifier_swin(net: nn.Module) -> Tuple[nn.Module, int]:
    in_f = net.head.in_features; net.head = Identity(); return net, in_f


def _strip_classifier_generic_backbone(net: nn.Module, image_size: int, device: torch.device) -> int:
    # 常见 head 位点
    for attr in ["head", "fc", "classifier"]:
        if hasattr(net, attr):
            mod = getattr(net, attr)
            if isinstance(mod, nn.Linear):
                in_f = mod.in_features
                setattr(net, attr, Identity())
                return in_f
            if hasattr(mod, "in_features"):
                in_f = mod.in_features
                setattr(net, attr, Identity())
                return in_f
    # 兜底：找最后一个 Linear 替换，并做一次前向推断维度
    for name, module in list(net.named_modules())[::-1]:
        if isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            last_name = name.split(".")[-1]
            parent = net.get_submodule(parent_name) if parent_name else net
            setattr(parent, last_name, Identity())
            break
    net.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        y = net(x)
        in_f = y.shape[-1] if y.ndim == 2 else int(np.prod(y.shape[1:]))
    return in_f


# =============================
# MambaVision 构建（官方 API）
# =============================
def _build_mamba_with_api(variant: str, device: torch.device, cfg: Config) -> Tuple[nn.Module, int]:
    """
    variant: 'mambavision_t' / 'mambavision_s'
    使用官方 API: mv_create_model('mamba_vision_T/S', pretrained=..., model_path=...)
    """
    if mv_create_model is None:
        raise ImportError("未安装 mambavision。请先安装该包或从源码安装后再使用 mambavision_t/s。")

    mv_name = "mamba_vision_T" if variant == "mambavision_t" else "mamba_vision_S"
    weight_path = cfg.MAMBA_T_WEIGHTS if variant == "mambavision_t" else cfg.MAMBA_S_WEIGHTS
    use_pretrained = bool(weight_path)

    net = mv_create_model(mv_name, pretrained=use_pretrained, model_path=weight_path)
    net.to(device)

    image_size = get_input_size(variant)
    in_f = _strip_classifier_generic_backbone(net, image_size, device)
    return net, in_f


# =============================
# 创建模型（骨干 + 统一头）
# =============================
def create_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    key = MODEL_ALIASES.get(model_name.lower())
    if key is None:
        raise ValueError(f"不支持的模型: {model_name}")
    image_size = get_input_size(model_name)

    if key == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        net, in_f = _strip_classifier_resnet(net)
        return BackboneWithHead(net.to(device), in_f, num_classes)

    if key.startswith("efficientnet_b"):
        weight_map = {
            "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "efficientnet_b1": models.EfficientNet_B1_Weights.IMAGENET1K_V2,
            "efficientnet_b2": models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        }
        net = getattr(models, key)(weights=weight_map[key])
        net, in_f = _strip_classifier_efficientnet(key, net)
        return BackboneWithHead(net.to(device), in_f, num_classes)

    if key.startswith("efficientnet_v2"):
        weight_map = {
            "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            "efficientnet_v2_m": models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        }
        net = getattr(models, key)(weights=weight_map[key])
        net, in_f = _strip_classifier_efficientnet(key, net)
        return BackboneWithHead(net.to(device), in_f, num_classes)

    if key in ("swin_t", "swin_s", "swin_v2_t", "swin_v2_s"):
        weight_map = {
            "swin_t": models.Swin_T_Weights.IMAGENET1K_V1,
            "swin_s": models.Swin_S_Weights.IMAGENET1K_V1,
            "swin_v2_t": models.Swin_V2_T_Weights.IMAGENET1K_V1,
            "swin_v2_s": models.Swin_V2_S_Weights.IMAGENET1K_V1,
        }
        net = getattr(models, key)(weights=weight_map[key])
        net, in_f = _strip_classifier_swin(net)
        return BackboneWithHead(net.to(device), in_f, num_classes)

    if key in ("mambavision_t", "mambavision_s"):
        net, in_f = _build_mamba_with_api(key, device, cfg_global)
        return BackboneWithHead(net, in_f, num_classes)

    raise ValueError(f"未实现模型: {model_name}")


# =============================
# 分阶段不同学习率的优化器/调度器
# =============================
def _defaults_for_family(cfg: Config) -> Tuple[str, float, float]:
    name = MODEL_ALIASES[cfg.MODEL_NAME.lower()]
    if is_swin_family(name) or is_mamba_family(name):
        return "adamw", 1e-4, 5e-2
    if is_effnet_family(name):
        return "adamw", 1e-3, 1e-2
    if name == "resnet101":
        base_lr = 0.1 * (cfg.BATCH_SIZE / 256)
        return "sgd", base_lr if base_lr > 0 else 0.05, 1e-4
    return "adamw", 1e-3, 1e-4


def _build_param_groups(model: BackboneWithHead, lr_head: float, lr_backbone: float, wd: float, opt_name: str):
    head_params = [p for p in model.cls_head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if opt_name == "sgd":
        optimizer = optim.SGD(
            [{"params": backbone_params, "lr": lr_backbone},
             {"params": head_params, "lr": lr_head}],
            momentum=0.9, weight_decay=wd, nesterov=False
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            [{"params": backbone_params, "lr": lr_backbone},
             {"params": head_params, "lr": lr_head}],
            weight_decay=wd, betas=(0.9, 0.999)
        )
    else:
        optimizer = optim.RMSprop(
            [{"params": backbone_params, "lr": lr_backbone},
             {"params": head_params, "lr": lr_head}],
            alpha=0.9, momentum=0.9, weight_decay=wd
        )
    return optimizer


def build_optimizer_and_scheduler(cfg: Config, model: BackboneWithHead, stage: int):
    opt_family, default_lr, default_wd = _defaults_for_family(cfg)
    wd = default_wd if cfg.WEIGHT_DECAY is None else cfg.WEIGHT_DECAY

    if stage == 1:
        lr_head = cfg.LR_HEAD_STAGE1 if cfg.LR_HEAD_STAGE1 is not None else (10 * default_lr)
        lr_backbone = 0.0
    else:
        lr_head = cfg.LR_HEAD if cfg.LR_HEAD is not None else default_lr
        lr_backbone = cfg.LR_BACKBONE if cfg.LR_BACKBONE is not None else (default_lr * 0.1)

    optimizer = _build_param_groups(model, lr_head, lr_backbone, wd, opt_family)

    # 调度器（默认 cos，全程 T_max=NUM_EPOCHS；如需分段 T_max 可再细化）
    if cfg.SCHEDULER == "none":
        scheduler = None
    else:
        if cfg.SCHEDULER == "multistep" and opt_family == "sgd":
            scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
        else:
            base_lr = cfg.LR if cfg.LR is not None else default_lr
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=base_lr * 0.01)
    return optimizer, scheduler


# =============================
# Dataloader
# =============================
def build_dataloaders(cfg: Config, model_name: str):
    image_size = get_input_size(model_name)
    tfs = build_transforms(cfg, image_size)

    train_set = datasets.ImageFolder(cfg.TRAIN_DIR, transform=tfs["train"])
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)

    val_set = datasets.ImageFolder(cfg.VAL_DIR, transform=tfs["valid"])
    val_loader = DataLoader(val_set, batch_size=max(32, cfg.BATCH_SIZE), shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

    print(f"[Data] InputSize={image_size} | Train={len(train_set)} | Val={len(val_set)} | AUGMENT={cfg.AUGMENT}")
    return train_loader, val_loader, image_size


# =============================
# 训练 / 验证
# =============================
def make_criterion(cfg: Config):
    if is_swin_family(cfg.MODEL_NAME):
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    return nn.CrossEntropyLoss()


def set_backbone_trainable(model: BackboneWithHead, train_backbone: bool, freeze_bn: bool = True):
    for p in model.backbone.parameters():
        p.requires_grad = train_backbone
    for p in model.cls_head.parameters():
        p.requires_grad = True
    if freeze_bn:
        def _set_bn_eval(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                m.eval()
        if not train_backbone:
            model.backbone.apply(_set_bn_eval)


def train_one_epoch(cfg: Config, model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0; total = 0; correct1 = 0
    pbar = tqdm(loader, desc="Train", ncols=100, leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if cfg.USE_AMP:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if is_swin_family(cfg.MODEL_NAME) or is_mamba_family(cfg.MODEL_NAME):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = criterion(logits, y)
            loss.backward()
            if is_swin_family(cfg.MODEL_NAME) or is_mamba_family(cfg.MODEL_NAME):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct1 += (pred.eq(y)).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=f"{running_loss/max(1,total):.4f}",
                         acc=f"{correct1/max(1,total)*100:.2f}%",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return running_loss / max(1, total), correct1 / max(1, total) * 100.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str):
    model.eval()
    running_loss = 0.0; total = 0; top1 = 0.0; top5 = 0.0
    pbar = tqdm(loader, desc="Valid", ncols=100, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)

        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)

        _, pred = outputs.topk(5, 1, True, True)
        total += targets.size(0)
        top1 += (pred[:, :1].squeeze(1) == targets).float().sum().item()
        top5 += (pred == targets.unsqueeze(1)).any(dim=1).float().sum().item()

        pbar.set_postfix(loss=f"{running_loss/max(1,total):.4f}",
                         top1=f"{top1/max(1,total)*100:.2f}%",
                         top5=f"{top5/max(1,total)*100:.2f}%")
    return running_loss / max(1, total), top1 / max(1, total) * 100.0, top5 / max(1, total) * 100.0


# =============================
# 保存
# =============================
def _name_key(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name.lower(), model_name.lower())


def save_checkpoint(cfg, epoch, model, optimizer, scheduler, scaler, best_top1,
                    train_loss=None, val_loss=None):
    name = _name_key(cfg.MODEL_NAME)
    ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'scaler_state': scaler.state_dict() if scaler else None,
        'best_top1': best_top1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'cfg': asdict(cfg),
    }, ckpt_path)


def save_best_weights(cfg, model, best_top1, train_loss=None, val_loss=None):
    name = _name_key(cfg.MODEL_NAME)
    os.makedirs(cfg.BEST_DIR, exist_ok=True)
    best_path = os.path.join(cfg.BEST_DIR, f"{name}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'best_top1': best_top1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'cfg': asdict(cfg),
    }, best_path)


# =============================
# 主流程
# =============================
def main(cfg: Config):
    cfg.ensure()
    device = torch.device(f"cuda:{cfg.DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader, image_size = build_dataloaders(cfg, cfg.MODEL_NAME)


    # 修改为（新增一行）
    model = create_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, device)
    model.to(device)  # ★ 关键：把 cls_head 一起搬到 GPU
    print(f"[Model] {cfg.MODEL_NAME} 构建完成，输入尺寸={image_size}")


    scaler = GradScaler(enabled=cfg.USE_AMP)
    criterion = make_criterion(cfg)
    best_top1 = 0.0

    # Stage-1：只训练头
    if cfg.STAGED_TRAIN:
        set_backbone_trainable(model, train_backbone=False, freeze_bn=cfg.FREEZE_BN)
        stage = 1
    else:
        set_backbone_trainable(model, train_backbone=True, freeze_bn=False)
        stage = 2

    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, stage=stage)

    print("Training Start")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        if cfg.STAGED_TRAIN and epoch == (cfg.STAGE1_EPOCHS + 1) and stage == 1:
            print(f"\n[Stage Switch] 解冻骨干，从第 {epoch} 个 epoch 开始全模型训练（差异LR：head/backbone）")
            set_backbone_trainable(model, train_backbone=True, freeze_bn=False)
            stage = 2
            optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, stage=2)

        print(f"\n===== Epoch {epoch}/{cfg.NUM_EPOCHS} "
              f"{'(HeadOnly)' if stage==1 else '(Full)'} | AUG={cfg.AUGMENT} =====")

        train_loss, train_acc1 = train_one_epoch(cfg, model, train_loader, criterion, optimizer, scaler, device)
        if scheduler is not None:
            scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)

        print(f"[Train] loss={train_loss:.4f} acc@1={train_acc1:.2f}%")
        print(f"[Eval ] loss={val_loss:.4f} acc@1={val_acc1:.2f}% acc@5={val_acc5:.2f}%")

        save_checkpoint(cfg, epoch, model, optimizer, scheduler, scaler, best_top1, train_loss, val_loss)
        if val_acc1 > best_top1:
            best_top1 = val_acc1
            save_best_weights(cfg, model, best_top1, train_loss, val_loss)
            print(f"[Best Model] New best acc@1={best_top1:.2f}% saved!")

    print(f"\n训练完成 | 最佳 acc@1={best_top1:.2f}% | 最佳权重: "
          f"{os.path.join(cfg.BEST_DIR, _name_key(cfg.MODEL_NAME) + '.pt')}")


# 全局 cfg（用于 Mamba 权重路径）
cfg_global: Config

if __name__ == "__main__":
    cfg = Config(
        TRAIN_DIR="Train",
        VAL_DIR="Val",
        MODEL_NAME="efficientnet_v2_m",  # 可选：mambavision_t / mambavision_s / 其它支持模型
        NUM_CLASSES=19,
        BATCH_SIZE=16,
        NUM_EPOCHS=50,
        USE_AMP=False,

        # 数据增强
        AUGMENT=True,

        # 分阶段 + 不同LR
        STAGED_TRAIN=False,
        STAGE1_EPOCHS=10,
        FREEZE_BN=True,
        LR_HEAD_STAGE1=None,   # 若不填：默认家族lr的 10x
        LR_HEAD=None,          # 若不填：默认家族lr
        LR_BACKBONE=None,      # 若不填：默认家族lr的 0.1x

        # Mamba 权重（如有）
        MAMBA_T_WEIGHTS="utils_model_checkpoint\mambavision_tiny_1k.pth.tar",  # 可留空或填目录
        MAMBA_S_WEIGHTS="utils_model_checkpoint\mambavision_small_1k.pth.tar",
    )
    cfg_global = cfg
    main(cfg)
