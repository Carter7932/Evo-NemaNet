

"""
Eval_All_Models (CKPT 目录批量评测版 + 记录最优)
- 从指定 CKPT_DIR（可选递归）读取所有 .pt 权重，逐个在测试集上评估
- 仅记录：Precision、Recall、Accuracy（多分类）
- 产物：
  1) OUTPUT_DIR/summary_overall.csv（每个“权重文件”一行：Overall Precision/Recall/Accuracy）
  2) OUTPUT_DIR/best_overall.csv（全局最优；优先 Accuracy，P/R 作为次级tie-break）
  3) OUTPUT_DIR/best_per_model.csv（每个 model_key 的最优结果）
  4) OUTPUT_DIR/<model_key>__<weight_stem>/
        ├─ per_class_metrics.csv（各类别 + Overall 的 P/R/Acc）
        └─ confusion_matrix.csv（混淆矩阵，含类名表头）
说明：
- Overall Precision/Recall = 宏平均（macro-avg）
- Overall Accuracy = 全局准确率（micro-avg）
"""

import os, re, json, random, warnings, glob
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# =============================
# 配置（已按你的要求填写）
# =============================
@dataclass
class Config:
    TEST_DIR: str = "Test_SAM"  # 测试集（ImageFolder 结构）
    CKPT_DIR: str = "/root/autodl-tmp/I-Nema/Model_CheckPoint/efficientnet_v2_m"
    OUTPUT_DIR: str = "Eval_Outputs_SAM_EfficientNetV2"

    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 2
    DEVICE_ID: int = 0
    NUM_CLASSES: int = 19
    SEED: int = 42

    RECURSIVE: bool = True
    FORCE_MODEL_NAME: Optional[str] = "efficientnet_v2_m"  # 优先使用该模型名
    ONLY_MODELS: List[str] = None  # 留空=不做过滤

    # MambaVision 预训练权重（若在 create_model 中需要）
    MAMBA_T_WEIGHTS: Optional[str] = None
    MAMBA_S_WEIGHTS: Optional[str] = None

    PAD_FILL: tuple = (128, 128, 128)

    def ensure(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
        print("[Eval Config]", {k: v for k, v in asdict(self).items()
                               if k not in ("PAD_FILL", "MAMBA_T_WEIGHTS", "MAMBA_S_WEIGHTS")})

cfg = Config()

# =============================
# 与训练脚本一致的模型别名/输入尺寸
# =============================
MODEL_ALIASES = {
    "resnet101": "resnet101",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_v2_s": "efficientnet_v2_s",
    "efficientnet_v2_m": "efficientnet_v2_m",
    "swin_t": "swin_t",
    "swin_s": "swin_s",
    "swin_v2_t": "swin_v2_t",
    "swin_v2_s": "swin_v2_s",
    "mambavision_t": "mambavision_t",
    "mambavision_s": "mambavision_s",
}
MODEL_DEFAULT_SIZE = {
    "resnet101": 224,
    "efficientnet_b0": 224, "efficientnet_b1": 240, "efficientnet_b2": 260,
    "efficientnet_v2_s": 384, "efficientnet_v2_m": 480,
    "swin_t": 224, "swin_s": 224,
    "swin_v2_t": 256, "swin_v2_s": 256,
    "mambavision_t": 256, "mambavision_s": 256,
}

def get_input_size(model_name: str) -> int:
    key = MODEL_ALIASES.get(model_name.lower())
    if key is None or key not in MODEL_DEFAULT_SIZE:
        raise ValueError(f"未配置默认输入尺寸: {model_name}")
    return MODEL_DEFAULT_SIZE[key]

def _name_key(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name.lower(), model_name.lower())

def is_swin_family(n: str) -> bool: return n.lower().startswith("swin")
def is_effnet_family(n: str) -> bool: return n.lower().startswith("efficientnet")
def is_mamba_family(n: str) -> bool: return n.lower().startswith("mambavision")

# =============================
# 预处理（与训练一致）
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
        top  = (self.size - h) // 2
        canvas.paste(img, (left, top))
        return canvas

class MinMaxNormalize(object):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.amin(dim=(1, 2), keepdim=True)
        max_val = tensor.amax(dim=(1, 2), keepdim=True)
        return (tensor - min_val) / (max_val - min_val + 1e-8)

def build_eval_transform(image_size: int):
    return transforms.Compose([
        ResizeByLongerSide(image_size),
        CenterPadToSquare(image_size, fill=cfg.PAD_FILL),
        transforms.ToTensor(),
        MinMaxNormalize(),
    ])

# =============================
# MambaVision（若需要）
# =============================
try:
    from mambavision import create_model as mv_create_model
except Exception:
    mv_create_model = None

def _build_mamba_backbone(variant: str, device: torch.device):
    if mv_create_model is None:
        raise ImportError("未安装 mambavision，无法构建 MambaVision 模型")
    mv_name = "mamba_vision_T" if variant == "mambavision_t" else "mamba_vision_S"
    weight_path = cfg.MAMBA_T_WEIGHTS if variant == "mambavision_t" else cfg.MAMBA_S_WEIGHTS
    net = mv_create_model(mv_name, pretrained=bool(weight_path), model_path=weight_path)
    net.to(device)
    return net

# =============================
# 统一分类头（两层FC）
# =============================
class Identity(nn.Module):
    def forward(self, x): return x

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, p: float = 0.2):
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
    table = {
        "efficientnet_b0": 1280, "efficientnet_b1": 1280, "efficientnet_b2": 1408,
        "efficientnet_v2_s": 1280, "efficientnet_v2_m": 1280,
    }
    in_f = table[name]; net.classifier = Identity(); return net, in_f

def _strip_classifier_swin(net: nn.Module) -> Tuple[nn.Module, int]:
    in_f = net.head.in_features; net.head = Identity(); return net, in_f

def _strip_classifier_generic_backbone(net: nn.Module, image_size: int, device: torch.device) -> int:
    for attr in ["head", "fc", "classifier"]:
        if hasattr(net, attr):
            mod = getattr(net, attr)
            if isinstance(mod, nn.Linear):
                in_f = mod.in_features
                setattr(net, attr, Identity()); return in_f
            if hasattr(mod, "in_features"):
                in_f = mod.in_features
                setattr(net, attr, Identity()); return in_f
    net.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        y = net(x)
        in_f = y.shape[-1] if y.ndim == 2 else int(np.prod(y.shape[1:]))
    return in_f

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
        net = _build_mamba_backbone(key, device)
        in_f = _strip_classifier_generic_backbone(net, image_size, device)
        return BackboneWithHead(net, in_f, num_classes)

    raise ValueError(f"未实现模型: {model_name}")

# =============================
# 数据
# =============================
def build_test_loader(model_name: str):
    image_size = get_input_size(model_name)
    tfm = build_eval_transform(image_size)
    test_set = datasets.ImageFolder(cfg.TEST_DIR, transform=tfm)
    loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)
    return loader, test_set.classes

# =============================
# 评估工具
# =============================
def compute_confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_metrics_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall    = np.zeros(num_classes, dtype=np.float64)
    acc_cls   = np.zeros(num_classes, dtype=np.float64)
    total = cm.sum()
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = total - TP - FP - FN
        precision[i] = TP / (TP + FP + 1e-12)
        recall[i]    = TP / (TP + FN + 1e-12)
        acc_cls[i]   = (TP + TN) / (total + 1e-12)
    return precision, recall, acc_cls

def overall_from_per_class(prec: np.ndarray, rec: np.ndarray, cm: np.ndarray) -> Tuple[float, float, float]:
    macro_p = float(np.mean(prec))
    macro_r = float(np.mean(rec))
    overall_acc = float(np.trace(cm) / (cm.sum() + 1e-12))
    return macro_p, macro_r, overall_acc

# =============================
# 权重 & 模型名解析
# =============================
MODEL_KEY_PATTERN = re.compile(r"(resnet101|efficientnet_b[0-2]|efficientnet_v2_[sm]|swin_(?:t|s)|swin_v2_(?:t|s)|mambavision_(?:t|s))", re.I)
EPOCH_NUM_PATTERN = re.compile(r"epoch[_\-]?(\d+)", re.I)

def guess_model_name_from_string(s: str) -> Optional[str]:
    m = MODEL_KEY_PATTERN.search(os.path.basename(s))
    if m:
        key = m.group(1).lower()
        return MODEL_ALIASES.get(key, key)
    return None

def guess_model_name(weight_path: str) -> Optional[str]:
    # 1) 从文件名
    nm = guess_model_name_from_string(weight_path)
    if nm: return nm
    # 2) 从上级目录名
    parent = os.path.basename(os.path.dirname(weight_path))
    nm = guess_model_name_from_string(parent)
    if nm: return nm
    return None

def get_epoch_from_path(p: str) -> Optional[int]:
    m = EPOCH_NUM_PATTERN.search(os.path.basename(p))
    return int(m.group(1)) if m else None

def list_weight_files() -> List[str]:
    pattern = "**/*.pt" if cfg.RECURSIVE else "*.pt"
    all_files = glob.glob(os.path.join(cfg.CKPT_DIR, pattern), recursive=cfg.RECURSIVE)
    all_files = [f for f in all_files if os.path.isfile(f)]

    # 仅保留 ONLY_MODELS（若设置）
    if cfg.ONLY_MODELS:
        keys = set(m.lower() for m in cfg.ONLY_MODELS)
        keep = []
        for f in all_files:
            nm = guess_model_name(f)
            if nm and nm.lower() in keys:
                keep.append(f)
        all_files = keep

    # 自然序排序：优先识别 epoch_###，否则按文件名排序
    def sort_key(p):
        b = os.path.basename(p)
        m = EPOCH_NUM_PATTERN.search(b)
        if m:
            return (0, int(m.group(1)))
        return (1, b.lower())

    all_files = sorted(all_files, key=sort_key)
    return all_files

# =============================
# 主评估逻辑
# =============================
@torch.no_grad()
def evaluate_one_model(weight_path: str) -> Dict:
    device = torch.device(f"cuda:{cfg.DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")
    payload = torch.load(weight_path, map_location=device)

    # 确定模型名 & 类别数
    model_name = cfg.FORCE_MODEL_NAME if cfg.FORCE_MODEL_NAME else None
    num_classes = cfg.NUM_CLASSES

    saved_cfg = payload.get("cfg", None)
    if model_name is None and isinstance(saved_cfg, dict):
        model_name = saved_cfg.get("MODEL_NAME", None)
        num_classes = int(saved_cfg.get("NUM_CLASSES", num_classes))

    if model_name is None:
        model_name = guess_model_name(weight_path)

    if model_name is None:
        raise ValueError(f"无法确定模型名（建议设置 cfg.FORCE_MODEL_NAME）。文件：{weight_path}")

    # 数据
    loader, class_names = build_test_loader(model_name)
    if len(class_names) != num_classes:
        print(f"[Warn] 类别数不一致：数据集={len(class_names)}，权重/配置={num_classes}。以数据集为准。")
        num_classes = len(class_names)

    # 构建模型并加载权重
    model = create_model(model_name, num_classes, device).to(device)
    state = payload.get("model_state", None)
    if state is None:
        if "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")

    # 推理
    y_true_all, y_pred_all = [], []
    model.eval()
    pbar = tqdm(loader, desc=f"Eval[{_name_key(model_name)}|{os.path.basename(weight_path)}]", ncols=110, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred_all.append(preds)
        y_true_all.append(targets.numpy())
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    # 混淆矩阵 & 指标
    cm = compute_confusion_matrix(num_classes, y_true, y_pred)
    prec, rec, acc_cls = per_class_metrics_from_cm(cm)
    oP, oR, oA = overall_from_per_class(prec, rec, cm)

    run_key = f"{_name_key(model_name)}__{os.path.splitext(os.path.basename(weight_path))[0]}"
    epoch_num = get_epoch_from_path(weight_path)

    return {
        "model_key": _name_key(model_name),
        "weight_file": os.path.basename(weight_path),
        "run_key": run_key,
        "epoch": epoch_num,
        "class_names": class_names,
        "cm": cm,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_accuracy": acc_cls,
        "overall_precision": oP,
        "overall_recall": oR,
        "overall_accuracy": oA,
    }

def save_model_reports(res: Dict):
    # 每个权重单独一个目录，避免覆盖
    out_dir = os.path.join(cfg.OUTPUT_DIR, res["run_key"])
    os.makedirs(out_dir, exist_ok=True)

    # 1) per_class_metrics.csv
    per_cls_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(per_cls_path, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,accuracy\n")
        for i, name in enumerate(res["class_names"]):
            f.write(f"{name},{res['per_class_precision'][i]:.6f},{res['per_class_recall'][i]:.6f},{res['per_class_accuracy'][i]:.6f}\n")
        f.write(f"Overall,{res['overall_precision']:.6f},{res['overall_recall']:.6f},{res['overall_accuracy']:.6f}\n")

    # 2) confusion_matrix.csv（带行列头）
    cm_path = os.path.join(out_dir, "confusion_matrix.csv")
    cls = res["class_names"]
    cm = res["cm"]
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("label/" + ",".join(cls) + "\n")
        for i, row in enumerate(cm):
            row_str = ",".join(str(int(v)) for v in row.tolist())
            f.write(f"{cls[i]},{row_str}\n")

    # 3) 类名顺序（可选）
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(cls, f, ensure_ascii=False, indent=2)

def append_summary_row(summary_rows: List[str], res: Dict):
    # summary 里带上具体的权重文件名和 epoch
    epoch_str = "" if res["epoch"] is None else res["epoch"]
    summary_rows.append("{},{},{},{:.6f},{:.6f},{:.6f}".format(
        res["model_key"], res["weight_file"], epoch_str,
        res["overall_precision"], res["overall_recall"], res["overall_accuracy"]
    ))

def main():
    cfg.ensure()
    weight_files = list_weight_files()
    if not weight_files:
        print(f"[Error] 在 {cfg.CKPT_DIR} 未找到 .pt 权重文件（RECURSIVE={cfg.RECURSIVE}）")
        return

    summary_rows = ["model,weight_file,epoch,overall_precision,overall_recall,overall_accuracy"]
    all_results = []  # 收集全部结果用于最优统计

    for w in weight_files:
        print("\n" + "="*100)
        print(f"▶ 评估权重：{w}")
        print("="*100)
        try:
            res = evaluate_one_model(w)
            save_model_reports(res)
            append_summary_row(summary_rows, res)
            all_results.append(res)
        except Exception as e:
            print(f"[Skip] 评估失败：{repr(e)}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- 写 summary_overall.csv ----
    summary_path = os.path.join(cfg.OUTPUT_DIR, "summary_overall.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")
    print(f"\n[Done] 总表已保存：{summary_path}")

    # ---- 统计最优（Accuracy 优先，P/R 次级排序）----
    if all_results:
        def best_key(r):
            return (r["overall_accuracy"], r["overall_precision"], r["overall_recall"])

        # 全局最优
        best_global = max(all_results, key=best_key)
        best_global_csv = os.path.join(cfg.OUTPUT_DIR, "best_overall.csv")
        with open(best_global_csv, "w", encoding="utf-8") as f:
            f.write("model,weight_file,epoch,overall_precision,overall_recall,overall_accuracy\n")
            f.write("{},{},{},{:.6f},{:.6f},{:.6f}\n".format(
                best_global["model_key"],
                best_global["weight_file"],
                "" if best_global["epoch"] is None else best_global["epoch"],
                best_global["overall_precision"],
                best_global["overall_recall"],
                best_global["overall_accuracy"],
            ))
        print(f"[Best] 全局最优已保存：{best_global_csv}")

        # 各 model_key 各自最优
        best_per_model = {}
        for r in all_results:
            k = r["model_key"]
            if (k not in best_per_model) or (best_key(r) > best_key(best_per_model[k])):
                best_per_model[k] = r

        best_per_model_csv = os.path.join(cfg.OUTPUT_DIR, "best_per_model.csv")
        with open(best_per_model_csv, "w", encoding="utf-8") as f:
            f.write("model,weight_file,epoch,overall_precision,overall_recall,overall_accuracy\n")
            for k, r in sorted(best_per_model.items()):
                f.write("{},{},{},{:.6f},{:.6f},{:.6f}\n".format(
                    r["model_key"],
                    r["weight_file"],
                    "" if r["epoch"] is None else r["epoch"],
                    r["overall_precision"],
                    r["overall_recall"],
                    r["overall_accuracy"],
                ))
        print(f"[Best] 各模型最优已保存：{best_per_model_csv}")

if __name__ == "__main__":
    main()
