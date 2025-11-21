

import cv2
import time, math, random
import numpy as np
from PIL import Image
from pathlib import Path
import sys, site
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ==============================
# 配置
# ==============================
class Config:
    # I/O
    INPUT_DIR  = "Test_Raw_Data"
    OUTPUT_DIR = "Test_SAM"
    PRESERVE_FOLDER_STRUCTURE = True

    # 代码1（补集法）参数
    PADDING_RATIO = 0.01           # 宽边带 = 距图像边缘向内 1%
    BG_LINE_RATIO_THRESH = 0.40    # 在边带上的覆盖率阈值（≥ 两条边）
    MIN_COMPONENT_AREA = 256       # 候选连通域最小像素
    COMPLEMENT_DILATE_RATIO = 0.005  # 补集膨胀半径 = 短边 0.5%

    # 复查边带（贴边必须触达“更厚的review边带”）
    REVIEW_BAND_SCALE = 1.5
    REVIEW_BAND_MIN_ADD = 2

    # 裁剪外扩比例
    CROP_PAD_RATIO = 0.02

    # SAM
    MAX_SIZE = 1024
    MODEL_TYPE = "vit_h"
    CHECKPOINT = "utils_model_checkpoint\sam_vit_h_4b8939.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    POINTS_PER_SIDE = 8
    PRED_IOU_THRESH = 0.8
    STABILITY_SCORE_THRESH = 0.9
    CROP_N_LAYERS = 1
    CROP_N_POINTS_DOWN = 2
    MIN_MASK_REGION_AREA = 0
    OUTPUT_MASKS = True

    # 最终复查（阈值）
    FINAL_TWO_EDGES_RATIO = 0.40   # ≥ 两条边各>=40%
    FINAL_ONE_EDGE_RATIO  = 0.70   # 或 任意一条边>=70%


# ==============================
# 主体
# ==============================
class WormSelector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.OUTPUT_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(123)
        self._load_sam()

    # ---------- SAM ----------
    def _load_sam(self):
        ckpt = Path(self.cfg.CHECKPOINT)
        assert ckpt.exists(), f"SAM 权重不存在: {ckpt}"
        sam = sam_model_registry[self.cfg.MODEL_TYPE](checkpoint=str(ckpt))
        sam.to(self.cfg.DEVICE)
        self._mask_gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.cfg.POINTS_PER_SIDE,
            pred_iou_thresh=self.cfg.PRED_IOU_THRESH,
            stability_score_thresh=self.cfg.STABILITY_SCORE_THRESH,
            crop_n_layers=self.cfg.CROP_N_LAYERS,
            crop_n_points_downscale_factor=self.cfg.CROP_N_POINTS_DOWN,
            min_mask_region_area=self.cfg.MIN_MASK_REGION_AREA,
            output_mode="binary_mask" if self.cfg.OUTPUT_MASKS else "uncompressed_rle",
        )

    # ---------- I/O ----------
    def _load_image(self, path: Path):
        try:
            with Image.open(path) as im:
                if im.mode != "RGB": im = im.convert("RGB")
                return np.array(im, dtype=np.uint8)
        except Exception:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    def _resize_if_needed(self, img):
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= self.cfg.MAX_SIZE:
            return img, 1.0
        s = self.cfg.MAX_SIZE / float(m)
        new = (int(round(w * s)), int(round(h * s)))
        small = np.array(Image.fromarray(img).resize(new, Image.Resampling.LANCZOS))
        return small, s

    def _scale_masks_back(self, masks, s, shape):
        if s == 1.0:
            return masks
        oh, ow = shape[:2]
        out = []
        for mi in masks:
            m = cv2.resize(mi["segmentation"].astype(np.uint8), (ow, oh),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            ni = mi.copy()
            ni["segmentation"] = m
            out.append(ni)
        return out

    # ---------- 工具 ----------
    def _build_padding_band(self, h, w, pad_px):
        band = np.zeros((h, w), dtype=bool)
        band[:pad_px, :] = True
        band[-pad_px:, :] = True
        band[:, :pad_px] = True
        band[:, -pad_px:] = True
        return band

    def _edge_length_cover_ratios(self, mask: np.ndarray, H: int, W: int, review_pad: int):
        """
        用“线长占比”评估四边覆盖率（行/列是否被占用）。
        """
        m = mask.astype(bool)
        k = int(review_pad)

        top_hit    = m[:k, :]
        bottom_hit = m[-k:, :]
        left_hit   = m[:, :k]
        right_hit  = m[:, -k:]

        top_ratio    = np.any(top_hit,    axis=0).sum() / max(1, W)
        bottom_ratio = np.any(bottom_hit, axis=0).sum() / max(1, W)
        left_ratio   = np.any(left_hit,   axis=1).sum() / max(1, H)
        right_ratio  = np.any(right_hit,  axis=1).sum() / max(1, H)

        return {
            'top': top_ratio, 'bottom': bottom_ratio,
            'left': left_ratio, 'right': right_ratio
        }

    # ---------- 代码1：补集膨胀 + 边带 → 候选 ----------
    def _candidate_code1(self, masks, H, W):
        union = np.zeros((H, W), dtype=bool)
        for m in masks:
            union |= m["segmentation"].astype(bool)

        rad = max(1, int(round(self.cfg.COMPLEMENT_DILATE_RATIO * min(H, W))))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad+1, 2*rad+1))
        complement_thick = cv2.dilate((~union).astype(np.uint8), ker, 1).astype(bool)

        pad_px = max(1, int(round(self.cfg.PADDING_RATIO * min(H, W))))
        border_band = self._build_padding_band(H, W, pad_px)
        complement_cut = complement_thick | border_band

        inside = ~complement_cut
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            inside.astype(np.uint8), connectivity=8)
        if num <= 1:
            return None

        comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)
                 if stats[i, cv2.CC_STAT_AREA] >= self.cfg.MIN_COMPONENT_AREA]
        if not comps:
            return None
        comps.sort(key=lambda x: x[1], reverse=True)
        top2 = [c[0] for c in comps[:2]]

        thr = self.cfg.BG_LINE_RATIO_THRESH
        def cover(mask_bool):
            return {
                'top':    np.count_nonzero(mask_bool[:pad_px, :])  / max(1, W),
                'bottom': np.count_nonzero(mask_bool[-pad_px:, :]) / max(1, W),
                'left':   np.count_nonzero(mask_bool[:, :pad_px])  / max(1, H),
                'right':  np.count_nonzero(mask_bool[:, -pad_px:]) / max(1, H),
            }
        def is_bg(cov):
            return sum(v >= thr for v in cov.values()) >= 2

        if len(top2) == 1:
            return (labels == top2[0])

        a = (labels == top2[0]); b = (labels == top2[1])
        ca, cb = cover(a), cover(b)
        a_bg, b_bg = is_bg(ca), is_bg(cb)
        if a_bg and not b_bg: return b
        if b_bg and not a_bg: return a
        if a_bg and b_bg:
            cnt_a = sum(v >= thr for v in ca.values())
            cnt_b = sum(v >= thr for v in cb.values())
            if cnt_a != cnt_b:
                return (a if cnt_a < cnt_b else b)
            return (a if a.sum() < b.sum() else b)
        return (a if a.sum() < b.sum() else b)

    # ---------- 最终复查 ----------
    def _final_review_should_use_original(self, mask: np.ndarray, H: int, W: int) -> bool:
        area = np.count_nonzero(mask)
        img_area = H * W
        if area < self.cfg.MIN_COMPONENT_AREA:   # 太小
            return True
        if area > 0.9 * img_area:                # 几乎覆盖整图
            return True

        pad_px = max(1, int(round(self.cfg.PADDING_RATIO * min(H, W))))
        review_pad = max(
            pad_px + int(self.cfg.REVIEW_BAND_MIN_ADD),
            int(round(pad_px * float(self.cfg.REVIEW_BAND_SCALE)))
        )

        review_band = self._build_padding_band(H, W, review_pad)
        if not np.any(mask & review_band):
            return True

        len_cov = self._edge_length_cover_ratios(mask, H, W, review_pad)
        if any(v >= self.cfg.FINAL_ONE_EDGE_RATIO for v in len_cov.values()):
            return True
        if sum(v >= self.cfg.FINAL_TWO_EDGES_RATIO for v in len_cov.values()) >= 2:
            return True
        return False

    # ---------- 处理目录 ----------
    def process_dir(self, inp_dir):
        inp = Path(inp_dir)
        assert inp.exists(), f"输入目录不存在：{inp}"
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = [p for p in inp.rglob("*") if p.suffix.lower() in exts]
        print(f"[INFO] 待处理 {len(images)} 张图")
        if images:
            stat = {}
            for p in images:
                key = p.suffix.lower()
                stat[key] = stat.get(key, 0) + 1
            detail = ", ".join(f"{k}:{v}" for k, v in sorted(stat.items()))
            print(f"[INFO] 类型统计 {detail}")

        for i, p in enumerate(images, 1):
            rel = p.relative_to(inp) if self.cfg.PRESERVE_FOLDER_STRUCTURE else Path(p.name)
            out_path = (self.out_dir / rel).with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{i}/{len(images)}] {p}")
            self._process_one(p, out_path)

    # ---------- 单图 ----------
    def _process_one(self, path: Path, out_path: Path):
        img = self._load_image(path)
        if img is None:
            print(f"[WARN] 读图失败: {path}")
            return
        H, W = img.shape[:2]
        small, s = self._resize_if_needed(img)

        try:
            masks = self._mask_gen.generate(small)
        except Exception as e:
            print(f"[ERROR] SAM 失败: {path}, {e}")
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return

        masks = self._scale_masks_back(masks, s, img.shape)

        mask_c1 = self._candidate_code1(masks, H, W)

        if mask_c1 is None or not mask_c1.any():
            print(f"[INFO] 无候选，保存原图: {path}")
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return

        if self._final_review_should_use_original(mask_c1, H, W):
            print(f"[INFO] 最终复查失败 → 保存原图: {path}")
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return

        # ======= 通过复查 → 裁剪保存 =======
        x, y, w, h = cv2.boundingRect(mask_c1.astype(np.uint8))
        pad = int(round(self.cfg.CROP_PAD_RATIO * min(H, W)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        crop = img[y1:y2, x1:x2]

        cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"[OK] 裁剪保存: {out_path}")


# ==============================
# 入口
# ==============================
def main():
    cfg = Config()
    app = WormSelector(cfg)
    app.process_dir(cfg.INPUT_DIR)

if __name__ == "__main__":
    main()
