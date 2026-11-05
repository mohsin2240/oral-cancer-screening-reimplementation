"""Offline SAM masks for the segmentation-based augmentation (Section II-C.2).

The paper runs SAM in "everything" mode because there are no lesion boxes, and keeps coarse masks that remove
non-diagnostic regions. Heuristic used here: keep the union of SAM segments that cover between 1% and 90% of the image
and whose centroid lies in the central 70% of the frame (where the oral cavity is framed), then fill holes with a
morphological close. If the union is under 15% of the image the mask falls back to the whole image.
Training-only; inference never uses SAM.

Needs the ViT-B checkpoint (sam_vit_b_01ec64.pth, ~375 MB):
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  ->  checkpoints/
Usage: python src/sam_masks.py --manifest data/manifest.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


def select_mask(masks, h, w):
    union = np.zeros((h, w), bool)
    for m in masks:
        frac = m["area"] / (h * w)
        if not 0.01 <= frac <= 0.9:
            continue
        ys, xs = np.nonzero(m["segmentation"])
        cy, cx = ys.mean() / h, xs.mean() / w
        if 0.15 <= cy <= 0.85 and 0.15 <= cx <= 0.85:
            union |= m["segmentation"]
    if union.mean() < 0.15:
        return np.ones((h, w), bool)
    k = max(3, int(0.03 * max(h, w))) | 1
    return cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    ap.add_argument("--out", type=Path, default=Path("data/sam_masks"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.9)
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    coverage = []
    for i, p in tqdm(zip(df["id"], df["proc_path"]), total=len(df)):
        dst = args.out / f"{i:06d}.png"
        img = np.asarray(Image.open(args.data_root / p).convert("RGB"))
        if not dst.exists():
            mask = select_mask(gen.generate(img), *img.shape[:2])
            Image.fromarray(mask.astype(np.uint8) * 255).save(dst)
        coverage.append((np.asarray(Image.open(dst)) > 127).mean())
    print(f"mask coverage: mean {np.mean(coverage):.2f}, whole-image fallbacks {np.mean(np.array(coverage) > 0.999):.0%}")


if __name__ == "__main__":
    main()
