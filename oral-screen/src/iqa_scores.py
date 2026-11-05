"""Precompute no-reference image-quality scores for every manifest image (Section II-C.3).

NIQE: lower = better quality. CLIP-IQA: higher = better. Both via pyiqa, on the cached processed images.
Writes data/iqa_scores.csv with columns id, niqe, clipiqa.

Usage: python src/iqa_scores.py --manifest data/manifest.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


def iqa_weights(scores: pd.DataFrame, ids, metric: str):
    """Per-sample loss weights, normalised to mean 1 over `ids`.
    NIQE: inverse score (down-weights low-quality images). CLIP-IQA: the score itself."""
    s = scores.set_index("id").loc[list(ids), metric].astype(float)
    w = 1.0 / s.clip(lower=1e-3) if metric == "niqe" else s.clip(lower=1e-3)
    return (w / w.mean()).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/iqa_scores.csv"))
    args = ap.parse_args()

    import pyiqa  # only needed to compute scores; train.py just reads the CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = {m: pyiqa.create_metric(m, device=device) for m in ["niqe", "clipiqa"]}
    df = pd.read_csv(args.manifest)
    rows = []
    for i, p in tqdm(zip(df["id"], df["proc_path"]), total=len(df)):
        img = Image.open(args.data_root / p).convert("RGB")
        if min(img.size) < 192:  # NIQE works on 96-px blocks and fails on very small images
            s = 192 / min(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.BICUBIC)
        x = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            rows.append({"id": i, **{m: float(f(x)) for m, f in metrics.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.describe())


if __name__ == "__main__":
    main()
