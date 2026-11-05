"""Hardware-noise stress tests on the test split (Section V, Table IX).

Noise is applied to the full-resolution image (our cached copies, longest side 640) before the resize to the model
input, as the paper does on raw images. Intensities are percentages:
  gaussian  - additive N(0, sigma) with sigma = p% of the [0, 1] range (low-light sensor noise)
  stripe    - p% of image columns get a constant offset ~ U(-0.3, 0.3) (read-out stripe artefacts)
  saltpepper- p% of pixels set to 0 or 1 with equal probability (stuck pixels / bit errors)
Noise RNG is seeded per image so every model sees identical corruptions.

Usage:
  python src/robustness.py --run-dir results/manifest_train5/mobilevitv2_geometric_wce --manifest data/manifest_train5.csv
(the run must have been trained with --keep-ckpt)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import OralDataset, build_transform, load_manifest
from models import build_model
from train import compute_metrics, predict

LEVELS = {"none": [0], "gaussian": [15, 20, 25], "stripe": [2, 5, 7], "saltpepper": [2, 5, 7]}


class Noise(torch.nn.Module):
    def __init__(self, kind: str, pct: float, seed: int = 0):
        super().__init__()
        self.kind, self.p, self.seed, self.calls = kind, pct / 100.0, seed, 0

    def forward(self, x):
        if self.kind == "none" or self.p == 0:
            return x
        g = torch.Generator().manual_seed(self.seed + self.calls)
        self.calls += 1
        c, h, w = x.shape
        if self.kind == "gaussian":
            x = x + torch.randn(x.shape, generator=g) * self.p
        elif self.kind == "stripe":
            cols = torch.randperm(w, generator=g)[: max(1, int(round(self.p * w)))]
            offs = (torch.rand(len(cols), generator=g) - 0.5) * 0.6
            x = x.clone()
            x[:, :, cols] += offs.view(1, 1, -1)
        elif self.kind == "saltpepper":
            r = torch.rand((1, h, w), generator=g)
            x = x.clone()
            x = torch.where(r < self.p / 2, torch.zeros_like(x), x)
            x = torch.where((r >= self.p / 2) & (r < self.p), torch.ones_like(x), x)
        return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: every seed with a saved checkpoint")
    args = ap.parse_args()

    cfg = json.load(open(args.run_dir / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_manifest(args.manifest)
    test = df[df["split"] == "test"].reset_index(drop=True)
    seeds = args.seeds if args.seeds is not None else sorted(int(p.stem[4:]) for p in args.run_dir.glob("seed*.pt"))
    if not seeds:
        raise SystemExit(f"no checkpoints in {args.run_dir}; retrain with --keep-ckpt")

    rows = []
    for seed in seeds:
        model, spec = build_model(cfg["model"], pretrained=False)
        model.load_state_dict(torch.load(args.run_dir / f"seed{seed}.pt", map_location="cpu"))
        model.to(device)
        size = int(cfg["input_size"]) if cfg.get("input_size") not in (None, "None") else spec.input_size
        for kind, levels in LEVELS.items():
            for pct in levels:
                tf = build_transform(size, noise=Noise(kind, pct))
                # workers=0 keeps the noise call counter (and so the per-image RNG) deterministic
                loader = DataLoader(OralDataset(test, args.data_root, tf), batch_size=64, num_workers=0)
                y, p = predict(model, loader, device, device.type == "cuda")
                m = compute_metrics(y, p)
                rows.append({"seed": seed, "noise": kind, "intensity": pct, **m})
                print(f"seed {seed} {kind:10s} {pct:3d}%  sens {m['sens']:.3f} spec {m['spec']:.3f} auroc {m['auroc']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(args.run_dir / "robustness_seeds.csv", index=False)
    agg = out.groupby(["noise", "intensity"], sort=False)[["sens", "spec", "auroc"]].agg(["mean", "std"]) * 100
    agg.to_csv(args.run_dir / "robustness.csv")
    print(agg.round(1))


if __name__ == "__main__":
    main()
