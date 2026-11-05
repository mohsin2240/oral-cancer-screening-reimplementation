"""Interpretability for MobileViTv2 (Section IV, Figs 1-2): Grad-CAM++ and multi-scale attention rollout.

Grad-CAM++ targets the last MobileViTv2 stage (8x8 at 256 input).
Attention rollout: MobileViTv2 uses separable linear attention, so there is no N x N attention matrix. Each block
computes context scores = softmax over patches of a 1-channel query (B, 1, P, N), P = pixels per 2x2 patch,
N = patches. We hook the qkv projection, recompute those scores, average them over the blocks of a stage and fold them
back onto the stage's feature map. The three transformer stages give the paper's 256 / 64 / 16 patch scales
(2, 4 and 3 blocks).

The public datasets have no lesion annotations, so this is qualitative only (as in the paper).

Usage:
  python src/explain.py --run-dir results/manifest_train5/mobilevitv2_geometric_wce --manifest data/manifest_train5.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from datasets import build_transform, load_manifest
from models import build_model


class TwoD(torch.nn.Module):
    """grad-cam expects (B, n_classes) outputs."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).unsqueeze(1)


def attention_maps(model, x):
    """Return {n_patches: (H_in, W_in) map in [0, 1]} for each transformer stage."""
    enc = model.net.mobilevitv2.encoder
    records, hooks = {}, []
    for stage in (2, 3, 4):
        layer = enc.layer[stage]
        ph, pw = layer.patch_height, layer.patch_width
        for blk in layer.transformer.layer:
            def hook(_m, _inp, out, stage=stage, ph=ph, pw=pw):
                scores = torch.softmax(out[:, :1].float(), dim=-1)  # (B, 1, P, N)
                records.setdefault(stage, []).append((scores, ph, pw))
            hooks.append(blk.attention.qkv_proj.register_forward_hook(hook))
    feats = {}
    for stage in (2, 3, 4):  # spatial size of the feature map entering each stage's transformer
        hooks.append(enc.layer[stage].conv_1x1.register_forward_hook(
            lambda _m, _i, out, stage=stage: feats.__setitem__(stage, out.shape[-2:])))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    maps = {}
    for stage, recs in records.items():
        s = torch.stack([r[0] for r in recs]).mean(0)  # (B, 1, P, N)
        ph, pw = recs[0][1], recs[0][2]
        b, _, p, n = s.shape
        hw = feats[stage]
        m = F.fold(s.reshape(b, p, n), output_size=tuple(hw), kernel_size=(ph, pw), stride=(ph, pw))
        m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        maps[n] = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=None, help="default: the seed with the best test GM")
    ap.add_argument("--n", type=int, default=6, help="number of images per category")
    args = ap.parse_args()

    cfg = json.load(open(args.run_dir / "config.json"))
    assert cfg["model"] == "mobilevitv2", "explain.py targets the MobileViTv2 hybrid"
    seeds = pd.read_csv(args.run_dir / "seeds.csv")
    seed = args.seed if args.seed is not None else int(seeds.loc[seeds["test_gm"].idxmax(), "seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec = build_model("mobilevitv2", pretrained=False)
    model.load_state_dict(torch.load(args.run_dir / f"seed{seed}.pt", map_location="cpu"))
    model.to(device).eval()
    cam = GradCAMPlusPlus(model=TwoD(model), target_layers=[model.net.mobilevitv2.encoder.layer[4]])

    preds = pd.read_csv(args.run_dir / f"seed{seed}_test_preds.csv")
    df = load_manifest(args.manifest).set_index("id")
    tp = preds[(preds.label == 1) & (preds.prob >= 0.5)].sort_values("prob", ascending=False).head(args.n)
    fn = preds[(preds.label == 1) & (preds.prob < 0.5)].sort_values("prob").head(args.n // 2)
    fp = preds[(preds.label == 0) & (preds.prob >= 0.5)].sort_values("prob", ascending=False).head(args.n // 2)
    tf = build_transform(spec.input_size)
    out_dir = args.run_dir / "explain"
    out_dir.mkdir(exist_ok=True)

    for name, sel in [("true_positives", tp), ("false_negatives", fn), ("false_positives", fp)]:
        if sel.empty:
            continue
        cols = ["Image", "Grad-CAM++", "Attn 256", "Attn 64", "Attn 16"]
        fig, axes = plt.subplots(len(sel), 5, figsize=(12, 2.5 * len(sel)), squeeze=False)
        for r, (_, row) in enumerate(sel.iterrows()):
            row_id = int(row["id"])
            img = Image.open(args.data_root / df.loc[row_id, "proc_path"]).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            rgb = x[0].permute(1, 2, 0).cpu().numpy()
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
            att = attention_maps(model, x)
            panels = [None, g, att.get(256), att.get(64), att.get(16)]
            for c, (ax, hm) in enumerate(zip(axes[r], panels)):
                ax.imshow(rgb)
                if hm is not None:
                    ax.imshow(hm, cmap="jet", alpha=0.45)
                ax.set_xticks([]), ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=10)
            axes[r, 0].set_ylabel(f"id {row_id}\np={row['prob']:.2f}", fontsize=8)
        fig.suptitle(f"MobileViTv2 seed {seed}: {name.replace('_', ' ')}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
        print("wrote", out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
