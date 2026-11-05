"""Export a trained student to ONNX for the Android app.

Two graphs are written to --out-dir, and the app runs them in sequence:
  preprocess.onnx  RGB float32 [1, 3, H, W] in [0, 1] at any size -> bicubic resize with corner alignment (as in
                   training) -> [1, 3, S, S]. The app sends the photo with its longest side scaled to 640 px, like
                   the cached training copies.
  model.onnx       [1, 3, S, S] -> P(suspicious) as [1]; the model's channel order / normalisation is inside.
They are separate because MobileViTv2's unfold/fold only exports with static shapes.

The script then checks the ONNX pair against PyTorch on the test split and against the saved test predictions.

Usage: python src/export_onnx.py --run-dir results/manifest_train5/mobilevitv2_geometric_wce --seed 3 \
           --out-dir ../oral-screen-android/app/src/main/assets
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from datasets import build_transform, load_split
from models import build_model


class Resize(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=True).clamp(0, 1)


def _unfolding(self, x):
    # Same layout as F.unfold with kernel = stride = patch size: [B, C, ph*pw, n_patches], patches row-major.
    b, c, h, w = x.shape
    ph, pw = self.patch_height, self.patch_width
    x = x.reshape(b, c, h // ph, ph, w // pw, pw).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c, ph * pw, (h // ph) * (w // pw)), (h, w)


def _folding(self, patches, output_size):
    b, c, _, _ = patches.shape
    (h, w), ph, pw = output_size, self.patch_height, self.patch_width
    x = patches.reshape(b, c, ph, pw, h // ph, w // pw).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c, h, w)


def patch_fold_for_export(model):
    """Swap MobileViTv2's F.unfold / F.fold (aten::col2im, which the ONNX exporter can't convert here) for the
    equivalent reshapes, and check the outputs don't change."""
    from transformers.models.mobilevitv2 import modeling_mobilevitv2 as m

    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        before = model(x)
        m.MobileViTV2Layer.unfolding, m.MobileViTV2Layer.folding = _unfolding, _folding
        after = model(x)
    assert torch.allclose(before, after, atol=1e-5), (before - after).abs().max()


class Prob(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest_train5.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / "config.json").read_text())
    model, spec = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(torch.load(args.run_dir / f"seed{args.seed}.pt", map_location="cpu"))
    model.eval()
    s = spec.input_size
    if cfg["model"] == "mobilevitv2":
        patch_fold_for_export(model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pre, net = args.out_dir / "preprocess.onnx", args.out_dir / "model.onnx"
    torch.onnx.export(Resize(s).eval(), torch.rand(1, 3, 480, 640), pre, input_names=["image"], output_names=["resized"],
                      dynamic_axes={"image": {2: "height", 3: "width"}}, opset_version=18, dynamo=False)
    torch.onnx.export(Prob(model).eval(), torch.rand(1, 3, s, s), net, input_names=["resized"], output_names=["prob"],
                      opset_version=18, dynamo=False)
    for f in (pre, net):
        print(f"wrote {f} ({f.stat().st_size / 2**20:.1f} MB)")

    import onnxruntime as ort

    sess_pre = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"])
    sess_net = ort.InferenceSession(str(net), providers=["CPUExecutionProvider"])
    test = load_split(args.manifest, "test")
    tf = build_transform(s)
    saved = pd.read_csv(args.run_dir / f"seed{args.seed}_test_preds.csv").set_index("id")["prob"]
    diffs, saved_diffs = [], []
    for _, r in test.iterrows():
        img = Image.open(args.data_root / r["proc_path"]).convert("RGB")
        x = (np.asarray(img, dtype=np.float32) / 255).transpose(2, 0, 1)[None]
        p_onnx = sess_net.run(None, {"resized": sess_pre.run(None, {"image": x})[0]})[0].item()
        with torch.no_grad():
            p_torch = torch.sigmoid(model(tf(img)[None])).item()
        diffs.append(abs(p_onnx - p_torch))
        saved_diffs.append(abs(p_onnx - saved[r["id"]]))
    print(f"{len(test)} test images: max |onnx - torch| = {max(diffs):.2e}, "
          f"max |onnx - saved preds| = {max(saved_diffs):.2e}")
    (args.out_dir / "model_info.json").write_text(json.dumps(
        {"model": cfg["model"], "run": args.run_dir.name, "seed": args.seed, "input_size": s, "threshold": 0.5}))


if __name__ == "__main__":
    main()
