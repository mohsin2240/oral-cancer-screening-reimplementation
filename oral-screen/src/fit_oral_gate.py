"""Fit the "is this an oral photo?" gate for the Android app and export it as gate.onnx.

The screening model only ever saw oral-cavity photos, so it scores anything (a cartoon house came back Suspicious).
Its own fine-tuned features cannot tell oral from non-oral photos, so the gate uses the ImageNet-pretrained
MobileViTv2 backbone: pooled features -> standardisation -> logistic regression -> P(oral photo).

Positives: the oral training split (data/manifest.csv). Negatives: Kaggle "prasunroy/natural-images" (airplane, car,
cat, dog, flower, fruit, motorbike, person; 80 per class). Evaluated on the oral val/test splits, unseen natural
images, scikit-image sample photos, synthetic images (noise, solid colours) and the dental-photo set
"salmansajid05/oral-diseases" (intraoral, so accepting it is fine). A leave-classes-out run checks that the gate
also rejects kinds of photo it never saw.

Input of gate.onnx: the 256 x 256 output of preprocess.onnx. Output: P(oral) as [1]; the app rejects below 0.5.

Usage: python src/fit_oral_gate.py --out-dir ../oral-screen-android/app/src/main/assets
"""
import argparse
import json
import random
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import build_transform
from export_onnx import patch_fold_for_export
from models import build_model

N_TRAIN, N_EVAL = 80, 40  # natural images per class


class Gate(torch.nn.Module):
    def __init__(self, model, w, b):
        super().__init__()
        self.model = model
        self.linear = torch.nn.Linear(len(w), 1)
        self.linear.weight.data = torch.tensor(w, dtype=torch.float32)[None]
        self.linear.bias.data = torch.tensor([b], dtype=torch.float32)

    def forward(self, x):
        f = self.model.net.mobilevitv2(pixel_values=self.model.preprocess(x)).pooler_output
        return torch.sigmoid(self.linear(f)).squeeze(1)


def images(paths):
    return [Image.open(p) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    model, spec = build_model("mobilevitv2", pretrained=True)
    model.eval()
    tf = build_transform(spec.input_size)

    @torch.no_grad()
    def feats(imgs):
        out = []
        for i in range(0, len(imgs), 32):
            batch = []
            for im in imgs[i:i + 32]:
                im = im.convert("RGB")
                im.thumbnail((640, 640), Image.Resampling.BICUBIC)  # what the app does before inference
                batch.append(tf(im))
            out.append(model.net.mobilevitv2(pixel_values=model.preprocess(torch.stack(batch))).pooler_output)
        return torch.cat(out).numpy()

    m = pd.read_csv(args.manifest)
    oral = {s: feats(images(args.data_root / p for p in m[m.split == s].proc_path)) for s in ("train", "val", "test")}
    random.seed(0)
    nat_dir = Path(kagglehub.dataset_download("prasunroy/natural-images")) / "natural_images"
    nat = {c.name: feats(images(random.sample(sorted(c.iterdir()), N_TRAIN + N_EVAL))) for c in sorted(nat_dir.iterdir())}
    try:
        dent_dir = Path(kagglehub.dataset_download("salmansajid05/oral-diseases"))
    except OSError:  # one folder name ends in a space, which Windows cannot create; use what did extract
        dent_dir = Path.home() / ".cache/kagglehub/datasets/salmansajid05/oral-diseases/versions/3"
    dent = [p for p in dent_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    import skimage.data as sd
    sk = [getattr(sd, n)() for n in ("astronaut", "coffee", "chelsea", "rocket", "camera", "coins",
                                     "hubble_deep_field", "immunohistochemistry", "page", "moon")]
    rng = np.random.default_rng(0)
    synth = [Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))]
    synth += [Image.new("RGB", (640, 480), c) for c in ["white", "black", (200, 120, 120), (230, 180, 170), "red"]]
    other = {
        "dental (intraoral, accepting is fine)": feats(images(random.sample(dent, 150))),
        "scikit-image photos": feats([Image.fromarray(a if a.ndim == 3 else np.stack([a] * 3, -1)) for a in sk]),
        "synthetic (noise, solid colours)": feats(synth),
    }

    def fit(classes):
        X = np.r_[oral["train"], np.concatenate([nat[c][:N_TRAIN] for c in classes])]
        y = np.r_[np.ones(len(oral["train"])), np.zeros(N_TRAIN * len(classes))]
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")).fit(X, y)

    def accept(clf, f):
        return float(np.mean(clf.predict_proba(f)[:, 1] >= 0.5))

    report = {}
    classes = sorted(nat)
    for seen in (classes[::2], classes[1::2]):
        clf = fit(seen)
        report[f"leave-classes-out, trained on {'/'.join(seen)}"] = {
            "oral test accepted": accept(clf, oral["test"]),
            **{f"unseen {c} rejected": 1 - accept(clf, nat[c][N_TRAIN:]) for c in classes if c not in seen},
        }
    clf = fit(classes)
    report["final gate (all 8 classes)"] = {
        "oral val accepted": accept(clf, oral["val"]),
        "oral test accepted": accept(clf, oral["test"]),
        **{f"held-out {c} rejected": 1 - accept(clf, nat[c][N_TRAIN:]) for c in classes},
        **{f"{k} rejected": 1 - accept(clf, f) for k, f in other.items()},
    }
    for name, r in report.items():
        print(name)
        for k, v in r.items():
            print(f"  {k}: {v:.1%}")

    # fold the scaler into the logistic regression: w' = w / sd, b' = b - sum(w * mean / sd)
    scaler, lr = clf[0], clf[1]
    w = lr.coef_[0] / scaler.scale_
    b = lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_)
    patch_fold_for_export(model)
    gate = Gate(model, w, b).eval()
    with torch.no_grad():
        x = torch.rand(4, 3, spec.input_size, spec.input_size)
        f = model.net.mobilevitv2(pixel_values=model.preprocess(x)).pooler_output.numpy()
        assert np.allclose(gate(x).numpy(), clf.predict_proba(f)[:, 1], atol=1e-4)
    out = args.out_dir / "gate.onnx"
    torch.onnx.export(gate, torch.rand(1, 3, spec.input_size, spec.input_size), out, input_names=["resized"],
                      output_names=["p_oral"], opset_version=18, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)")
    (args.out_dir / "gate_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
