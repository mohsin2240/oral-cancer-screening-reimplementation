# oral-screen

Re-implementation of *Robust Lightweight Deep Learning Models for Oral Cancer Screening* (arXiv 2608.21583) on public
data. The plan is in [../IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md). Every experiment from the paper (Tables
IV–IX, Figs 1–2) is implemented.

## Setup (Windows, uv)
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Data
```bash
.venv/Scripts/python.exe src/download_data.py --out data/raw
# balanced (natural) split
.venv/Scripts/python.exe src/prepare_data.py --raw data/raw --out data
# main experimental manifest: training split at the paper's 5:1 imbalance, val/test keep every suspicious image
.venv/Scripts/python.exe src/prepare_data.py --raw data/raw --out data --neg-pos-ratio 5 --rebalance-scope train --name manifest_train5.csv
# held-out source: Kaggle images become an "external" test split
.venv/Scripts/python.exe src/prepare_data.py --raw data/raw --out data --neg-pos-ratio 5 --rebalance-scope train --external-source kaggle_lips_tongue --name manifest_external.csv
# offline inputs for Tables IV and V
.venv/Scripts/python.exe src/sam_masks.py --manifest data/manifest.csv     # needs checkpoints/sam_vit_b_01ec64.pth
.venv/Scripts/python.exe src/iqa_scores.py --manifest data/manifest.csv    # NIQE + CLIP-IQA
```
Sources:
- `mendeley`: Oral Images Dataset (doi 10.17632/mhjyrn35p4.2), `original_data` only. 165 benign → 0, 158 malignant → 1.
- `kaggle_lips_tongue`: `shivam17299/oral-cancer-lips-and-tongue-images`. `cancer` → 1, `non-cancer` → 0.

New sources (SMART, Biocon) go in `data/raw/<name>/`, with a rule added to `SOURCE_RULES` in `prepare_data.py`.

Why `--rebalance-scope train`: sensitivity and specificity do not depend on prevalence. Imbalancing only the training
set reproduces the paper's training conditions and keeps all 43 suspicious images in val/test, instead of 8. NPV is
also reported re-weighted to the paper's 16.5% prevalence (`npv_adj`).

## Experiments
`src/run_grid.py` runs everything sequentially and skips configurations that already have a `summary.json`:
```bash
.venv/Scripts/python.exe src/run_grid.py --seeds 0 1 2 3 4 --epochs 50
.venv/Scripts/python.exe src/run_grid.py --list          # show what is left
```

| Group | Paper | What |
|---|---|---|
| `table4` | Table IV | 6 students × {none, geometric, zoom, colour, SAM} augmentation |
| `table5` | Table V | 6 students × {NIQE, CLIP-IQA} loss reweighting |
| `table6` | Table VI | 6 students × {under-, over-sampling, WCE, focal, balanced mix-up α ∈ {0.1, 0.2, 0.3}} |
| `recipe` | — | MobileViTv2 + geometric + WCE, checkpoints kept |
| `table7` | Table VII | 5-fold CV of the recipe (train+val pooled, test fixed) |
| `teachers` | Table VIII | ViT-B/16, EfficientNetV2-L (384 px), Swin-B teachers |
| `table8` | Table VIII | KD into MobileViTv2 at β ∈ {0.5, 1.0}, T = 4 |
| `external` | Limitations | Train on Mendeley, test on the unseen Kaggle source |

After the `recipe` run:
```bash
.venv/Scripts/python.exe src/robustness.py --run-dir results/manifest_train5/mobilevitv2_geometric_wce --manifest data/manifest_train5.csv   # Table IX
.venv/Scripts/python.exe src/explain.py    --run-dir results/manifest_train5/mobilevitv2_geometric_wce --manifest data/manifest_train5.csv   # Figs 1-2
.venv/Scripts/python.exe src/report.py     # results/tables/*.csv|md, laid out like the paper's tables
```

A single configuration can also be run directly, e.g.
`src/train.py --manifest data/manifest_train5.csv --model mobilevitv2 --aug geometric --loss wce`.
See `src/train.py --help` for all switches.

## Differences from the paper
- **Data:** about 420 public images after cleaning, against ~29.6k private ones. Absolute numbers are not comparable; only trends are.
- **Seeds:** 5 per configuration by default, against the paper's 10, so the full grid fits in roughly 18 h on an RTX 3050 4 GB. Pass `--seeds 0 … 9` for the full protocol.
- **Preprocessing:** follows each checkpoint's own convention, applied inside the model. For HF MobileViTv2 that is BGR input in [0, 1] with no mean/std, as in the official script.
- **NASNet-Mobile:** has no reachable PyTorch ImageNet weights, so it is replaced by MnasNet-1.0 (`timm mnasnet_100`), another NAS-designed mobile CNN.
- **ViT-B/16 and Swin-B teachers:** trained at lr 1e-5 (`TEACHER_LR` in `run_grid.py`). At the default 1e-3 the loss explodes and predictions collapse to one class (test AUROC ~55), and they still collapse at 1e-4. The paper does not state a learning rate.
- **EfficientNetV2-L teacher:** trained at 384 px instead of 480 to fit a 4 GB GPU.
- **512-px CNNs and teachers:** use batch 16 × 2 accumulation steps (effective batch 32).
- **SAM augmentation:** each training image is masked with probability 0.5. Masks come from SAM ViT-B in "everything" mode with a central-segments heuristic (see `sam_masks.py`). As the paper reports, they often cut away lesions.
- **KD:** the loss is BCE between temperature-softened sigmoid outputs, × T² with T = 4. The paper gives neither the form nor T.
- **Robustness:** noise is applied to the cached 640-px copies rather than the camera originals.
- **Interpretability:** the public sets have no lesion annotations, so it is qualitative only.

## Reproduction notebook
`reproduce.ipynb` runs the whole pipeline in order. `MODE = "smoke"` checks it in minutes (1 seed, 2 epochs, outputs
in `results_smoke/`); `MODE = "full"` reproduces the tables (5 seeds × 50 epochs, roughly 15–20 GPU-hours on 2× T4).

## Kaggle (2× T4)
`kaggle/make_bundle.py` packs `src/`, the manifests and the processed images into a private Kaggle dataset, and
`kaggle/kernel/run_on_kaggle.py` runs two grid shards in parallel (one per GPU) with a 9.5 h time budget and a
90-minute per-configuration timeout, then zips `results/`. Edit the dataset and kernel ids in
`kaggle/kernel/kernel-metadata.json` to your own account.

## Export for the Android app
```bash
# screening model: preprocess.onnx (bicubic resize, any input size) + model.onnx (P(suspicious))
.venv/Scripts/python.exe src/export_onnx.py --run-dir results/manifest_train5/mobilevitv2_geometric_wce --seed 3 --out-dir ../oral-screen-android/app/src/main/assets
# oral-photo gate: gate.onnx (P(the photo shows the oral cavity)), trained against Kaggle prasunroy/natural-images
.venv/Scripts/python.exe src/fit_oral_gate.py --out-dir ../oral-screen-android/app/src/main/assets
```
Both scripts check the ONNX output against PyTorch. Seed 3 is the recipe seed with the best validation AUROC.
