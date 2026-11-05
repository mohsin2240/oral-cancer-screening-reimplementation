# Oral cancer screening with lightweight models — a public-data re-implementation

Re-implementation of *Robust Lightweight Deep Learning Models for Oral Cancer Screening* (arXiv 2608.21583). The
paper's ~29,600 field photographs are private, so every experiment is rebuilt on two open datasets (about 420 images
after cleaning). The aim is to reproduce the paper's **trends** — which augmentations, losses and architectures win —
not its absolute numbers.

**Research code, not a medical device.**

## Repository layout

| Folder | What it holds |
|---|---|
| [`oral-screen/`](oral-screen/) | The PyTorch pipeline: data download and cleaning, training, the experiment grid (paper Tables IV–IX), explainability, noise robustness, result tables, Kaggle runner, ONNX export, reproduction notebook |
| [`oral-screen/results/tables/`](oral-screen/results/tables/) | The final result tables (CSV and Markdown), laid out like the paper's |
| [`oral-screen-android/`](oral-screen-android/) | Kotlin / Jetpack Compose app that runs the final model on the phone with ONNX Runtime, plus an oral-photo check |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | The plan the implementation followed |

Images, trained checkpoints and per-run outputs are not in the repository: the scripts download the data and
regenerate everything. The manifests in `oral-screen/data/*.csv` fix the exact splits that were used.

## Quick start

```bash
cd oral-screen
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Then follow [`oral-screen/README.md`](oral-screen/README.md), or open `oral-screen/reproduce.ipynb` and run it with
`MODE = "smoke"` first. The Android app builds with `./gradlew assembleDebug` in `oral-screen-android/`; its models
are already in `app/src/main/assets/`.

## Data

| Source | Use |
|---|---|
| Mendeley Oral Images Dataset (doi 10.17632/mhjyrn35p4.2), `original_data` | 165 benign, 158 malignant |
| Kaggle `shivam17299/oral-cancer-lips-and-tongue-images` | cancer vs non-cancer |
| Kaggle `prasunroy/natural-images` | negatives for the app's oral-photo check only |

The datasets keep their own licences and are not redistributed here.

## Main changes from the paper

- ~420 public images instead of ~29.6k private ones; the training split is subsampled to the paper's 5:1 imbalance
  (24 suspicious images), while validation and test keep every suspicious image.
- NASNet-Mobile is replaced by MnasNet-1.0 (no usable PyTorch ImageNet weights).
- The ViT-B/16 and Swin-B teachers train at lr 1e-5; they collapse at 1e-3 and 1e-4.
- EfficientNetV2-L teacher at 384 px instead of 480; 5 seeds per configuration instead of 10.
- An extra held-out-source experiment (train on Mendeley, test on Kaggle) that the paper could not run.

## Headline results

| | Result |
|---|---|
| Recipe (MobileViTv2 + geometric + WCE), 5-fold CV | sensitivity 72.0, specificity 91.0, AUROC 91.2 (paper: 83.2 / 86.0) |
| Best imbalance method | random undersampling, GM 82.1 for MobileViTv2 |
| Best overall | MobileViTv2 distilled from Swin-B (β = 1.0): GM 82.8, against 74.7 without distillation |
| Held-out source | AUROC 87.6 on the training source, 62.0 on the unseen one |

Full tables: [`oral-screen/results/tables/`](oral-screen/results/tables/).
