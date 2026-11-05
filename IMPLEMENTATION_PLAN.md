# Implementation Plan — "Robust Lightweight Deep Learning Models for Oral Cancer Screening" (arXiv 2608.21583)

## 0. What the paper does (in one paragraph)
Binary classification of smartphone photos of the oral cavity into **Suspicious** (OPMDs + malignant) vs **Non-suspicious** (normal, normal variation, benign, other), on ~29.6k private Biocon Foundation images with a ~1:5 class imbalance and a 60/20/20 split. Six lightweight ImageNet-pretrained models (<7.5M params) are compared across ablations:
augmentations → image-quality loss reweighting → class-imbalance handling → 5-fold CV of the winner → knowledge distillation → interpretability → noise stress tests.
Winner: **MobileViTv2-1.0 + geometric augmentation (flips + ±10° rotation) + weighted cross-entropy (WCE)**, best fold sens 87.4% / spec 86.5% / NPV 97.2%.

Official code (partial): https://github.com/OCS-Tanuh/Conference-AIMLSystems2026 — one script with the final recipe. Details it reveals that the paper omits:
- HuggingFace `MobileViTV2ForImageClassification` from `apple/mobilevitv2-1.0-imagenet1k-256`, head replaced with `Linear(in, 1)` → `BCEWithLogitsLoss(pos_weight = w1/w0)`.
- Adam, lr 1e-3, `ReduceLROnPlateau(mode='max', factor=0.1, patience=5, min_lr=1e-7)` stepped on val sens×spec.
- Data stored as HDF5, `(N,3,256,256)`; a BGR→RGB channel swap in the loop.
- Checkpoint saved when val sens×spec improves **and** train loss improves.
- Batch 64 in code vs 32 in paper; 50 epochs; 10 seeds. → Use 32 (paper) as the default; make it configurable.

---

## 1. Substitute datasets (the Biocon data is private)

No public dataset matches the paper's size (~30k) or its field-captured triage distribution. Combine several smaller **clinical photograph** sets (not histopathology) and map their labels onto the paper's binary schema.

| Dataset | Content | Size | Access | Use |
|---|---|---|---|---|
| **SMART Intraoral Images** (Ragas Dental College; Sci. Data 2026, doi 10.1038/s41597-026-06954-5) — [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/smart_intraoral_images_dataset_1.html) | Smartphone, field conditions, India; Normal / Variation from normal / OPMD; VIA region annotations | 1,850 images, 235 patients (growing) | Restricted; request through AIKosh | **Primary**: closest to the paper's domain. Patient IDs allow patient-level splits. Region annotations allow quantitative Grad-CAM evaluation. |
| **Oral Images Dataset** (Chandrashekar et al.) — [Mendeley mhjyrn35p4](https://data.mendeley.com/datasets/mhjyrn35p4) | Mobile + intraoral camera, Karnataka India; benign vs malignant | 165 benign + 158 malignant (use `original_data` only) | Open | Secondary source / external test |
| **Oral Cancer (Lips and Tongue) Images** — Kaggle `shivam17299/oral-cancer-lips-and-tongue-images` | Cancer vs non-cancer photos | ~131 | Open | Small add-on; web-scraped, likely duplicates elsewhere |
| **Mouth and Oral Diseases (MOD)** and **Oral Diseases** — Kaggle | Multi-class oral diseases (OLP, ulcers, gingivitis, etc.) | Hundreds per class | Open | Extra Non-suspicious "benign" and Suspicious OLP/leukoplakia images |
| **Biocon/IIIT-H OPMD set** (Talwar et al., *Cancers* 2023, 15:4120) | 1,120 suspicious + 1,058 non-suspicious smartphone images, same Biocon source as the paper | 2,178 | Not public; email the authors (Birur / Shetty are co-authors of this paper too) | Best possible match; worth asking for |

The paper's data statement says research access is "subject to approval from ethics committees", so it is also worth emailing the corresponding author (siddhant0701@gmail.com) to request the original dataset.

Avoid H&E histopathology sets (NDB-UFES, ORCHID, the Kaggle "OCDB"/histopathology database). They are a different imaging modality.

### Label mapping → binary
- **Suspicious (1):** OPMD, leukoplakia (homogeneous or not), verrucous leukoplakia, OLP, tobacco-pouch keratosis, OSMF, erythroplakia, OSCC/malignant.
- **Non-suspicious (0):** normal, variation from normal, benign lesions, aphthous ulcer, gingivitis, candidiasis, etc.
- Keep a `source` column and the original fine-grained label for every image.

### Data hygiene (important with Kaggle sets)
1. Remove exact and near duplicates **across all sources** using a perceptual hash (`imagehash.phash`, Hamming distance ≤ 6). The paper also removed 126 duplicates.
2. Drop non-diagnostic images. Use a Laplacian-variance blur score plus manual review. This mirrors the paper's QC step.
3. Split by **patient** where IDs exist (SMART). Otherwise split by source-image group.
4. **Recreate the imbalance:** public sets are roughly balanced. Build a "paper-like" training split by subsampling Suspicious to a 1:5 ratio, so that the class-imbalance ablation still means something. Keep a balanced variant as well.
5. Splits: 60/20/20 stratified, plus an **external test** that holds out one whole source (e.g. Mendeley) to measure domain shift. The paper names this as a limitation.

Expected outcome: about 3–5k images in total, so absolute numbers will not match the paper. The goal is to reproduce the **relative trends** (which augmentations, losses and architectures win).

---

## 2. Project layout
```
oral-screen/
  configs/            # one YAML per experiment (model, aug, loss, sampler, seed list)
  data/
    raw/<source>/     # each download in its own folder (untrusted data)
    manifest.csv      # path, source, orig_label, binary_label, patient_id, split, phash, blur
  src/
    prepare_data.py   # dedup, QC, label mapping, splits, optional HDF5 export
    datasets.py       # Dataset + transforms (+ optional SAM-masked variant)
    models.py         # model factory (6 students + 3 teachers)
    losses.py         # WCE, Focal, IQA-weighted, KD loss
    samplers.py       # under/over-sampling, balanced mix-up
    train.py          # train/val loop, checkpoint selection, test eval
    iqa_scores.py     # precompute NIQE / CLIP-IQA per image
    sam_masks.py      # offline SAM "everything" masks
    distill.py        # teacher training + KD student training
    explain.py        # Grad-CAM++ + attention rollout
    robustness.py     # Gaussian / stripe / salt-and-pepper tests
    report.py         # aggregate seeds → mean±std, best pair, GM tables
  results/
```
Stack: PyTorch 2.3+, `timm`, `transformers` (MobileViTv2 as in the official code), `scikit-learn`, `torchmetrics`, `imblearn`, `pyiqa` (NIQE, CLIP-IQA), `segment-anything`, `pytorch-grad-cam`, `imagehash`, `opencv-python`, `pandas`, `pyyaml`.

---

## 3. Models (Table III)
| Paper name | Source | Input |
|---|---|---|
| MobileViTv2-1.0 | HF `apple/mobilevitv2-1.0-imagenet1k-256` (or timm `mobilevitv2_100`) | 256 |
| EdgeNeXt-S | timm `edgenext_small` | 256 |
| DeiT-Ti | timm `deit_tiny_patch16_224` | 224 |
| MobileNetV3-L | timm `mobilenetv3_large_100` | 512 |
| EfficientNetV2-B0 | timm `tf_efficientnetv2_b0` | 512 |
| NASNet-Mobile | not in timm. Use the Keras `NASNetMobile` weights through a port, or the `pretrainedmodels` package. If neither works, document a substitution (e.g. `mnasnet_100`). | 512 |
| Teachers: ViT-B/16, EfficientNetV2-L, Swin-B | timm `vit_base_patch16_224`, `tf_efficientnetv2_l`, `swin_base_patch4_window7_224` | 224 / 480 / 224 |

All models use a single-logit head, ImageNet mean/std normalisation, and bicubic resize with `align_corners=True`.

---

## 4. Training core (`train.py`)
- 50 epochs, batch 32, Adam lr 1e-3, ReduceLROnPlateau on val sens×spec (matches the code).
- Per epoch: compute val sensitivity, specificity, GM = √(sens·spec), AUROC. Keep the checkpoint with the best val sens×spec. Optionally add the code's "and train loss improved" rule behind a flag.
- Test with the selected checkpoint and log sens, spec, GM, NPV, PPV and AUROC.
- **10 seeds per configuration.** Report mean±std, plus the **"best pair"**: the (sens, spec) of the seed with the highest test GM, exactly as the paper defines it.
- Use AMP and log to CSV (or W&B). On a single consumer GPU, a 6-model × 10-seed grid is the bottleneck, so start with 3 seeds while debugging.

---

## 5. Experiments in paper order

**E1 – Augmentation ablation (Table IV)**: baseline (none), SAM-masked input, zoom-in (RandomResizedCrop scale 0.7–1), geometric (H/V flip + RandomRotation ±10°), colour jitter. All six models.
SAM: run `sam_masks.py` once offline in "everything" mode, keep the mask covering the mouth region, and black out the rest. This is training-time only.

**E2 – Image-quality reweighting (Table V)**: precompute NIQE and CLIP-IQA for each image with `pyiqa`. Per-sample loss weight is `w_i ∝ 1/NIQE_i` (lower NIQE = better) or `w_i ∝ CLIPIQA_i`, normalised so the batch mean is 1.

**E3 – Class imbalance (Table VI)**: random undersampling, random oversampling, WCE with `w_c = N/(2·N_c)` (implemented as `pos_weight = w1/w0`), focal loss (γ=2), Balanced Mix-up (α ∈ {0.1, 0.2, 0.3}): mix one instance-sampled batch with one class-balanced batch, λ ~ Beta(α, 1). Run this on the 1:5 imbalanced training split.

**E4 – Final recipe + 5-fold CV (Table VII)**: MobileViTv2 + geometric aug + WCE. Stratified 5-fold CV over train+val, 10 seeds per fold; report per-fold mean±std and NPV.

**E5 – Knowledge distillation (Table VIII)**: train the three teachers with WCE + geometric aug. Then train the student with
`L = (1−β)·WCE(y, s) + β·T²·BCE(σ(t/T), σ(s/T))`, β ∈ {0.5, 1.0}, T = 4 (the paper doesn't give T, so tune it in {2, 4}).
Expect β = 1.0 with the CNN teacher to be unstable, which is what the paper reports.

**E6 – Interpretability (Figs 1–2)**:
- Grad-CAM++ (`pytorch-grad-cam`) on MobileViTv2's last conv stage.
- "Attention rollout": MobileViTv2 uses *separable* linear attention, so there is no N×N attention matrix. Hook the context-score softmax (over patches) in the 3 / 4 / 2 attention blocks of stages 3, 4 and 5. Average within each stage and upsample to get maps at the paper's 16 / 64 / 256 scales.
- Bonus beyond the paper: SMART has expert region annotations, so you can report pointing-game accuracy or IoU between the thresholded CAM and the annotation. The paper could only do this qualitatively.

**E7 – Robustness (Table IX)**: apply the noise to raw images before resize, test set only.
- Gaussian: σ = p% of 255, with p ∈ {15, 20, 25}.
- Stripe: p% of columns (or rows) get a random constant offset, with p ∈ {2, 5, 7}.
- Salt and pepper: p% of pixels set to 0/255, with p ∈ {2, 5, 7}.
Optional extension (the paper's future work): median-filter pre-processing, and noise augmentation during training.

**E8 – Edge deployment (optional)**: export to ONNX → TFLite/CoreML, check that the model is about 17 MB, and measure CPU latency.

---

## 6. Milestones
1. Data acquisition and the manifest. Request SMART and the Biocon data in parallel, and start with the open Mendeley and Kaggle sets.
2. `prepare_data.py`: dedup, QC, mapping, splits. Sanity-check class counts per source.
3. Train core + MobileViTv2 baseline. Check that it beats a majority-class baseline and inspect a few predictions.
4. E1–E3 grids across all six models.
5. E4 CV, then E5 KD.
6. E6 explainability, E7 robustness.
7. `report.py` to regenerate Tables IV–IX, and a write-up comparing the trends with the paper (expect lower absolute numbers and higher variance because the data is smaller).

## 7. Risks / differences to document
- Much smaller, multi-source, partly web-scraped data. Source can leak into the label (e.g. all malignant images from one camera), so always report the held-out-source test.
- Different label mix: the paper's Suspicious class is mostly leukoplakia, tobacco-pouch keratosis and OLP, and only 144 images are malignant.
- Some hyperparameters are unstated (KD temperature, how the noise % is defined, the NASNet-Mobile port). Record each choice in the config.
