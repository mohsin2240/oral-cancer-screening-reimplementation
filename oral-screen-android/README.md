# Oral Screen (Android)

On-device oral-lesion triage with the MobileViTv2 recipe model from the `oral-screen` re-implementation
(geometric augmentation + weighted cross-entropy, seed 3: the seed with the best validation AUROC). Take a photo with
the system camera or choose one from the gallery; the app returns *Suspicious* / *Non-suspicious* with the model score.
Everything runs locally with ONNX Runtime; the app has no internet or camera permission.

**Research prototype, not a medical device.**

## Build and run

Requires the Android SDK (compileSdk 36) and JDK 17.

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or open this folder in Android Studio and press Run. minSdk is 28 (Android 9).

## How it works

| Step | Where |
|---|---|
| Decode the photo with EXIF rotation applied, longest side scaled to 640 px (like the training cache) | `ScreenViewModel.decode` |
| RGB → float32 `[1, 3, H, W]` in [0, 1] | `OralClassifier.predict` |
| Bicubic resize to 256 × 256 with corner alignment, exactly as in training | `assets/preprocess.onnx` |
| MobileViTv2 (channel flip to BGR inside) → sigmoid → P(suspicious) | `assets/model.onnx` |
| Suspicious if P ≥ 0.5 (the threshold used for every reported metric) | `OralClassifier.THRESHOLD` |
| Oral-photo gate: ImageNet MobileViTv2 features → logistic regression → P(oral photo) | `assets/gate.onnx` |
| Below 0.5: "Not recognised as an oral photo", with a retake prompt and *Show the screening result anyway* | `NotOralCard` |

The graphs are produced by two scripts in `oral-screen/`:

```bash
cd ../oral-screen
.venv/Scripts/python.exe src/export_onnx.py --run-dir results/manifest_train5/mobilevitv2_geometric_wce \
    --seed 3 --out-dir ../oral-screen-android/app/src/main/assets
.venv/Scripts/python.exe src/fit_oral_gate.py --out-dir ../oral-screen-android/app/src/main/assets
```

The export checks the screening graphs against PyTorch on all 84 test images (max difference 2e-6) and against the
training run's saved test predictions (max 1.2e-3). On the emulator, the app's scores for test images 341, 364 and
277 (0.74, 0.60, 0.18) match the saved predictions.

### The oral-photo gate

The screening model alone scores any picture (the emulator's cartoon camera scene came back *Suspicious*, 0.59), and
its own fine-tuned features cannot separate oral from non-oral photos. The gate is a logistic regression on the
ImageNet-pretrained MobileViTv2 features, trained on the oral training split against 640 photos from Kaggle
`prasunroy/natural-images` (airplane, car, cat, dog, flower, fruit, motorbike, person). `assets/gate_report.json`
holds the full evaluation:

| Check | Result |
|---|---|
| Oral val / test photos accepted | 98.8% / 98.8% (1 test photo rejected: a lip carcinoma with an arrow drawn on it) |
| Held-out natural photos rejected (40 per class) | 97.5% (flower) to 100% (all other classes) |
| Categories never seen in training (leave-classes-out) | 85–100% rejected, except unseen *person* photos: 67.5% |
| scikit-image sample photos, noise and solid colours | 100% rejected |
| Dental photos (`salmansajid05/oral-diseases`, intraoral) | 96.7% accepted |

On the emulator: the camera's cartoon scene (P = 0.09), a cat photo (0.00) and the rejected lip photo (0.10) are
refused; the oral test photos pass. A rejection is never presented as an all-clear, because real lesions can be
rejected (as the lip photo shows).

## Limitations

- The gate only knows the kinds of non-oral photo it was trained on. It rejected unseen categories well, except
  people (67.5% when no person photos were in training), and close-ups of skin or other body parts are untested.
- The screening model saw only ~420 oral-cavity photos from two public sources.
- On a different source (camera, clinic) its AUROC fell to 62% in the held-out-source experiment, so scores on new
  phones are much less reliable than the 81.4% sensitivity / 92.7% specificity measured on its own test set.
- Inference latency shown in the app was measured on an x86 emulator (0.4–5 s); real ARM phones are typically faster.
- The debug APK is 141 MB: two 17 MB models plus ONNX Runtime built for four CPU types (13–22 MB each). Keeping
  only `arm64-v8a` (`abiFilters`) would remove about 56 MB, but the x86_64 emulator then cannot run it.
