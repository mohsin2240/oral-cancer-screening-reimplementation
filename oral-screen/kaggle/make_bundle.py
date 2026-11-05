"""Prepare the Kaggle dataset bundle and notebook metadata for training on Kaggle GPUs.

Builds kaggle/bundle/ (src/ + data/processed + data/sam_masks + manifests/scores, ~115 MB) with dataset-metadata.json,
and writes kaggle/kernel/kernel-metadata.json. The Kaggle username is read from the Kaggle API credentials
(~/.kaggle/kaggle.json or KAGGLE_USERNAME), which you set up yourself.

--with-results also copies finished runs from results/ (and the downloaded kaggle/output/ of a previous session), so a
follow-up session resumes the grid instead of repeating it.

Then:
  kaggle datasets create  -p kaggle/bundle --dir-mode zip            # first time (private)
  kaggle datasets version -p kaggle/bundle --dir-mode zip -m "update" # later
  kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4    # GPU T4 x2
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "kaggle" / "bundle"
KERNEL = ROOT / "kaggle" / "kernel"
DATASET_SLUG = "oral-screen-bundle"
KERNEL_SLUG = "oral-screen-grid"


def kaggle_username():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for p in [Path.home() / ".kaggle" / "kaggle.json", Path.home() / ".config" / "kaggle" / "kaggle.json"]:
        if p.exists():
            return json.loads(p.read_text())["username"]
    raise SystemExit("No Kaggle credentials found: put kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create New Token)")


def copy_results(src_root: Path, dst_root: Path):
    """Copy finished run dirs (those with summary.json); include checkpoints only where later steps need them."""
    n = 0
    for summary in src_root.glob("manifest_*/*/summary.json"):
        run = summary.parent
        dst = dst_root / run.relative_to(src_root)
        shutil.copytree(run, dst, dirs_exist_ok=True,
                        ignore=None if ("geometric_wce" in run.name and "kd" not in run.name and "fold" not in run.name)
                        else shutil.ignore_patterns("*.pt"))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-results", action="store_true")
    args = ap.parse_args()
    user = kaggle_username()

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "data").mkdir(parents=True)
    shutil.copytree(ROOT / "src", BUNDLE / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for d in ("processed", "sam_masks"):
        shutil.copytree(ROOT / "data" / d, BUNDLE / "data" / d)
    for f in (ROOT / "data").glob("*.csv"):
        shutil.copy2(f, BUNDLE / "data" / f.name)
    if args.with_results:
        n = copy_results(ROOT / "results", BUNDLE / "results")
        prev = ROOT / "kaggle" / "output" / "results"  # unzipped results.zip of a previous Kaggle session
        if prev.exists():
            n += copy_results(prev, BUNDLE / "results")
        print(f"included {n} finished runs")
    json.dump({"title": "oral-screen bundle", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]},
              open(BUNDLE / "dataset-metadata.json", "w"), indent=2)

    KERNEL.mkdir(parents=True, exist_ok=True)
    json.dump({
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "run_on_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # pretrained weights come from HuggingFace / timm
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }, open(KERNEL / "kernel-metadata.json", "w"), indent=2)
    size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file()) / 1e6
    print(f"bundle: {BUNDLE} ({size:.0f} MB) -> {user}/{DATASET_SLUG}")
    print(f"kernel: {KERNEL} -> {user}/{KERNEL_SLUG}")


if __name__ == "__main__":
    main()
