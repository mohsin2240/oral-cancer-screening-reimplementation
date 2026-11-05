"""Download the open substitute datasets into data/raw/<source>/.

Sources:
  mendeley  - Oral Images Dataset (Chandrashekar et al.), doi 10.17632/mhjyrn35p4.2.
              Only the two `original_data` folders are kept; the authors' own augmented copies are skipped.
  kaggle_*  - public Kaggle datasets fetched with kagglehub (needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
              only if Kaggle refuses anonymous downloads).

Usage: python src/download_data.py --out data/raw [--only mendeley kaggle_lips_tongue ...]
"""
import argparse
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

MENDELEY_ID = "mhjyrn35p4"
# Folder ids of `original_data/*` in version 2. The API does not expose folder names; the per-folder file counts match
# the dataset description exactly (165 benign, 158 malignant), and the two other folders (1115/1155) are augmented_data.
MENDELEY_FOLDERS = {
    "a70af066-cdfe-431b-87d2-afe05d054df2": "benign",
    "71f08377-eac5-4fe0-af08-e278cb3a0615": "malignant",
}

KAGGLE_DATASETS = {
    "kaggle_lips_tongue": "shivam17299/oral-cancer-lips-and-tongue-images",
}
# Not used: salmansajid05/oral-diseases is dental photography (caries, calculus, tooth discoloration, gingivitis,
# hypodontia) plus mouth ulcers, with no OPMD/malignant class. As a single-label source it would let the model learn
# "teeth / this camera -> non-suspicious" instead of lesion features.


def _download(url, dest: Path, sha256=None):
    if dest.exists() and (sha256 is None or hashlib.sha256(dest.read_bytes()).hexdigest() == sha256):
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    if sha256 and hashlib.sha256(r.content).hexdigest() != sha256:
        raise ValueError(f"checksum mismatch for {dest}")
    dest.write_bytes(r.content)


def get_mendeley(out: Path):
    meta = requests.get(f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}", timeout=60).json()
    jobs = []
    for f in meta["files"]:
        cls = MENDELEY_FOLDERS.get(f["folder_id"])
        if cls is None:
            continue
        d = out / "mendeley" / cls
        d.mkdir(parents=True, exist_ok=True)
        c = f["content_details"]
        jobs.append((c["download_url"], d / f["filename"], c.get("sha256_hash")))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: _download(*j), jobs))
    print(f"mendeley: {len(jobs)} files")


def get_kaggle(name: str, slug: str, out: Path):
    import kagglehub

    src = Path(kagglehub.dataset_download(slug))
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"{name}: {n} files from {slug}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    wanted = args.only or ["mendeley", *KAGGLE_DATASETS]
    for name in wanted:
        try:
            if name == "mendeley":
                get_mendeley(args.out)
            else:
                get_kaggle(name, KAGGLE_DATASETS[name], args.out)
        except Exception as e:  # keep going so one unavailable source doesn't block the rest
            print(f"{name}: FAILED ({e})")


if __name__ == "__main__":
    main()
