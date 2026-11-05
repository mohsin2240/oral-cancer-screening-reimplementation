"""Build data/manifest.csv from the raw downloads (milestone 2).

Steps (mirroring Section II-A of the paper where possible):
  1. scan data/raw/<source>/ and map each image's folder label to the binary schema (1 = suspicious, 0 = non-suspicious);
     folders that do not map are reported and excluded
  2. drop unreadable / tiny images
  3. near-duplicate removal across all sources with a perceptual hash, made invariant to flips and 90-degree rotations
     (several Kaggle sets ship augmented copies). Groups whose members disagree on the label are dropped entirely.
  4. quality flag: Laplacian-variance blur score; images below --blur-thresh are excluded (all scores are kept in
     data/qc_report.csv for manual review)
  5. optional: subsample suspicious images per split to reach a non-suspicious:suspicious ratio (paper: ~5:1)
  6. stratified 60/20/20 train/val/test split; --external-source moves one whole source into an "external" split
  7. write resized copies (longest side --proc-size) to data/processed/ so training doesn't decode large photos

Usage: python src/prepare_data.py --raw data/raw --out data [--neg-pos-ratio 5] [--external-source mendeley]
"""
import argparse
import re
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Per-source rules: (regex on the lowercased relative directory path, label). First match wins; 1 = suspicious.
# Suspicious = OPMDs (leukoplakia, OLP, OSMF, erythroplakia, tobacco-pouch keratosis) and malignancy;
# non-suspicious = normal, normal variations and benign conditions (Table I).
SOURCE_RULES = {
    "mendeley": [(r"\bmalignant\b", 1), (r"\bbenign\b", 0)],
    "kaggle_lips_tongue": [(r"non[\s_-]*cancer", 0), (r"cancer", 1)],
}


def label_for(source: str, rel_dir: str):
    for pattern, label in SOURCE_RULES.get(source, []):
        if re.search(pattern, rel_dir):
            return label
    return None


def scan(raw: Path) -> pd.DataFrame:
    rows, unmapped = [], {}
    for src_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        source = src_dir.name
        for p in src_dir.rglob("*"):
            if p.suffix.lower() not in IMG_EXT:
                continue
            rel_dir = p.parent.relative_to(src_dir).as_posix().lower()
            label = label_for(source, rel_dir)
            if label is None:
                unmapped[(source, rel_dir)] = unmapped.get((source, rel_dir), 0) + 1
                continue
            rows.append({"source": source, "raw_path": p.relative_to(raw.parent).as_posix(), "orig_label": rel_dir,
                         "label": label})
    for (source, d), n in sorted(unmapped.items()):
        print(f"  unmapped, excluded: {source}/{d} ({n} images)")
    return pd.DataFrame(rows)


def dihedral_hashes(img: Image.Image):
    """pHash of the 8 flip/90-degree-rotation variants, as uint64."""
    out = []
    for flip in (False, True):
        im = ImageOps.mirror(img) if flip else img
        for k in range(4):
            h = imagehash.phash(im.rotate(90 * k, expand=True))
            out.append(int(str(h), 16))
    return out


def blur_score(img: Image.Image, side=512) -> float:
    g = np.asarray(img.convert("L"))
    s = side / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def analyse(df: pd.DataFrame, base: Path, min_side: int) -> pd.DataFrame:
    hashes, blur, w, h, ok = [], [], [], [], []
    for rp in df["raw_path"]:
        try:
            img = ImageOps.exif_transpose(Image.open(base / rp)).convert("RGB")
        except Exception:
            hashes.append([0] * 8), blur.append(np.nan), w.append(0), h.append(0), ok.append(False)
            continue
        hashes.append(dihedral_hashes(img))
        blur.append(blur_score(img))
        w.append(img.width), h.append(img.height)
        ok.append(min(img.size) >= min_side)
    df = df.assign(width=w, height=h, blur=blur, readable=ok)
    df["phash"] = [f"{hs[0]:016x}" for hs in hashes]
    return df, np.array(hashes, dtype=np.uint64)


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def duplicate_groups(hashes: np.ndarray, max_dist: int) -> np.ndarray:
    """Union-find over pairs whose min Hamming distance (canonical hash vs any dihedral variant) <= max_dist."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    canon = hashes[:, 0]
    for i in range(n - 1):
        d = popcount64(canon[i] ^ hashes[i + 1:]).min(axis=1)
        for j in np.nonzero(d <= max_dist)[0] + i + 1:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def dedup(df: pd.DataFrame, hashes: np.ndarray, max_dist: int) -> pd.DataFrame:
    df = df.copy()
    df["dup_group"] = duplicate_groups(hashes, max_dist)
    df["status"] = "keep"
    for _, g in df.groupby("dup_group"):
        if len(g) == 1:
            continue
        if g["label"].nunique() > 1:
            df.loc[g.index, "status"] = "dup_label_conflict"
            continue
        keep = (g["width"] * g["height"]).idxmax()  # keep the highest-resolution copy
        df.loc[g.index.difference([keep]), "status"] = "duplicate"
    return df


def rebalance(df: pd.DataFrame, ratio: float, seed: int, splits=None) -> pd.DataFrame:
    """Within each split (or only those in `splits`), drop random suspicious images until n_neg / n_pos >= ratio."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for split, g in df[df["split"] != "unused"].groupby("split"):
        if splits is not None and split not in splits:
            continue
        pos = g.index[g["label"] == 1]
        target = int(round((g["label"] == 0).sum() / ratio))
        if len(pos) > target:
            df.loc[rng.choice(pos, len(pos) - target, replace=False), "split"] = "unused"
    return df


def split(df: pd.DataFrame, external_source: str | None, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["split"] = ""
    pool = df
    if external_source:
        df.loc[df["source"] == external_source, "split"] = "external"
        pool = df[df["source"] != external_source]
    # stratify on source x label so every source is represented in every split
    strat = pool["source"] + "_" + pool["label"].astype(str)
    tr, rest = train_test_split(pool.index, test_size=0.4, stratify=strat, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=strat.loc[rest], random_state=seed)
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    return df


def write_processed(df: pd.DataFrame, base: Path, out: Path, size: int) -> pd.DataFrame:
    proc = out / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, rp in zip(df.index, df["raw_path"]):
        dst = proc / f"{i:06d}.png"
        if not dst.exists():
            img = ImageOps.exif_transpose(Image.open(base / rp)).convert("RGB")
            img.thumbnail((size, size), Image.Resampling.BICUBIC)
            img.save(dst)
        paths.append(dst.relative_to(out).as_posix())
    return df.assign(proc_path=paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--hash-dist", type=int, default=6, help="max pHash Hamming distance for a near-duplicate")
    ap.add_argument("--blur-thresh", type=float, default=10.0, help="exclude images with Laplacian variance below this")
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--neg-pos-ratio", type=float, default=None, help="e.g. 5 to mimic the paper's imbalance")
    ap.add_argument("--rebalance-scope", choices=["all", "train"], default="all",
                    help="'train': imbalance only the training split. Sensitivity/specificity do not depend on "
                         "prevalence, so val/test can keep every suspicious image for tighter estimates.")
    ap.add_argument("--external-source", default=None)
    ap.add_argument("--proc-size", type=int, default=640, help="longest side of the cached copies")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="manifest.csv")
    args = ap.parse_args()

    base = args.raw.parent
    print("scanning", args.raw)
    df = scan(args.raw)
    print(f"  {len(df)} labelled images")
    df, hashes = analyse(df, base, args.min_side)
    df = dedup(df, hashes, args.hash_dist)
    df.loc[~df["readable"], "status"] = "unreadable_or_tiny"
    df.loc[(df["status"] == "keep") & (df["blur"] < args.blur_thresh), "status"] = "blurry"
    df.to_csv(args.out / "qc_report.csv", index_label="id")
    print(pd.crosstab([df["source"], df["label"]], df["status"]))

    kept = df[df["status"] == "keep"].copy()
    kept = split(kept, args.external_source, args.seed)
    if args.neg_pos_ratio:
        kept = rebalance(kept, args.neg_pos_ratio, args.seed, ["train"] if args.rebalance_scope == "train" else None)
    kept = write_processed(kept, base, args.out, args.proc_size)
    cols = ["source", "raw_path", "proc_path", "orig_label", "label", "split", "width", "height", "blur", "phash",
            "dup_group"]
    kept[cols].to_csv(args.out / args.name, index_label="id")
    print(pd.crosstab(kept["split"], kept["label"], margins=True))
    print(f"wrote {args.out / args.name}")


if __name__ == "__main__":
    main()
