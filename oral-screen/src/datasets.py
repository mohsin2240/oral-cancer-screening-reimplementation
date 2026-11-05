"""Dataset and transforms. Reads the manifest written by prepare_data.py.

Transforms output RGB float tensors in [0, 1]; normalisation happens inside the model (see models.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


class BicubicResize(torch.nn.Module):
    """Resize to size x size with bicubic interpolation and corner alignment (Section II-C.1)."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(0), size=(self.size, self.size), mode="bicubic", align_corners=True)
        return x.squeeze(0).clamp(0, 1)


def augmentation_ops(aug: str, size: int):
    if aug in ("none", "sam"):  # SAM masking is applied by the dataset, before the transform
        return []
    if aug == "geometric":  # horizontal/vertical flips + random rotation in [-10, 10] degrees (the paper's winner)
        return [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10)]
    if aug == "zoom":
        return [T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(1.0, 1.0), antialias=True)]
    if aug == "colour":
        return [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
    raise ValueError(aug)


def build_transform(size: int, aug: str = "none", train: bool = False, noise=None):
    """noise: optional callable on the full-resolution [0, 1] tensor, applied before resizing (Section V)."""
    ops = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if noise is not None:
        ops.append(noise)
    ops.append(BicubicResize(size))
    if train:
        ops += augmentation_ops(aug, size)
    return T.Compose(ops)


class OralDataset(Dataset):
    """Yields (image, label, sample_weight).

    sample_weight: per-image loss weight (image-quality reweighting); 1 when not given.
    mask_dir / mask_prob: SAM-based augmentation; with probability mask_prob the image is multiplied by its offline
    SAM mask (data/sam_masks/<id>.png) so only the oral-cavity region remains.
    Decoded images are cached in memory; the cleaned corpus is a few hundred images.
    """

    def __init__(self, df: pd.DataFrame, root: Path, transform, weights=None, mask_dir=None, mask_prob=0.0):
        self.paths = [Path(root) / p for p in df["proc_path"]]
        self.ids = df["id"].tolist()
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
        self.weights = torch.ones(len(df)) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        self.transform = transform
        self.mask_dir, self.mask_prob = (Path(mask_dir) if mask_dir else None), mask_prob
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        if i not in self._cache:
            self._cache[i] = Image.open(self.paths[i]).convert("RGB")
        return self._cache[i]

    def __getitem__(self, i):
        img = self._load(i)
        if self.mask_dir is not None and torch.rand(1).item() < self.mask_prob:  # torch RNG is seeded per worker
            mpath = self.mask_dir / f"{self.ids[i]:06d}.png"
            if mpath.exists():
                m = np.asarray(Image.open(mpath).convert("L").resize(img.size, Image.Resampling.NEAREST)) > 127
                img = Image.fromarray(np.asarray(img) * m[..., None].astype(np.uint8))
        return self.transform(img), self.labels[i], self.weights[i]


def load_manifest(manifest: Path) -> pd.DataFrame:
    return pd.read_csv(manifest)


def load_split(manifest: Path, split: str) -> pd.DataFrame:
    df = load_manifest(manifest)
    return df[df["split"] == split].reset_index(drop=True)
