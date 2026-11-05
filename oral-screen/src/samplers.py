"""Class-imbalance sampling strategies (Section II-C.4)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def resample(df: pd.DataFrame, how: str, seed: int) -> pd.DataFrame:
    """Random under-/over-sampling of the training split, drawn once per seed (as imblearn's samplers do)."""
    if how == "none":
        return df
    rng = np.random.default_rng(seed)
    pos, neg = df.index[df["label"] == 1], df.index[df["label"] == 0]
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    if how == "under":  # majority down to the minority count
        keep = np.concatenate([minority, rng.choice(majority, len(minority), replace=False)])
    elif how == "over":  # minority up to the majority count, sampling with replacement
        keep = np.concatenate([majority, minority, rng.choice(minority, len(majority) - len(minority), replace=True)])
    else:
        raise ValueError(how)
    return df.loc[rng.permutation(keep)].reset_index(drop=True)


def class_balanced_sampler(labels, num_samples: int, seed: int):
    """Draws each class with equal probability; the class-balanced stream of Balanced Mix-up."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    w = 1.0 / counts[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples, replacement=True, generator=g)


def balanced_mixup(x_inst, y_inst, x_bal, y_bal, alpha: float):
    """Eq. (3): x = lam * x_I + (1 - lam) * x_C with lam ~ Beta(alpha, 1) (Galdran et al., MICCAI 2021)."""
    lam = float(np.random.beta(alpha, 1.0))
    n = min(len(x_inst), len(x_bal))
    return lam * x_inst[:n] + (1 - lam) * x_bal[:n], lam * y_inst[:n] + (1 - lam) * y_bal[:n]
