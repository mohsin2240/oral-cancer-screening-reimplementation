"""Losses from Section II-C.4 and III-E. All take raw logits and float targets in {0, 1} (1 = suspicious)."""
import torch
import torch.nn.functional as F
from torch import nn


def inverse_frequency_weights(n_neg: int, n_pos: int):
    """w_c = N / (2 N_c). With the paper's training split (14812 / 2932) this gives w0 = 0.60, w1 = 3.03."""
    n = n_neg + n_pos
    return n / (2 * n_neg), n / (2 * n_pos)


class WeightedBCE(nn.Module):
    """Eq. (1). w0 = w1 = 1 gives plain BCE (the paper's baseline). Accepts soft targets (mix-up)."""

    def __init__(self, w0=1.0, w1=1.0):
        super().__init__()
        self.w0, self.w1 = w0, w1

    def forward(self, logits, y, sample_weight=None):
        w = self.w1 * y + self.w0 * (1 - y)
        if sample_weight is not None:
            w = w * sample_weight
        return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()


class FocalLoss(nn.Module):
    """Eq. (2), gamma = 2."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y, sample_weight=None):
        p = torch.sigmoid(logits)
        p_t = y * p + (1 - y) * (1 - p)
        loss = (1 - p_t) ** self.gamma * F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def build_loss(name: str, n_neg: int, n_pos: int):
    if name == "bce":
        return WeightedBCE()
    if name == "wce":
        return WeightedBCE(*inverse_frequency_weights(n_neg, n_pos))
    if name == "focal":
        return FocalLoss()
    raise ValueError(name)


def kd_loss(student_logits, teacher_logits, T: float):
    """Hinton-style distillation for a single sigmoid output: BCE between temperature-softened probabilities, x T^2."""
    soft = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy_with_logits(student_logits / T, soft) * T * T
