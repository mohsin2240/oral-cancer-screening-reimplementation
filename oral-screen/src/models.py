"""Model factory: lightweight students from Table III and the KD teachers, each with a single-logit head.

Every model takes RGB images in [0, 1] and applies the input preprocessing its pretrained weights expect inside
forward(). The paper says "ImageNet mean/std", but the official script feeds the HuggingFace MobileViTv2 checkpoint BGR
images in [0, 1] without mean/std normalisation, which is what that checkpoint was trained with, so we follow each
checkpoint's own preprocessing. Keeping it inside the model lets the data pipeline, distillation (teacher and student
see the same augmented image) and the noise stress tests all work on plain [0, 1] RGB tensors.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelSpec:
    backend: str  # "hf" or "timm"
    name: str
    input_size: int


MODEL_ZOO = {
    # students (Table III)
    "mobilevitv2": ModelSpec("hf", "apple/mobilevitv2-1.0-imagenet1k-256", 256),
    "edgenext_s": ModelSpec("timm", "edgenext_small", 256),
    "deit_ti": ModelSpec("timm", "deit_tiny_patch16_224", 224),
    "mobilenetv3_l": ModelSpec("timm", "mobilenetv3_large_100", 512),
    "efficientnetv2_b0": ModelSpec("timm", "tf_efficientnetv2_b0", 512),
    # NASNet-Mobile has no reachable PyTorch ImageNet weights (the pretrainedmodels host is offline and timm only ships
    # NASNet-A Large). MnasNet-1.0 is the closest available substitute: also NAS-designed for mobile, 4.4M params.
    "nasnet_mobile": ModelSpec("timm", "mnasnet_100", 512),
    # KD teachers (Section III-E). EfficientNetV2-L is trained at 384 instead of 480 to fit a 4 GB GPU.
    "vit_b16": ModelSpec("timm", "vit_base_patch16_224", 224),
    "effv2_l": ModelSpec("timm", "tf_efficientnetv2_l", 384),
    "swin_b": ModelSpec("timm", "swin_base_patch4_window7_224", 224),
}
STUDENTS = ["mobilevitv2", "edgenext_s", "deit_ti", "mobilenetv3_l", "efficientnetv2_b0", "nasnet_mobile"]
TEACHERS = ["vit_b16", "effv2_l", "swin_b"]


class Classifier(nn.Module):
    """Wraps a backbone: [0, 1] RGB in, one logit per image out (positive = suspicious)."""

    def __init__(self, net, backend, mean, std, bgr):
        super().__init__()
        self.net, self.backend, self.bgr = net, backend, bgr
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, x):
        x = (x - self.mean) / self.std
        return x.flip(1) if self.bgr else x

    def forward(self, x):
        x = self.preprocess(x)
        out = self.net(pixel_values=x).logits if self.backend == "hf" else self.net(x)
        return out.squeeze(1)


def build_model(key: str, pretrained: bool = True):
    """Return (model, spec)."""
    spec = MODEL_ZOO[key]
    if spec.backend == "hf":
        from transformers import MobileViTV2Config, MobileViTV2ForImageClassification

        if pretrained:
            net = MobileViTV2ForImageClassification.from_pretrained(spec.name, num_labels=1, ignore_mismatched_sizes=True)
        else:
            net = MobileViTV2ForImageClassification(MobileViTV2Config.from_pretrained(spec.name, num_labels=1))
        return Classifier(net, "hf", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), bgr=True), spec

    import timm

    kw = {"img_size": spec.input_size} if key in ("vit_b16", "deit_ti", "swin_b") else {}
    net = timm.create_model(spec.name, pretrained=pretrained, num_classes=1, **kw)
    cfg = net.pretrained_cfg
    return Classifier(net, "timm", tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD)), False), spec


def resize_to(x, size):
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=True).clamp(0, 1)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    for k in MODEL_ZOO:
        m, s = build_model(k, pretrained=False)
        with torch.no_grad():
            out = m(torch.zeros(1, 3, s.input_size, s.input_size))
        print(f"{k:18s} {count_params(m):6.2f}M  in={s.input_size}  out={tuple(out.shape)}")
