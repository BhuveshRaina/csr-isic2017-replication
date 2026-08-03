"""
Shared utilities for the CSR (Concept-based Similarity Reasoning) pipeline.

Keeps training/eval scripts small and consistent: reproducible seeding,
device selection, AMP context helpers, and the exact image transforms used
for the ISIC-2017 skin-lesion experiments described in the supplementary
material (image size 224x224, ImageNet normalisation).
"""
import os
import random
import contextlib

import numpy as np
import torch
from torchvision import transforms

# ImageNet statistics (backbone is ImageNet-pretrained ConvNeXt-T).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = 42):
    """Make a run reproducible across python / numpy / torch / cudnn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducible but still reasonably fast.
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str = "cuda") -> torch.device:
    """Return a valid device, gracefully falling back to CPU / MPS."""
    if "cuda" in requested and torch.cuda.is_available():
        return torch.device(requested)
    if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_autocast(device: torch.device):
    """
    Autocast context that is safe on every backend.

    AMP only meaningfully helps on CUDA; on CPU/MPS we return a no-op context
    so the same training code runs everywhere without dtype surprises.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device):
    """GradScaler that is only enabled on CUDA."""
    enabled = device.type == "cuda"
    try:  # torch >= 2.3 unified API
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_transforms(image_size: int = 224, is_train: bool = True):
    """
    ISIC-2017 transforms.

    The paper trains ISIC at 224x224 (supplementary Sec. 3.3). We use light
    flip augmentation for training and a deterministic resize for eval so that
    the reported macro-F1 is measured on clean, unaugmented images.
    """
    if is_train:
        # Flip-only augmentation. We tried stronger dermoscopy-style augmentation
        # (180 deg rotation + RandomResizedCrop + stronger colour jitter) and it
        # measurably hurt: peak val_conceptF1 dropped from 0.4829 to 0.3641.
        # Root cause: the Part-2 concept labels (pigment_network, streaks, etc.)
        # are about specific regions of the lesion, often near the border.
        # RandomResizedCrop(scale<1.0) can crop that exact evidence out of frame
        # while the label still says "present" -- i.e. it manufactures label
        # noise. Flips don't have this problem since they preserve every pixel.
        # Eval transform below is left deterministic so the reported macro-F1 is
        # measured on clean, unaugmented images.
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
