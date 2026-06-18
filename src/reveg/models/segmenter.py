"""Segmentation models: SegFormer-B0 (4-channel) + U-Net baseline, with MC-dropout.

SegFormer (transformers `nvidia/mit-b0`) transfers well under weak/noisy labels; the
first patch-embed conv is widened 3->4 channels (NIR initialised from the red weights).
A classifier dropout is kept ACTIVE at inference for MC-dropout uncertainty.

HF weights on a China GPU box: set `HF_ENDPOINT=https://hf-mirror.com` before run.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import NUM_CLASSES


def _widen_to_4ch(conv: nn.Conv2d) -> nn.Conv2d:
    """Return a Conv2d like `conv` but with 4 input channels; NIR <- red weights."""
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:4] = conv.weight[:, :1]  # NIR init from red
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_segformer(num_classes: int = NUM_CLASSES, *, dropout: float = 0.1):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        classifier_dropout_prob=dropout,
    )
    enc = model.segformer.encoder
    proj = enc.patch_embeddings[0].proj
    enc.patch_embeddings[0].proj = _widen_to_4ch(proj)
    return model


def build_unet(num_classes: int = NUM_CLASSES, *, dropout: float = 0.1):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                    in_channels=4, classes=num_classes, aux_params=None)


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Unify SegFormer (HF) and smp outputs to full-res logits (N,K,H,W)."""
    out = model(x)
    logits = out.logits if hasattr(out, "logits") else out
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def enable_mc_dropout(model) -> None:
    """Keep dropout layers stochastic at inference (model otherwise in eval mode)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model, x: torch.Tensor, *, passes: int = 20):
    """MC-dropout: mean softmax + per-pixel predictive entropy (normalised 0..1).
    Returns (pred [N,H,W] long, entropy [N,H,W] float)."""
    model.eval()
    enable_mc_dropout(model)
    probs = None
    for _ in range(passes):
        p = F.softmax(forward_logits(model, x), dim=1)
        probs = p if probs is None else probs + p
    probs = probs / passes
    pred = probs.argmax(dim=1)
    ent = -(probs * torch.clamp(probs, 1e-8).log()).sum(dim=1)
    ent = ent / np.log(probs.shape[1])  # normalise by log(K)
    return pred, ent
