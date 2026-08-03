"""
Phase-1 Concept model.

Paper (Sec. 2.2 + supplementary Sec. 1):
  F : ConvNeXt-T backbone, ImageNet pretrained -> feature map f in R^{C x H x W}.
  C : concept head = 1x1 conv WITHOUT bias -> K channels (the class activation
      maps cam_k), then Global Average Pooling, then Sigmoid.
The model is trained for multi-label concept classification with BCE.

Backbone note
-------------
The supplementary says "ConvNeXt-T ... ImageNet pretrained", i.e. plain IN-1k.
We default to that for faithfulness (`convnext_tiny.fb_in1k`). A stronger
IN-12k->IN-1k checkpoint is available and usually helps on small datasets; pass
`backbone="convnext_tiny.in12k_ft_in1k"` to use it. Both output 768 channels
at a 7x7 spatial grid for 224x224 input.
"""
import torch
import torch.nn as nn
import timm

DEFAULT_BACKBONE = "convnext_tiny.fb_in1k"


class ConceptModel(nn.Module):
    def __init__(self, num_concepts=4, backbone=DEFAULT_BACKBONE, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.F = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        # Infer channel dim of the feature map robustly across timm versions.
        feature_dim = getattr(self.F, "num_features", 768)

        # Concept head C: 1x1 conv (no bias) -> K class activation maps.
        self.cam_conv = nn.Conv2d(feature_dim, num_concepts, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts

    def forward(self, x):
        f = self.F.forward_features(x)          # (B, C, H, W)
        cam = self.cam_conv(f)                  # (B, K, H, W)  class activation maps
        c_logits = self.gap(cam).flatten(1)     # (B, K)        concept logits
        c = torch.sigmoid(c_logits)             # (B, K)        concept probabilities
        return c_logits, c, cam, f
