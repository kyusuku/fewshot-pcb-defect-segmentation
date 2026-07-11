"""Frozen Wide-ResNet features for a PatchCore-style memory-bank baseline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from features.dinov2 import (
    PatchFeatureMap,
    _image_to_normalized_tensor,
    _resolve_device,
    resize_and_pad_square,
)


class PatchCoreFeatureExtractor:
    """Combine stride-8 and stride-16 Wide-ResNet feature maps on one grid."""

    def __init__(
        self,
        image_size: int = 512,
        device: str = "auto",
        backbone=None,
    ) -> None:
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        import torch

        self.image_size = image_size
        self.patch_size = 8
        self.device = _resolve_device(device, torch)
        if backbone is None:
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
            from torchvision.models.feature_extraction import create_feature_extractor

            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
            backbone = create_feature_extractor(
                model,
                return_nodes={"layer2": "layer2", "layer3": "layer3"},
            )
        self.backbone = backbone.to(self.device) if hasattr(backbone, "to") else backbone
        if hasattr(self.backbone, "eval"):
            self.backbone.eval()

    def extract(self, image: Image.Image) -> PatchFeatureMap:
        import torch
        import torch.nn.functional as functional

        prepared = resize_and_pad_square(image.convert("RGB"), self.image_size)
        tensor = _image_to_normalized_tensor(prepared, torch).to(self.device)
        with torch.no_grad():
            output = self.backbone(tensor)
        if not isinstance(output, dict) or not {"layer2", "layer3"}.issubset(output):
            raise RuntimeError("PatchCore backbone must return layer2 and layer3 feature maps")

        layer2 = output["layer2"]
        layer3 = functional.interpolate(
            output["layer3"],
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat([layer2, layer3], dim=1)
        features = functional.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        array = features[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(array).all():
            raise RuntimeError("PatchCore backbone returned non-finite features")
        return PatchFeatureMap(
            features=array,
            image_size=prepared.size,
            patch_size=self.patch_size,
        )
