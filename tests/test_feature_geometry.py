from __future__ import annotations

from unittest import mock

import numpy as np
from PIL import Image

from anomaly.memory_bank import build_memory_bank
from anomaly.multiscale import compute_anomaly_heatmap
from features.dinov2 import (
    ColorPatchFeatureExtractor,
    DINOv2PatchFeatureExtractor,
    PatchFeatureMap,
)
from features.patchcore import PatchCoreFeatureExtractor


class _FakeDINOv2:
    def to(self, _device):
        return self

    def eval(self) -> None:
        return None

    def forward_features(self, tensor):
        import torch

        batch = tensor.shape[0]
        tokens = torch.arange(16, dtype=torch.float32, device=tensor.device)
        tokens = tokens.reshape(1, 16, 1).repeat(batch, 1, 1)
        return {"x_norm_patchtokens": tokens}


class _FalseyFakeDINOv2(_FakeDINOv2):
    def __bool__(self) -> bool:
        return False


class _FakePatchCoreBackbone:
    def __call__(self, tensor):
        import torch

        batch = tensor.shape[0]
        return {
            "layer2": torch.ones((batch, 2, 8, 8), device=tensor.device),
            "layer3": torch.ones((batch, 3, 4, 4), device=tensor.device),
        }


class _FixedGeometryExtractor:
    patch_size = 10

    def __init__(self, feature_map: PatchFeatureMap) -> None:
        self.feature_map = feature_map

    def extract(self, _image: Image.Image) -> PatchFeatureMap:
        return self.feature_map


def test_color_patch_records_asymmetric_content_geometry() -> None:
    extractor = ColorPatchFeatureExtractor(image_size=40, patch_size=10)

    result = extractor.extract(Image.new("RGB", (100, 40)))

    assert result.source_size == (100, 40)
    assert result.image_size == (40, 40)
    assert result.content_box == (0, 12, 40, 28)
    np.testing.assert_array_equal(
        result.valid_patch_mask(),
        np.array(
            [
                [False, False, False, False],
                [True, True, True, True],
                [True, True, True, True],
                [False, False, False, False],
            ]
        ),
    )


def test_dinov2_fake_model_records_asymmetric_content_geometry() -> None:
    extractor = DINOv2PatchFeatureExtractor(
        image_size=56,
        patch_size=14,
        device="cpu",
        model=_FakeDINOv2(),
    )

    result = extractor.extract(Image.new("RGB", (80, 20)))

    assert result.source_size == (80, 20)
    assert result.content_box == (0, 21, 56, 35)
    assert result.valid_patch_mask().sum() == 4


def test_falsey_injected_dinov2_model_does_not_trigger_network_loading() -> None:
    with mock.patch("torch.hub.load", side_effect=AssertionError("network load attempted")):
        extractor = DINOv2PatchFeatureExtractor(
            image_size=56,
            patch_size=14,
            device="cpu",
            model=_FalseyFakeDINOv2(),
        )

    assert isinstance(extractor.model, _FalseyFakeDINOv2)


def test_patchcore_fake_backbone_records_asymmetric_content_geometry() -> None:
    extractor = PatchCoreFeatureExtractor(
        image_size=64,
        device="cpu",
        backbone=_FakePatchCoreBackbone(),
    )

    result = extractor.extract(Image.new("RGB", (80, 20)))

    assert result.source_size == (80, 20)
    assert result.content_box == (0, 24, 64, 40)
    assert result.valid_patch_mask().sum() == 16


def test_support_memory_bank_excludes_padding_only_patch_centers() -> None:
    feature_map = PatchFeatureMap(
        features=np.arange(16, dtype=np.float32).reshape(4, 4, 1),
        image_size=(40, 40),
        patch_size=10,
        source_size=(100, 40),
        content_box=(0, 12, 40, 28),
    )

    bank = build_memory_bank([feature_map], normalize=False)

    np.testing.assert_array_equal(
        bank,
        np.arange(4, 12, dtype=np.float32).reshape(8, 1),
    )


def test_query_heatmap_unpads_before_resizing_to_asymmetric_source() -> None:
    grid = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [100.0, 100.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )
    feature_map = PatchFeatureMap(
        features=grid[..., None],
        image_size=(40, 40),
        patch_size=10,
        source_size=(100, 40),
        content_box=(0, 12, 40, 28),
    )

    heatmap = compute_anomaly_heatmap(
        image=Image.new("RGB", (100, 40)),
        extractor=_FixedGeometryExtractor(feature_map),
        memory_bank=np.zeros((1, 1), dtype=np.float32),
        normalize_features=False,
    )

    assert heatmap.shape == (40, 100)
    assert float(heatmap.max()) < 5.0
    assert float(heatmap[:, -1].mean()) > float(heatmap[:, 0].mean())


def test_legacy_feature_map_treats_entire_prepared_frame_as_content() -> None:
    feature_map = PatchFeatureMap(
        features=np.ones((2, 3, 1), dtype=np.float32),
        image_size=(30, 20),
        patch_size=10,
    )

    assert feature_map.source_size == (30, 20)
    assert feature_map.content_box == (0, 0, 30, 20)
    assert feature_map.valid_patch_mask().all()
