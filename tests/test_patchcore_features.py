from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from anomaly.memory_bank import select_greedy_coreset
from features.dinov2 import build_feature_extractor
from features.patchcore import PatchCoreFeatureExtractor
from scripts.run_dinov2_baseline import parse_args, prepare_memory_bank
from scripts.run_patchcore_baseline import _append_default


class _FakeBackbone:
    def __call__(self, tensor):
        import torch

        batch = tensor.shape[0]
        return {
            "layer2": torch.ones((batch, 4, 8, 8), device=tensor.device),
            "layer3": torch.full((batch, 6, 4, 4), 2.0, device=tensor.device),
        }


def test_combines_layer2_and_upsampled_layer3() -> None:
    extractor = PatchCoreFeatureExtractor(
        image_size=64,
        device="cpu",
        backbone=_FakeBackbone(),
    )

    result = extractor.extract(Image.new("RGB", (48, 32), (20, 40, 60)))

    assert result.features.shape == (8, 8, 10)
    assert result.image_size == (64, 64)
    assert result.patch_size == 8
    assert result.features.dtype == np.float32
    assert np.isfinite(result.features).all()


def test_build_feature_extractor_dispatches_patchcore_lazily() -> None:
    sentinel = object()
    with mock.patch("features.patchcore.PatchCoreFeatureExtractor", return_value=sentinel) as ctor:
        result = build_feature_extractor(
            "patchcore_wrn50",
            image_size=256,
            patch_size=14,
            device="cpu",
        )

    assert result is sentinel
    ctor.assert_called_once_with(image_size=256, device="cpu")


def test_greedy_coreset_is_deterministic_subset() -> None:
    bank = np.arange(24, dtype=np.float32).reshape(8, 3)

    first = select_greedy_coreset(bank, ratio=0.25, seed=4880)
    second = select_greedy_coreset(bank, ratio=0.25, seed=4880)

    assert first.shape == (2, 3)
    np.testing.assert_array_equal(first, second)


def test_greedy_coreset_full_ratio_returns_independent_copy() -> None:
    bank = np.arange(12, dtype=np.float32).reshape(4, 3)

    selected = select_greedy_coreset(bank, ratio=1.0)

    np.testing.assert_array_equal(selected, bank)
    assert selected is not bank
    selected[0, 0] = -1.0
    assert bank[0, 0] == 0.0


@pytest.mark.parametrize(
    "ratio",
    [0.0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"],
)
def test_greedy_coreset_rejects_invalid_ratio(ratio) -> None:
    with pytest.raises(ValueError, match="ratio"):
        select_greedy_coreset(np.ones((2, 3), dtype=np.float32), ratio=ratio)


@pytest.mark.parametrize(
    ("bank", "message"),
    [
        (np.empty((0, 3), dtype=np.float32), "non-empty"),
        (np.ones((3,), dtype=np.float32), "2D"),
        (np.ones((2, 0), dtype=np.float32), "feature dimension"),
        (np.array([[1.0, np.nan]], dtype=np.float32), "finite"),
        (np.array([[1.0, np.inf]], dtype=np.float32), "finite"),
    ],
)
def test_greedy_coreset_rejects_invalid_bank(bank: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_greedy_coreset(bank)


@pytest.mark.parametrize("projection_dim", [0, -1, 1.5, True])
def test_greedy_coreset_rejects_invalid_projection_dimension(projection_dim) -> None:
    with pytest.raises(ValueError, match="projection_dim"):
        select_greedy_coreset(
            np.ones((2, 3), dtype=np.float32),
            projection_dim=projection_dim,
        )


@pytest.mark.parametrize("seed", [-1, 1.5, True, "4880"])
def test_greedy_coreset_rejects_invalid_seed(seed) -> None:
    with pytest.raises(ValueError, match="seed"):
        select_greedy_coreset(np.ones((2, 3), dtype=np.float32), seed=seed)


def test_prepare_memory_bank_applies_coreset_only_to_patchcore() -> None:
    bank = np.arange(24, dtype=np.float32).reshape(8, 3)

    patchcore_bank, patchcore_provenance = prepare_memory_bank(
        bank,
        feature_backbone="patchcore_wrn50",
        coreset_ratio=0.25,
        coreset_seed=4880,
        coreset_projection_dim=5,
    )
    dino_bank, dino_provenance = prepare_memory_bank(
        bank,
        feature_backbone="dinov2_vits14",
        coreset_ratio=0.25,
        coreset_seed=4880,
        coreset_projection_dim=5,
    )

    assert patchcore_bank.shape == (2, 3)
    assert patchcore_provenance == {
        "feature_backbone": "patchcore_wrn50",
        "memory_bank_size_full": 8,
        "memory_bank_size_used": 2,
        "coreset_applied": True,
        "coreset_ratio": 0.25,
        "coreset_projection_dim": 5,
        "coreset_seed": 4880,
    }
    np.testing.assert_array_equal(dino_bank, bank)
    assert dino_bank is bank
    assert dino_provenance["coreset_applied"] is False
    assert dino_provenance["memory_bank_size_used"] == 8


def test_compatibility_cli_appends_patchcore_defaults_without_overriding_user_values() -> None:
    argv = ["run_patchcore_baseline.py", "--image-size", "320"]
    with mock.patch.object(sys, "argv", argv):
        _append_default("--feature-backbone", "patchcore_wrn50")
        _append_default("--image-size", "512")
        _append_default("--patch-size", "8")

    assert argv == [
        "run_patchcore_baseline.py",
        "--image-size",
        "320",
        "--feature-backbone",
        "patchcore_wrn50",
        "--patch-size",
        "8",
    ]


def test_compatibility_cli_help_is_network_free_and_lists_shared_arguments() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_patchcore_baseline.py"), "--help"],
        check=False,
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PatchCore-style baseline" in result.stdout
    assert "--k" in result.stdout
    assert "--feature-backbone" in result.stdout
    assert "--coreset-ratio" in result.stdout
    assert "--coreset-projection-dim" in result.stdout


def test_shared_runner_keeps_dino_and_coreset_defaults() -> None:
    with mock.patch.object(sys, "argv", ["run_dinov2_baseline.py"]):
        args = parse_args()

    assert args.feature_backbone == "dinov2_vits14"
    assert args.image_size == 518
    assert args.patch_size == 14
    assert args.coreset_ratio == 0.01
    assert args.coreset_projection_dim == 64
