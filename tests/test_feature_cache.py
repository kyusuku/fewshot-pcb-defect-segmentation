from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anomaly.multiscale import compute_anomaly_heatmap
from features import cache as cache_module
from features.cache import (
    FeatureCache,
    FeatureCacheError,
    extractor_revision_identity,
    extractor_source_revision,
    feature_cache_key,
    numerical_runtime_identity,
)
from features.dinov2 import ColorPatchFeatureExtractor, PatchFeatureMap


class FeatureCacheTest(unittest.TestCase):
    def test_raw_digest_cannot_bypass_identity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            with self.assertRaisesRegex(TypeError, "FeatureCacheIdentity"):
                cache.load(_cache_key())
            with self.assertRaisesRegex(TypeError, "FeatureCacheIdentity"):
                cache.save(_cache_key(), _feature_map())
            with self.assertRaisesRegex(TypeError, "FeatureCacheIdentity"):
                cache.get_or_compute(_cache_key(), _feature_map)

    def test_wrong_small_map_cannot_load_under_518_identity(self) -> None:
        large_identity = cache_module.FeatureCacheIdentity(
            sample_id="pcb1/a",
            image_sha256="a" * 64,
            extractor_revision="b" * 64,
            backbone="dinov2_vits14",
            image_size=518,
            patch_size=14,
            view="query:global:0,0,40,20",
            source_size=(40, 20),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "image_size.*identity"):
                cache.save(large_identity, _feature_map())
            with self.assertRaisesRegex(TypeError, "FeatureCacheIdentity"):
                cache.load(large_identity.key)

    def test_lock_files_live_outside_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "features"
            first = FeatureCache(root)
            second = FeatureCache(root)
            first.get_or_compute(_identity(), _feature_map)

            self.assertEqual(first.lock_root, second.lock_root)
            self.assertNotEqual(first.lock_root.parent, root)
            self.assertEqual(sorted(path.suffix for path in root.iterdir()), [".npz"])

    def test_round_trip_preserves_features_and_geometry_metadata(self) -> None:
        original = PatchFeatureMap(
            features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            image_size=(28, 28),
            patch_size=14,
            source_size=(40, 20),
            content_box=(0, 7, 28, 21),
        )
        identity = _identity()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            cache.save(identity, original)
            restored = cache.load(identity)

        self.assertIsNotNone(restored)
        assert restored is not None
        np.testing.assert_array_equal(restored.features, original.features)
        self.assertEqual(restored.image_size, original.image_size)
        self.assertEqual(restored.patch_size, original.patch_size)
        self.assertEqual(restored.source_size, original.source_size)
        self.assertEqual(restored.content_box, original.content_box)

    def test_key_changes_for_every_result_affecting_field_and_view(self) -> None:
        base = {
            "sample_id": "pcb1/a",
            "image_sha256": "a" * 64,
            "extractor_revision": "extractor-rev-a",
            "backbone": "dinov2_vits14",
            "image_size": 518,
            "patch_size": 14,
            "view": "query:global:0,0,1024,768",
        }
        variants = []
        replacements = {
            "sample_id": "pcb1/b",
            "image_sha256": "b" * 64,
            "extractor_revision": "extractor-rev-b",
            "backbone": "patchcore_wrn50",
            "image_size": 512,
            "patch_size": 8,
            "view": "query:crop:0,0,768,768",
        }
        for field, replacement in replacements.items():
            changed = dict(base)
            changed[field] = replacement
            variants.append(feature_cache_key(**changed))

        keys = {_cache_key(), *variants}

        self.assertEqual(len(keys), 1 + len(replacements))
        self.assertTrue(all(len(key) == 64 for key in keys))

    def test_cache_hit_avoids_compute(self) -> None:
        calls = 0
        feature_map = _feature_map()

        def compute() -> PatchFeatureMap:
            nonlocal calls
            calls += 1
            return feature_map

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            first = cache.get_or_compute(_identity(), compute)
            second = cache.get_or_compute(_identity(), compute)

        self.assertEqual(calls, 1)
        np.testing.assert_array_equal(first.features, second.features)

    def test_corrupt_entry_raises_without_recomputing(self) -> None:
        calls = 0
        identity = _identity()

        def compute() -> PatchFeatureMap:
            nonlocal calls
            calls += 1
            return _feature_map()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            cache.path_for_key(identity.key).parent.mkdir(parents=True, exist_ok=True)
            cache.path_for_key(identity.key).write_bytes(b"not an npz archive")

            with self.assertRaisesRegex(FeatureCacheError, "Could not read feature cache"):
                cache.get_or_compute(identity, compute)

        self.assertEqual(calls, 0)

    def test_invalid_geometry_payload_is_rejected(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            path = cache.path_for_key(identity.key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                cache_format_version=np.asarray(2, dtype=np.int32),
                cache_key=np.asarray(identity.key),
                identity_json=np.asarray(identity.canonical_json),
                features=np.ones((2, 2, 3), dtype=np.float32),
                image_size=np.asarray((28, 28), dtype=np.int32),
                patch_size=np.asarray(14, dtype=np.int32),
                source_size=np.asarray((40, 20), dtype=np.int32),
                content_box=np.asarray((0, 7, 29, 21), dtype=np.int32),
            )

            with self.assertRaisesRegex(FeatureCacheError, "Invalid feature cache"):
                cache.load(identity)

    def test_transplanted_payload_is_rejected(self) -> None:
        first_identity = _identity()
        second_identity = cache_module.FeatureCacheIdentity(
            sample_id="pcb1/b",
            image_sha256="a" * 64,
            extractor_revision="extractor-rev-a",
            backbone="dinov2_vits14",
            image_size=28,
            patch_size=14,
            view="query:global:0,0,40,20",
            source_size=(40, 20),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            first_path = cache.save(first_identity, _feature_map())
            second_path = cache.path_for_key(second_identity.key)
            second_path.write_bytes(first_path.read_bytes())

            with self.assertRaisesRegex(FeatureCacheError, "cache key does not match"):
                cache.load(second_identity)

    def test_nonfinite_features_are_not_cached(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            PatchFeatureMap(
                features=np.asarray([[[np.nan]]], dtype=np.float32),
                image_size=(1, 1),
                patch_size=1,
            )

    def test_identity_rejects_feature_map_with_wrong_source_geometry(self) -> None:
        identity = _identity()
        feature_map = PatchFeatureMap(
            features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            image_size=(28, 28),
            patch_size=14,
            source_size=(41, 20),
            content_box=(0, 7, 28, 21),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "source_size.*identity"):
                cache.save(identity, feature_map)

    def test_identity_rejects_feature_map_with_wrong_content_geometry(self) -> None:
        identity = _identity()
        feature_map = PatchFeatureMap(
            features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            image_size=(28, 28),
            patch_size=14,
            source_size=(40, 20),
            content_box=(0, 6, 28, 20),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "content_box.*identity"):
                cache.save(identity, feature_map)

    def test_identity_rejects_view_geometry_inconsistent_with_source_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "view.*source_size"):
            cache_module.FeatureCacheIdentity(
                sample_id="pcb1/a",
                image_sha256="a" * 64,
                extractor_revision="b" * 64,
                backbone="dinov2_vits14",
                image_size=28,
                patch_size=14,
                view="query:crop:0,0,20,20",
                source_size=(40, 20),
            )

    def test_save_revalidates_mutated_feature_values(self) -> None:
        feature_map = _feature_map()
        feature_map.features[0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "finite"):
                cache.save(_identity(), feature_map)

    def test_payload_identity_must_match_requested_identity(self) -> None:
        first = _identity()
        second = cache_module.FeatureCacheIdentity(
            sample_id="pcb1/b",
            image_sha256="a" * 64,
            extractor_revision="extractor-rev-a",
            backbone="dinov2_vits14",
            image_size=28,
            patch_size=14,
            view="query:global:0,0,40,20",
            source_size=(40, 20),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            first_path = cache.save(first, _feature_map())
            with np.load(first_path, allow_pickle=False) as stored:
                payload = {name: np.asarray(stored[name]).copy() for name in stored.files}
            payload["cache_key"] = np.asarray(second.key)
            np.savez_compressed(cache.path_for_key(second.key), **payload)

            with self.assertRaisesRegex(FeatureCacheError, "identity does not match"):
                cache.load(second)

    def test_concurrent_identical_requests_compute_once(self) -> None:
        calls = 0
        calls_lock = threading.Lock()
        both_ready = threading.Barrier(3)
        compute_started = threading.Event()
        release_compute = threading.Event()

        def compute() -> PatchFeatureMap:
            nonlocal calls
            with calls_lock:
                calls += 1
            compute_started.set()
            self.assertTrue(release_compute.wait(timeout=5))
            return _feature_map()

        def worker(cache: FeatureCache) -> PatchFeatureMap:
            both_ready.wait(timeout=5)
            return cache.get_or_compute(_identity(), compute)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker, cache) for _ in range(2)]
                both_ready.wait(timeout=5)
                self.assertTrue(compute_started.wait(timeout=5))
                release_compute.set()
                results = [future.result(timeout=5) for future in futures]

        self.assertEqual(calls, 1)
        np.testing.assert_array_equal(results[0].features, results[1].features)


class ExtractorRevisionTest(unittest.TestCase):
    def test_batchnorm_eps_changes_output_and_revision_with_identical_state(self) -> None:
        import torch

        first_model = torch.nn.BatchNorm1d(2, eps=1e-5, affine=False).eval()
        second_model = torch.nn.BatchNorm1d(2, eps=0.5, affine=False).eval()
        second_model.load_state_dict(first_model.state_dict())
        inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        self.assertFalse(torch.equal(first_model(inputs), second_model(inputs)))

        first = extractor_source_revision(
            _TorchExtractor(first_model),
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )
        second = extractor_source_revision(
            _TorchExtractor(second_model),
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )

        self.assertNotEqual(first, second)

    def test_custom_module_behavior_config_and_source_are_identified(self) -> None:
        import torch

        first_model = _ScaleModule(1.0)
        second_model = _ScaleModule(2.0)
        inputs = torch.ones(1)
        self.assertFalse(torch.equal(first_model(inputs), second_model(inputs)))

        first_identity = extractor_revision_identity(
            _TorchExtractor(first_model),
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )
        second_identity = extractor_revision_identity(
            _TorchExtractor(second_model),
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )

        self.assertNotEqual(
            cache_module.extractor_revision_digest(first_identity),
            cache_module.extractor_revision_digest(second_identity),
        )
        module_entry = first_identity["model_structure"]["named_modules"][0]
        self.assertEqual(module_entry["extra_repr"], "scale=1.0")
        self.assertIn(module_entry["class"], first_identity["source_sha256"])

    def test_runtime_identity_includes_hardware_and_tf32_policies(self) -> None:
        runtime = numerical_runtime_identity()

        self.assertIn("hardware", runtime)
        self.assertIn("torch_cuda_matmul_allow_tf32", runtime)
        self.assertIn("torch_cudnn_allow_tf32", runtime)
        self.assertIn("cpu", runtime["hardware"])
        self.assertIn("cuda_devices", runtime["hardware"])

    def test_revision_changes_with_injected_hardware_or_tf32_identity(self) -> None:
        extractor = _FakeExtractor()
        first_runtime = {
            **_runtime_identity(),
            "hardware": {"cpu": "cpu-a", "cuda_devices": []},
            "torch_cuda_matmul_allow_tf32": False,
            "torch_cudnn_allow_tf32": False,
        }
        hardware_changed = {
            **first_runtime,
            "hardware": {"cpu": "cpu-b", "cuda_devices": []},
        }
        tf32_changed = {**first_runtime, "torch_cuda_matmul_allow_tf32": True}

        revisions = {
            extractor_source_revision(
                extractor,
                runtime_identity=runtime,
                precision_identity={"policy": "float32"},
            )
            for runtime in (first_runtime, hardware_changed, tf32_changed)
        }

        self.assertEqual(len(revisions), 3)

    def test_revision_changes_when_injected_model_weight_changes(self) -> None:
        extractor = _FakeExtractor()
        first = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )
        extractor.model.weight[0, 0] += np.float32(1.0)
        second = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )

        self.assertNotEqual(first, second)

    def test_revision_changes_with_numerical_runtime_identity(self) -> None:
        extractor = _FakeExtractor()
        first_runtime = _runtime_identity()
        second_runtime = {**first_runtime, "numpy": "changed"}

        first = extractor_source_revision(
            extractor,
            runtime_identity=first_runtime,
            precision_identity={"policy": "float32"},
        )
        second = extractor_source_revision(
            extractor,
            runtime_identity=second_runtime,
            precision_identity={"policy": "float32"},
        )

        self.assertNotEqual(first, second)

    def test_revision_changes_with_device_or_precision_policy(self) -> None:
        extractor = _FakeExtractor(device="cpu")
        cpu = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )
        extractor.device = "cuda:0"
        cuda = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )
        mixed_precision = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "autocast-float16"},
        )

        self.assertEqual(len({cpu, cuda, mixed_precision}), 3)

    def test_revision_changes_with_model_graph_at_identical_weights(self) -> None:
        extractor = _FakeExtractor()
        first = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )
        extractor.model.graph = "different-forward-graph"
        second = extractor_source_revision(
            extractor,
            runtime_identity=_runtime_identity(),
            precision_identity={"policy": "float32"},
        )

        self.assertNotEqual(first, second)

    def test_color_patch_revision_is_stable_and_configuration_sensitive(self) -> None:
        first = ColorPatchFeatureExtractor(image_size=28, patch_size=14)
        equivalent = ColorPatchFeatureExtractor(image_size=28, patch_size=14)
        changed = ColorPatchFeatureExtractor(image_size=56, patch_size=14)

        first_revision = extractor_source_revision(first, runtime_identity=_runtime_identity())
        equivalent_revision = extractor_source_revision(
            equivalent,
            runtime_identity=_runtime_identity(),
        )
        changed_revision = extractor_source_revision(changed, runtime_identity=_runtime_identity())

        self.assertEqual(first_revision, equivalent_revision)
        self.assertNotEqual(first_revision, changed_revision)


class PatchFeatureMapValidationTest(unittest.TestCase):
    def test_rejects_nonpositive_patch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "patch_size.*positive integer"):
            PatchFeatureMap(
                features=np.ones((1, 1, 1), dtype=np.float32),
                image_size=(1, 1),
                patch_size=0,
            )

    def test_rejects_empty_grid_or_feature_dimension(self) -> None:
        for shape in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    PatchFeatureMap(
                        features=np.ones(shape, dtype=np.float32),
                        image_size=(1, 1),
                        patch_size=1,
                    )

    def test_rejects_feature_grid_inconsistent_with_image_and_patch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid.*image_size.*patch_size"):
            PatchFeatureMap(
                features=np.ones((1, 1, 3), dtype=np.float32),
                image_size=(28, 28),
                patch_size=14,
            )

    def test_rejects_non_float32_or_nonfinite_features(self) -> None:
        for features in (
            np.ones((1, 1, 1), dtype=np.float64),
            np.asarray([[[np.inf]]], dtype=np.float32),
        ):
            with self.subTest(dtype=features.dtype, value=float(features[0, 0, 0])):
                with self.assertRaisesRegex(ValueError, "float32|finite"):
                    PatchFeatureMap(features=features, image_size=(1, 1), patch_size=1)

    def test_rejects_malformed_sizes_and_content_box(self) -> None:
        cases = (
            {"image_size": (True, 1)},
            {"source_size": (1.5, 1)},
            {"content_box": (0.0, 0, 1, 1)},
        )
        for overrides in cases:
            arguments = {
                "features": np.ones((1, 1, 1), dtype=np.float32),
                "image_size": (1, 1),
                "patch_size": 1,
                **overrides,
            }
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "integer"):
                    PatchFeatureMap(**arguments)


class MultiScaleFeatureCacheTest(unittest.TestCase):
    def test_global_and_crop_views_use_distinct_entries_and_then_hit_cache(self) -> None:
        image = Image.new("RGB", (8, 8), (16, 32, 48))
        extractor = _CountingExtractor(image_size=4, patch_size=2)
        memory_bank = np.zeros((1, 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            def key_for_view(view: str):
                x1, y1, x2, y2 = (int(value) for value in view.rsplit(":", 1)[1].split(","))
                return cache_module.FeatureCacheIdentity(
                    sample_id="pcb1/query",
                    image_sha256="a" * 64,
                    extractor_revision="extractor-rev-a",
                    backbone="color_patch",
                    image_size=4,
                    patch_size=2,
                    view=f"query:{view}",
                    source_size=(x2 - x1, y2 - y1),
                )

            first = compute_anomaly_heatmap(
                image=image,
                extractor=extractor,
                memory_bank=memory_bank,
                crop_sizes=[4],
                crop_overlap=0.0,
                normalize_features=False,
                feature_cache=cache,
                cache_key_for_view=key_for_view,
            )
            first_call_count = extractor.calls
            second = compute_anomaly_heatmap(
                image=image,
                extractor=extractor,
                memory_bank=memory_bank,
                crop_sizes=[4],
                crop_overlap=0.0,
                normalize_features=False,
                feature_cache=cache,
                cache_key_for_view=key_for_view,
            )
            entry_count = len(list(Path(tmpdir).glob("*.npz")))

        self.assertEqual(first_call_count, 5)
        self.assertEqual(extractor.calls, first_call_count)
        self.assertEqual(entry_count, 5)
        np.testing.assert_array_equal(first, second)


class _CountingExtractor(ColorPatchFeatureExtractor):
    def __init__(self, image_size: int, patch_size: int) -> None:
        super().__init__(image_size=image_size, patch_size=patch_size)
        self.calls = 0

    def extract(self, image: Image.Image) -> PatchFeatureMap:
        self.calls += 1
        return super().extract(image)


def _cache_key() -> str:
    return feature_cache_key(
        "pcb1/a",
        "a" * 64,
        "extractor-rev-a",
        "dinov2_vits14",
        518,
        14,
        "query:global:0,0,1024,768",
    )


def _identity():
    return cache_module.FeatureCacheIdentity(
        sample_id="pcb1/a",
        image_sha256="a" * 64,
        extractor_revision="extractor-rev-a",
        backbone="dinov2_vits14",
        image_size=28,
        patch_size=14,
        view="query:global:0,0,40,20",
        source_size=(40, 20),
    )


def _feature_map() -> PatchFeatureMap:
    return PatchFeatureMap(
        features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        image_size=(28, 28),
        patch_size=14,
        source_size=(40, 20),
        content_box=(0, 7, 28, 21),
    )


class _FakeModel:
    def __init__(self) -> None:
        self.weight = np.arange(4, dtype=np.float32).reshape(2, 2)
        self.graph = "initial-forward-graph"

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weight": self.weight}


class _FakeExtractor:
    def __init__(self, device: str = "cpu") -> None:
        self.model = _FakeModel()
        self.model_name = "fake_backbone"
        self.image_size = 28
        self.patch_size = 14
        self.device = device


class _TorchExtractor:
    def __init__(self, model) -> None:
        self.model = model
        self.model_name = "injected_torch_model"
        self.image_size = 28
        self.patch_size = 14
        self.device = "cpu"


class _ScaleModule(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, inputs):
        return inputs * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


def _runtime_identity() -> dict[str, str]:
    return {
        "python": "3.11-test",
        "torch": "2-test",
        "torchvision": "0.18-test",
        "numpy": "1.26-test",
        "pillow": "10-test",
    }


if __name__ == "__main__":
    unittest.main()
