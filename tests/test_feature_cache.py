from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly.multiscale import compute_anomaly_heatmap
from features.cache import FeatureCache, FeatureCacheError, feature_cache_key
from features.dinov2 import ColorPatchFeatureExtractor, PatchFeatureMap


class FeatureCacheTest(unittest.TestCase):
    def test_round_trip_preserves_features_and_geometry_metadata(self) -> None:
        original = PatchFeatureMap(
            features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            image_size=(28, 28),
            patch_size=14,
            source_size=(40, 20),
            content_box=(0, 7, 28, 21),
        )
        key = _cache_key()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            cache.save(key, original)
            restored = cache.load(key)

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
            first = cache.get_or_compute(_cache_key(), compute)
            second = cache.get_or_compute(_cache_key(), compute)

        self.assertEqual(calls, 1)
        np.testing.assert_array_equal(first.features, second.features)

    def test_corrupt_entry_raises_without_recomputing(self) -> None:
        calls = 0
        key = _cache_key()

        def compute() -> PatchFeatureMap:
            nonlocal calls
            calls += 1
            return _feature_map()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            cache.path_for_key(key).parent.mkdir(parents=True, exist_ok=True)
            cache.path_for_key(key).write_bytes(b"not an npz archive")

            with self.assertRaisesRegex(FeatureCacheError, "Could not read feature cache"):
                cache.get_or_compute(key, compute)

        self.assertEqual(calls, 0)

    def test_invalid_geometry_payload_is_rejected(self) -> None:
        key = _cache_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            path = cache.path_for_key(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                cache_format_version=np.asarray(1, dtype=np.int32),
                cache_key=np.asarray(key),
                features=np.ones((2, 2, 3), dtype=np.float32),
                image_size=np.asarray((28, 28), dtype=np.int32),
                patch_size=np.asarray(14, dtype=np.int32),
                source_size=np.asarray((40, 20), dtype=np.int32),
                content_box=np.asarray((0, 7, 29, 21), dtype=np.int32),
            )

            with self.assertRaisesRegex(FeatureCacheError, "Invalid feature cache"):
                cache.load(key)

    def test_transplanted_payload_is_rejected(self) -> None:
        first_key = _cache_key()
        second_key = feature_cache_key(
            "pcb1/b",
            "a" * 64,
            "extractor-rev-a",
            "dinov2_vits14",
            518,
            14,
            "query:global:0,0,1024,768",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            first_path = cache.save(first_key, _feature_map())
            second_path = cache.path_for_key(second_key)
            second_path.write_bytes(first_path.read_bytes())

            with self.assertRaisesRegex(FeatureCacheError, "cache key does not match"):
                cache.load(second_key)

    def test_nonfinite_features_are_not_cached(self) -> None:
        feature_map = PatchFeatureMap(
            features=np.asarray([[[np.nan]]], dtype=np.float32),
            image_size=(1, 1),
            patch_size=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "finite"):
                cache.save(_cache_key(), feature_map)


class MultiScaleFeatureCacheTest(unittest.TestCase):
    def test_global_and_crop_views_use_distinct_entries_and_then_hit_cache(self) -> None:
        image = Image.new("RGB", (8, 8), (16, 32, 48))
        extractor = _CountingExtractor(image_size=4, patch_size=2)
        memory_bank = np.zeros((1, 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            key_for_view = lambda view: feature_cache_key(
                "pcb1/query",
                "a" * 64,
                "extractor-rev-a",
                "color_patch",
                4,
                2,
                f"query:{view}",
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


def _feature_map() -> PatchFeatureMap:
    return PatchFeatureMap(
        features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        image_size=(28, 28),
        patch_size=14,
        source_size=(40, 20),
        content_box=(0, 7, 28, 21),
    )


if __name__ == "__main__":
    unittest.main()
