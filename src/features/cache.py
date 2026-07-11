"""Deterministic, validated on-disk cache for patch feature maps."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from features.dinov2 import PatchFeatureMap


_CACHE_FORMAT_VERSION = 1
_CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_FIELDS = {
    "cache_format_version",
    "cache_key",
    "features",
    "image_size",
    "patch_size",
    "source_size",
    "content_box",
}


class FeatureCacheError(RuntimeError):
    """Raised when an existing cache artifact cannot be trusted."""


def feature_cache_key(
    sample_id: str,
    image_sha256: str,
    extractor_revision: str,
    backbone: str,
    image_size: int,
    patch_size: int,
    view: str,
) -> str:
    """Hash every input that can change one extracted feature map."""

    string_fields = {
        "sample_id": sample_id,
        "image_sha256": image_sha256,
        "extractor_revision": extractor_revision,
        "backbone": backbone,
        "view": view,
    }
    for name, value in string_fields.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    for name, value in (("image_size", image_size), ("patch_size", patch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    payload = {
        **string_fields,
        "image_size": image_size,
        "patch_size": patch_size,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a source image without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extractor_source_revision(extractor: object) -> str:
    """Fingerprint the local source modules that implement an extractor."""

    from features import dinov2

    modules = {inspect.getmodule(type(extractor)), dinov2}
    digest = hashlib.sha256()
    digest.update(f"{type(extractor).__module__}.{type(extractor).__qualname__}\n".encode())
    for module in sorted((module for module in modules if module), key=lambda item: item.__name__):
        source_path = inspect.getsourcefile(module)
        if source_path is None:
            raise RuntimeError(f"Could not locate extractor source module {module.__name__}")
        digest.update(f"{module.__name__}\n".encode())
        digest.update(Path(source_path).read_bytes())
    return digest.hexdigest()


class FeatureCache:
    """Store trusted ``PatchFeatureMap`` values as non-pickle NPZ artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for_key(self, key: str) -> Path:
        if not isinstance(key, str) or _CACHE_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("feature cache key must be a lowercase SHA-256 digest")
        return self.root / f"{key}.npz"

    def load(self, key: str) -> PatchFeatureMap | None:
        path = self.path_for_key(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                fields = set(payload.files)
                if not _REQUIRED_FIELDS.issubset(fields):
                    missing = sorted(_REQUIRED_FIELDS - fields)
                    raise FeatureCacheError(f"Invalid feature cache {path}: missing {missing}")
                return self._restore_feature_map(key, path, payload)
        except FeatureCacheError:
            raise
        except Exception as exc:
            raise FeatureCacheError(f"Could not read feature cache {path}: {exc}") from exc

    def save(self, key: str, feature_map: PatchFeatureMap) -> Path:
        path = self.path_for_key(key)
        features = np.asarray(feature_map.features)
        if features.dtype != np.float32:
            raise ValueError("cached features must have dtype float32")
        if not np.isfinite(features).all():
            raise ValueError("cached features must contain only finite values")

        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=f".{key}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    cache_format_version=np.asarray(_CACHE_FORMAT_VERSION, dtype=np.int32),
                    cache_key=np.asarray(key),
                    features=features,
                    image_size=np.asarray(feature_map.image_size, dtype=np.int32),
                    patch_size=np.asarray(feature_map.patch_size, dtype=np.int32),
                    source_size=np.asarray(feature_map.source_size, dtype=np.int32),
                    content_box=np.asarray(feature_map.content_box, dtype=np.int32),
                )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], PatchFeatureMap],
    ) -> PatchFeatureMap:
        cached = self.load(key)
        if cached is not None:
            return cached
        result = compute()
        if not isinstance(result, PatchFeatureMap):
            raise TypeError("feature cache compute callback must return PatchFeatureMap")
        self.save(key, result)
        return result

    @staticmethod
    def _restore_feature_map(key: str, path: Path, payload) -> PatchFeatureMap:
        version = _scalar_integer(payload["cache_format_version"], "cache_format_version")
        if version != _CACHE_FORMAT_VERSION:
            raise FeatureCacheError(
                f"Invalid feature cache {path}: unsupported format version {version}"
            )

        stored_key_array = np.asarray(payload["cache_key"])
        if stored_key_array.shape != () or stored_key_array.dtype.kind not in {"U", "S"}:
            raise FeatureCacheError(f"Invalid feature cache {path}: malformed cache_key")
        stored_key = str(stored_key_array.item())
        if stored_key != key:
            raise FeatureCacheError(
                f"Invalid feature cache {path}: cache key does not match requested key"
            )

        features = np.asarray(payload["features"])
        if features.dtype != np.float32 or features.ndim != 3:
            raise FeatureCacheError(
                f"Invalid feature cache {path}: features must be a float32 3D array"
            )
        if not np.isfinite(features).all():
            raise FeatureCacheError(f"Invalid feature cache {path}: features must be finite")

        try:
            return PatchFeatureMap(
                features=features.copy(),
                image_size=_integer_tuple(payload["image_size"], "image_size", 2),
                patch_size=_scalar_integer(payload["patch_size"], "patch_size"),
                source_size=_integer_tuple(payload["source_size"], "source_size", 2),
                content_box=_integer_tuple(payload["content_box"], "content_box", 4),
            )
        except ValueError as exc:
            raise FeatureCacheError(f"Invalid feature cache {path}: {exc}") from exc


def _scalar_integer(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u"}:
        raise FeatureCacheError(f"{name} must be an integer scalar")
    return int(array.item())


def _integer_tuple(value: np.ndarray, name: str, length: int) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in {"i", "u"}:
        raise FeatureCacheError(f"{name} must contain exactly {length} integers")
    return tuple(int(item) for item in array)
