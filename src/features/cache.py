"""Deterministic, validated on-disk cache for patch feature maps."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import tempfile
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import PIL

from features.dinov2 import PatchFeatureMap

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback retains thread safety.
    fcntl = None


_CACHE_FORMAT_VERSION = 2
_CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_FIELDS = {
    "cache_format_version",
    "cache_key",
    "identity_json",
    "features",
    "image_size",
    "patch_size",
    "source_size",
    "content_box",
}
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class FeatureCacheError(RuntimeError):
    """Raised when an existing cache artifact cannot be trusted."""


@dataclass(frozen=True)
class FeatureCacheIdentity:
    """Canonical identity and expected geometry for one extracted view."""

    sample_id: str
    image_sha256: str
    extractor_revision: str
    backbone: str
    image_size: int
    patch_size: int
    view: str
    source_size: tuple[int, int]

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "image_sha256",
            "extractor_revision",
            "backbone",
            "view",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("image_size", "patch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        source_size = _positive_integer_tuple(self.source_size, "source_size", 2)
        if _source_size_from_view(self.view) != source_size:
            raise ValueError("view geometry must match source_size")

    @property
    def canonical_json(self) -> str:
        return _canonical_json(
            {
                "sample_id": self.sample_id,
                "image_sha256": self.image_sha256,
                "extractor_revision": self.extractor_revision,
                "backbone": self.backbone,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "view": self.view,
                "source_size": list(self.source_size),
            }
        )

    @property
    def key(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


FeatureCacheReference: TypeAlias = str | FeatureCacheIdentity


def feature_cache_key(
    sample_id: str,
    image_sha256: str,
    extractor_revision: str,
    backbone: str,
    image_size: int,
    patch_size: int,
    view: str,
) -> str:
    """Compatibility helper returning a key for the original seven fields."""

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
    payload = {**string_fields, "image_size": image_size, "patch_size": patch_size}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a source image without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numerical_runtime_identity() -> dict[str, str | None]:
    """Return versions of libraries and backends that can alter numerical output."""

    import torch

    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": _distribution_version("torchvision"),
        "numpy": str(np.__version__),
        "pillow": str(PIL.__version__),
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "torch_cudnn": str(cudnn_version) if cudnn_version is not None else None,
    }


def extractor_revision_identity(
    extractor: object,
    runtime_identity: Mapping[str, Any] | None = None,
    precision_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe exact extractor code, model state, configuration, and numerics."""

    model, model_role = _extractor_model(extractor)
    state_identity = _state_dict_identity(model)
    precision = (
        dict(precision_identity)
        if precision_identity is not None
        else _default_precision_identity(state_identity)
    )
    runtime = (
        dict(runtime_identity)
        if runtime_identity is not None
        else numerical_runtime_identity()
    )
    return {
        "identity_version": 1,
        "extractor_class": _qualified_class_name(extractor),
        "model_role": model_role,
        "model_class": _qualified_class_name(model) if model is not None else None,
        "model_structure": _model_structure_identity(model),
        "configuration": {
            "model_name": getattr(extractor, "model_name", None),
            "image_size": getattr(extractor, "image_size", None),
            "patch_size": getattr(extractor, "patch_size", None),
        },
        "device": str(getattr(extractor, "device", "cpu")),
        "precision": precision,
        "runtime": runtime,
        "source_sha256": _extractor_source_hashes(extractor, model),
        "model_state": state_identity,
    }


def extractor_revision_digest(identity: Mapping[str, Any]) -> str:
    """Hash a complete extractor revision identity."""

    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def extractor_source_revision(
    extractor: object,
    runtime_identity: Mapping[str, Any] | None = None,
    precision_identity: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint exact model state, implementation, runtime, device, and precision."""

    identity = extractor_revision_identity(
        extractor,
        runtime_identity=runtime_identity,
        precision_identity=precision_identity,
    )
    return extractor_revision_digest(identity)


class FeatureCache:
    """Store trusted ``PatchFeatureMap`` values as non-pickle NPZ artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for_key(self, key: str) -> Path:
        _validate_cache_key(key)
        return self.root / f"{key}.npz"

    def load(self, reference: FeatureCacheReference) -> PatchFeatureMap | None:
        key, identity_json, identity = _reference_parts(reference)
        path = self.path_for_key(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                fields = set(payload.files)
                if not _REQUIRED_FIELDS.issubset(fields):
                    missing = sorted(_REQUIRED_FIELDS - fields)
                    raise FeatureCacheError(f"Invalid feature cache {path}: missing {missing}")
                feature_map = self._restore_feature_map(
                    key,
                    identity_json,
                    path,
                    payload,
                )
            _validate_identity_geometry(identity, feature_map)
            return feature_map
        except FeatureCacheError:
            raise
        except Exception as exc:
            raise FeatureCacheError(f"Could not read feature cache {path}: {exc}") from exc

    def save(
        self,
        reference: FeatureCacheReference,
        feature_map: PatchFeatureMap,
    ) -> Path:
        key, identity_json, identity = _reference_parts(reference)
        _validate_identity_geometry(identity, feature_map)
        _revalidate_feature_map(feature_map)
        path = self.path_for_key(key)
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
                    identity_json=np.asarray(identity_json),
                    features=feature_map.features,
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
        reference: FeatureCacheReference,
        compute: Callable[[], PatchFeatureMap],
    ) -> PatchFeatureMap:
        cached = self.load(reference)
        if cached is not None:
            return cached
        key, _, _ = _reference_parts(reference)
        with self._key_lock(key):
            cached = self.load(reference)
            if cached is not None:
                return cached
            result = compute()
            if not isinstance(result, PatchFeatureMap):
                raise TypeError("feature cache compute callback must return PatchFeatureMap")
            self.save(reference, result)
            return result

    @contextmanager
    def _key_lock(self, key: str):
        self.root.mkdir(parents=True, exist_ok=True)
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(str(self.root.resolve() / key), threading.Lock())
        with thread_lock:
            lock_path = self.root / f".{key}.lock"
            with lock_path.open("a+b") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _restore_feature_map(
        key: str,
        identity_json: str,
        path: Path,
        payload,
    ) -> PatchFeatureMap:
        version = _scalar_integer(payload["cache_format_version"], "cache_format_version")
        if version != _CACHE_FORMAT_VERSION:
            raise FeatureCacheError(
                f"Invalid feature cache {path}: unsupported format version {version}"
            )
        stored_key = _scalar_string(payload["cache_key"], "cache_key")
        if stored_key != key:
            raise FeatureCacheError(
                f"Invalid feature cache {path}: cache key does not match requested key"
            )
        stored_identity = _scalar_string(payload["identity_json"], "identity_json")
        if stored_identity != identity_json:
            raise FeatureCacheError(
                f"Invalid feature cache {path}: identity does not match requested identity"
            )
        features = np.asarray(payload["features"])
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


def _reference_parts(
    reference: FeatureCacheReference,
) -> tuple[str, str, FeatureCacheIdentity | None]:
    if isinstance(reference, FeatureCacheIdentity):
        return reference.key, reference.canonical_json, reference
    _validate_cache_key(reference)
    return reference, _canonical_json({"cache_key": reference}), None


def _validate_identity_geometry(
    identity: FeatureCacheIdentity | None,
    feature_map: PatchFeatureMap,
) -> None:
    if identity is None:
        return
    if feature_map.image_size != (identity.image_size, identity.image_size):
        raise ValueError("feature map image_size does not match cache identity")
    if feature_map.patch_size != identity.patch_size:
        raise ValueError("feature map patch_size does not match cache identity")
    if feature_map.source_size != identity.source_size:
        raise ValueError("feature map source_size does not match cache identity")
    expected_content_box = _expected_content_box(identity.source_size, identity.image_size)
    if feature_map.content_box != expected_content_box:
        raise ValueError("feature map content_box does not match cache identity")


def _revalidate_feature_map(feature_map: PatchFeatureMap) -> None:
    PatchFeatureMap(
        features=feature_map.features,
        image_size=feature_map.image_size,
        patch_size=feature_map.patch_size,
        source_size=feature_map.source_size,
        content_box=feature_map.content_box,
    )


def _expected_content_box(
    source_size: tuple[int, int],
    image_size: int,
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    scale = min(image_size / source_width, image_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    return (
        offset_x,
        offset_y,
        offset_x + resized_width,
        offset_y + resized_height,
    )


def _source_size_from_view(view: str) -> tuple[int, int]:
    try:
        x1, y1, x2, y2 = (int(value) for value in view.rsplit(":", 1)[1].split(","))
    except (IndexError, ValueError) as exc:
        raise ValueError("view must end with exact x1,y1,x2,y2 geometry") from exc
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("view must contain non-empty geometry")
    return (x2 - x1, y2 - y1)


def _extractor_model(extractor: object) -> tuple[object | None, str | None]:
    for attribute in ("model", "backbone"):
        model = getattr(extractor, attribute, None)
        if model is not None:
            return model, attribute
    return None, None


def _model_structure_identity(model: object | None) -> dict[str, Any] | None:
    if model is None:
        return None
    named_modules = []
    if hasattr(model, "named_modules"):
        named_modules = [
            [name, _qualified_class_name(module)]
            for name, module in model.named_modules()
        ]
    graph = getattr(model, "graph", None)
    return {
        "training": getattr(model, "training", None),
        "graph": str(graph) if graph is not None else None,
        "named_modules": named_modules,
    }


def _state_dict_identity(model: object | None) -> dict[str, Any]:
    digest = hashlib.sha256()
    if model is None or not hasattr(model, "state_dict"):
        return {
            "available": False,
            "sha256": digest.hexdigest(),
            "tensor_count": 0,
            "num_bytes": 0,
            "dtypes": [],
        }
    state_dict = model.state_dict()
    if not isinstance(state_dict, Mapping):
        raise TypeError("model state_dict() must return a mapping")
    total_bytes = 0
    dtypes = set()
    for name in sorted(state_dict):
        if not isinstance(name, str):
            raise TypeError("model state_dict keys must be strings")
        dtype, shape, raw = _state_value_bytes(state_dict[name])
        dtypes.add(dtype)
        total_bytes += len(raw)
        for field in (name.encode(), dtype.encode(), _canonical_json(shape).encode(), raw):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return {
        "available": True,
        "sha256": digest.hexdigest(),
        "tensor_count": len(state_dict),
        "num_bytes": total_bytes,
        "dtypes": sorted(dtypes),
    }


def _default_precision_identity(state_identity: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {
        "input_dtype": "float32",
        "output_dtype": "float32",
        "autocast": "disabled",
        "model_state_dtypes": state_identity["dtypes"],
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _state_value_bytes(value: object) -> tuple[str, list[int], bytes]:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return str(array.dtype), list(array.shape), array.tobytes(order="C")
    if all(hasattr(value, attribute) for attribute in ("detach", "cpu", "contiguous")):
        tensor = value.detach().cpu().contiguous()
        dtype = str(tensor.dtype)
        shape = list(tensor.shape)
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            raw = tensor.view(dtype=getattr(__import__("torch"), "uint8")).numpy().tobytes()
        return dtype, shape, raw
    raise TypeError(f"Unsupported model state value type: {type(value).__name__}")


def _extractor_source_hashes(extractor: object, model: object | None) -> dict[str, str]:
    from features import dinov2

    modules = {inspect.getmodule(type(extractor)), dinov2}
    if model is not None:
        modules.add(inspect.getmodule(type(model)))
    hashes = {}
    for module in sorted((module for module in modules if module), key=lambda item: item.__name__):
        try:
            source_path = inspect.getsourcefile(module)
        except TypeError:
            source_path = None
        if source_path is None:
            continue
        hashes[module.__name__] = sha256_file(source_path)
    if "features.dinov2" not in hashes:
        raise RuntimeError("Could not fingerprint shared feature preprocessing source")
    return hashes


def _qualified_class_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_cache_key(key: str) -> None:
    if not isinstance(key, str) or _CACHE_KEY_PATTERN.fullmatch(key) is None:
        raise ValueError("feature cache key must be a lowercase SHA-256 digest")


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise FeatureCacheError(f"{name} must be a unicode scalar")
    return str(array.item())


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


def _positive_integer_tuple(value, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    return tuple(value)
