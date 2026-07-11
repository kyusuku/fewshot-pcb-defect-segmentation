"""Canonical, privacy-safe provenance records for experiment outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from experiments.spec import RunSpec


DEFAULT_LIBRARIES = ("numpy", "Pillow", "PyYAML", "torch", "torchvision")
_SENSITIVE_FRAGMENTS = ("secret", "token", "password", "api_key", "credential")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"checksum source is not a regular file: {source.name}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> str:
    """Serialize finite JSON with deterministic key and whitespace rules."""

    _validate_json_value(payload, "payload")
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Atomically replace a JSON artifact using canonical serialization."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def git_state(repo_root: str | Path) -> dict[str, object]:
    """Read the exact commit and tracked/untracked dirty state of a repository."""

    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("git rev-parse returned an invalid commit SHA")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    return {"commit": commit, "dirty": bool(status.strip())}


def build_provenance(
    run_spec: RunSpec,
    manifest_path: str | Path,
    support_ids: Sequence[str],
    git_commit: str | None = None,
    *,
    git_dirty: bool | None = None,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    config: Mapping[str, object] | None = None,
    checkpoint_paths: Mapping[str, str | Path] | None = None,
    model_config_paths: Mapping[str, str | Path] | None = None,
    cache_identity: Mapping[str, object] | None = None,
    artifact_identities: Mapping[str, object] | None = None,
    library_names: Sequence[str] = DEFAULT_LIBRARIES,
) -> dict[str, object]:
    """Build one immutable provenance record without retaining private paths."""

    if not isinstance(run_spec, RunSpec):
        raise TypeError("run_spec must be a RunSpec")
    support = _validate_support_ids(support_ids)
    expected_support_count = 0 if run_spec.method == "sam2_only" else run_spec.k
    if len(support) != expected_support_count:
        raise ValueError(
            f"run {run_spec.run_id} declares k={run_spec.k} but has "
            f"{len(support)} unique support IDs"
        )
    git = _resolve_git_state(git_commit, git_dirty, repo_root)
    root = Path(repo_root).resolve() if repo_root is not None else None
    manifest = _file_identity(manifest_path, root)
    experiment_config = _config_identity(config_path, config, root)
    checkpoints = _file_identity_map(checkpoint_paths or {}, root, "checkpoints")
    model_configs = _file_identity_map(model_config_paths or {}, root, "model_configs")
    cache = _mapping_identity(cache_identity or {}, "cache_identity")
    artifacts = _mapping_identity(artifact_identities or {}, "artifact_identities")
    libraries = _library_versions(library_names)
    overrides = run_spec.overrides

    record: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_spec.run_id,
        "run_spec": run_spec.to_dict(),
        "run_spec_sha256": run_spec.identity_sha256,
        "overrides": overrides,
        "overrides_sha256": sha256_json(overrides),
        "manifest": manifest,
        "support_ids": support,
        "support_ids_sha256": sha256_json(support),
        "git": git,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "platform": {
                "machine": platform.machine(),
                "release": platform.release(),
                "system": platform.system(),
            },
            "libraries": libraries,
        },
        "experiment_config": experiment_config,
        "checkpoints": checkpoints,
        "model_configs": model_configs,
        "cache_identity": cache,
        "artifact_identities": artifacts,
    }
    # This final serialization gate also prevents an extension from adding NaN.
    canonical_json(record)
    return record


def _resolve_git_state(
    commit: str | None, dirty: bool | None, repo_root: str | Path | None
) -> dict[str, object]:
    if commit is None and dirty is None:
        if repo_root is None:
            raise ValueError("repo_root is required when git state is not supplied")
        return git_state(repo_root)
    if commit is None or dirty is None:
        raise ValueError("git_commit and git_dirty must be supplied together")
    if (
        not isinstance(commit, str)
        or not commit
        or any(character.isspace() for character in commit)
    ):
        raise ValueError("git_commit must be a non-empty token")
    if not isinstance(dirty, bool):
        raise ValueError("git_dirty must be a boolean")
    supplied = {"commit": commit, "dirty": dirty}
    if repo_root is not None:
        observed = git_state(repo_root)
        if supplied != observed:
            raise ValueError(f"supplied git state does not match repository state: {observed}")
    return supplied


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"unable to inspect git repository {root.name!r}") from exc
    return result.stdout


def _file_identity(path: str | Path, root: Path | None) -> dict[str, str]:
    source = Path(path)
    return {"path": _public_path(source, root), "sha256": sha256_file(source)}


def _file_identity_map(
    paths: Mapping[str, str | Path], root: Path | None, context: str
) -> dict[str, object]:
    _validate_identity_keys(paths, context)
    return {
        name: _file_identity(path, root)
        for name, path in sorted(paths.items(), key=lambda item: item[0])
    }


def _config_identity(
    path: str | Path | None,
    config: Mapping[str, object] | None,
    root: Path | None,
) -> dict[str, object]:
    if path is None and config is None:
        return {}
    if path is None or config is None:
        raise ValueError("config_path and config must be supplied together")
    _validate_identity_mapping(config, "experiment_config")
    canonical_content = json.loads(canonical_json(dict(config)))
    file_identity = _file_identity(path, root)
    return {
        **file_identity,
        "canonical": canonical_content,
        "canonical_sha256": sha256_json(canonical_content),
    }


def _mapping_identity(payload: Mapping[str, object], context: str) -> dict[str, object]:
    _validate_identity_mapping(payload, context)
    canonical = json.loads(canonical_json(dict(payload)))
    if not canonical:
        return {}
    return {"canonical": canonical, "sha256": sha256_json(canonical)}


def _public_path(path: Path, root: Path | None) -> str:
    if path.is_absolute():
        resolved = path.resolve()
        if root is not None:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                relative = Path(resolved.name)
        else:
            relative = Path(resolved.name)
    else:
        relative = path
    value = relative.as_posix()
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or any(part in {"", "."} for part in pure.parts):
        raise ValueError("artifact path must be a normalized relative public path")
    return value


def _validate_support_ids(support_ids: Sequence[str]) -> list[str]:
    if isinstance(support_ids, (str, bytes)):
        raise ValueError("support_ids must be a sequence of support IDs")
    normalized = list(support_ids)
    if not all(isinstance(value, str) and value for value in normalized):
        raise ValueError("every support ID must be a non-empty string")
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate support IDs are not allowed")
    for value in normalized:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or any(
            character.isspace() and character not in {" "} for character in value
        ):
            raise ValueError(f"unsafe support ID: {value!r}")
    return sorted(normalized)


def _validate_identity_mapping(payload: Mapping[str, object], context: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a mapping")
    _validate_identity_keys(payload, context)
    for key, value in payload.items():
        if isinstance(value, Mapping):
            _validate_identity_mapping(value, f"{context}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    _validate_identity_mapping(item, f"{context}.{key}[{index}]")
                elif isinstance(item, str) and Path(item).is_absolute():
                    raise ValueError(f"{context} must not expose absolute/private paths")
        elif isinstance(value, str) and Path(value).is_absolute():
            raise ValueError(f"{context} must not expose absolute/private paths")


def _validate_identity_keys(payload: Mapping[str, object], context: str) -> None:
    for key in payload:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        lowered = key.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
            raise ValueError(f"{context} contains sensitive identity key {key!r}")


def _library_versions(names: Sequence[str]) -> dict[str, str | None]:
    if isinstance(names, (str, bytes)) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("library_names must be a sequence of non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("library_names must not contain duplicates")
    versions: dict[str, str | None] = {}
    for name in sorted(names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_json_value(value: object, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must contain only finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{context} must use string JSON object keys")
        for key, item in value.items():
            _validate_json_value(item, f"{context}.{key}")
        return
    raise TypeError(f"{context} contains a non-JSON value of type {type(value).__name__}")
