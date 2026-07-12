"""Canonical, privacy-safe provenance records for experiment outputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from experiments.spec import ReferencedArtifact, RunSpec, load_yaml_mapping


DEFAULT_LIBRARIES = ("numpy", "Pillow", "PyYAML", "torch", "torchvision")
_SENSITIVE_FRAGMENTS = ("secret", "token", "password", "api_key", "credential")
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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
    state: dict[str, object] = {"commit": commit, "dirty": bool(status.strip())}
    if state["dirty"]:
        state["dirty_identity"] = _dirty_identity(root)
    return state


def _dirty_identity(root: Path) -> dict[str, object]:
    staged = _git_bytes(root, "diff", "--cached", "--binary", "--no-ext-diff")
    unstaged = _git_bytes(root, "diff", "--binary", "--no-ext-diff")
    untracked_output = _git_bytes(
        root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked: list[dict[str, str]] = []
    for raw_path in sorted(value for value in untracked_output.split(b"\0") if value):
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("untracked paths must be valid UTF-8") from exc
        _validate_public_artifact_string(relative, "untracked path")
        if _contains_sensitive_fragment(relative):
            raise ValueError(f"sensitive untracked path cannot enter provenance: {relative!r}")
        source = root / relative
        if not source.is_file():
            raise ValueError(f"untracked provenance source is not a file: {relative!r}")
        untracked.append({"path": relative, "sha256": sha256_file(source)})
    return {
        "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
        "unstaged_diff_sha256": hashlib.sha256(unstaged).hexdigest(),
        "untracked": untracked,
    }


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
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Build one immutable provenance record without retaining private paths."""

    if not isinstance(run_spec, RunSpec):
        raise TypeError("run_spec must be a RunSpec")
    support = validate_support_manifest(manifest_path, run_spec, support_ids)
    if not isinstance(allow_dirty, bool):
        raise ValueError("allow_dirty must be a boolean")
    git = _resolve_git_state(git_commit, git_dirty, repo_root, allow_dirty)
    root = Path(repo_root).resolve() if repo_root is not None else None
    manifest = _file_identity(manifest_path, root)
    experiment_config = _config_identity(config_path, config, root)
    checkpoints = _file_identity_map(checkpoint_paths or {}, root, "checkpoints")
    model_configs = _file_identity_map(model_config_paths or {}, root, "model_configs")
    cache = _mapping_identity(cache_identity or {}, "cache_identity")
    artifacts = _mapping_identity(artifact_identities or {}, "artifact_identities")
    libraries = _library_versions(library_names)
    environment = {
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
    }
    overrides = run_spec.overrides
    dependency_identities = {
        dependency.run_id: dependency.expected_run_spec_sha256
        for dependency in run_spec.dependencies
    }
    if any(identity is None for identity in dependency_identities.values()):
        raise ValueError("every dependency must declare an expected run-spec identity")

    record: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_spec.run_id,
        "run_spec": run_spec.to_dict(),
        "run_spec_sha256": run_spec.identity_sha256,
        "overrides": overrides,
        "overrides_sha256": sha256_json(overrides),
        "dependency_identities": dependency_identities,
        "dependency_identities_sha256": sha256_json(dependency_identities),
        "manifest": manifest,
        "support_ids": support,
        "support_ids_sha256": sha256_json(support),
        "git": git,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": environment,
        "experiment_config": experiment_config,
        "checkpoints": checkpoints,
        "model_configs": model_configs,
        "cache_identity": cache,
        "artifact_identities": artifacts,
        "execution": {},
        "observed_identities": {},
    }
    record["effective_execution_sha256"] = compute_effective_execution_sha256(record)
    canonical_json(record)
    return record


def _resolve_git_state(
    commit: str | None,
    dirty: bool | None,
    repo_root: str | Path | None,
    allow_dirty: bool,
) -> dict[str, object]:
    if commit is None and dirty is None:
        if repo_root is None:
            raise ValueError("repo_root is required when git state is not supplied")
        observed = git_state(repo_root)
        return _enforce_dirty_policy(observed, allow_dirty, repo_root)
    if commit is None or dirty is None:
        raise ValueError("git_commit and git_dirty must be supplied together")
    _validate_git_commit(commit)
    if not isinstance(dirty, bool):
        raise ValueError("git_dirty must be a boolean")
    supplied = {"commit": commit, "dirty": dirty}
    if repo_root is not None:
        observed = git_state(repo_root)
        if supplied != {"commit": observed["commit"], "dirty": observed["dirty"]}:
            raise ValueError(f"supplied git state does not match repository state: {observed}")
        supplied = observed
    return _enforce_dirty_policy(supplied, allow_dirty, repo_root)


def _enforce_dirty_policy(
    state: dict[str, object], allow_dirty: bool, repo_root: str | Path | None
) -> dict[str, object]:
    if not state["dirty"]:
        return state
    if not allow_dirty:
        raise ValueError("dirty repository provenance is rejected by default")
    if repo_root is None or "dirty_identity" not in state:
        raise ValueError("allow_dirty requires repo_root to hash all repository changes")
    return state


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


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
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
    file_content = load_yaml_mapping(path)
    if canonical_json(file_content) != canonical_json(canonical_content):
        raise ValueError("experiment config file does not match supplied config")
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
    _validate_public_artifact_string(value, "artifact path")
    return value


def validate_support_manifest(
    manifest_path: str | Path,
    run_spec: RunSpec,
    support_ids: Sequence[str],
) -> list[str]:
    """Verify that a run's exact support set is legal in its frozen fold manifest."""

    if not isinstance(run_spec, RunSpec):
        raise TypeError("run_spec must be a RunSpec")
    normalized = _validate_support_ids(support_ids)
    expected_count = 0 if run_spec.method == "sam2_only" else run_spec.k
    if len(normalized) != expected_count:
        raise ValueError(
            f"run {run_spec.run_id} declares k={run_spec.k} but has "
            f"{len(normalized)} unique support IDs"
        )
    source = Path(manifest_path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("manifest must have a CSV header")
        if len(headers) != len(set(headers)):
            raise ValueError("manifest must not contain duplicate CSV columns")
        required = {"dataset", "sample_id", "category", "fold_id", "fold_split", "label"}
        missing = sorted(required - set(headers))
        if missing:
            raise ValueError(f"manifest is missing support-validation columns: {missing}")
        rows = list(reader)

    fold_value = str(run_spec.fold_id)
    for support_id in normalized:
        matching_id = [row for row in rows if row.get("sample_id") == support_id]
        matching_fold = [row for row in matching_id if row.get("fold_id") == fold_value]
        if len(matching_fold) != 1:
            raise ValueError(
                f"support ID {support_id!r} must exist exactly once for fold {run_spec.fold_id}"
            )
        row = matching_fold[0]
        if row.get("dataset") != "visa_pcb":
            raise ValueError(f"support ID {support_id!r} must have dataset=visa_pcb")
        if row.get("category") != run_spec.category:
            raise ValueError(
                f"support ID {support_id!r} category does not match {run_spec.category}"
            )
        if row.get("label") != "0":
            raise ValueError(f"support ID {support_id!r} must be normal with label=0")
        if row.get("fold_split") != "dev":
            raise ValueError(f"support ID {support_id!r} must have fold_split=dev")
    return normalized


def _validate_support_ids(support_ids: Sequence[str]) -> list[str]:
    if isinstance(support_ids, (str, bytes)):
        raise ValueError("support_ids must be a sequence of support IDs")
    normalized = list(support_ids)
    if not all(isinstance(value, str) and value for value in normalized):
        raise ValueError("every support ID must be a non-empty string")
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate support IDs are not allowed")
    for value in normalized:
        try:
            _validate_public_artifact_string(value, "support ID")
        except ValueError as exc:
            raise ValueError(f"unsafe support ID: {value!r}") from exc
    return sorted(normalized)


def resolve_referenced_artifacts(
    run_root: str | Path,
    reference: ReferencedArtifact,
) -> list[dict[str, str]]:
    """Resolve and hash every file declared by a writer-owned CSV reference."""

    if not isinstance(reference, ReferencedArtifact):
        raise TypeError("reference must be a ReferencedArtifact")
    root = Path(run_root).resolve()
    csv_path = (root / reference.csv_path).resolve()
    _require_within_base(csv_path, root, "referenced artifact CSV", "run_root")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if headers is None or reference.path_column not in headers:
            raise ValueError(
                f"{reference.csv_path} is missing referenced column "
                f"{reference.path_column!r}"
            )
        if len(headers) != len(set(headers)):
            raise ValueError(f"{reference.csv_path} has duplicate CSV columns")
        rows = list(reader)
    if reference.required_per_row and not rows:
        raise ValueError(f"{reference.csv_path} required index has zero rows")
    resolved: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        value = (row.get(reference.path_column) or "").strip()
        if not value:
            if reference.required_per_row:
                raise ValueError(
                    f"{reference.csv_path} row {index} requires {reference.path_column}"
                )
            continue
        _validate_public_artifact_string(value, f"{reference.path_column} row {index}")
        base = csv_path.parent if reference.path_scope == "csv_parent" else root
        artifact = (base / value).resolve()
        _require_within_base(
            artifact,
            base,
            f"{reference.path_column} row {index}",
            reference.path_scope,
        )
        if not artifact.is_file():
            raise ValueError(
                f"referenced artifact does not exist for {reference.path_column} row {index}"
            )
        resolved.append({"path": str(artifact), "sha256": sha256_file(artifact)})
    paths = [identity["path"] for identity in resolved]
    if len(paths) != len(set(paths)):
        raise ValueError("referenced artifact CSV contains duplicate output paths")
    return resolved


def _require_within_base(path: Path, base: Path, context: str, scope: str) -> None:
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{context} resolves outside declared {scope} base") from exc


def compute_effective_execution_sha256(provenance: Mapping[str, object]) -> str:
    """Hash every recorded input that can affect experiment outputs."""

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    _validate_effective_hash_inputs(provenance)
    fields = (
        "run_spec_sha256",
        "experiment_config",
        "manifest",
        "support_ids",
        "support_ids_sha256",
        "git",
        "dependency_identities",
        "dependency_identities_sha256",
        "checkpoints",
        "model_configs",
        "cache_identity",
        "artifact_identities",
        "environment",
        "python",
        "platform",
        "execution",
        "observed_identities",
    )
    missing = [field for field in fields if field not in provenance]
    if missing:
        raise ValueError(f"effective execution identity is missing fields: {missing}")
    return sha256_json({field: provenance[field] for field in fields})


def validate_resume_identity(
    provenance: Mapping[str, object],
    *,
    expected_effective_execution_sha256: str,
) -> None:
    """Reject resume unless the full effective execution identity matches."""

    _validate_sha256(
        expected_effective_execution_sha256,
        "expected effective execution identity",
    )
    observed = compute_effective_execution_sha256(provenance)
    stored = provenance.get("effective_execution_sha256")
    _validate_sha256(stored, "stored effective execution identity")
    if stored != observed:
        raise ValueError("stored effective execution identity is internally inconsistent")
    if observed != expected_effective_execution_sha256:
        raise ValueError("effective execution identity does not match requested run")


def _validate_effective_hash_inputs(provenance: Mapping[str, object]) -> None:
    _validate_sha256(provenance.get("run_spec_sha256"), "run_spec")
    execution = provenance.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("execution identity must be a mapping")
    observed = provenance.get("observed_identities")
    if not isinstance(observed, Mapping):
        raise ValueError("observed identities must be a mapping")
    run_spec = provenance.get("run_spec")
    if not isinstance(run_spec, Mapping) or sha256_json(run_spec) != provenance.get(
        "run_spec_sha256"
    ):
        raise ValueError("run_spec SHA-256 does not match canonical run_spec")
    manifest = _required_mapping(provenance.get("manifest"), "manifest")
    _validate_sha256(manifest.get("sha256"), "manifest")
    support_ids = provenance.get("support_ids")
    if not isinstance(support_ids, list) or support_ids != sorted(support_ids):
        raise ValueError("support_ids must be a sorted list")
    _validate_sha256(provenance.get("support_ids_sha256"), "support IDs")
    if sha256_json(support_ids) != provenance.get("support_ids_sha256"):
        raise ValueError("support IDs SHA-256 does not match support_ids")
    config = _required_mapping(provenance.get("experiment_config"), "experiment config")
    _validate_sha256(config.get("sha256"), "experiment config file")
    _validate_sha256(config.get("canonical_sha256"), "canonical experiment config")
    if sha256_json(config.get("canonical")) != config.get("canonical_sha256"):
        raise ValueError("canonical experiment config SHA-256 does not match content")
    dependencies = _required_mapping(
        provenance.get("dependency_identities"), "dependency identities"
    )
    for run_id, identity in dependencies.items():
        _validate_sha256(identity, f"dependency {run_id}")
    _validate_sha256(
        provenance.get("dependency_identities_sha256"), "dependency identities"
    )
    if sha256_json(dependencies) != provenance.get("dependency_identities_sha256"):
        raise ValueError("dependency identities SHA-256 does not match content")
    git = _required_mapping(provenance.get("git"), "git")
    _validate_git_commit(git.get("commit"))
    if not isinstance(git.get("dirty"), bool):
        raise ValueError("git dirty state must be boolean")
    if git["dirty"]:
        dirty = _required_mapping(git.get("dirty_identity"), "git dirty identity")
        _validate_sha256(dirty.get("staged_diff_sha256"), "staged git diff")
        _validate_sha256(dirty.get("unstaged_diff_sha256"), "unstaged git diff")
        untracked = dirty.get("untracked")
        if not isinstance(untracked, list):
            raise ValueError("git dirty untracked identity must be a list")
        for index, identity in enumerate(untracked):
            item = _required_mapping(identity, f"untracked identity {index}")
            _validate_sha256(item.get("sha256"), f"untracked identity {index}")
    for field_name in ("checkpoints", "model_configs"):
        identities = _required_mapping(provenance.get(field_name), field_name)
        for name, identity in identities.items():
            item = _required_mapping(identity, f"{field_name}.{name}")
            _validate_sha256(item.get("sha256"), f"{field_name}.{name}")
    for field_name in ("cache_identity", "artifact_identities"):
        identity = _required_mapping(provenance.get(field_name), field_name)
        if identity:
            _validate_sha256(identity.get("sha256"), field_name)
            if sha256_json(identity.get("canonical")) != identity.get("sha256"):
                raise ValueError(f"{field_name} SHA-256 does not match canonical content")
    _required_mapping(provenance.get("environment"), "environment")
    for field_name in ("python", "platform"):
        if not isinstance(provenance.get(field_name), str) or not provenance[field_name]:
            raise ValueError(f"{field_name} runtime identity must be a non-empty string")


def _required_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _validate_sha256(value: object, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in _SHA256_CHARACTERS for character in value
    ):
        raise ValueError(f"{context} must be a lowercase 64-character SHA-256")


def _validate_git_commit(value: object) -> None:
    if not isinstance(value, str) or len(value) not in {40, 64} or any(
        character not in _SHA256_CHARACTERS for character in value
    ):
        raise ValueError("git_commit must be a lowercase 40- or 64-character hash")


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
                else:
                    _validate_identity_scalar(item, f"{context}.{key}[{index}]")
        else:
            _validate_identity_scalar(value, f"{context}.{key}")


def _validate_identity_keys(payload: Mapping[str, object], context: str) -> None:
    for key in payload:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        _validate_ascii_text(key, f"{context} key")
        if _contains_sensitive_fragment(key):
            raise ValueError(f"{context} contains sensitive identity key {key!r}")


def _validate_identity_scalar(value: object, context: str) -> None:
    if isinstance(value, str):
        _validate_ascii_text(value, context)
        if (
            value in {".", ".."}
            or value.startswith(("/", "//", "~"))
            or value.lower().startswith("file:")
            or (len(value) >= 3 and value[0].isalpha() and value[1:3] in {":/", ":\\"})
            or "\\" in value
        ):
            raise ValueError(f"{context} must not expose absolute/private paths or traversal")
        if "/" in value and ".." in PurePosixPath(value).parts:
            raise ValueError(f"{context} must not expose path traversal")
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"{context} must contain JSON scalar values")


def _validate_public_artifact_string(value: str, context: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    _validate_ascii_text(value, context)
    if (
        value.startswith(("/", "//", "~"))
        or value.lower().startswith("file:")
        or (len(value) >= 3 and value[0].isalpha() and value[1:3] in {":/", ":\\"})
        or "\\" in value
    ):
        raise ValueError(f"{context} must be a relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or ".." in path.parts:
        raise ValueError(f"{context} must not contain path traversal")
    if path.as_posix() != value or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{context} must be normalized")


def _validate_ascii_text(value: str, context: str) -> None:
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        raise ValueError(f"{context} must contain printable ASCII text")


def _contains_sensitive_fragment(value: str) -> bool:
    lowered = value.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS)


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
