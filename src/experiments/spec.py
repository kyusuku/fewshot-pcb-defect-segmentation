"""Strict, immutable experiment specifications and matrix expansion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


PCB_CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
PAIRED_METHODS = (
    "patchcore",
    "dinov2_single",
    "dinov2_multi",
    "dinov2_single_sam2",
    "dinov2_multi_sam2",
    "anomaly_consistent_sam2",
)
METHODS = (*PAIRED_METHODS[:3], "sam2_only", *PAIRED_METHODS[3:])

_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_COLUMN_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_RUN_ID_RE = re.compile(
    r"^[a-z0-9_]+__pcb[1-4]__fold[0-9]+__k[0-9]+__seed[0-9]+"
    r"(?:__[a-z0-9][a-z0-9_-]*)?$"
)
_PRIMARY_FIELDS = {
    "name",
    "manifest",
    "fold_id",
    "categories",
    "shots",
    "seeds",
    "methods",
    "dinov2",
    "multi_scale",
    "calibration",
    "sam2",
    "patchcore",
    "sam2_only",
}
_ABLATION_FIELDS = {
    "name",
    "manifest",
    "base_config",
    "base_config_canonical_sha256",
    "dependency_output_root",
    "fold_id",
    "categories",
    "k",
    "seed",
    "ablations",
}
_FROZEN_CONFIG_SHA256 = {
    "arxiv_primary": "1e06656f525e4130f3b8685aa60afcc26a1ec5ffa48ab122c8ac4ed436df8127",
    "arxiv_ablations": "2489dbe090c615678695f401614805829a4e399fb933b2512643ab96e1b1aeee",
    "arxiv_smoke": "2467bab225c1c9cb0a50bc928cb45a5b9fcf8fc3ef79661a5cec84bab7069f28",
}

class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that fails rather than silently replacing duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable YAML key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ReferencedArtifact:
    """Files referenced by a path column in a dependency-owned CSV index."""

    csv_path: str
    path_column: str
    required_per_row: bool = True
    path_scope: str = "csv_parent"
    checksum: str = "sha256_file"

    def __post_init__(self) -> None:
        _validate_relative_path(self.csv_path, "referenced artifact CSV")
        if not isinstance(self.path_column, str) or not _COLUMN_RE.fullmatch(self.path_column):
            raise ValueError(f"invalid referenced artifact path column: {self.path_column!r}")
        if not isinstance(self.required_per_row, bool):
            raise ValueError("referenced artifact required_per_row must be boolean")
        if self.path_scope not in {"csv_parent", "run_root"}:
            raise ValueError("referenced artifact path_scope must be csv_parent or run_root")
        if self.checksum != "sha256_file":
            raise ValueError("referenced artifact checksum must be sha256_file")

    def to_dict(self) -> dict[str, object]:
        return {
            "csv_path": self.csv_path,
            "path_column": self.path_column,
            "required_per_row": self.required_per_row,
            "path_scope": self.path_scope,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ReferencedArtifact:
        expected = {"csv_path", "path_column", "required_per_row", "path_scope", "checksum"}
        _require_exact_fields(payload, expected, "referenced artifact")
        return cls(**payload)


@dataclass(frozen=True)
class RunDependency:
    """An upstream run and the relative artifacts required from it."""

    run_id: str
    artifacts: tuple[str, ...] = ()
    referenced_artifacts: tuple[ReferencedArtifact, ...] = ()
    expected_run_spec_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError(f"invalid dependency run_id: {self.run_id!r}")
        if isinstance(self.artifacts, (str, bytes)) or not isinstance(self.artifacts, Sequence):
            raise ValueError("dependency artifacts must be a sequence and not a string")
        normalized = tuple(self.artifacts)
        if not all(isinstance(artifact, str) for artifact in normalized):
            raise ValueError("dependency artifacts must contain strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("dependency artifacts must not contain duplicates")
        for artifact in normalized:
            _validate_relative_path(artifact, "dependency artifact")
        if isinstance(self.referenced_artifacts, (str, bytes)) or not isinstance(
            self.referenced_artifacts, Sequence
        ):
            raise ValueError("referenced_artifacts must be a sequence and not a string")
        references = tuple(self.referenced_artifacts)
        if not all(isinstance(reference, ReferencedArtifact) for reference in references):
            raise ValueError("referenced_artifacts must contain ReferencedArtifact values")
        reference_keys = [(reference.csv_path, reference.path_column) for reference in references]
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError("referenced_artifacts must not contain duplicate CSV columns")
        if self.expected_run_spec_sha256 is not None and (
            not isinstance(self.expected_run_spec_sha256, str)
            or not _SHA256_RE.fullmatch(self.expected_run_spec_sha256)
        ):
            raise ValueError("expected_run_spec_sha256 must be a lowercase SHA-256")
        object.__setattr__(self, "artifacts", normalized)
        object.__setattr__(self, "referenced_artifacts", references)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "artifacts": list(self.artifacts),
            "referenced_artifacts": [
                reference.to_dict() for reference in self.referenced_artifacts
            ],
            "expected_run_spec_sha256": self.expected_run_spec_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunDependency:
        expected = {
            "run_id",
            "artifacts",
            "referenced_artifacts",
            "expected_run_spec_sha256",
        }
        _require_exact_fields(payload, expected, "dependency")
        artifacts = _require_list(payload["artifacts"], "dependency.artifacts")
        if not all(isinstance(value, str) for value in artifacts):
            raise ValueError("dependency.artifacts must contain strings")
        references = _require_list(
            payload["referenced_artifacts"], "dependency.referenced_artifacts"
        )
        return cls(
            str(payload["run_id"]),
            tuple(artifacts),
            tuple(
                ReferencedArtifact.from_dict(_require_mapping(value, "referenced artifact"))
                for value in references
            ),
            payload["expected_run_spec_sha256"],
        )


@dataclass(frozen=True, init=False)
class RunSpec:
    """One immutable run, including canonical overrides and dependencies."""

    method: str
    category: str
    fold_id: int
    k: int
    seed: int
    variant: str = "primary"
    _overrides_json: str = field(default="{}", repr=False)
    dependencies: tuple[RunDependency, ...] = ()

    def __init__(
        self,
        method: str,
        category: str,
        fold_id: int,
        k: int,
        seed: int,
        variant: str = "primary",
        overrides: Mapping[str, object] | None = None,
        dependencies: Sequence[RunDependency] = (),
    ) -> None:
        _validate_run_identity(method, category, fold_id, k, seed, variant)
        if overrides is not None and not isinstance(overrides, Mapping):
            raise ValueError("overrides must be a mapping")
        override_mapping = dict(overrides or {})
        _validate_safe_identity_json(override_mapping, "overrides")
        _validate_overrides(override_mapping, method, "overrides")
        canonical_overrides = _canonical_json(override_mapping, "overrides")
        normalized_dependencies = tuple(dependencies)
        if not all(isinstance(value, RunDependency) for value in normalized_dependencies):
            raise TypeError("dependencies must contain RunDependency values")
        dependency_ids = [value.run_id for value in normalized_dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependencies must not contain duplicate run IDs")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "fold_id", fold_id)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "_overrides_json", canonical_overrides)
        object.__setattr__(self, "dependencies", normalized_dependencies)

    @property
    def run_id(self) -> str:
        base = (
            f"{self.method}__{self.category}__fold{self.fold_id}"
            f"__k{self.k}__seed{self.seed}"
        )
        return base if self.variant == "primary" else f"{base}__{self.variant}"

    @property
    def overrides(self) -> dict[str, object]:
        """Return an isolated JSON copy of the canonical override mapping."""

        return json.loads(self._overrides_json)

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "category": self.category,
            "fold_id": self.fold_id,
            "k": self.k,
            "seed": self.seed,
            "variant": self.variant,
            "overrides": self.overrides,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunSpec:
        expected = {
            "method",
            "category",
            "fold_id",
            "k",
            "seed",
            "variant",
            "overrides",
            "dependencies",
        }
        _require_exact_fields(payload, expected, "run spec")
        overrides = _require_mapping(payload["overrides"], "run spec.overrides")
        dependencies = _require_list(payload["dependencies"], "run spec.dependencies")
        return cls(
            method=payload["method"],
            category=payload["category"],
            fold_id=payload["fold_id"],
            k=payload["k"],
            seed=payload["seed"],
            variant=payload["variant"],
            overrides=overrides,
            dependencies=tuple(
                RunDependency.from_dict(_require_mapping(value, "dependency"))
                for value in dependencies
            ),
        )


def load_experiment_config(path: str | Path) -> dict[str, object]:
    """Load and strictly validate a primary, smoke, or ablation YAML config."""

    config = load_yaml_mapping(path)
    _validate_config(config)
    return json.loads(_canonical_json(config, "experiment config"))


def load_yaml_mapping(path: str | Path) -> dict[str, object]:
    """Load one duplicate-free YAML mapping without applying experiment schemas."""

    config_path = Path(path)
    try:
        payload = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path.name}: {exc}") from exc
    config = _require_mapping(payload, "YAML document")
    return json.loads(_canonical_json(config, "YAML document"))


def expand_matrix(config: Mapping[str, object]) -> list[RunSpec]:
    """Expand a validated config into sorted, duplicate-free immutable runs."""

    config_copy = json.loads(_canonical_json(dict(config), "experiment config"))
    _validate_config(config_copy)
    if "ablations" in config_copy:
        runs = _expand_ablations(config_copy)
    else:
        runs = _expand_primary(config_copy)
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
        raise ValueError(f"duplicate run IDs after matrix expansion: {duplicates}")
    return sorted(runs, key=lambda run: run.run_id)


def _expand_primary(config: Mapping[str, object]) -> list[RunSpec]:
    categories = config["categories"]
    shots = config["shots"]
    seeds = config["seeds"]
    fold_id = config["fold_id"]
    runs: list[RunSpec] = []
    for method in config["methods"]:
        if method == "sam2_only":
            runs.extend(RunSpec(method, category, fold_id, 0, 0) for category in categories)
            continue
        for category in categories:
            for k in shots:
                for seed in seeds:
                    runs.append(_make_primary_run_spec(method, category, fold_id, k, seed))
    return runs


def _expand_ablations(config: Mapping[str, object]) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for ablation in config["ablations"]:
        for category in config["categories"]:
            dependencies = _primary_dependencies(
                ablation["method"], category, config["fold_id"], config["k"], config["seed"]
            )
            runs.append(
                RunSpec(
                    method=ablation["method"],
                    category=category,
                    fold_id=config["fold_id"],
                    k=config["k"],
                    seed=config["seed"],
                    variant=ablation["name"],
                    overrides=ablation["overrides"],
                    dependencies=dependencies,
                )
            )
    return runs


def _primary_dependencies(
    method: str, category: str, fold_id: int, k: int, seed: int
) -> tuple[RunDependency, ...]:
    heatmap_artifacts = ("test/scores.csv", "calibration.json")
    heatmap_references = (
        ReferencedArtifact(
            csv_path="test/scores.csv",
            path_column="heatmap_path",
            path_scope="csv_parent",
        ),
    )
    if method == "dinov2_single_sam2":
        dependency = _make_primary_run_spec("dinov2_single", category, fold_id, k, seed)
        return (
            RunDependency(
                dependency.run_id,
                heatmap_artifacts,
                heatmap_references,
                dependency.identity_sha256,
            ),
        )
    if method == "dinov2_multi_sam2":
        dependency = _make_primary_run_spec("dinov2_multi", category, fold_id, k, seed)
        return (
            RunDependency(
                dependency.run_id,
                heatmap_artifacts,
                heatmap_references,
                dependency.identity_sha256,
            ),
        )
    if method == "anomaly_consistent_sam2":
        heatmap = _make_primary_run_spec("dinov2_multi", category, fold_id, k, seed)
        masks = _make_primary_run_spec("dinov2_multi_sam2", category, fold_id, k, seed)
        return (
            RunDependency(
                heatmap.run_id,
                heatmap_artifacts,
                heatmap_references,
                heatmap.identity_sha256,
            ),
            RunDependency(
                masks.run_id,
                ("test/mask_scores.csv",),
                (
                    ReferencedArtifact(
                        csv_path="test/mask_scores.csv",
                        path_column="sam2_mask_path",
                        path_scope="csv_parent",
                    ),
                ),
                masks.identity_sha256,
            ),
        )
    return ()


def _make_primary_run_spec(
    method: str, category: str, fold_id: int, k: int, seed: int
) -> RunSpec:
    return RunSpec(
        method,
        category,
        fold_id,
        k,
        seed,
        dependencies=_primary_dependencies(method, category, fold_id, k, seed),
    )


def _validate_config(config: Mapping[str, object]) -> None:
    name = config.get("name")
    _require_choice(
        name,
        "name",
        {"arxiv_primary", "arxiv_ablations", "arxiv_smoke"},
    )
    if name == "arxiv_ablations":
        _validate_ablation_config(config)
    else:
        _validate_primary_config(config)
    observed = hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()
    if observed != _FROZEN_CONFIG_SHA256[name]:
        raise ValueError(
            f"{name} differs from its frozen, pre-registered experiment specification"
        )


def _validate_primary_config(config: Mapping[str, object]) -> None:
    _require_exact_fields(config, _PRIMARY_FIELDS, "experiment config")
    _validate_common(config)
    shots = _integer_list(config["shots"], "shots", minimum=1)
    if any(value not in {1, 2, 4} for value in shots):
        raise ValueError("shots may only contain the pre-registered values 1, 2, and 4")
    _integer_list(config["seeds"], "seeds", minimum=0)
    methods = _string_list(config["methods"], "methods")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"methods contain unsupported values: {unknown}")
    _validate_dinov2(_require_mapping(config["dinov2"], "dinov2"))
    _validate_multi_scale(_require_mapping(config["multi_scale"], "multi_scale"))
    _validate_calibration(_require_mapping(config["calibration"], "calibration"))
    _validate_sam2(_require_mapping(config["sam2"], "sam2"))
    _validate_patchcore(_require_mapping(config["patchcore"], "patchcore"))
    _validate_sam2_only(_require_mapping(config["sam2_only"], "sam2_only"))


def _validate_ablation_config(config: Mapping[str, object]) -> None:
    _require_exact_fields(config, _ABLATION_FIELDS, "experiment config")
    _validate_common(config)
    _validate_relative_path(config["base_config"], "base_config")
    canonical_sha = config["base_config_canonical_sha256"]
    if not isinstance(canonical_sha, str) or not _SHA256_RE.fullmatch(canonical_sha):
        raise ValueError("base_config_canonical_sha256 must be a lowercase SHA-256")
    base_path = Path(__file__).resolve().parents[2] / config["base_config"]
    base = load_yaml_mapping(base_path)
    _validate_primary_config(base)
    observed_base_sha = hashlib.sha256(_canonical_json(base).encode("utf-8")).hexdigest()
    if observed_base_sha != canonical_sha:
        raise ValueError("base config canonical SHA-256 does not match declared identity")
    _validate_relative_path(config["dependency_output_root"], "dependency_output_root")
    _require_int(config["k"], "k", minimum=1)
    _require_int(config["seed"], "seed", minimum=0)
    ablations = _require_list(config["ablations"], "ablations")
    if not ablations:
        raise ValueError("ablations must not be empty")
    names: list[str] = []
    for index, raw_ablation in enumerate(ablations):
        ablation = _require_mapping(raw_ablation, f"ablations[{index}]")
        _require_exact_fields(ablation, {"name", "method", "overrides"}, f"ablations[{index}]")
        name = ablation["name"]
        _validate_component(name, f"ablations[{index}].variant")
        if name == "primary" or "__" in name:
            raise ValueError(f"invalid ablation variant: {name!r}")
        names.append(name)
        method = ablation["method"]
        if not isinstance(method, str) or method not in PAIRED_METHODS:
            raise ValueError(f"ablations[{index}].method is unsupported: {method!r}")
        overrides = _require_mapping(ablation["overrides"], f"ablations[{index}].overrides")
        if not overrides:
            raise ValueError(f"ablations[{index}].overrides must not be empty")
        _validate_safe_identity_json(overrides, f"ablations[{index}].overrides")
        _validate_overrides(overrides, method, f"ablations[{index}].overrides")
    if len(names) != len(set(names)):
        raise ValueError("duplicate ablation variant names are not allowed")


def _validate_common(config: Mapping[str, object]) -> None:
    _validate_relative_path(config["manifest"], "manifest")
    _require_int(config["fold_id"], "fold_id", minimum=0)
    categories = _string_list(config["categories"], "categories")
    unknown = sorted(set(categories) - set(PCB_CATEGORIES))
    if unknown:
        raise ValueError(f"categories contain unsupported values: {unknown}")


def _validate_dinov2(config: Mapping[str, object]) -> None:
    _require_exact_fields(config, {"backbone", "image_size", "patch_size"}, "dinov2")
    if not isinstance(config["backbone"], str) or not config["backbone"].strip():
        raise ValueError("dinov2.backbone must be a non-empty string")
    _require_int(config["image_size"], "dinov2.image_size", minimum=1)
    _require_int(config["patch_size"], "dinov2.patch_size", minimum=1)


def _validate_multi_scale(config: Mapping[str, object]) -> None:
    _require_exact_fields(config, {"crop_sizes", "crop_overlap", "fusion"}, "multi_scale")
    _integer_list(config["crop_sizes"], "multi_scale.crop_sizes", minimum=1)
    _require_number(config["crop_overlap"], "multi_scale.crop_overlap", minimum=0, maximum=1)
    if config["crop_overlap"] == 1:
        raise ValueError("multi_scale.crop_overlap must be less than 1")
    _require_choice(config["fusion"], "multi_scale.fusion", {"max", "mean"})


def _validate_calibration(config: Mapping[str, object]) -> None:
    _require_exact_fields(config, {"quantile"}, "calibration")
    value = _require_number(config["quantile"], "calibration.quantile", minimum=0, maximum=1)
    if value in {0.0, 1.0}:
        raise ValueError("calibration.quantile must be in (0, 1)")


def _validate_sam2(config: Mapping[str, object]) -> None:
    allowed = {
        "refiner",
        "checkpoint",
        "model_config",
        "prompt_mode",
        "point_mode",
        "max_mask_area_fraction",
    }
    required = {"refiner", "prompt_mode", "point_mode", "max_mask_area_fraction"}
    _require_fields(config, required, allowed, "sam2")
    _require_choice(config["refiner"], "sam2.refiner", {"sam2", "fallback"})
    if config["refiner"] == "sam2":
        for field_name in ("checkpoint", "model_config"):
            if field_name not in config:
                raise ValueError(f"sam2.{field_name} is required for the sam2 refiner")
            _validate_relative_path(config[field_name], f"sam2.{field_name}")
    elif "checkpoint" in config or "model_config" in config:
        raise ValueError("fallback refinement must not declare checkpoint or model_config")
    _require_choice(config["prompt_mode"], "sam2.prompt_mode", {"point", "box", "point_box"})
    _require_choice(config["point_mode"], "sam2.point_mode", {"anomaly_max", "box_center"})
    area = config["max_mask_area_fraction"]
    if area is not None:
        value = _require_number(area, "sam2.max_mask_area_fraction", minimum=0, maximum=1)
        if value == 0:
            raise ValueError("sam2.max_mask_area_fraction must be positive")


def _validate_patchcore(config: Mapping[str, object]) -> None:
    expected = {
        "backbone",
        "weights",
        "source",
        "layers",
        "image_size",
        "patch_size",
        "coreset_ratio",
        "projection_dim",
        "normalization",
        "seed_policy",
    }
    _require_exact_fields(config, expected, "patchcore")
    fixed = {
        "backbone": "wide_resnet50_2",
        "weights": "IMAGENET1K_V2",
        "source": "torchvision",
        "layers": ["layer2", "layer3"],
        "image_size": 512,
        "patch_size": 8,
        "normalization": "imagenet",
        "seed_policy": "run_seed",
    }
    for field_name, expected_value in fixed.items():
        if config[field_name] != expected_value:
            raise ValueError(f"patchcore.{field_name} must be {expected_value!r}")
    ratio = _require_number(config["coreset_ratio"], "patchcore.coreset_ratio", 0, 1)
    if ratio == 0:
        raise ValueError("patchcore.coreset_ratio must be positive")
    _require_int(config["projection_dim"], "patchcore.projection_dim", minimum=1)


def _validate_sam2_only(config: Mapping[str, object]) -> None:
    common = {
        "refiner",
        "query_policy",
        "query_fold_split",
        "limit",
        "prompt_longest_side",
        "grid_size",
        "max_regions",
        "box_scale",
        "max_mask_area_fraction",
        "device",
    }
    sam2_fields = {"checkpoint", "model_config"}
    fallback_fields = {"fallback_threshold_fraction"}
    refiner = _require_choice(config.get("refiner"), "sam2_only.refiner", {"sam2", "fallback"})
    if refiner == "sam2":
        _require_exact_fields(config, common | sam2_fields, "sam2_only")
        _validate_relative_path(config["checkpoint"], "sam2_only.checkpoint")
        _validate_relative_path(config["model_config"], "sam2_only.model_config")
    else:
        _require_exact_fields(config, common | fallback_fields, "sam2_only")
        _require_number(
            config["fallback_threshold_fraction"],
            "sam2_only.fallback_threshold_fraction",
            0,
            1,
        )
    _require_choice(config["query_fold_split"], "sam2_only.query_fold_split", {"test"})
    policy = _require_choice(
        config["query_policy"],
        "sam2_only.query_policy",
        {"full_test", "bounded_fixture"},
    )
    if policy == "full_test":
        if config["limit"] is not None:
            raise ValueError("sam2_only full_test policy requires limit: null")
    elif policy == "bounded_fixture":
        _require_int(config["limit"], "sam2_only.limit", minimum=1)
    _require_int(config["prompt_longest_side"], "sam2_only.prompt_longest_side", minimum=1)
    _require_int(config["grid_size"], "sam2_only.grid_size", minimum=1)
    _require_int(config["max_regions"], "sam2_only.max_regions", minimum=1)
    box_scale = _require_number(config["box_scale"], "sam2_only.box_scale", 0, 1)
    if box_scale == 0:
        raise ValueError("sam2_only.box_scale must be positive")
    area = config["max_mask_area_fraction"]
    if area is not None:
        value = _require_number(area, "sam2_only.max_mask_area_fraction", 0, 1)
        if value == 0:
            raise ValueError("sam2_only.max_mask_area_fraction must be positive")
    _require_choice(config["device"], "sam2_only.device", {"auto", "cpu", "cuda"})


def _validate_overrides(
    overrides: Mapping[str, object], method: str, context: str
) -> None:
    if not overrides:
        return
    allowed_by_method = {
        "dinov2_multi": {"crop_sizes", "crop_overlap", "fusion"},
        "dinov2_single_sam2": {
            "prompt_mode",
            "point_mode",
            "max_mask_area_fraction",
        },
        "dinov2_multi_sam2": {
            "prompt_mode",
            "point_mode",
            "max_mask_area_fraction",
        },
        "anomaly_consistent_sam2": {"mask_output", "min_iou", "max_expansion"},
    }
    allowed = allowed_by_method.get(method, set())
    if not allowed:
        raise ValueError(f"method {method!r} does not allow overrides")
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"{context} has method-incompatible fields: {unknown}")
    if "crop_sizes" in overrides:
        _integer_list(overrides["crop_sizes"], "overrides.crop_sizes", minimum=1)
    if "crop_overlap" in overrides:
        value = _require_number(overrides["crop_overlap"], "overrides.crop_overlap", 0, 1)
        if value == 1:
            raise ValueError("overrides.crop_overlap must be less than 1")
    if "fusion" in overrides:
        _require_choice(overrides["fusion"], "overrides.fusion", {"max", "mean"})
    if "prompt_mode" in overrides:
        _require_choice(
            overrides["prompt_mode"],
            "overrides.prompt_mode",
            {"point", "box", "point_box"},
        )
    if "point_mode" in overrides:
        _require_choice(
            overrides["point_mode"],
            "overrides.point_mode",
            {"anomaly_max", "box_center"},
        )
    area = overrides.get("max_mask_area_fraction")
    if area is not None:
        value = _require_number(area, "overrides.max_mask_area_fraction", 0, 1)
        if value == 0:
            raise ValueError("overrides.max_mask_area_fraction must be positive")
    if "mask_output" in overrides:
        _require_choice(
            overrides["mask_output"],
            "overrides.mask_output",
            {"union", "selective"},
        )
    if "min_iou" in overrides:
        _require_number(overrides["min_iou"], "overrides.min_iou", 0, 1)
    if "max_expansion" in overrides:
        _require_number(overrides["max_expansion"], "overrides.max_expansion", 1, None)


def _validate_run_identity(
    method: object,
    category: object,
    fold_id: object,
    k: object,
    seed: object,
    variant: object,
) -> None:
    if method not in METHODS:
        raise ValueError(f"invalid method: {method!r}")
    if category not in PCB_CATEGORIES:
        raise ValueError(f"invalid category: {category!r}")
    _require_int(fold_id, "fold_id", minimum=0)
    _require_int(k, "k", minimum=0)
    _require_int(seed, "seed", minimum=0)
    _validate_component(variant, "variant")
    if "__" in variant:
        raise ValueError("variant must not contain the run ID separator '__'")
    if method == "sam2_only" and (k != 0 or seed != 0):
        raise ValueError("sam2_only must use k=0 and seed=0")
    if method != "sam2_only" and k == 0:
        raise ValueError("k must be positive for support-based methods")


def _validate_component(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):
        raise ValueError(f"invalid {field_name}: {value!r}")


def _validate_relative_path(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty POSIX relative path")
    _validate_ascii_text(value, field_name)
    if value.lower().startswith("file:"):
        raise ValueError(f"{field_name} must not be a local URI")
    if value.startswith("~"):
        raise ValueError(f"{field_name} must not use a home path")
    if "\\" in value or _WINDOWS_ABSOLUTE_RE.match(value) or value.startswith("//"):
        raise ValueError(f"{field_name} must be a POSIX relative public path")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be a relative public path")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain path traversal")
    if not path.parts:
        raise ValueError(f"{field_name} must be a normalized relative public path")
    if path.as_posix() != value or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field_name} must be a normalized relative public path")


def _validate_safe_identity_json(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context} keys must be non-empty strings")
            _validate_ascii_text(key, f"{context} key")
            lowered = key.lower().replace("-", "_")
            if any(
                fragment in lowered
                for fragment in ("secret", "token", "password", "api_key", "credential")
            ):
                raise ValueError(f"{context} contains sensitive key {key!r}")
            _validate_safe_identity_json(item, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe_identity_json(item, f"{context}[{index}]")
        return
    if isinstance(value, str):
        _validate_ascii_text(value, context)
        if (
            value in {".", ".."}
            or value.startswith(("/", "//", "~"))
            or value.lower().startswith("file:")
            or _WINDOWS_ABSOLUTE_RE.match(value)
        ):
            label = "path traversal" if value in {".", ".."} else "absolute/private path"
            raise ValueError(f"{context} contains {label}: {value!r}")
        if "\\" in value:
            raise ValueError(f"{context} contains an absolute/private path")
        if "/" in value and ".." in PurePosixPath(value).parts:
            raise ValueError(f"{context} contains path traversal")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise ValueError(f"{context} must contain only JSON values")


def _validate_ascii_text(value: str, field_name: str) -> None:
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        raise ValueError(f"{field_name} must contain printable ASCII text")


def _require_exact_fields(
    payload: Mapping[str, object], expected: set[str], context: str
) -> None:
    _require_fields(payload, expected, expected, context)


def _require_fields(
    payload: Mapping[str, object], required: set[str], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError(f"{context} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a string-keyed mapping")
    return value


def _require_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    values = _require_list(value, field_name)
    if not values or not all(isinstance(item, str) and item for item in values):
        raise ValueError(f"{field_name} must be a non-empty list of strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return values


def _integer_list(value: object, field_name: str, minimum: int) -> list[int]:
    values = _require_list(value, field_name)
    if not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    normalized = [_require_int(item, field_name, minimum) for item in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return normalized


def _require_int(value: object, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_number(
    value: object,
    field_name: str,
    minimum: float | None,
    maximum: float | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return number


def _require_choice(value: object, field_name: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return value


def _canonical_json(value: object, context: str = "value") -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain finite JSON values: {exc}") from exc


SAM2_ONLY_OUTPUT_REFERENCES = (
    ReferencedArtifact(
        csv_path="test/mask_scores.csv",
        path_column="pred_mask_path",
        path_scope="csv_parent",
    ),
    ReferencedArtifact(
        csv_path="test/mask_scores.csv",
        path_column="heatmap_path",
        path_scope="csv_parent",
    ),
)
