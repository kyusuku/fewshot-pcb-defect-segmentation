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
}
_ABLATION_FIELDS = {
    "name",
    "manifest",
    "dependency_output_root",
    "fold_id",
    "categories",
    "k",
    "seed",
    "ablations",
}
_FROZEN_CONFIG_SHA256 = {
    "arxiv_primary": "488b3a7da0b0f469cd4cc88a20ffc696287673ca1d63e16d3bdb9f778b1085fb",
    "arxiv_ablations": "cf7791daf3d9fb670a2036735fad3477d0158db1d79bf91c40f4bdba3db08fdf",
    "arxiv_smoke": "04a7d88b5ec19fc550ecbf5712c5e1a0cf28df40e44e11a6fec30ac978220f67",
}


@dataclass(frozen=True)
class RunDependency:
    """An upstream run and the relative artifacts required from it."""

    run_id: str
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError(f"invalid dependency run_id: {self.run_id!r}")
        normalized = tuple(self.artifacts)
        if len(normalized) != len(set(normalized)):
            raise ValueError("dependency artifacts must not contain duplicates")
        for artifact in normalized:
            _validate_relative_path(artifact, "dependency artifact")
        object.__setattr__(self, "artifacts", normalized)

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "artifacts": list(self.artifacts)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunDependency:
        _require_exact_fields(payload, {"run_id", "artifacts"}, "dependency")
        artifacts = _require_list(payload["artifacts"], "dependency.artifacts")
        if not all(isinstance(value, str) for value in artifacts):
            raise ValueError("dependency.artifacts must contain strings")
        return cls(str(payload["run_id"]), tuple(artifacts))


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
        canonical_overrides = _canonical_json(dict(overrides or {}), "overrides")
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

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path.name}: {exc}") from exc
    config = _require_mapping(payload, "experiment config")
    _validate_config(config)
    return json.loads(_canonical_json(config, "experiment config"))


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
                    dependencies = _primary_dependencies(method, category, fold_id, k, seed)
                    runs.append(
                        RunSpec(
                            method,
                            category,
                            fold_id,
                            k,
                            seed,
                            dependencies=dependencies,
                        )
                    )
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
    if method == "dinov2_single_sam2":
        dependency = RunSpec("dinov2_single", category, fold_id, k, seed)
        return (RunDependency(dependency.run_id, heatmap_artifacts),)
    if method == "dinov2_multi_sam2":
        dependency = RunSpec("dinov2_multi", category, fold_id, k, seed)
        return (RunDependency(dependency.run_id, heatmap_artifacts),)
    if method == "anomaly_consistent_sam2":
        heatmap = RunSpec("dinov2_multi", category, fold_id, k, seed)
        masks = RunSpec("dinov2_multi_sam2", category, fold_id, k, seed)
        return (
            RunDependency(heatmap.run_id, heatmap_artifacts),
            RunDependency(masks.run_id, ("test/mask_scores.csv", "test/raw_masks")),
        )
    return ()


def _validate_config(config: Mapping[str, object]) -> None:
    name = config.get("name")
    if name not in {"arxiv_primary", "arxiv_ablations", "arxiv_smoke"}:
        raise ValueError("name must be arxiv_primary, arxiv_ablations, or arxiv_smoke")
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


def _validate_ablation_config(config: Mapping[str, object]) -> None:
    _require_exact_fields(config, _ABLATION_FIELDS, "experiment config")
    _validate_common(config)
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
        _validate_overrides(
            _require_mapping(ablation["overrides"], f"ablations[{index}].overrides"),
            index,
        )
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
    if config["fusion"] not in {"max", "mean"}:
        raise ValueError("multi_scale.fusion must be max or mean")


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
    if config["refiner"] not in {"sam2", "fallback"}:
        raise ValueError("sam2.refiner must be sam2 or fallback")
    if config["refiner"] == "sam2":
        for field_name in ("checkpoint", "model_config"):
            if field_name not in config:
                raise ValueError(f"sam2.{field_name} is required for the sam2 refiner")
            _validate_relative_path(config[field_name], f"sam2.{field_name}")
    elif "checkpoint" in config or "model_config" in config:
        raise ValueError("fallback refinement must not declare checkpoint or model_config")
    if config["prompt_mode"] not in {"point", "box", "point_box"}:
        raise ValueError("sam2.prompt_mode is invalid")
    if config["point_mode"] not in {"anomaly_max", "box_center"}:
        raise ValueError("sam2.point_mode is invalid")
    area = config["max_mask_area_fraction"]
    if area is not None:
        value = _require_number(area, "sam2.max_mask_area_fraction", minimum=0, maximum=1)
        if value == 0:
            raise ValueError("sam2.max_mask_area_fraction must be positive")


def _validate_overrides(overrides: Mapping[str, object], index: int) -> None:
    if not overrides:
        raise ValueError(f"ablations[{index}].overrides must not be empty")
    allowed = {
        "crop_sizes",
        "crop_overlap",
        "fusion",
        "prompt_mode",
        "point_mode",
        "max_mask_area_fraction",
        "mask_output",
        "min_iou",
        "max_expansion",
    }
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"ablations[{index}].overrides has unknown fields: {unknown}")
    if "crop_sizes" in overrides:
        _integer_list(overrides["crop_sizes"], "overrides.crop_sizes", minimum=1)
    if "crop_overlap" in overrides:
        value = _require_number(overrides["crop_overlap"], "overrides.crop_overlap", 0, 1)
        if value == 1:
            raise ValueError("overrides.crop_overlap must be less than 1")
    if "fusion" in overrides and overrides["fusion"] not in {"max", "mean"}:
        raise ValueError("overrides.fusion must be max or mean")
    if "prompt_mode" in overrides and overrides["prompt_mode"] not in {"point", "box", "point_box"}:
        raise ValueError("overrides.prompt_mode is invalid")
    if "point_mode" in overrides and overrides["point_mode"] not in {"anomaly_max", "box_center"}:
        raise ValueError("overrides.point_mode is invalid")
    area = overrides.get("max_mask_area_fraction")
    if area is not None:
        value = _require_number(area, "overrides.max_mask_area_fraction", 0, 1)
        if value == 0:
            raise ValueError("overrides.max_mask_area_fraction must be positive")
    if "mask_output" in overrides and overrides["mask_output"] not in {"union", "selective"}:
        raise ValueError("overrides.mask_output must be union or selective")
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
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be a relative public path")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain path traversal")
    if path.parts[0].startswith("~") or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field_name} must be a normalized relative public path")


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
