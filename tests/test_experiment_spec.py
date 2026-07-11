from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import pytest
import yaml

from experiments.spec import RunSpec, expand_matrix, load_experiment_config


CONFIG_ROOT = Path("configs/experiments")


def test_frozen_matrices_have_exact_deduplicated_cardinalities() -> None:
    primary = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml"))
    ablations = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_ablations.yaml"))

    assert len(primary) == len({run.run_id for run in primary}) == 364
    assert len(ablations) == len({run.run_id for run in ablations}) == 48


def test_sam2_only_is_deduplicated_across_shots_and_seeds() -> None:
    runs = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml"))
    sam2_only = [run for run in runs if run.method == "sam2_only"]

    assert len(sam2_only) == 4
    assert {(run.k, run.seed) for run in sam2_only} == {(0, 0)}


def test_run_id_and_order_are_stable() -> None:
    run = RunSpec("dinov2_multi", "pcb2", fold_id=0, k=2, seed=4881)
    assert run.run_id == "dinov2_multi__pcb2__fold0__k2__seed4881"

    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    first = expand_matrix(config)
    second = expand_matrix(copy.deepcopy(config))
    assert [run.run_id for run in first] == sorted(run.run_id for run in first)
    assert first == second
    assert [run.identity_sha256 for run in first] == [run.identity_sha256 for run in second]


def test_ablation_preserves_overrides_without_mutable_aliases() -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_ablations.yaml")
    run = next(run for run in expand_matrix(config) if run.variant == "fusion_selective")

    assert run.overrides == {
        "mask_output": "selective",
        "max_expansion": 2.0,
        "min_iou": 0.25,
    }
    returned = run.overrides
    returned["min_iou"] = 0.99
    assert run.overrides["min_iou"] == 0.25
    assert RunSpec.from_dict(json.loads(json.dumps(run.to_dict()))) == run


def test_ablation_dependencies_are_machine_readable() -> None:
    runs = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_ablations.yaml"))
    prompt = next(run for run in runs if run.category == "pcb1" and run.variant == "prompt_box")
    fusion = next(
        run for run in runs if run.category == "pcb1" and run.variant == "fusion_union"
    )

    assert [dependency.run_id for dependency in prompt.dependencies] == [
        "dinov2_multi__pcb1__fold0__k4__seed4880"
    ]
    assert [dependency.run_id for dependency in fusion.dependencies] == [
        "dinov2_multi__pcb1__fold0__k4__seed4880",
        "dinov2_multi_sam2__pcb1__fold0__k4__seed4880",
    ]
    assert fusion.dependencies[1].artifacts == ("test/mask_scores.csv",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("categories", ["pcb1", "board"], "categories"),
        ("categories", ["pcb1", "pcb1"], "duplicate"),
        ("shots", [0], "shots"),
        ("shots", [1, True], "shots"),
        ("seeds", [-1], "seeds"),
        ("seeds", [4880, 4880], "duplicate"),
        ("methods", ["dinov2_multi", "unknown"], "methods"),
        ("methods", ["dinov2_multi", "dinov2_multi"], "duplicate"),
    ],
)
def test_primary_config_rejects_invalid_matrix_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    payload[field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unexpected": 1}, "unknown fields"),
        ({"manifest": "/Users/name/private.csv"}, "relative"),
        ({"manifest": "../private.csv"}, "traversal"),
        ({"fold_id": -1}, "fold_id"),
        ({"name": "wrong_name"}, "name"),
    ],
)
def test_config_rejects_unknown_fields_unsafe_paths_and_invalid_values(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    payload = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    payload.update(mutation)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "payload",
    [None, [], "primary", 3],
)
def test_config_requires_a_mapping(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config(path)


def test_nested_schema_and_duplicate_variants_are_strict(tmp_path: Path) -> None:
    primary = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    primary["dinov2"]["typo"] = True
    primary_path = tmp_path / "primary.yaml"
    primary_path.write_text(yaml.safe_dump(primary, sort_keys=False))
    with pytest.raises(ValueError, match="dinov2.*unknown fields"):
        load_experiment_config(primary_path)

    ablations = load_experiment_config(CONFIG_ROOT / "arxiv_ablations.yaml")
    ablations["ablations"].append(copy.deepcopy(ablations["ablations"][0]))
    ablation_path = tmp_path / "ablations.yaml"
    ablation_path.write_text(yaml.safe_dump(ablations, sort_keys=False))
    with pytest.raises(ValueError, match="duplicate.*variant"):
        load_experiment_config(ablation_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "unknown"}, "method"),
        ({"category": "pcb5"}, "category"),
        ({"fold_id": -1}, "fold_id"),
        ({"k": 0}, "k"),
        ({"seed": -1}, "seed"),
        ({"variant": "../../escape"}, "variant"),
        ({"variant": "has__separator"}, "variant"),
    ],
)
def test_run_spec_rejects_invalid_or_path_unsafe_identity(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "method": "dinov2_multi",
        "category": "pcb1",
        "fold_id": 0,
        "k": 1,
        "seed": 4880,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RunSpec(**values)


def test_sam2_only_requires_zero_pairing_dimensions() -> None:
    with pytest.raises(ValueError, match="sam2_only"):
        RunSpec("sam2_only", "pcb1", 0, 1, 4880)
    assert RunSpec("sam2_only", "pcb1", 0, 0, 0).run_id.endswith("__k0__seed0")


def test_expand_matrix_revalidates_programmatic_configs() -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    config["categories"] = ["pcb1", "pcb1"]
    with pytest.raises(ValueError, match="duplicate"):
        expand_matrix(config)


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        ("arxiv_primary.yaml", lambda payload: payload.update(seeds=[4880])),
        (
            "arxiv_ablations.yaml",
            lambda payload: payload["ablations"][0]["overrides"].update(crop_sizes=[640]),
        ),
        (
            "arxiv_smoke.yaml",
            lambda payload: payload.update(methods=["dinov2_single", "dinov2_multi"]),
        ),
    ],
)
def test_named_paper_configs_reject_valid_but_unregistered_changes(
    tmp_path: Path, filename: str, mutate: Callable[[dict[str, object]], None]
) -> None:
    payload = load_experiment_config(CONFIG_ROOT / filename)
    mutate(payload)
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="frozen"):
        load_experiment_config(path)
