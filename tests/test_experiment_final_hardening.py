from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest
import yaml

from experiments.provenance import (
    build_provenance,
    compute_effective_execution_sha256,
    resolve_referenced_artifacts,
    sha256_json,
    validate_resume_identity,
)
from experiments.spec import (
    ReferencedArtifact,
    SAM2_ONLY_OUTPUT_REFERENCES,
    expand_matrix,
    load_experiment_config,
)
from sam_refine.artifacts import write_mask_scores as write_refinement_scores
from scripts.run_dinov2_baseline import write_scores_csv
from scripts.run_sam2_baseline import write_mask_scores as write_sam2_only_scores


CONFIG_ROOT = Path("configs/experiments")


def test_declared_references_resolve_real_writer_fixtures(tmp_path: Path) -> None:
    primary = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml"))
    guided = next(
        run
        for run in primary
        if run.method == "dinov2_multi_sam2"
        and run.category == "pcb1"
        and run.k == 1
        and run.seed == 4880
    )
    fusion = next(
        run
        for run in primary
        if run.method == "anomaly_consistent_sam2"
        and run.category == "pcb1"
        and run.k == 1
        and run.seed == 4880
    )

    heatmap_root = tmp_path / guided.dependencies[0].run_id
    heatmap_dir = heatmap_root / "test"
    heatmap_dir.mkdir(parents=True)
    heatmap = heatmap_dir / "000_heatmap.npy"
    heatmap.write_bytes(b"heatmap")
    write_scores_csv(
        [
            {
                "sample_id": "pcb1/a",
                "category": "pcb1",
                "label": "1",
                "fold_split": "test",
                "image_score": "1.0",
                "image_path": "",
                "mask_path": "",
                "heatmap_path": str(heatmap),
                "debug_path": "",
            }
        ],
        heatmap_dir / "scores.csv",
    )
    heatmap_reference = guided.dependencies[0].referenced_artifacts[0]
    assert heatmap_reference.path_scope == "csv_parent"
    assert resolve_referenced_artifacts(heatmap_root, heatmap_reference)[0]["path"] == str(
        heatmap
    )

    mask_root = tmp_path / fusion.dependencies[1].run_id
    mask_dir = mask_root / "test"
    mask_dir.mkdir(parents=True)
    sam2_mask = mask_dir / "000_sam2_mask.png"
    sam2_mask.write_bytes(b"sam2-mask")
    write_refinement_scores(
        [{"sample_id": "pcb1/a", "sam2_mask_path": str(sam2_mask)}],
        mask_dir / "mask_scores.csv",
    )
    mask_reference = fusion.dependencies[1].referenced_artifacts[0]
    assert mask_reference.path_column == "sam2_mask_path"
    assert mask_reference.path_scope == "csv_parent"
    assert resolve_referenced_artifacts(mask_root, mask_reference)[0]["path"] == str(
        sam2_mask
    )
    assert "test/raw_masks" not in fusion.dependencies[1].artifacts

    sam2_only_root = tmp_path / "sam2_only"
    sam2_only_dir = sam2_only_root / "test"
    sam2_only_dir.mkdir(parents=True)
    pred_mask = sam2_only_dir / "000_pred_mask.png"
    neutral_heatmap = sam2_only_dir / "000_neutral_heatmap.npy"
    pred_mask.write_bytes(b"pred-mask")
    neutral_heatmap.write_bytes(b"neutral")
    scores_path = sam2_only_dir / "mask_scores.csv"
    write_sam2_only_scores(
        [
            {
                "sample_id": "pcb1/a",
                "pred_mask_path": str(pred_mask),
                "heatmap_path": str(neutral_heatmap),
            }
        ],
        scores_path,
    )
    rows = _read_csv(scores_path)
    assert rows[0]["pred_mask_path"] == "000_pred_mask.png"
    assert rows[0]["heatmap_path"] == "000_neutral_heatmap.npy"
    for reference in SAM2_ONLY_OUTPUT_REFERENCES:
        assert resolve_referenced_artifacts(sam2_only_root, reference)


def test_effective_execution_hash_covers_every_result_affecting_identity(
    tmp_path: Path,
) -> None:
    record = _build_complete_record(tmp_path)
    effective = record["effective_execution_sha256"]
    assert len(effective) == 64
    assert compute_effective_execution_sha256(record) == effective
    validate_resume_identity(record, expected_effective_execution_sha256=effective)

    mutations = [
        lambda value: value["manifest"].update(sha256="1" * 64),
        _mutate_support,
        lambda value: value["git"].update(commit="1" * 40),
        _mutate_dependency,
        lambda value: value["checkpoints"]["sam2"].update(sha256="3" * 64),
        lambda value: value["model_configs"]["sam2"].update(sha256="4" * 64),
        lambda value: _mutate_mapping_identity(value, "cache_identity", "changed-cache"),
        lambda value: _mutate_mapping_identity(
            value, "artifact_identities", "changed-source"
        ),
        lambda value: value["environment"]["libraries"].update(numpy="changed"),
        lambda value: value.update(platform="changed-platform"),
    ]
    for mutate in mutations:
        expected = copy.deepcopy(record)
        mutate(expected)
        expected_hash = compute_effective_execution_sha256(expected)
        with pytest.raises(ValueError, match="effective execution identity"):
            validate_resume_identity(
                record,
                expected_effective_execution_sha256=expected_hash,
            )


def test_effective_hash_rejects_malformed_or_missing_hashes(tmp_path: Path) -> None:
    record = _build_complete_record(tmp_path)
    record["dependency_identities"][next(iter(record["dependency_identities"]))] = None
    with pytest.raises(ValueError, match="dependency.*SHA-256"):
        compute_effective_execution_sha256(record)

    record = _build_complete_record(tmp_path / "second")
    record["manifest"]["sha256"] = "ABC"
    with pytest.raises(ValueError, match="manifest.*SHA-256"):
        compute_effective_execution_sha256(record)


def test_config_canonical_comparison_distinguishes_bool_from_int(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("value: true\n")
    with pytest.raises(ValueError, match="does not match supplied config"):
        build_provenance(
            _guided_run(),
            manifest,
            ["pcb1/a"],
            "a" * 40,
            git_dirty=False,
            config_path=config_path,
            config={"value": 1},
            library_names=(),
        )


def test_manifest_requires_visa_pcb_dataset(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv", dataset="other")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: debug\n")
    with pytest.raises(ValueError, match="dataset=visa_pcb"):
        build_provenance(
            _guided_run(),
            manifest,
            ["pcb1/a"],
            "a" * 40,
            git_dirty=False,
            config_path=config_path,
            config={"name": "debug"},
            library_names=(),
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        "file:///tmp/private",
        "FiLe:/tmp/private",
        "FILE:relative-private",
        "~",
        "~user/data",
    ],
)
def test_local_uri_and_home_shortcuts_are_rejected(tmp_path: Path, unsafe: str) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    config["manifest"] = unsafe
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="public path|local URI|home"):
        load_experiment_config(path)


def test_required_reference_rejects_header_only_and_duplicate_header_csv(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    test_dir = run_root / "test"
    test_dir.mkdir(parents=True)
    reference = ReferencedArtifact("test/scores.csv", "heatmap_path")
    scores = test_dir / "scores.csv"

    scores.write_text("sample_id,heatmap_path\n")
    with pytest.raises(ValueError, match="zero rows|required index"):
        resolve_referenced_artifacts(run_root, reference)

    scores.write_text("sample_id,heatmap_path,heatmap_path\na,x.npy,x.npy\n")
    with pytest.raises(ValueError, match="duplicate CSV columns"):
        resolve_referenced_artifacts(run_root, reference)


def test_referenced_artifact_cannot_escape_base_through_symlink(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    test_dir = run_root / "test"
    test_dir.mkdir(parents=True)
    outside = tmp_path / "private.npy"
    outside.write_bytes(b"private")
    (test_dir / "escaped.npy").symlink_to(outside)
    (test_dir / "scores.csv").write_text(
        "sample_id,heatmap_path\npcb1/a,escaped.npy\n"
    )
    reference = ReferencedArtifact("test/scores.csv", "heatmap_path")

    with pytest.raises(ValueError, match="outside declared csv_parent base"):
        resolve_referenced_artifacts(run_root, reference)


def _build_complete_record(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(root / "manifest.csv")
    config_path = root / "config.yaml"
    checkpoint = root / "sam2.pt"
    model_config = root / "sam2.yaml"
    config = {"name": "debug", "numerics": {"deterministic": True}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    checkpoint.write_bytes(b"checkpoint")
    model_config.write_text("model: tiny\n")
    return build_provenance(
        _guided_run(),
        manifest,
        ["pcb1/a"],
        "a" * 40,
        git_dirty=False,
        config_path=config_path,
        config=config,
        checkpoint_paths={"sam2": checkpoint},
        model_config_paths={"sam2": model_config},
        cache_identity={"extractor_sha256": "b" * 64},
        artifact_identities={"source_sha256": "c" * 64},
        library_names=("numpy",),
    )


def _mutate_support(value: dict[str, object]) -> None:
    value["support_ids"] = ["pcb1/other"]
    value["support_ids_sha256"] = sha256_json(value["support_ids"])


def _mutate_dependency(value: dict[str, object]) -> None:
    dependencies = value["dependency_identities"]
    dependencies[next(iter(dependencies))] = "2" * 64
    value["dependency_identities_sha256"] = sha256_json(dependencies)


def _mutate_mapping_identity(
    value: dict[str, object], field_name: str, replacement: str
) -> None:
    identity = value[field_name]
    identity["canonical"] = {"identity": replacement}
    identity["sha256"] = sha256_json(identity["canonical"])


def _guided_run():
    return next(
        run
        for run in expand_matrix(
            load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
        )
        if run.method == "dinov2_multi_sam2"
        and run.category == "pcb1"
        and run.k == 1
        and run.seed == 4880
    )


def _write_manifest(path: Path, dataset: str = "visa_pcb") -> Path:
    fields = ["dataset", "sample_id", "category", "fold_id", "fold_split", "label"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset,
                "sample_id": "pcb1/a",
                "category": "pcb1",
                "fold_id": "0",
                "fold_split": "dev",
                "label": "0",
            }
        )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
