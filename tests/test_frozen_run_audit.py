from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.audit_frozen_runs import write_audit_bundle
from evaluation.frozen_run_audit import (
    audit_method_invariants,
    paired_baseline_statistics,
    runtime_provenance_summary,
    validate_run_config_binding,
)
from experiments.provenance import sha256_json
from experiments.spec import RunSpec


def test_paired_baseline_statistics_use_candidate_minus_baseline_repeated_measures(
    tmp_path: Path,
) -> None:
    values = {
        "patchcore": (0.10, 0.20),
        "dinov2_single": (0.20, 0.30),
        "dinov2_multi": (0.40, 0.50),
        "dinov2_single_sam2": (0.30, 0.40),
        "dinov2_multi_sam2": (0.60, 0.70),
    }
    for seed in (4880, 4881):
        for method, method_values in values.items():
            run = RunSpec(method, "pcb1", 0, 1, seed)
            filename = "per_image.csv" if "sam2" not in method else "mask_per_image.csv"
            _write_csv(
                tmp_path / run.run_id / "test" / filename,
                [
                    {
                        "sample_id": sample_id,
                        "label": "1",
                        "mask_f1": value + 0.01 * (seed - 4880),
                        "mask_iou": value / 2 + 0.01 * (seed - 4880),
                    }
                    for sample_id, value in zip(("pcb1/a", "pcb1/b"), method_values)
                ],
            )

    report = paired_baseline_statistics(
        tmp_path,
        categories=["pcb1"],
        shots=[1],
        seeds=[4880, 4881],
        fold_id=0,
        bootstrap_samples=100,
    )

    assert report["delta_definition"] == "candidate_minus_baseline"
    comparisons = report["comparisons"]
    assert comparisons["dinov2_multi_minus_dinov2_single"]["overall"]["f1"][
        "mean_delta"
    ] == pytest.approx(0.2)
    assert comparisons["dinov2_multi_minus_patchcore_style"]["overall"]["f1"][
        "mean_delta"
    ] == pytest.approx(0.3)
    assert comparisons["dinov2_multi_sam2_minus_dinov2_single_sam2"]["overall"]["f1"][
        "mean_delta"
    ] == pytest.approx(0.3)
    assert all(
        item["scope"]["sample_inclusion"] == "anomaly_images_only" for item in comparisons.values()
    )


def test_method_invariants_check_raw_sam2_equality_and_intersection_subsets(
    tmp_path: Path,
) -> None:
    multi = RunSpec("dinov2_multi", "pcb1", 0, 1, 4880)
    guided = RunSpec("dinov2_multi_sam2", "pcb1", 0, 1, 4880)
    fusion = RunSpec("anomaly_consistent_sam2", "pcb1", 0, 1, 4880)
    multi_test = tmp_path / multi.run_id / "test"
    guided_test = tmp_path / guided.run_id / "test"
    fusion_test = tmp_path / fusion.run_id / "test"
    multi_test.mkdir(parents=True)
    guided_test.mkdir(parents=True)
    fusion_test.mkdir(parents=True)
    np.save(
        multi_test / "sample_heatmap.npy", np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    )
    sam2_mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    fusion_mask = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    _write_mask(guided_test / "sample_sam2.png", sam2_mask)
    _write_mask(guided_test / "sample_pred.png", sam2_mask)
    _write_mask(fusion_test / "sample_pred.png", fusion_mask)
    common = {
        "sample_id": "pcb1/a",
        "label": "1",
        "evidence_class": "paper_evidence",
        "refiner": "sam2",
        "raw_mask_source": "sam2",
        "proposal_threshold": "0.5",
    }
    _write_csv(
        guided_test / "mask_scores.csv",
        [
            {
                **common,
                "mask_output": "sam2",
                "selected_source": "sam2",
                "sam2_mask_path": "sample_sam2.png",
                "pred_mask_path": "sample_pred.png",
                "heatmap_path": f"../../{multi.run_id}/test/sample_heatmap.npy",
            }
        ],
    )
    _write_csv(
        fusion_test / "mask_scores.csv",
        [
            {
                **common,
                "mask_output": "intersection",
                "selected_source": "intersection",
                "sam2_mask_path": f"../../{guided.run_id}/test/sample_sam2.png",
                "pred_mask_path": "sample_pred.png",
                "heatmap_path": f"../../{multi.run_id}/test/sample_heatmap.npy",
            }
        ],
    )

    report = audit_method_invariants(
        tmp_path,
        categories=["pcb1"],
        shots=[1],
        seeds=[4880],
        fold_id=0,
    )

    assert report["ok"] is True
    assert report["runs_checked"] == 1
    assert report["images_checked"] == 1
    assert report["guided_pred_not_raw_sam2_pixels"] == 0
    assert report["fusion_outside_raw_sam2_pixels"] == 0
    assert report["fusion_outside_anomaly_proposal_pixels"] == 0
    assert report["fusion_not_exact_intersection_pixels"] == 0

    _write_mask(fusion_test / "sample_pred.png", np.zeros((2, 2), dtype=np.uint8))
    failed = audit_method_invariants(
        tmp_path,
        categories=["pcb1"],
        shots=[1],
        seeds=[4880],
        fold_id=0,
    )
    assert failed["ok"] is False
    assert failed["fusion_not_exact_intersection_pixels"] == 1

    _write_mask(fusion_test / "sample_pred.png", np.zeros((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="native mask shapes differ"):
        audit_method_invariants(
            tmp_path,
            categories=["pcb1"],
            shots=[1],
            seeds=[4880],
            fold_id=0,
        )


def test_runtime_provenance_summary_compacts_frozen_model_and_environment_identities(
    tmp_path: Path,
) -> None:
    commit = "f" * 40
    for method in ("patchcore", "dinov2_multi", "dinov2_multi_sam2"):
        run = RunSpec(method, "pcb1", 0, 1, 4880)
        run_dir = tmp_path / run.run_id
        run_dir.mkdir(parents=True)
        payload = {
            "run_id": run.run_id,
            "git_commit": commit,
            "git_dirty": False,
            "manifest": {"path": "data/manifests/visa.csv", "sha256": "a" * 64},
            "effective_execution_sha256": method[0] * 64,
            "run_spec_sha256": method[-1] * 64,
            "environment": {
                "python": {"version": "3.10.8", "implementation": "CPython"},
                "libraries": {"torch": "2.5.1+cu121"},
            },
            "execution": {
                "selected_device": "cuda",
                "device_identity": {"kind": "cuda", "name": "NVIDIA GeForce RTX 4090"},
                "runtime": {
                    "python_version": "3.10.8",
                    "libraries": {"torch": "2.5.1+cu121"},
                },
            },
            "observed_identities": {
                "source_revisions": {"repository": commit, "sam2": "b" * 40},
                "extractors": [
                    {
                        "identity": {
                            "extractor_identity": {
                                "extractor_class": f"features.{method}.Extractor",
                                "device": "cuda",
                                "configuration": {"image_size": 518},
                                "model_state": {"sha256": "c" * 64, "num_bytes": 10},
                            }
                        }
                    }
                ],
            },
            "checkpoints": {"sam2": {"path": "weights/sam2.pt", "sha256": "d" * 64}},
            "model_configs": {"sam2": {"path": "configs/sam2.yaml", "sha256": "e" * 64}},
        }
        (run_dir / "provenance.json").write_text(json.dumps(payload))

    summary = runtime_provenance_summary(
        tmp_path,
        category="pcb1",
        k=1,
        seed=4880,
        fold_id=0,
        source_commit=commit,
    )

    assert summary["source_commit"] == commit
    assert summary["manifest_sha256"] == "a" * 64
    assert summary["environment"]["python"]["version"] == "3.10.8"
    assert summary["execution_environment"]["selected_device"] == "cuda"
    assert summary["execution_environment"]["device_identity"]["name"] == (
        "NVIDIA GeForce RTX 4090"
    )
    assert summary["execution_environment"]["cuda_version"] == "12.1"
    assert set(summary["representative_runs"]) == {
        "patchcore",
        "dinov2_multi",
        "dinov2_multi_sam2",
    }
    dino = summary["representative_runs"]["dinov2_multi"]
    assert dino["extractors"][0]["model_state"]["sha256"] == "c" * 64
    guided = summary["representative_runs"]["dinov2_multi_sam2"]
    assert guided["checkpoints"]["sam2"]["sha256"] == "d" * 64


def test_validate_run_config_binding_checks_every_provenance_identity(tmp_path: Path) -> None:
    commit = "f" * 40
    config_sha = "a" * 64
    canonical_sha = "b" * 64
    runs = [RunSpec("dinov2_multi", "pcb1", 0, 1, seed) for seed in (4880, 4881)]
    for index, run in enumerate(runs):
        run_dir = tmp_path / run.run_id
        run_dir.mkdir(parents=True)
        payload = {
            "run_id": run.run_id,
            "git_commit": commit,
            "git_dirty": False,
            "run_spec_sha256": run.identity_sha256,
            "effective_execution_sha256": f"{index + 1}" * 64,
            "experiment_config": {"sha256": config_sha},
            "execution": {
                "config_sha256": config_sha,
                "config_canonical_sha256": canonical_sha,
                "run_spec_sha256": run.identity_sha256,
            },
        }
        (run_dir / "provenance.json").write_text(json.dumps(payload))

    binding = validate_run_config_binding(
        tmp_path,
        runs=runs,
        source_commit=commit,
        config_sha256=config_sha,
        config_canonical_sha256=canonical_sha,
    )

    assert binding["run_count"] == 2
    assert binding["run_ids_sha256"] == sha256_json(sorted(run.run_id for run in runs))
    assert binding["config_sha256"] == config_sha
    assert binding["config_canonical_sha256"] == canonical_sha

    bad_path = tmp_path / runs[-1].run_id / "provenance.json"
    bad = json.loads(bad_path.read_text())
    bad["execution"]["config_sha256"] = "c" * 64
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="config identity mismatch"):
        validate_run_config_binding(
            tmp_path,
            runs=runs,
            source_commit=commit,
            config_sha256=config_sha,
            config_canonical_sha256=canonical_sha,
        )


def test_write_audit_bundle_is_checksum_backed_and_non_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    manifest = write_audit_bundle(
        output,
        source_commit="f" * 40,
        generation_command="test audit",
        baseline_statistics={"comparisons": {}},
        method_invariants={"ok": True, "images_checked": 1},
        runtime_provenance={"environment": {}},
        config_bindings={"primary": {"run_count": 364}, "ablations": {"run_count": 48}},
    )

    assert manifest["source_commit"] == "f" * 40
    assert {item["path"] for item in manifest["generated_files"]} == {
        "baseline_paired_statistics.json",
        "method_invariants.json",
        "runtime_provenance.json",
        "config_bindings.json",
    }
    for item in manifest["generated_files"]:
        assert item["sha256"] == hashlib.sha256((output / item["path"]).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_audit_bundle(
            output,
            source_commit="f" * 40,
            generation_command="test audit",
            baseline_statistics={},
            method_invariants={},
            runtime_provenance={},
            config_bindings={},
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)
