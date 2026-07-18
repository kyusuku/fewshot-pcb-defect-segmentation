from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from scripts.finalize_paper_evidence import (
    _copy_method_figure,
    _copy_qualitative_figures,
    _reject_private_paths,
    _require_current_commit,
    finalize_paper_evidence,
)
from scripts.render_method_figure import BLOCKS
from experiments.provenance import sha256_file, sha256_json
from experiments.spec import RunSpec, expand_matrix, load_experiment_config


def test_finalize_paper_evidence_validates_freeze_and_builds_revision_package(
    tmp_path: Path,
) -> None:
    source_commit = "f" * 40
    postprocessor_commit = "a" * 40
    repo_root = Path(__file__).resolve().parents[1]
    primary_config = repo_root / "configs/experiments/arxiv_primary.yaml"
    ablation_config = repo_root / "configs/experiments/arxiv_ablations.yaml"
    primary_config_data = load_experiment_config(primary_config)
    ablation_config_data = load_experiment_config(ablation_config)
    primary_specs = expand_matrix(primary_config_data)
    ablation_specs = expand_matrix(ablation_config_data)
    primary_ids = [run.run_id for run in primary_specs]
    ablation_ids = [run.run_id for run in ablation_specs]
    identities = {
        run_id: hashlib.sha256(run_id.encode()).hexdigest()
        for run_id in (*primary_ids, *ablation_ids)
    }
    source = tmp_path / "source"
    source.mkdir()
    _write_csv(
        source / "primary_results.csv",
        [_result_row(run, source_commit, identities[run.run_id]) for run in primary_specs],
    )
    _write_csv(
        source / "ablation_results.csv",
        [
            _result_row(run, source_commit, identities[run.run_id], variant=run.variant)
            for run in ablation_specs
        ],
    )
    (source / "paired_statistics.json").write_text(
        json.dumps(
            {
                "comparisons": {
                    "dinov2_multi_vs_anomaly_consistent_sam2": {
                        "overall": {
                            "f1": {"mean_delta": 0.1, "ci_low": 0.05, "ci_high": 0.15},
                            "iou": {"mean_delta": 0.08, "ci_low": 0.03, "ci_high": 0.12},
                        }
                    }
                }
            }
        )
        + "\n"
    )
    qualitative_selection = source / "qualitative_manifest.csv"
    _write_csv(
        qualitative_selection,
        [
            {
                "category": category,
                "role": role,
                "sample_id": f"{category}/{role}",
                "sam2_delta_f1": 0.1 if role == "success" else -0.1,
                **{
                    f"sha256_{panel}": hashlib.sha256(
                        f"{category}/{role}/{panel}".encode()
                    ).hexdigest()
                    for panel in (
                        "image",
                        "mask",
                        "anomaly_panel",
                        "sam2_panel",
                        "fusion_panel",
                    )
                },
            }
            for category in primary_config_data["categories"]
            for role in ("success", "failure")
        ],
    )
    source_generated = [
        source / "primary_results.csv",
        source / "ablation_results.csv",
        source / "paired_statistics.json",
        qualitative_selection,
    ]
    source_manifest = {
        "schema_version": 2,
        "ready_for_writing": True,
        "readiness_scope": "complete_paper_evidence",
        "source_runs": [
            {
                "run_id": run_id,
                "git_commit": source_commit,
                "git_dirty": False,
                "manifest_sha256": "b" * 64,
                "effective_execution_sha256": identities[run_id],
            }
            for run_id in (*primary_ids, *ablation_ids)
        ],
        "generated_files": [{"path": path.name, "sha256": _sha(path)} for path in source_generated],
    }
    (source / "completion_manifest.json").write_text(json.dumps(source_manifest) + "\n")

    audit = tmp_path / "audit"
    audit.mkdir()
    audit_files = {
        "baseline_paired_statistics.json": {"comparisons": {}},
        "method_invariants.json": {"ok": True, "images_checked": 1},
        "runtime_provenance.json": {"source_commit": source_commit},
        "config_bindings.json": {
            "primary": {
                "config_sha256": sha256_file(primary_config),
                "config_canonical_sha256": sha256_json(primary_config_data),
                "run_count": len(primary_ids),
                "run_ids_sha256": sha256_json(sorted(primary_ids)),
                "run_spec_identities_sha256": sha256_json(
                    {run.run_id: run.identity_sha256 for run in primary_specs}
                ),
                "effective_execution_identities_sha256": sha256_json(
                    {run_id: identities[run_id] for run_id in sorted(primary_ids)}
                ),
            },
            "ablations": {
                "config_sha256": sha256_file(ablation_config),
                "config_canonical_sha256": sha256_json(ablation_config_data),
                "run_count": len(ablation_ids),
                "run_ids_sha256": sha256_json(sorted(ablation_ids)),
                "run_spec_identities_sha256": sha256_json(
                    {run.run_id: run.identity_sha256 for run in ablation_specs}
                ),
                "effective_execution_identities_sha256": sha256_json(
                    {run_id: identities[run_id] for run_id in sorted(ablation_ids)}
                ),
            },
        },
    }
    for name, payload in audit_files.items():
        (audit / name).write_text(json.dumps(payload) + "\n")
    audit_manifest = {
        "schema_version": 2,
        "source_commit": source_commit,
        "postprocessor_commit": postprocessor_commit,
        "generated_files": [{"path": name, "sha256": _sha(audit / name)} for name in audit_files],
    }
    (audit / "audit_manifest.json").write_text(json.dumps(audit_manifest) + "\n")

    primary_check = tmp_path / "primary_check.json"
    ablation_check = tmp_path / "ablation_check.json"
    primary_check.write_text(
        json.dumps(
            {
                "ok": True,
                "runs": [{"run_id": run_id, "ok": True, "errors": []} for run_id in primary_ids],
            }
        )
    )
    ablation_check.write_text(
        json.dumps(
            {
                "ok": True,
                "runs": [{"run_id": run_id, "ok": True, "errors": []} for run_id in ablation_ids],
            }
        )
    )

    guided_ids = [run.run_id for run in primary_specs if run.method == "dinov2_multi_sam2"]
    fusion_ids = [run.run_id for run in primary_specs if run.method == "anomaly_consistent_sam2"]
    audit_files["method_invariants.json"].update(
        {
            "runs_checked": len(guided_ids),
            "images_checked": 1,
            "guided_run_ids_checked": guided_ids,
            "fusion_run_ids_checked": fusion_ids,
        }
    )
    (audit / "method_invariants.json").write_text(
        json.dumps(audit_files["method_invariants.json"]) + "\n"
    )
    for item in audit_manifest["generated_files"]:
        if item["path"] == "method_invariants.json":
            item["sha256"] = _sha(audit / "method_invariants.json")
    (audit / "audit_manifest.json").write_text(json.dumps(audit_manifest) + "\n")

    method_assets = tmp_path / "method-assets"
    method_assets.mkdir()
    method_png = method_assets / "method_figure.png"
    Image.new("RGB", (64, 32), "white").save(method_png)
    method_layout = method_assets / "method_figure_layout.json"
    method_layout.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "output_png": method_png.name,
                "output_png_sha256": _sha(method_png),
                "generation_command": "test method renderer",
                "renderer_source_sha256": _sha(repo_root / "scripts/render_method_figure.py"),
                "blocks": BLOCKS,
            }
        )
        + "\n"
    )
    qualitative_figures = tmp_path / "qualitative-figures"
    qualitative_figures.mkdir()
    figure_entries = []
    for category in primary_config_data["categories"]:
        for role in ("success", "failure"):
            path = qualitative_figures / f"{category}_{role}.png"
            Image.new("RGB", (64, 32), "white").save(path)
            figure_entries.append(
                {
                    "category": category,
                    "role": role,
                    "sample_id": f"{category}/{role}",
                    "sam2_delta_f1": 0.1 if role == "success" else -0.1,
                    "path": path.name,
                    "sha256": _sha(path),
                    "source_panel_sha256": {
                        panel: hashlib.sha256(f"{category}/{role}/{panel}".encode()).hexdigest()
                        for panel in (
                            "image",
                            "mask",
                            "anomaly_panel",
                            "sam2_panel",
                            "fusion_panel",
                        )
                    },
                }
            )
    (qualitative_figures / "qualitative_figure_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_manifest_sha256": "1" * 64,
                "renderer_source_sha256": _sha(repo_root / "scripts/render_qualitative_figures.py"),
                "figures": figure_entries,
            }
        )
        + "\n"
    )

    output = tmp_path / "final"
    manifest = finalize_paper_evidence(
        source_evidence_dir=source,
        audit_dir=audit,
        primary_check=primary_check,
        ablation_check=ablation_check,
        output_dir=output,
        source_commit=source_commit,
        postprocessor_commit=postprocessor_commit,
        generation_command="test finalizer",
        primary_config=primary_config,
        ablation_config=ablation_config,
        method_figure_layout=method_layout,
        qualitative_figure_dir=qualitative_figures,
    )

    assert manifest["ready_for_writing"] is True
    assert manifest["readiness_scope"] == "complete_paper_evidence"
    assert manifest["evidence_revision"]["source_commit"] == source_commit
    assert manifest["evidence_revision"]["postprocessor_commit"] == postprocessor_commit
    conclusion = json.loads((output / "evidence_conclusion.json").read_text())
    assert conclusion["outcome"] == "anomaly_consistent_sam2_improves_robustly"
    assert conclusion["delta_definition"] == "candidate_minus_baseline"
    assert (output / "primary_summary.csv").is_file()
    assert (output / "ablation_summary.csv").is_file()
    assert (output / "matrix_primary_check.json").is_file()
    assert (output / "method_figure.png").is_file()
    assert (output / "qualitative_figures/pcb1_success.png").is_file()
    assert (output / f"source_completion_manifest_{source_commit[:7]}.json").is_file()
    for item in manifest["generated_files"]:
        assert item["sha256"] == _sha(output / item["path"])
    assert "/root/" not in "".join(
        path.read_text(errors="ignore") for path in output.rglob("*") if path.is_file()
    )

    wrong_primary = tmp_path / "wrong_primary_check.json"
    wrong_primary.write_text(
        json.dumps(
            {
                "ok": True,
                "runs": [
                    {"run_id": f"unrelated-{index}", "ok": True, "errors": []}
                    for index in range(len(primary_ids))
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="checker run IDs do not match"):
        finalize_paper_evidence(
            source_evidence_dir=source,
            audit_dir=audit,
            primary_check=wrong_primary,
            ablation_check=ablation_check,
            output_dir=tmp_path / "bad-final",
            source_commit=source_commit,
            postprocessor_commit=postprocessor_commit,
            generation_command="test finalizer",
            primary_config=primary_config,
            ablation_config=ablation_config,
            method_figure_layout=method_layout,
            qualitative_figure_dir=qualitative_figures,
        )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_paper_evidence(
            source_evidence_dir=source,
            audit_dir=audit,
            primary_check=primary_check,
            ablation_check=ablation_check,
            output_dir=output,
            source_commit=source_commit,
            postprocessor_commit=postprocessor_commit,
            generation_command="test finalizer",
            primary_config=primary_config,
            ablation_config=ablation_config,
            method_figure_layout=method_layout,
            qualitative_figure_dir=qualitative_figures,
        )

    private_source = tmp_path / "private-source"
    shutil.copytree(source, private_source)
    private_statistics = json.loads((private_source / "paired_statistics.json").read_text())
    private_statistics["private_note"] = "/root/private-output"
    (private_source / "paired_statistics.json").write_text(json.dumps(private_statistics) + "\n")
    private_manifest_path = private_source / "completion_manifest.json"
    private_manifest = json.loads(private_manifest_path.read_text())
    for item in private_manifest["generated_files"]:
        if item["path"] == "paired_statistics.json":
            item["sha256"] = _sha(private_source / "paired_statistics.json")
    private_manifest_path.write_text(json.dumps(private_manifest) + "\n")
    private_output = tmp_path / "private-final"
    with pytest.raises(ValueError, match="private paths leaked"):
        finalize_paper_evidence(
            source_evidence_dir=private_source,
            audit_dir=audit,
            primary_check=primary_check,
            ablation_check=ablation_check,
            output_dir=private_output,
            source_commit=source_commit,
            postprocessor_commit=postprocessor_commit,
            generation_command="test private-path rejection",
            primary_config=primary_config,
            ablation_config=ablation_config,
            method_figure_layout=method_layout,
            qualitative_figure_dir=qualitative_figures,
        )
    assert not private_output.exists()


def test_require_current_commit_rejects_tracked_dirtiness(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40

    def fake_check_output(command: list[str], **_: object) -> str:
        if command == ["git", "rev-parse", "HEAD"]:
            return commit + "\n"
        if command == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return " M scripts/finalize_paper_evidence.py\n"
        raise AssertionError(command)

    monkeypatch.setattr(
        "scripts.finalize_paper_evidence.subprocess.check_output", fake_check_output
    )
    with pytest.raises(ValueError, match="tracked worktree is dirty"):
        _require_current_commit(commit)


def test_reject_private_paths_ignores_binary_png_chunks(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    png = public_dir / "figure.png"
    Image.new("RGB", (16, 16), "white").save(png)

    chunk_type = b"raNd"
    chunk_data = b"C:\\compressed-pixel-bytes-are-not-a-path"
    chunk = (
        len(chunk_data).to_bytes(4, "big")
        + chunk_type
        + chunk_data
        + zlib.crc32(chunk_type + chunk_data).to_bytes(4, "big")
    )
    content = png.read_bytes()
    png.write_bytes(content[:-12] + chunk + content[-12:])

    _reject_private_paths(public_dir)


def test_reject_private_paths_checks_textual_png_metadata(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    png = public_dir / "figure.png"
    metadata = PngInfo()
    metadata.add_text("source", "/Users/private/figure.png")
    Image.new("RGB", (16, 16), "white").save(png, pnginfo=metadata)

    with pytest.raises(ValueError, match="private paths leaked"):
        _reject_private_paths(public_dir)


def test_finalize_paper_evidence_cli_help_runs_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/finalize_paper_evidence.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--postprocessor-commit" in result.stdout


def test_copy_method_figure_rejects_stale_layout_even_with_valid_png_hash(
    tmp_path: Path,
) -> None:
    png = tmp_path / "method_figure.png"
    Image.new("RGB", (16, 16), "white").save(png)
    layout = tmp_path / "method_figure_layout.json"
    layout.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "output_png": png.name,
                "output_png_sha256": _sha(png),
                "renderer_source_sha256": _sha(
                    Path(__file__).resolve().parents[1] / "scripts/render_method_figure.py"
                ),
                "blocks": [],
            }
        )
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="current renderer contract"):
        _copy_method_figure(layout, output)


def test_copy_qualitative_figures_rejects_stale_source_panel_hashes(
    tmp_path: Path,
) -> None:
    panel_names = ("image", "mask", "anomaly_panel", "sam2_panel", "fusion_panel")
    selection = tmp_path / "qualitative_manifest.csv"
    selection_rows = []
    figures = []
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    for role, delta in (("success", 0.1), ("failure", -0.1)):
        source_hashes = {
            panel: hashlib.sha256(f"{role}/{panel}".encode()).hexdigest() for panel in panel_names
        }
        selection_rows.append(
            {
                "category": "pcb1",
                "role": role,
                "sample_id": f"pcb1/{role}",
                "sam2_delta_f1": delta,
                **{f"sha256_{panel}": value for panel, value in source_hashes.items()},
            }
        )
        png = figure_dir / f"pcb1_{role}.png"
        Image.new("RGB", (16, 16), "white").save(png)
        figures.append(
            {
                "category": "pcb1",
                "role": role,
                "sample_id": f"pcb1/{role}",
                "sam2_delta_f1": delta,
                "path": png.name,
                "sha256": _sha(png),
                "source_panel_sha256": {
                    **source_hashes,
                    "image": "0" * 64,
                },
            }
        )
    _write_csv(selection, selection_rows)
    (figure_dir / "qualitative_figure_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_manifest_sha256": _sha(selection),
                "renderer_source_sha256": _sha(
                    Path(__file__).resolve().parents[1] / "scripts/render_qualitative_figures.py"
                ),
                "figures": figures,
            }
        )
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="source panel identity mismatch"):
        _copy_qualitative_figures(
            figure_dir,
            output,
            categories=["pcb1"],
            source_selection=selection,
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _result_row(
    run: RunSpec,
    source_commit: str,
    effective_identity: str,
    *,
    variant: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run.run_id,
        "method": run.method,
        "category": run.category,
        "k": run.k,
        "seed": run.seed,
        "image_auroc": 0.8 if run.method in {"patchcore", "dinov2_single", "dinov2_multi"} else "",
        "pixel_auroc": 0.7 if run.method in {"patchcore", "dinov2_single", "dinov2_multi"} else "",
        "aupro": 0.6 if run.method in {"patchcore", "dinov2_single", "dinov2_multi"} else "",
        "aggregate_pixel_f1": 0.5
        if run.method in {"patchcore", "dinov2_single", "dinov2_multi"}
        else "",
        "aggregate_pixel_iou": 0.4
        if run.method in {"patchcore", "dinov2_single", "dinov2_multi"}
        else "",
        "mean_anomaly_mask_f1": 0.5,
        "mean_anomaly_mask_iou": 0.4,
        "mean_anomaly_mask_precision": 0.6,
        "mean_anomaly_mask_recall": 0.5,
        "git_commit": source_commit,
        "git_dirty": False,
        "manifest_sha256": "b" * 64,
        "effective_execution_sha256": effective_identity,
        "run_spec_sha256": run.identity_sha256,
    }
    if variant is not None:
        row = {"variant": variant, **row}
    return row


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
