#!/usr/bin/env python
"""Audit frozen paper runs without changing their commit-bound provenance."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.frozen_run_audit import (
    audit_method_invariants,
    paired_baseline_statistics,
    runtime_provenance_summary,
    validate_run_config_binding,
)
from experiments.provenance import sha256_file, sha256_json
from experiments.spec import expand_matrix, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-config", type=Path, required=True)
    parser.add_argument("--ablation-config", type=Path, required=True)
    parser.add_argument("--primary-output-root", type=Path, required=True)
    parser.add_argument("--ablation-output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--postprocessor-commit", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        _require_current_commit(args.postprocessor_commit)
        primary_config = load_experiment_config(args.primary_config)
        ablation_config = load_experiment_config(args.ablation_config)
        primary_runs = expand_matrix(primary_config)
        ablation_runs = expand_matrix(ablation_config)
        categories = [str(value) for value in primary_config["categories"]]
        shots = [int(value) for value in primary_config["shots"]]
        seeds = [int(value) for value in primary_config["seeds"]]
        fold_id = int(primary_config["fold_id"])
        baseline = paired_baseline_statistics(
            args.primary_output_root,
            categories=categories,
            shots=shots,
            seeds=seeds,
            fold_id=fold_id,
            bootstrap_samples=args.bootstrap_samples,
        )
        invariants = audit_method_invariants(
            args.primary_output_root,
            categories=categories,
            shots=shots,
            seeds=seeds,
            fold_id=fold_id,
        )
        runtime = runtime_provenance_summary(
            args.primary_output_root,
            category=categories[0],
            k=shots[0],
            seed=seeds[0],
            fold_id=fold_id,
            source_commit=args.source_commit,
        )
        config_bindings = {
            "primary": validate_run_config_binding(
                args.primary_output_root,
                runs=primary_runs,
                source_commit=args.source_commit,
                config_sha256=sha256_file(args.primary_config),
                config_canonical_sha256=sha256_json(primary_config),
            ),
            "ablations": validate_run_config_binding(
                args.ablation_output_root,
                runs=ablation_runs,
                source_commit=args.source_commit,
                config_sha256=sha256_file(args.ablation_config),
                config_canonical_sha256=sha256_json(ablation_config),
            ),
        }
        write_audit_bundle(
            args.audit_dir,
            source_commit=args.source_commit,
            postprocessor_commit=args.postprocessor_commit,
            generation_command=shlex.join(sys.argv),
            baseline_statistics=baseline,
            method_invariants=invariants,
            runtime_provenance=runtime,
            config_bindings=config_bindings,
        )
        if not invariants["ok"]:
            raise ValueError("frozen method invariants failed")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def write_audit_bundle(
    output_dir: str | Path,
    *,
    source_commit: str,
    postprocessor_commit: str,
    generation_command: str,
    baseline_statistics: Mapping[str, object],
    method_invariants: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    config_bindings: Mapping[str, object],
) -> dict[str, object]:
    """Write one immutable compact audit directory and its hash manifest."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {output_dir}")
    output_dir.mkdir(parents=True)
    payloads = {
        "baseline_paired_statistics.json": baseline_statistics,
        "method_invariants.json": method_invariants,
        "runtime_provenance.json": runtime_provenance,
        "config_bindings.json": config_bindings,
    }
    generated = []
    for name, payload in payloads.items():
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated.append({"path": name, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 2,
        "source_commit": source_commit,
        "postprocessor_commit": postprocessor_commit,
        "generation_command": generation_command,
        "generated_files": generated,
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _require_current_commit(expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("postprocessor commit must be a lowercase 40-character Git SHA")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if observed != expected:
        raise ValueError("postprocessor commit does not match current repository HEAD")
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if tracked_status:
        raise ValueError("postprocessor tracked worktree is dirty")


if __name__ == "__main__":
    main()
