#!/usr/bin/env python
"""Run a frozen paper experiment matrix without shell interpretation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.runner import run_matrix
from experiments.spec import expand_matrix, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--method")
    parser.add_argument("--category")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    runs = _filter_runs(expand_matrix(config), args.run_id, args.method, args.category)
    dependency_root = args.dependency_root or _default_dependency_root(config, args.output_root)
    cache = args.feature_cache_dir or args.output_root / ".feature_cache"
    run_matrix(
        runs,
        config,
        args.config,
        args.output_root,
        dependency_root,
        args.device,
        cache,
        dry_run=args.dry_run,
        resume=args.resume,
        allow_dirty=args.allow_dirty,
    )


def _filter_runs(runs, run_id: str | None, method: str | None, category: str | None):
    selected = [
        run
        for run in runs
        if (run_id is None or run.run_id == run_id)
        and (method is None or run.method == method)
        and (category is None or run.category == category)
    ]
    if not selected:
        raise ValueError("matrix filters selected zero runs")
    return selected


def _default_dependency_root(config: dict[str, object], output_root: Path) -> Path:
    configured = config.get("dependency_output_root")
    return PROJECT_ROOT / str(configured) if configured else output_root


if __name__ == "__main__":
    main()
