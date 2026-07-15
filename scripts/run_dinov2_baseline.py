#!/usr/bin/env python
"""Run a minimal DINOv2-only few-shot anomaly heatmap baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anomaly.heatmap import save_heatmap_debug_panel
from anomaly.memory_bank import build_memory_bank, select_greedy_coreset
from anomaly.multiscale import compute_anomaly_heatmap_components, parse_crop_sizes
from features.cache import (
    FeatureCache,
    FeatureCacheIdentity,
    extractor_revision_digest,
    extractor_revision_identity,
    sha256_file,
)
from features.dinov2 import build_feature_extractor
from utils.image import load_rgb_image
from utils.heatmap_io import HEATMAP_STORAGE_FORMATS, save_heatmap
from utils.visualize import safe_filename


def parse_args(
    description: str | None = None,
    argument_defaults: dict[str, object] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/visa_pcb_folds.csv"))
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--category", default="pcb1")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Few-shot normal support images per category.",
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit",
        type=_positive_int,
        default=8,
        help="Positive number of query images to score (default: 8).",
    )
    limit_group.add_argument(
        "--all",
        dest="limit",
        action="store_const",
        const=None,
        help="Process the full selected query split.",
    )
    parser.add_argument("--query-fold-split", default="test", choices=("test", "val", "dev"))
    parser.add_argument("--feature-backbone", default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=4880)
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.01,
        help="PatchCore memory-bank retention ratio; ignored by other backbones.",
    )
    parser.add_argument(
        "--coreset-projection-dim",
        type=int,
        default=64,
        help="PatchCore random-projection dimension; ignored by other backbones.",
    )
    parser.add_argument("--patchcore-weights", choices=("IMAGENET1K_V2",), default="IMAGENET1K_V2")
    parser.add_argument("--patchcore-source", choices=("torchvision",), default="torchvision")
    parser.add_argument("--patchcore-layers", choices=("layer2,layer3",), default="layer2,layer3")
    parser.add_argument("--feature-normalization", choices=("imagenet",), default="imagenet")
    parser.add_argument("--coreset-seed-policy", choices=("run_seed",), default="run_seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov2_baseline_debug"))
    parser.add_argument(
        "--crop-sizes",
        default="",
        help="Comma-separated local crop sizes in pixels. Empty string disables multi-scale.",
    )
    parser.add_argument("--crop-overlap", type=float, default=0.25)
    parser.add_argument("--fusion", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="Optional ignored directory for deterministic patch-feature cache artifacts.",
    )
    parser.add_argument(
        "--debug-limit",
        type=_nonnegative_int,
        default=8,
        help="Render at most this many debug panels; use 0 to render none.",
    )
    parser.add_argument(
        "--heatmap-format",
        choices=HEATMAP_STORAGE_FORMATS,
        default="npy",
        help="Lossless heatmap storage encoding; compressed NPZ reduces matrix disk use.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 feature normalization.",
    )
    if argument_defaults:
        parser.set_defaults(**argument_defaults)
    return parser.parse_args()


def main(
    description: str | None = None,
    argument_defaults: dict[str, object] | None = None,
) -> None:
    args = parse_args(description=description, argument_defaults=argument_defaults)
    rows = read_manifest_rows(args.manifest)
    support_rows, query_rows = select_rows(
        rows=rows,
        fold_id=args.fold_id,
        category=args.category,
        k=args.k,
        query_fold_split=args.query_fold_split,
        limit=args.limit,
        seed=args.seed,
    )
    if not support_rows:
        raise RuntimeError("No support rows selected. Check fold_id, category, and manifest path.")
    if not query_rows:
        raise RuntimeError("No query rows selected. Check fold_id, category, and query split.")

    extractor = build_feature_extractor(
        feature_backbone=args.feature_backbone,
        image_size=args.image_size,
        patch_size=args.patch_size,
        device=args.device,
    )
    normalize = not args.no_normalize
    crop_sizes = parse_crop_sizes(args.crop_sizes)
    feature_cache = FeatureCache(args.feature_cache_dir) if args.feature_cache_dir else None
    extractor_identity = extractor_revision_identity(extractor)
    source_revision = extractor_revision_digest(extractor_identity)
    extractor_image_size = int(getattr(extractor, "image_size", args.image_size))
    extractor_patch_size = int(extractor.patch_size)
    support_features = []
    for row in support_rows:
        image = load_rgb_image(row["image_path"])
        cache_key = None
        if feature_cache:
            cache_key = _row_feature_cache_identity(
                row=row,
                image_sha256=sha256_file(row["image_path"]),
                source_revision=source_revision,
                feature_backbone=args.feature_backbone,
                image_size=extractor_image_size,
                patch_size=extractor_patch_size,
                view=f"support:global:0,0,{image.width},{image.height}",
            )
        support_features.append(
            feature_cache.get_or_compute(cache_key, lambda: extractor.extract(image))
            if feature_cache and cache_key
            else extractor.extract(image)
        )
    full_memory_bank = build_memory_bank(support_features, normalize=normalize)
    del support_features
    memory_bank, memory_bank_provenance = prepare_memory_bank(
        full_memory_bank,
        feature_backbone=args.feature_backbone,
        coreset_ratio=args.coreset_ratio,
        coreset_seed=args.seed,
        coreset_projection_dim=args.coreset_projection_dim,
    )
    memory_bank_provenance.update(
        {
            "feature_cache_enabled": feature_cache is not None,
            "extractor_source_revision": source_revision,
            "extractor_identity": extractor_identity,
        }
    )
    del full_memory_bank
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous_debug_panels(args.output_dir)
    (args.output_dir / "memory_bank_provenance.json").write_text(
        json.dumps(memory_bank_provenance, indent=2, sort_keys=True) + "\n"
    )

    score_rows = []
    for rank, row in enumerate(query_rows):
        image = load_rgb_image(row["image_path"])
        cache_key_for_view = None
        if feature_cache:
            image_sha256 = sha256_file(row["image_path"])
            cache_key_for_view = partial(
                _row_feature_cache_identity,
                row=row,
                image_sha256=image_sha256,
                source_revision=source_revision,
                feature_backbone=args.feature_backbone,
                image_size=extractor_image_size,
                patch_size=extractor_patch_size,
                view_prefix="query:",
            )
        heatmap_components = compute_anomaly_heatmap_components(
            image=image,
            extractor=extractor,
            memory_bank=memory_bank,
            crop_sizes=crop_sizes,
            crop_overlap=args.crop_overlap,
            fusion=args.fusion,
            normalize_features=normalize,
            feature_cache=feature_cache,
            cache_key_for_view=cache_key_for_view,
        )
        heatmap_for_eval = heatmap_components.render()
        image_score = float(heatmap_for_eval.max())
        output_stem = f"{rank:03d}_{safe_filename(row['sample_id'])}"
        output_path = args.output_dir / f"{output_stem}.png"
        heatmap_suffix = ".npy" if args.heatmap_format == "npy" else ".npz"
        heatmap_path = args.output_dir / f"{output_stem}_heatmap{heatmap_suffix}"
        heatmap_to_store = (
            heatmap_components if args.heatmap_format == "npz_components" else heatmap_for_eval
        )
        save_heatmap(heatmap_path, heatmap_to_store, storage=args.heatmap_format)
        debug_path = ""
        if rank < args.debug_limit:
            save_heatmap_debug_panel(
                image_path=row["image_path"],
                mask_path=row.get("mask_path") or None,
                heatmap=heatmap_for_eval,
                output_path=output_path,
                title=f"{row['sample_id']} | score={image_score:.4f}",
            )
            debug_path = str(output_path)
        score_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "sample_id": row["sample_id"],
                "category": row["category"],
                "label": row["label"],
                "fold_split": args.query_fold_split,
                "image_score": f"{image_score:.8f}",
                "image_path": row["image_path"],
                "mask_path": row.get("mask_path") or "",
                "heatmap_path": str(heatmap_path),
                "debug_path": debug_path,
            }
        )
        print(f"[{rank + 1}/{len(query_rows)}] {row['sample_id']} score={image_score:.4f}")

    write_scores_csv(score_rows, args.output_dir / "scores.csv")
    print(f"support images: {len(support_rows)}")
    print(f"query images: {len(query_rows)}")
    print(f"outputs: {args.output_dir}")


def read_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _row_feature_cache_identity(
    view: str,
    *,
    row: dict[str, str],
    image_sha256: str,
    source_revision: str,
    feature_backbone: str,
    image_size: int,
    patch_size: int,
    view_prefix: str = "",
) -> FeatureCacheIdentity:
    full_view = f"{view_prefix}{view}"
    return FeatureCacheIdentity(
        sample_id=row["sample_id"],
        image_sha256=image_sha256,
        extractor_revision=source_revision,
        backbone=feature_backbone,
        image_size=image_size,
        patch_size=patch_size,
        view=full_view,
        source_size=_view_source_size(view),
    )


def _view_source_size(view: str) -> tuple[int, int]:
    try:
        x1, y1, x2, y2 = (int(value) for value in view.rsplit(":", 1)[1].split(","))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cache view does not contain exact geometry: {view!r}") from exc
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError(f"cache view has invalid geometry: {view!r}")
    return (x2 - x1, y2 - y1)


def _remove_previous_debug_panels(output_dir: Path) -> None:
    """Remove only debug panels referenced by the previous runner scores file."""

    scores_path = output_dir / "scores.csv"
    if not scores_path.is_file():
        return
    output_root = output_dir.resolve()
    for row in read_manifest_rows(scores_path):
        debug_value = row.get("debug_path") or ""
        if not debug_value:
            continue
        debug_path = Path(debug_value)
        candidate = (
            debug_path.resolve()
            if debug_path.is_absolute()
            else (output_dir / debug_path).resolve()
        )
        if candidate.is_relative_to(output_root) and candidate.suffix.lower() == ".png":
            candidate.unlink(missing_ok=True)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return parsed


def prepare_memory_bank(
    memory_bank: np.ndarray,
    feature_backbone: str,
    coreset_ratio: float,
    coreset_seed: int,
    coreset_projection_dim: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply PatchCore compression and return auditable bank provenance."""

    full_size = int(memory_bank.shape[0])
    coreset_applied = feature_backbone == "patchcore_wrn50"
    selected = memory_bank
    if coreset_applied:
        selected = select_greedy_coreset(
            memory_bank,
            ratio=coreset_ratio,
            seed=coreset_seed,
            projection_dim=coreset_projection_dim,
        )
    provenance: dict[str, str | int | float | bool] = {
        "feature_backbone": feature_backbone,
        "memory_bank_size_full": full_size,
        "memory_bank_size_used": int(selected.shape[0]),
        "coreset_applied": coreset_applied,
        "coreset_ratio": coreset_ratio,
        "coreset_projection_dim": coreset_projection_dim,
        "coreset_seed": coreset_seed,
    }
    return selected, provenance


def select_rows(
    rows: list[dict[str, str]],
    fold_id: int,
    category: str,
    k: int,
    query_fold_split: str,
    limit: int | None,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fold_value = str(fold_id)
    category_rows = [
        row
        for row in rows
        if row["dataset"] == "visa_pcb"
        and row["category"] == category
        and row.get("fold_id") == fold_value
    ]
    support_pool = [
        row for row in category_rows if row.get("fold_split") == "dev" and row["label"] == "0"
    ]
    rng = random.Random(seed)
    support_rows = sorted(support_pool, key=lambda row: row["sample_id"])
    rng.shuffle(support_rows)
    support_rows = support_rows[:k]

    query_rows = [row for row in category_rows if row.get("fold_split") == query_fold_split]
    query_rows = sorted(query_rows, key=lambda row: (row["label"], row["sample_id"]))
    if query_fold_split == "test":
        query_rows = interleave_query_rows(query_rows)
    return support_rows, query_rows if limit is None else query_rows[:limit]


def interleave_query_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    anomalies = sorted(
        (row for row in rows if row["label"] == "1"),
        key=lambda row: row["sample_id"],
    )
    normals = sorted((row for row in rows if row["label"] == "0"), key=lambda row: row["sample_id"])
    interleaved = []
    for index in range(max(len(anomalies), len(normals))):
        if index < len(anomalies):
            interleaved.append(anomalies[index])
        if index < len(normals):
            interleaved.append(normals[index])
    return interleaved


def write_scores_csv(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    """Write paths relative to scores.csv so results are independent of caller cwd."""

    output_path = Path(output_path)
    base_dir = output_path.parent.resolve()
    fieldnames = [
        "dataset",
        "sample_id",
        "category",
        "label",
        "fold_split",
        "image_score",
        "image_path",
        "mask_path",
        "heatmap_path",
        "debug_path",
    ]
    portable_rows = []
    for row in rows:
        portable = dict(row)
        for key in ("image_path", "mask_path", "heatmap_path", "debug_path"):
            value = portable.get(key) or ""
            if value:
                portable[key] = os.path.relpath(Path(value).resolve(), start=base_dir)
        portable_rows.append(portable)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(portable_rows)
    return output_path


def np_save_heatmap(path: str | Path, heatmap) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, heatmap.astype(np.float32, copy=False))
    return path


if __name__ == "__main__":
    main()
