#!/usr/bin/env python
"""Run a minimal DINOv2-only few-shot anomaly heatmap baseline."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anomaly.heatmap import save_heatmap_debug_panel
from anomaly.memory_bank import build_memory_bank, score_patch_features
from features.dinov2 import build_feature_extractor
from utils.image import load_rgb_image
from utils.visualize import safe_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/visa_pcb_folds.csv"))
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--category", default="pcb1")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Few-shot normal support images per category.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of query images to score.")
    parser.add_argument("--query-fold-split", default="test", choices=("test", "val", "dev"))
    parser.add_argument("--feature-backbone", default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=4880)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov2_baseline_debug"))
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 feature normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    support_features = [
        extractor.extract(load_rgb_image(row["image_path"]))
        for row in support_rows
    ]
    memory_bank = build_memory_bank(support_features, normalize=normalize)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_rows = []
    for rank, row in enumerate(query_rows):
        query_features = extractor.extract(load_rgb_image(row["image_path"]))
        heatmap = score_patch_features(query_features, memory_bank, normalize=normalize)
        image_score = float(heatmap.max())
        output_path = args.output_dir / f"{rank:03d}_{safe_filename(row['sample_id'])}.png"
        save_heatmap_debug_panel(
            image_path=row["image_path"],
            mask_path=row.get("mask_path") or None,
            heatmap=heatmap,
            output_path=output_path,
            title=f"{row['sample_id']} | score={image_score:.4f}",
        )
        score_rows.append(
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "label": row["label"],
                "image_score": f"{image_score:.8f}",
                "debug_path": str(output_path),
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


def select_rows(
    rows: list[dict[str, str]],
    fold_id: int,
    category: str,
    k: int,
    query_fold_split: str,
    limit: int,
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
        row
        for row in category_rows
        if row.get("fold_split") == "dev" and row["label"] == "0"
    ]
    rng = random.Random(seed)
    support_rows = sorted(support_pool, key=lambda row: row["sample_id"])
    rng.shuffle(support_rows)
    support_rows = support_rows[:k]

    query_rows = [
        row for row in category_rows if row.get("fold_split") == query_fold_split
    ]
    query_rows = sorted(query_rows, key=lambda row: (row["label"], row["sample_id"]))
    if query_fold_split == "test":
        query_rows = sorted(query_rows, key=lambda row: (row["label"] != "1", row["sample_id"]))
    return support_rows, query_rows[:limit]


def write_scores_csv(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fieldnames = ["sample_id", "category", "label", "image_score", "debug_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    main()
