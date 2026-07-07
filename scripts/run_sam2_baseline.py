#!/usr/bin/env python
"""Run a SAM2-only/simple-prompt mask baseline without DINO anomaly heatmaps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sam_refine.prompts import PromptRegion
from sam_refine.refiner import FallbackMaskRefiner, SAM2MaskRefiner
from utils.image import load_rgb_image
from utils.visualize import safe_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/visa_pcb_folds.csv"))
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--category", default="pcb1")
    parser.add_argument("--limit", type=int, default=8, help="Number of query images to process.")
    parser.add_argument("--query-fold-split", default="test", choices=("test", "val", "dev"))
    parser.add_argument(
        "--prompt-longest-side",
        type=int,
        default=128,
        help="Longest side of the prompt/mask grid before masks are scaled for evaluation.",
    )
    parser.add_argument("--grid-size", type=int, default=3, help="Number of prompt boxes per side.")
    parser.add_argument("--max-regions", type=int, default=9)
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.0,
        help="Scale each grid box around its center. Use values in (0, 1].",
    )
    parser.add_argument("--refiner", choices=("fallback", "sam2"), default="fallback")
    parser.add_argument("--fallback-threshold-fraction", type=float, default=0.5)
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("weights/sam2_checkpoint.pt"))
    parser.add_argument("--sam2-model-config", default="")
    parser.add_argument(
        "--max-mask-area-fraction",
        type=float,
        default=None,
        help="Reject SAM2 mask candidates covering more than this image fraction.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam2_only_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest_rows(args.manifest)
    query_rows = select_query_rows(
        rows=rows,
        fold_id=args.fold_id,
        category=args.category,
        query_fold_split=args.query_fold_split,
        limit=args.limit,
    )
    if not query_rows:
        raise RuntimeError("No query rows selected. Check fold_id, category, and query split.")

    refiner = build_refiner(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    for index, row in enumerate(query_rows):
        image = load_rgb_image(row["image_path"])
        prompt_shape = prompt_shape_for_image(image.size, longest_side=args.prompt_longest_side)
        neutral_heatmap = np.ones(prompt_shape, dtype=np.float32)
        regions = build_grid_prompt_regions(
            prompt_shape=prompt_shape,
            grid_size=args.grid_size,
            max_regions=args.max_regions,
            box_scale=args.box_scale,
        )
        predictions = refiner.refine(image=image, heatmap=neutral_heatmap, regions=regions)
        pred_mask = union_masks(
            [prediction.mask for prediction in predictions],
            shape=neutral_heatmap.shape,
        )
        output_stem = f"{index:03d}_{safe_filename(row['sample_id'])}"
        pred_mask_path = args.output_dir / f"{output_stem}_pred_mask.png"
        heatmap_path = args.output_dir / f"{output_stem}_neutral_heatmap.npy"
        debug_path = args.output_dir / f"{output_stem}_sam2_only.png"
        np.save(heatmap_path, neutral_heatmap)
        save_mask(pred_mask, pred_mask_path)
        save_debug_panel(image, pred_mask, debug_path)
        mask_score = max((prediction.score for prediction in predictions), default=0.0)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "category": row.get("category", ""),
                "label": row.get("label", ""),
                "mask_path": row.get("mask_path", ""),
                "num_regions": str(len(regions)),
                "num_masks": str(len(predictions)),
                "mask_score": f"{mask_score:.8f}",
                "pred_mask_path": str(pred_mask_path),
                "heatmap_path": str(heatmap_path),
                "debug_path": str(debug_path),
            }
        )
        print(
            f"[{index + 1}/{len(query_rows)}] {row['sample_id']} "
            f"regions={len(regions)} masks={len(predictions)}"
        )

    write_mask_scores(output_rows, args.output_dir / "mask_scores.csv")
    print(f"query images: {len(query_rows)}")
    print(f"outputs: {args.output_dir}")


def read_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_query_rows(
    rows: list[dict[str, str]],
    fold_id: int,
    category: str,
    query_fold_split: str,
    limit: int,
) -> list[dict[str, str]]:
    fold_value = str(fold_id)
    query_rows = [
        row
        for row in rows
        if row.get("dataset") == "visa_pcb"
        and row.get("category") == category
        and row.get("fold_id") == fold_value
        and row.get("fold_split") == query_fold_split
    ]
    query_rows = sorted(query_rows, key=lambda row: (row.get("label", ""), row["sample_id"]))
    if query_fold_split == "test":
        query_rows = interleave_query_rows(query_rows)
    return query_rows[:limit]


def interleave_query_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    anomalies = sorted(
        (row for row in rows if row.get("label") == "1"),
        key=lambda row: row["sample_id"],
    )
    normals = sorted(
        (row for row in rows if row.get("label") == "0"),
        key=lambda row: row["sample_id"],
    )
    interleaved = []
    for index in range(max(len(anomalies), len(normals))):
        if index < len(anomalies):
            interleaved.append(anomalies[index])
        if index < len(normals):
            interleaved.append(normals[index])
    return interleaved


def prompt_shape_for_image(image_size: tuple[int, int], longest_side: int) -> tuple[int, int]:
    if longest_side <= 0:
        raise ValueError("longest_side must be positive")
    width, height = image_size
    scale = longest_side / float(max(width, height))
    prompt_width = max(1, round(width * scale))
    prompt_height = max(1, round(height * scale))
    return (prompt_height, prompt_width)


def build_grid_prompt_regions(
    prompt_shape: tuple[int, int],
    grid_size: int,
    max_regions: int,
    box_scale: float = 1.0,
) -> list[PromptRegion]:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if max_regions <= 0:
        return []
    if not 0.0 < box_scale <= 1.0:
        raise ValueError("box_scale must be in (0, 1]")

    height, width = prompt_shape
    regions = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = round(col * width / grid_size)
            x2 = round((col + 1) * width / grid_size)
            y1 = round(row * height / grid_size)
            y2 = round((row + 1) * height / grid_size)
            x1, y1, x2, y2 = _scale_box((x1, y1, x2, y2), box_scale, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            area = int((x2 - x1) * (y2 - y1))
            regions.append(
                PromptRegion(
                    box_xyxy=(x1, y1, x2, y2),
                    point_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    area=area,
                    score=1.0,
                )
            )
            if len(regions) >= max_regions:
                return regions
    return regions


def build_refiner(args: argparse.Namespace):
    if args.refiner == "fallback":
        return FallbackMaskRefiner(threshold_fraction=args.fallback_threshold_fraction)
    if not args.sam2_model_config:
        raise RuntimeError("--sam2-model-config is required when --refiner sam2")
    return SAM2MaskRefiner(
        checkpoint_path=args.sam2_checkpoint,
        model_config=args.sam2_model_config,
        device=args.device,
        max_mask_area_fraction=args.max_mask_area_fraction,
    )


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        if mask.shape != shape:
            mask = resize_mask(mask, shape)
        union |= (mask > 0).astype(np.uint8)
    return union


def save_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(output_path)
    return output_path


def save_debug_panel(image: Image.Image, mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    mask_resized = resize_mask(mask, (image.size[1], image.size[0]))
    overlay = image.convert("RGBA")
    red = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = (mask_resized > 0).astype(np.uint8) * 120
    Image.alpha_composite(overlay, Image.fromarray(red, mode="RGBA")).convert("RGB").save(
        output_path
    )
    return output_path


def write_mask_scores(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fieldnames = [
        "sample_id",
        "category",
        "label",
        "mask_path",
        "num_regions",
        "num_masks",
        "mask_score",
        "pred_mask_path",
        "heatmap_path",
        "debug_path",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)


def _scale_box(
    box_xyxy: tuple[int, int, int, int],
    box_scale: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if box_scale == 1.0:
        return box_xyxy
    x1, y1, x2, y2 = box_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_width = max(1, round((x2 - x1) * box_scale))
    box_height = max(1, round((y2 - y1) * box_scale))
    scaled_x1 = round(center_x - box_width / 2.0)
    scaled_y1 = round(center_y - box_height / 2.0)
    scaled_x2 = scaled_x1 + box_width
    scaled_y2 = scaled_y1 + box_height
    if scaled_x1 < 0:
        scaled_x2 -= scaled_x1
        scaled_x1 = 0
    if scaled_y1 < 0:
        scaled_y2 -= scaled_y1
        scaled_y1 = 0
    if scaled_x2 > width:
        shift = scaled_x2 - width
        scaled_x1 = max(0, scaled_x1 - shift)
        scaled_x2 = width
    if scaled_y2 > height:
        shift = scaled_y2 - height
        scaled_y1 = max(0, scaled_y1 - shift)
        scaled_y2 = height
    return (scaled_x1, scaled_y1, scaled_x2, scaled_y2)


if __name__ == "__main__":
    main()
