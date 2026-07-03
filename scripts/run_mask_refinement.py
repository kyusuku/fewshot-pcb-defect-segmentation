#!/usr/bin/env python
"""Generate mask predictions from saved anomaly heatmaps."""

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

from evaluation.metrics import resolve_score_row_paths
from sam_refine.prompts import heatmap_to_prompt_regions
from sam_refine.refiner import FallbackMaskRefiner, SAM2MaskRefiner
from utils.image import load_rgb_image
from utils.visualize import safe_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--refiner", choices=("fallback", "sam2"), default="fallback")
    parser.add_argument("--fallback-threshold-fraction", type=float, default=0.5)
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("weights/sam2_checkpoint.pt"))
    parser.add_argument("--sam2-model-config", default="")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_score_rows(args.scores_csv)
    rows = resolve_score_row_paths(rows, base_dir=args.scores_csv.parent)
    refiner = build_refiner(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    for index, row in enumerate(rows):
        image = load_rgb_image(row["image_path"])
        heatmap = np.load(row["heatmap_path"]).astype(np.float32, copy=False)
        regions = heatmap_to_prompt_regions(
            heatmap,
            threshold=args.threshold,
            percentile=args.percentile,
            min_area=args.min_area,
            max_regions=args.max_regions,
        )
        predictions = refiner.refine(image=image, heatmap=heatmap, regions=regions)
        pred_mask = union_masks(
            [prediction.mask for prediction in predictions],
            shape=heatmap.shape,
        )
        output_stem = f"{index:03d}_{safe_filename(row['sample_id'])}"
        pred_mask_path = args.output_dir / f"{output_stem}_pred_mask.png"
        debug_path = args.output_dir / f"{output_stem}_refined.png"
        save_mask(pred_mask, pred_mask_path)
        save_refinement_debug(image, heatmap, pred_mask, debug_path)
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
                "heatmap_path": row["heatmap_path"],
                "debug_path": str(debug_path),
            }
        )
        print(
            f"[{index + 1}/{len(rows)}] {row['sample_id']} "
            f"regions={len(regions)} masks={len(predictions)}"
        )

    write_mask_scores(output_rows, args.output_dir / "mask_scores.csv")
    print(f"outputs: {args.output_dir}")


def build_refiner(args: argparse.Namespace):
    if args.refiner == "fallback":
        return FallbackMaskRefiner(threshold_fraction=args.fallback_threshold_fraction)
    if not args.sam2_model_config:
        raise RuntimeError("--sam2-model-config is required when --refiner sam2")
    return SAM2MaskRefiner(
        checkpoint_path=args.sam2_checkpoint,
        model_config=args.sam2_model_config,
        device=args.device,
    )


def read_score_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        if mask.shape != shape:
            mask = _resize_mask(mask, shape)
        union |= (mask > 0).astype(np.uint8)
    return union


def save_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(output_path)
    return output_path


def save_refinement_debug(
    image: Image.Image,
    heatmap: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_rgb = _heatmap_to_rgb(heatmap).resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image.convert("RGB"), heatmap_rgb, alpha=0.35).convert("RGBA")
    mask_rgba = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_resized = _resize_mask(mask, (image.size[1], image.size[0]))
    red = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = (mask_resized > 0).astype(np.uint8) * 120
    mask_rgba = Image.fromarray(red, mode="RGBA")
    Image.alpha_composite(overlay, mask_rgba).convert("RGB").save(output_path)
    return output_path


def _heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    heatmap = heatmap.astype(np.float32, copy=False)
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum > minimum:
        heatmap = (heatmap - minimum) / (maximum - minimum)
    else:
        heatmap = np.zeros_like(heatmap)
    red = (heatmap * 255).astype(np.uint8)
    green = (np.clip(1.0 - np.abs(heatmap - 0.75) * 2.0, 0.0, 1.0) * 220).astype(
        np.uint8
    )
    blue = ((1.0 - heatmap) * 80).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)


if __name__ == "__main__":
    main()
