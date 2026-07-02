"""Generate tiny synthetic datasets for loader and visualization smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SyntheticDebugDatasets:
    """Root paths for generated VisA-like and DeepPCB-like debug data."""

    root: Path
    visa_root: Path
    deeppcb_root: Path


def create_synthetic_debug_datasets(root: str | Path) -> SyntheticDebugDatasets:
    """Create tiny VisA-like and DeepPCB-like datasets under `root`.

    The generated files are intentionally small and deterministic. They are not
    meant to model real PCB statistics; they only exercise path discovery,
    annotation loading, mask loading, box parsing, and visualization.
    """

    root = Path(root)
    visa_root = root / "VisA"
    deeppcb_root = root / "DeepPCB" / "PCBData"

    _create_synthetic_visa(visa_root)
    _create_synthetic_deeppcb(deeppcb_root)

    return SyntheticDebugDatasets(root=root, visa_root=visa_root, deeppcb_root=deeppcb_root)


def _create_synthetic_visa(root: Path) -> None:
    image_dir = root / "pcb1" / "test" / "anomaly"
    normal_dir = root / "pcb1" / "test" / "normal"
    mask_dir = root / "pcb1" / "ground_truth" / "anomaly"
    image_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    anomaly = _pcb_like_image(size=64, defect=True)
    normal = _pcb_like_image(size=64, defect=False)
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle((36, 28, 49, 41), fill=255)

    anomaly.save(image_dir / "pcb1_anomaly_000.png")
    normal.save(normal_dir / "pcb1_normal_000.png")
    mask.save(mask_dir / "pcb1_anomaly_000.png")


def _create_synthetic_deeppcb(root: Path) -> None:
    group_dir = root / "group00000"
    group_dir.mkdir(parents=True, exist_ok=True)

    template = _pcb_like_image(size=64, defect=False)
    test = _pcb_like_image(size=64, defect=True)
    template.save(group_dir / "00000000_temp.jpg")
    test.save(group_dir / "00000000_test.jpg")
    (group_dir / "00000000_test.txt").write_text("34,26,51,43,3\n10,10,20,20,6\n")


def _pcb_like_image(size: int, defect: bool) -> Image.Image:
    image = Image.new("RGB", (size, size), (28, 74, 57))
    draw = ImageDraw.Draw(image)

    # Simple trace grid and pads, enough structure for visual debugging.
    for offset in range(8, size, 14):
        draw.line((0, offset, size, offset), fill=(190, 160, 90), width=1)
        draw.line((offset, 0, offset, size), fill=(190, 160, 90), width=1)
    for y in range(14, size, 24):
        for x in range(14, size, 24):
            draw.rectangle((x, y, x + 6, y + 6), fill=(38, 42, 45))

    if defect:
        draw.rectangle((36, 28, 49, 41), fill=(230, 45, 55))
        draw.line((10, 10, 20, 20), fill=(238, 238, 225), width=3)
    return image
