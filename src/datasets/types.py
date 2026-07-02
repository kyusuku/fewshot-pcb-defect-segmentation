"""Shared sample types used by dataset loaders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoxAnnotation:
    """One DeepPCB bounding-box annotation."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self, width: int, height: int) -> "BoxAnnotation":
        """Return a copy clipped to image bounds."""

        x1 = min(max(self.x1, 0.0), float(width - 1))
        y1 = min(max(self.y1, 0.0), float(height - 1))
        x2 = min(max(self.x2, 0.0), float(width - 1))
        y2 = min(max(self.y2, 0.0), float(height - 1))
        return BoxAnnotation(
            x1=min(x1, x2),
            y1=min(y1, y2),
            x2=max(x1, x2),
            y2=max(y1, y2),
            class_id=self.class_id,
            class_name=self.class_name,
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRecord:
    """Path-level sample metadata before image decoding."""

    dataset: str
    category: str
    sample_id: str
    split: str
    image_path: Path
    label: int
    mask_path: Path | None = None
    box_path: Path | None = None
    template_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomaly(self) -> bool:
        return bool(self.label)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("image_path", "mask_path", "box_path", "template_path"):
            value = data[key]
            data[key] = str(value) if value is not None else None
        return data
