"""Helpers for writing portable evaluation reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence


def relativize_report_paths(
    rows: Sequence[Mapping[str, str | float]],
    path_fields: Sequence[str],
    report_path: str | Path,
) -> list[dict[str, str | float]]:
    """Make absolute artifact paths portable relative to the report directory."""

    report_base = Path(report_path).resolve().parent
    portable_rows: list[dict[str, str | float]] = []
    for row in rows:
        portable = dict(row)
        for field in path_fields:
            value = portable.get(field)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if path.is_absolute():
                portable[field] = Path(os.path.relpath(path, report_base)).as_posix()
        portable_rows.append(portable)
    return portable_rows
