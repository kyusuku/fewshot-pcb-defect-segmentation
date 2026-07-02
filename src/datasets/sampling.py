"""Few-shot support-set sampling utilities."""

from __future__ import annotations

import random
from collections import defaultdict

from datasets.types import DatasetRecord


def sample_few_shot_normals(
    records: list[DatasetRecord],
    k: int,
    seed: int = 4880,
    categories: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[DatasetRecord]]:
    """Sample up to `k` normal records per category.

    This is deterministic for a fixed seed and never samples anomalous records.
    It raises when a requested category has fewer than `k` normal records so a
    memory-bank run cannot silently use a weaker support set than intended.
    """

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    wanted = set(categories) if categories is not None else None
    grouped: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        if record.label != 0:
            continue
        if wanted is not None and record.category not in wanted:
            continue
        grouped[record.category].append(record)

    rng = random.Random(seed)
    support: dict[str, list[DatasetRecord]] = {}
    for category in sorted(grouped):
        candidates = list(grouped[category])
        if len(candidates) < k:
            raise ValueError(
                f"Category {category!r} has only {len(candidates)} normal samples, "
                f"cannot sample k={k}."
            )
        rng.shuffle(candidates)
        support[category] = sorted(candidates[:k], key=lambda record: record.sample_id)

    if wanted is not None:
        missing = sorted(wanted - set(support))
        if missing:
            raise ValueError(f"No normal samples available for categories: {missing}")

    return support
