"""Small dataset factory for scripts and experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from datasets.deeppcb import DeepPCBDataset
from datasets.visa import VisAPCBDataset


def build_dataset_from_config(config: dict[str, Any]):
    """Build a dataset from a YAML-style config dictionary."""

    dataset_config = deepcopy(config.get("dataset", config))
    name = dataset_config.pop("name")

    if name == "visa_pcb":
        return VisAPCBDataset(**dataset_config)
    if name == "deeppcb":
        return DeepPCBDataset(**dataset_config)

    raise ValueError(f"Unknown dataset name: {name!r}")
