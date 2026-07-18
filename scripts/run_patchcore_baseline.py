#!/usr/bin/env python
"""Run the PatchCore-style baseline with the shared anomaly runner."""

from __future__ import annotations

from pathlib import Path

if __package__:
    from scripts.run_dinov2_baseline import main
else:
    from run_dinov2_baseline import main


PATCHCORE_ARGUMENT_DEFAULTS: dict[str, object] = {
    "feature_backbone": "patchcore_wrn50",
    "image_size": 512,
    "patch_size": 8,
    "output_dir": Path("outputs/patchcore_baseline"),
}


if __name__ == "__main__":
    main(description=__doc__, argument_defaults=PATCHCORE_ARGUMENT_DEFAULTS)
