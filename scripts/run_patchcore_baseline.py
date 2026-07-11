#!/usr/bin/env python
"""Run the PatchCore-style baseline with the shared anomaly runner."""

from __future__ import annotations

import sys

if __package__:
    from scripts.run_dinov2_baseline import main
else:
    from run_dinov2_baseline import main


def _append_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _append_default("--feature-backbone", "patchcore_wrn50")
    _append_default("--image-size", "512")
    _append_default("--patch-size", "8")
    main(description=__doc__)
