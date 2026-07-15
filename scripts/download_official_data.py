#!/usr/bin/env python
"""Download official dataset archives into ignored local data folders."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

VISA_TAR_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
DEEPPCB_ZIP_URL = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("visa", "deeppcb", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--extract", action="store_true", help="Extract archives after download.")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing downloaded archives."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"visa", "all"}:
        visa_archive = args.output_dir / "VisA_20220922.tar"
        download_file(VISA_TAR_URL, visa_archive, force=args.force)
        if args.extract:
            safe_extract_tar(visa_archive, args.output_dir)

    if args.dataset in {"deeppcb", "all"}:
        deeppcb_archive = args.output_dir / "DeepPCB-master.zip"
        download_file(DEEPPCB_ZIP_URL, deeppcb_archive, force=args.force)
        if args.extract:
            safe_extract_zip(deeppcb_archive, args.output_dir)


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        print(f"exists: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}")
    print(f"to: {destination}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return destination


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            _ensure_inside(output_dir, output_dir / member.name)
        archive.extractall(output_dir)
    print(f"extracted: {archive_path} -> {output_dir}")


def safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            _ensure_inside(output_dir, output_dir / member)
        archive.extractall(output_dir)
    print(f"extracted: {archive_path} -> {output_dir}")


def _ensure_inside(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved != target_resolved and root_resolved not in target_resolved.parents:
        raise ValueError(f"Archive member would extract outside target directory: {target}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("cancelled")
