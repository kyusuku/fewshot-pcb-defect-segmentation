from __future__ import annotations

import subprocess
import re
from pathlib import Path


FORBIDDEN_ROOTS = (
    "data/",
    "outputs/",
    "weights/",
    "external/",
    "artifacts/",
    "reports/",
    "submissions/",
)
FORBIDDEN_EXTENSIONS = (
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
)
TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def test_public_repo_does_not_track_private_artifacts_or_secret_markers() -> None:
    repo = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        cwd=repo,
        text=True,
    ).splitlines()
    bad_paths = [
        path
        for path in tracked
        if path.startswith(FORBIDDEN_ROOTS) or Path(path).suffix.lower() in FORBIDDEN_EXTENSIONS
    ]
    assert bad_paths == []

    secret_patterns = [
        re.compile(r"s" + r"k-[A-Za-z0-9_-]{20,}"),
        re.compile(r"OPENAI_" + r"API_KEY\s*=\s*[A-Za-z0-9_./+-]{8,}"),
        re.compile(r"SUPABASE_" + r"SERVICE_ROLE_KEY\s*=\s*[A-Za-z0-9_./+-]{8,}"),
        re.compile(r"course_" + r"report_private"),
    ]
    leaks = []
    for path in tracked:
        if Path(path).suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = (repo / path).read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.search(text):
                leaks.append((path, pattern.pattern))
    assert leaks == []
