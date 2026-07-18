from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path


def test_presentation_notebook_is_evidence_bound_and_executes_top_to_bottom() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks/pcb_defect_pipeline.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    text = "\n".join("".join(cell.get("source", [])) for cell in cells)

    assert "docs/evidence/generated" in text
    assert "artifacts/paper_assets" not in text
    assert "unconditional SAM2" not in text
    assert "ready_for_writing" in text
    assert "anomaly_consistent_sam2_improves_robustly" in text
    assert "no model parameters are trained or fine-tuned" in text.lower()
    assert "../docs/evidence/generated/method_figure.png" in text
    for category in ("pcb1", "pcb2", "pcb3", "pcb4"):
        for role in ("success", "failure"):
            assert f"../docs/evidence/generated/qualitative_figures/{category}_{role}.png" in text

    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []

    namespace: dict[str, object] = {}
    previous_cwd = Path.cwd()
    output = io.StringIO()
    try:
        os.chdir(repo_root)
        with contextlib.redirect_stdout(output):
            for index, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell.get("source", []))
                exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous_cwd)

    rendered_output = output.getvalue()
    assert "ready_for_writing: True" in rendered_output
    assert "source runs: 412" in rendered_output
    assert "primary matrix: 364 / 364" in rendered_output
    assert "ablation matrix: 48 / 48" in rendered_output
