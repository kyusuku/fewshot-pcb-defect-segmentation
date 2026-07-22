from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re


def _load_notebook() -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks/pcb_defect_pipeline.ipynb"
    return notebook_path, json.loads(notebook_path.read_text(encoding="utf-8"))


def _joined_cell_text(notebook: dict[str, object], cell_type: str | None = None) -> str:
    cells = notebook["cells"]
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell_type is None or cell["cell_type"] == cell_type
    )


def test_presentation_notebook_is_evidence_bound_and_executes_top_to_bottom() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path, notebook = _load_notebook()
    cells = notebook["cells"]
    text = _joined_cell_text(notebook)

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


def test_teaching_notebook_covers_the_complete_project() -> None:
    _, notebook = _load_notebook()
    text = _joined_cell_text(notebook)
    required_headings = (
        "How to Use This Notebook",
        "Executive Story",
        "Conceptual Background",
        "Datasets and Benchmark Boundary",
        "Few-Shot Protocol",
        "Repository Map and Artifact Flow",
        "Image Preparation and Alignment",
        "DINOv2 Patch-Feature Extraction",
        "Normal Memory Bank and Anomaly Score",
        "PatchCore-Style Baseline",
        "Multi-Scale Anomaly Proposals",
        "Normal-Only Calibration",
        "Heatmap-to-Prompt Conversion",
        "SAM2 Refinement and Candidate Selection",
        "Anomaly-Consistent Fusion",
        "One-Image Trace",
        "Primary Experiment Matrix",
        "Ablation Matrix",
        "Metrics",
        "Statistical Design",
        "Frozen AutoDL Results",
        "Ablations and Failure Analysis",
        "Qualitative Evidence",
        "Research Answer and Novelty Boundary",
        "Limitations and Threats to Validity",
        "Class Presentation Guide",
        "Anticipated Q&A",
        "Glossary and Symbol Table",
        "Final Cheat Sheet",
        "Reproduction Appendix",
    )
    for heading in required_headings:
        assert heading in text, heading


def test_teaching_notebook_is_detailed_and_source_traceable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, notebook = _load_notebook()
    markdown = _joined_cell_text(notebook, "markdown")
    words = re.findall(r"\b[\w'-]+\b", markdown)
    assert len(words) >= 7000

    required_paths = (
        "src/datasets/visa.py",
        "src/datasets/sampling.py",
        "src/features/dinov2.py",
        "src/features/patchcore.py",
        "src/anomaly/memory_bank.py",
        "src/anomaly/multiscale.py",
        "src/evaluation/calibration.py",
        "src/sam_refine/prompts.py",
        "src/sam_refine/refiner.py",
        "src/sam_refine/fusion.py",
        "src/experiments/spec.py",
        "src/evaluation/statistics.py",
        "configs/experiments/arxiv_primary.yaml",
        "configs/experiments/arxiv_ablations.yaml",
    )
    for relative in required_paths:
        assert relative in markdown
        assert (repo_root / relative).is_file()


def test_teaching_notebook_contains_presentation_and_qa_material() -> None:
    _, notebook = _load_notebook()
    markdown = _joined_cell_text(notebook, "markdown")
    assert len(re.findall(r"^### Slide \d+:", markdown, flags=re.MULTILINE)) >= 10
    assert len(re.findall(r"^### Q\d+\.", markdown, flags=re.MULTILINE)) >= 25
    for topic in (
        "Why did you not train a model?",
        "Why DINOv2?",
        "Why use SAM2 if it is not anomaly-aware?",
        "Why intersection instead of union?",
        "Are the five seeds five folds?",
        "What is actually novel?",
        "What is the biggest limitation?",
    ):
        assert topic in markdown


def test_teaching_notebook_preserves_frozen_claims_and_assets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, notebook = _load_notebook()
    text = _joined_cell_text(notebook)
    for required in (
        "364 primary runs",
        "48 ablation runs",
        "412 source runs",
        "+0.0666",
        "[0.0538, 0.0788]",
        "+0.0565",
        "[0.0462, 0.0665]",
        "multi-scale versus single-scale DINOv2 is inconclusive",
        "not five-fold cross-validation",
        "No model parameters are trained or fine-tuned",
    ):
        assert required.lower() in text.lower()

    linked_pngs = set(re.findall(r"\((\.\./docs/evidence/generated/[^)]+\.png)\)", text))
    assert len(linked_pngs) >= 9
    for relative in linked_pngs:
        assert (repo_root / "notebooks" / relative).resolve().is_file()


def test_teaching_notebook_matches_deterministic_builder() -> None:
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/build_teaching_notebook.py"
    spec = importlib.util.spec_from_file_location("build_teaching_notebook", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook_path = repo_root / "notebooks/pcb_defect_pipeline.ipynb"
    assert notebook_path.read_text(encoding="utf-8") == module.serialized_notebook()
