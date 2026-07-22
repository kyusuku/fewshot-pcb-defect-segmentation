# Notebooks

The final project has one comprehensive teaching and presentation notebook:
`notebooks/pcb_defect_pipeline.ipynb`.

Read it from top to bottom to understand the complete project without opening
another explanatory document. It covers:

1. motivation, research question, contribution, and final answer;
2. DINOv2, PatchCore, SAM2, few-shot, and normal-only background;
3. VisA/DeepPCB dataset boundaries and the repeated-support protocol;
4. repository architecture, provenance, and AutoDL artifact flow;
5. every pipeline stage with equations, tensor shapes, code excerpts, and source
   paths;
6. experiment-matrix derivations, metrics, repeated-measures statistics, and
   frozen AutoDL result tables;
7. success/failure panels, ablations, limitations, and novelty boundaries;
8. a 12-slide class presentation guide, 30 anticipated Q&A answers, glossary,
   cheat sheet, and exact reproduction commands.

The notebook is readable without execution. Its three optional Python cells use
only the standard library to validate the tracked evidence package and print
additional details. They do not require a dataset, GPU, PyTorch, DINOv2 weights,
or a SAM2 checkpoint.

Core implementation remains in `src/`, experiment entry points remain in
`scripts/`, fixed choices remain in `configs/`, and frozen results remain in
`docs/evidence/generated/`. The notebook explains those sources; it does not
duplicate or alter their algorithms. Heavy experiments stay in scripts and
AutoDL.

The notebook is generated deterministically from the existing source and frozen
evidence by a documentation-only builder:

```bash
venv/bin/python scripts/build_teaching_notebook.py
venv/bin/python scripts/build_teaching_notebook.py --check
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
```

The executable contract verifies complete chapter coverage, narrative depth,
source traceability, presentation/Q&A material, linked figures, output-free
cells, evidence checksums, frozen matrix counts, final conclusion, deterministic
generation, and top-to-bottom execution.
