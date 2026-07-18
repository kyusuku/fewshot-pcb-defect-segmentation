# Notebooks

The final project includes one readable presentation notebook:
`notebooks/pcb_defect_pipeline.ipynb`.

It should be read as a report-facing layer, not as the implementation source of
truth. The notebook covers:

1. frozen research question and prior-test-exposure disclosure,
2. dataset and repeated few-shot protocol,
3. DINOv2, multi-scale, PatchCore, SAM2, and fusion method summary,
4. normal-only calibration versus oracle diagnostics,
5. primary table loading from `docs/evidence/generated/primary_summary.csv`,
6. paired confidence intervals and supporting comparisons,
7. selected success/failure figures,
8. limitations and decision-rule outcome,
9. exact reproduction commands.

Keep reusable implementation in `src/` and runnable entry points in `scripts/`.
The notebook should call those modules rather than becoming the source of truth.
Exploratory notebooks are fine during development, but they should not replace
tested code paths or hide heavy experiments. Heavy runs stay in scripts and
AutoDL; local notebook cells should only load compact tables and selected assets.

The notebook is bound to `docs/evidence/generated/completion_manifest.json`. It
fails closed if the package is not writing-ready or if any declared SHA-256 does
not match. It has no stored cell outputs or execution counts, so the public file
does not duplicate results or leak a local path. Its executable presentation
contract is checked with:

```bash
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
```

The notebook is a writing handoff, not a manuscript draft and not an experiment
runner. All heavy results remain sourced from the frozen modular pipeline.
