# Exact Compact Heatmap Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store exact anomaly-map projection components compactly while preserving bit-for-bit full-resolution heatmaps and all downstream results.

**Architecture:** Introduce validated immutable projection-component objects in `anomaly.heatmap`, make multi-scale inference produce an archive that renders through the existing projection math, and add a versioned non-pickle NPZ encoding whose loader reconstructs the full heatmap transparently. Existing APIs and legacy heatmap formats remain compatible; the frozen matrix command switches to the exact component encoding and is accepted only after local and real-data equivalence checks.

**Tech Stack:** Python 3.10+, NumPy, PyTorch feature extractors, pytest, Ruff, AutoDL RTX 4090

---

## File Map

- `src/anomaly/heatmap.py`: validated component/archive value objects and deterministic rendering.
- `src/anomaly/multiscale.py`: component-producing inference path; compatibility wrapper returning arrays.
- `src/utils/heatmap_io.py`: versioned component NPZ writer/reader plus legacy compatibility.
- `scripts/run_dinov2_baseline.py`: save component archives while using rendered arrays for scores/debugging.
- `src/experiments/runner.py`: request the exact compact encoding for matrix runs.
- `tests/test_feature_geometry.py`: projection archive validation and rendering exactness.
- `tests/test_multiscale_anomaly.py`: inference compatibility and component capture.
- `tests/test_heatmap_io.py`: archive round trip and malformed-schema rejection.
- `tests/test_experiment_runner.py`: frozen command identity.
- `README.md`, `docs/autodl_data_setup.md`, `docs/evidence/README.md`: representation and compatibility notes.

### Task 1: Projection components and exact rendering

**Files:**
- Modify: `src/anomaly/heatmap.py`
- Test: `tests/test_feature_geometry.py`

- [ ] **Step 1: Write failing single- and multi-component rendering tests**

Add tests that construct `PatchHeatmapComponent` values from small float32 patch
grids and `PatchFeatureMap` geometry, then assert exact array equality between
`projected.render()` and the current explicit projection/fusion:

```python
component = PatchHeatmapComponent.from_feature_map(patch_scores, feature_map)
projected = ProjectedHeatmap(
    source_size=(8, 6),
    fusion="max",
    components=(
        PositionedHeatmap((0, 0, 8, 6), component),
        PositionedHeatmap((2, 1, 6, 5), crop_component),
    ),
)
expected = project_patch_heatmap_to_source(patch_scores, feature_map)
expected[1:5, 2:6] = np.maximum(expected[1:5, 2:6], crop_expected)
np.testing.assert_array_equal(projected.render(), expected)
```

Cover `max`, `mean`, first-component-global ordering, crop bounds, float32 and
finite-value requirements, and inconsistent component source geometry.

- [ ] **Step 2: Run the new tests and confirm the missing types fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_feature_geometry.py -q -p no:cacheprovider
```

Expected: collection or import failure naming `PatchHeatmapComponent` or
`ProjectedHeatmap`.

- [ ] **Step 3: Implement the immutable validated value objects**

Add frozen dataclasses with these public interfaces:

```python
@dataclass(frozen=True)
class PatchHeatmapComponent:
    patch_scores: np.ndarray
    image_size: tuple[int, int]
    patch_size: int
    source_size: tuple[int, int]
    content_box: tuple[int, int, int, int]

    @classmethod
    def from_feature_map(cls, patch_scores, feature_map):
        return cls(
            patch_scores=np.asarray(patch_scores, dtype=np.float32),
            image_size=feature_map.image_size,
            patch_size=feature_map.patch_size,
            source_size=feature_map.source_size,
            content_box=feature_map.content_box,
        )

    def render(self) -> np.ndarray:
        return project_patch_heatmap_to_source(self.patch_scores, self)

    def valid_patch_mask(self) -> np.ndarray:
        grid_h, grid_w = self.patch_scores.shape
        width, height = self.image_size
        x1, y1, x2, y2 = self.content_box
        x_centers = (np.arange(grid_w, dtype=np.float32) + 0.5) * width / grid_w
        y_centers = (np.arange(grid_h, dtype=np.float32) + 0.5) * height / grid_h
        return (
            (y_centers[:, None] >= y1)
            & (y_centers[:, None] < y2)
            & (x_centers[None, :] >= x1)
            & (x_centers[None, :] < x2)
        )

@dataclass(frozen=True)
class PositionedHeatmap:
    box: tuple[int, int, int, int]
    component: PatchHeatmapComponent

@dataclass(frozen=True)
class ProjectedHeatmap:
    source_size: tuple[int, int]
    fusion: str
    components: Sequence[PositionedHeatmap]

    def render(self) -> np.ndarray:
        global_item, *crops = self.components
        fused = global_item.component.render()
        counts = np.ones_like(fused, dtype=np.float32) if self.fusion == "mean" else None
        for item in crops:
            x1, y1, x2, y2 = item.box
            crop = item.component.render()
            if self.fusion == "max":
                fused[y1:y2, x1:x2] = np.maximum(fused[y1:y2, x1:x2], crop)
            else:
                fused[y1:y2, x1:x2] += crop
                counts[y1:y2, x1:x2] += 1.0
        if counts is not None:
            fused /= np.maximum(counts, 1.0)
        return fused.astype(np.float32, copy=False)
```

`render()` must reuse `project_patch_heatmap_to_source`, initialize from the
first full-canvas component, and apply remaining components in tuple order with
the current float32 max/mean operations.

- [ ] **Step 4: Run geometry tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1 explicitly**

```bash
git add src/anomaly/heatmap.py tests/test_feature_geometry.py
git commit -m "feat: model exact projected heatmap components"
```

### Task 2: Versioned component archive storage

**Files:**
- Modify: `src/utils/heatmap_io.py`
- Test: `tests/test_heatmap_io.py`

- [ ] **Step 1: Write failing exact-round-trip and schema-rejection tests**

Create a projected archive with two components, save it with
`storage="npz_components"`, and require:

```python
saved = save_heatmap(path, projected, storage="npz_components")
np.testing.assert_array_equal(load_heatmap(saved), projected.render())
assert saved.stat().st_size < projected.render().nbytes // 4
```

Add parameterized malformed files for an unknown version, missing field, extra
field, object dtype, non-finite patch score, invalid fusion, and invalid crop
geometry. Preserve the three existing materialized `.npy`/`.npz` tests.

- [ ] **Step 2: Run storage tests and confirm the unsupported format fails**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_heatmap_io.py -q -p no:cacheprovider
```

Expected: failure reporting unsupported `npz_components` storage.

- [ ] **Step 3: Implement a strict non-pickle schema**

Extend `HEATMAP_STORAGE_FORMATS` and use these exact archive fields:

```python
{
    "format_version",
    "storage_kind",
    "source_size",
    "fusion",
    "boxes",
    "patch_scores",
    "image_sizes",
    "patch_sizes",
    "component_source_sizes",
    "content_boxes",
}
```

Use format version `1` and storage kind `projected_patch_components`. Stack
component grids only after verifying identical grid shapes. On load, require
the exact field set and primitive numeric/string dtypes, rebuild the value
objects, and return `ProjectedHeatmap.render()`. Do not enable pickle or an
approximate fallback.

- [ ] **Step 4: Run storage and geometry tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_heatmap_io.py tests/test_feature_geometry.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2 explicitly**

```bash
git add src/utils/heatmap_io.py tests/test_heatmap_io.py
git commit -m "feat: store exact heatmap component archives"
```

### Task 3: Capture components during anomaly inference

**Files:**
- Modify: `src/anomaly/multiscale.py`
- Test: `tests/test_multiscale_anomaly.py`
- Test: `tests/test_feature_cache.py`

- [ ] **Step 1: Write failing compatibility and exactness tests**

For single-scale, multi-scale max, and multi-scale mean, call both APIs and
require bit-for-bit equality:

```python
kwargs = {
    "image": image,
    "extractor": extractor,
    "memory_bank": memory_bank,
    "crop_sizes": [4],
    "crop_overlap": 0.5,
    "fusion": "max",
}
archive = compute_anomaly_heatmap_components(**kwargs)
array = compute_anomaly_heatmap(**kwargs)
np.testing.assert_array_equal(archive.render(), array)
```

Retain feature-cache tests and assert the component path does not perform an
extra extractor call.

- [ ] **Step 2: Run the focused tests and confirm the new API is absent**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_multiscale_anomaly.py tests/test_feature_cache.py -q -p no:cacheprovider
```

Expected: import failure naming `compute_anomaly_heatmap_components`.

- [ ] **Step 3: Refactor scoring without changing numerical order**

Change `_score_image` to return a `PatchHeatmapComponent` built from the scored
patch grid and feature-map geometry. Implement
`compute_anomaly_heatmap_components` with the same arguments as
`compute_anomaly_heatmap` and a `ProjectedHeatmap` return type, preserving crop
iteration order. Make `compute_anomaly_heatmap` return the rendered value from
the new component-producing function.

- [ ] **Step 4: Run focused anomaly/cache tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit Task 3 explicitly**

```bash
git add src/anomaly/multiscale.py tests/test_multiscale_anomaly.py tests/test_feature_cache.py
git commit -m "refactor: preserve heatmap projection components"
```

### Task 4: Experiment integration and frozen execution identity

**Files:**
- Modify: `scripts/run_dinov2_baseline.py`
- Modify: `src/experiments/runner.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_baseline_script.py`

- [ ] **Step 1: Write failing command and script tests**

Change command assertions to require `--heatmap-format npz_components`. Add a
tiny color-patch script test that loads every output archive and checks the
reported `image_score` equals `float(loaded.max())`.

- [ ] **Step 2: Run the focused tests and observe the old encoding**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_experiment_runner.py tests/test_baseline_script.py -q -p no:cacheprovider
```

Expected: command assertion reports `npz_compressed` instead of
`npz_components`.

- [ ] **Step 3: Save archives and render once per query**

In the baseline script, call `compute_anomaly_heatmap_components`, render it
once for scoring/debugging, and pass the component object to `save_heatmap`.
In the matrix runner, request `npz_components` for all heatmap methods.

- [ ] **Step 4: Run focused integration tests and the offline matrix smoke**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_experiment_runner.py tests/test_baseline_script.py \
  tests/test_arxiv_smoke.py -q -p no:cacheprovider
```

Expected: all tests pass and smoke archives are component-encoded.

- [ ] **Step 5: Commit Task 4 explicitly**

```bash
git add scripts/run_dinov2_baseline.py src/experiments/runner.py \
  tests/test_experiment_runner.py tests/test_baseline_script.py tests/test_arxiv_smoke.py
git commit -m "feat: use exact compact heatmaps in paper runs"
```

### Task 5: Documentation and complete local verification

**Files:**
- Modify: `README.md`
- Modify: `docs/autodl_data_setup.md`
- Modify: `docs/evidence/README.md`

- [ ] **Step 1: Document the representation boundary**

State that final runs retain checker-valid raw artifacts as exact component
archives, loaders reconstruct full-resolution float32 maps, legacy formats
remain readable, and no quantization/downsampling changes the protocol.

- [ ] **Step 2: Run all live CLI help and the full quality gate**

```bash
for script in scripts/run_dinov2_baseline.py scripts/run_experiment_matrix.py \
  scripts/check_experiment_matrix.py scripts/analyze_paper_results.py \
  scripts/build_paper_evidence.py; do
  PYTHONPATH=src venv/bin/python "$script" --help >/dev/null
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src venv/bin/python -m pytest -q -p no:cacheprovider
venv/bin/python -m ruff check --no-cache .
venv/bin/python -m ruff format --check src scripts tests
git diff --check
git status --short --branch
```

Expected: all tests and subtests pass; lint, format, and diff checks exit zero;
only intentional documentation changes remain.

- [ ] **Step 3: Commit docs and push the clean feature branch**

```bash
git add README.md docs/autodl_data_setup.md docs/evidence/README.md
git commit -m "docs: explain exact compact heatmap artifacts"
git push origin codex/arxiv-evidence-readiness
```

### Task 6: AutoDL bit-for-bit admission gate

**Files:**
- No tracked file changes unless a verified bug is found.
- Write ignored state only: `outputs/arxiv_loop_state.json`

- [ ] **Step 1: Pull the exact clean commit and repeat remote verification**

Confirm the remote SHA equals the pushed branch, Git is clean, CUDA identifies
the RTX 4090, and run the same full pytest/Ruff gate as Task 5.

- [ ] **Step 2: Run a new PCB1 representative output**

```bash
PYTHONPATH=src venv/bin/python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary_components_preflight \
  --feature-cache-dir outputs/.feature_cache \
  --device cuda \
  --run-id dinov2_multi__pcb1__fold0__k1__seed4880
```

Expected: complete run with no dirty/stale provenance.

- [ ] **Step 3: Compare every output against the materialized preflight**

Use a read-only comparison that joins validation and test rows by `sample_id`,
loads both heatmap paths through `load_heatmap`, and requires
`np.array_equal(old, new)` for all 381 images. Require exact JSON equality for
`calibration.json` and `test/metrics.json` after ignoring no fields.

- [ ] **Step 4: Measure and approve the disk budget**

Record new run size, feature-cache size, free disk, and conservative projected
primary+ablation size in `outputs/arxiv_loop_state.json`. Require at least 20%
free-space headroom before launching the 364-run matrix.

- [ ] **Step 5: Resume the main readiness execution queue**

Only after Steps 1-4 pass, freeze the new clean commit as the sole final-run
revision, complete the remaining PatchCore/SAM2/fusion representative paths,
and then launch the primary matrix in persistent `tmux` with `--resume`.
