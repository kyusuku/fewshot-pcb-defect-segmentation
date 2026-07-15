# Exact Compact Heatmap Storage Design

**Date:** 2026-07-16  
**Scope:** ArXiv evidence execution storage only  
**Status:** Approved by the autonomous execution mandate

## Problem

The frozen-commit AutoDL preflight produced one complete PCB1 multi-scale DINOv2
run in 13 minutes 43 seconds. Its metrics reproduced the earlier result, but its
381 losslessly compressed full-resolution heatmaps occupy about 1.8 GB. The
machine has 126 GB free, so retaining the required referenced artifacts for all
364 primary and 48 ablation runs would exhaust the disk before the matrix
checkers and evidence builder can validate them.

Deleting heatmaps after downstream use is not acceptable. The matrix checker
requires every CSV-referenced output to exist and match the SHA-256 identity
recorded in run provenance. The execution mandate also requires checker-valid
outputs to remain intact until compact evidence is built, transferred, and
independently verified.

## Decision

Store anomaly heatmaps as exact projected-component archives instead of
materialized full-resolution float arrays.

Each archive contains the patch-score grid for the global view and every local
crop, together with all geometry needed by the existing projection function:

- component order;
- source canvas width and height;
- crop boxes in source coordinates;
- prepared image sizes;
- patch sizes;
- component source sizes;
- content boxes used to remove padding;
- the registered `max` or `mean` fusion rule;
- a versioned storage-schema marker.

Loading the archive projects each component with the same float32 bilinear
implementation and applies the components in the same order as the current
inference path. Consumers continue to receive a two-dimensional float32 array.
Legacy `.npy` and materialized `.npz` heatmaps remain readable.

This is a storage representation change only. It must not change support
selection, extracted features, memory-bank construction, anomaly scores,
calibration, SAM2 prompts, predicted masks, metrics, matrix membership, or the
frozen YAML protocol.

## Alternatives Rejected

### Purchase or attach more storage

This preserves the current representation but needs new billable authority and
would retain hundreds of gigabytes of redundant upsampled values. It remains a
fallback only if exact reconstruction cannot be proven.

### Delete per-image intermediates after downstream runs

This fits the disk, but it makes completed runs artifact-missing and checksum
invalid. It contradicts both matrix completion gates and prevents independent
verification of the evidence freeze.

### Quantize or downsample materialized heatmaps

Float16, integer quantization, and lower-resolution evaluation may be compact,
but they can change thresholds, prompt geometry, masks, and metrics. They would
alter the frozen numerical protocol and are therefore out of scope.

## Components and Interfaces

### Projection model

`src/anomaly/heatmap.py` owns immutable component and archive value objects.
Each component validates its patch-score grid and projection geometry. The
archive validates component order, crop bounds, source geometry, and fusion
choice. Its render method is the single implementation of full-resolution
reconstruction.

### Inference

`src/anomaly/multiscale.py` exposes a component-producing inference path. The
existing `compute_anomaly_heatmap` API delegates to it and renders immediately,
preserving callers and tests. The experiment script uses the component result
both to obtain the image score/debug visualization and to write the compact
archive without recomputing features or scores.

### Storage

`src/utils/heatmap_io.py` adds a versioned `npz_components` encoding. Archives
contain only non-pickle NumPy arrays. The loader rejects unknown versions,
missing or extra fields, invalid scalar values, non-finite scores, malformed
geometry, and unsafe object arrays. Existing encodings remain unchanged.

### Matrix execution and provenance

The matrix runner requests `npz_components`. Referenced-artifact validation and
SHA-256 provenance continue to operate on the archive file itself. Because the
command changes, the effective execution identity changes and all final runs
must use the new clean commit. Existing preflight output remains preliminary
and must not be mixed into the final matrix.

## Exactness Invariants

For identical input features and memory bank:

1. rendered component output must be bit-for-bit equal to the existing
   full-resolution output for single-scale, multi-scale max, and multi-scale
   mean paths;
2. loading a saved component archive must be bit-for-bit equal to rendering its
   in-memory component object;
3. image scores, calibration thresholds, heatmap metrics, guided-SAM2 masks,
   and fusion masks must be unchanged;
4. archive paths must stay inside their owning run directory and remain covered
   by existing provenance checksums;
5. a representative real PCB1 run must reproduce the old preflight metrics and
   every loaded heatmap exactly before the representation is admitted to the
   full matrix.

## Error Handling

Malformed component archives fail closed with a descriptive `ValueError`.
There is no fallback to an approximate reconstruction. Matrix staging remains
atomic, so a failed write or validation cannot be promoted as a complete run.
Legacy materialized outputs continue to load, but final runs use one encoding
consistently.

## Verification

Implementation follows test-driven development:

1. unit tests for component validation and exact single/multi-scale rendering;
2. storage round-trip and malformed-schema tests;
3. experiment-command tests proving `npz_components` is requested;
4. the complete local suite, Ruff lint, Ruff format check, and diff check;
5. the complete remote suite on the RTX 4090;
6. a real PCB1 frozen-protocol run in a new output root;
7. bit-for-bit comparison of all 381 reconstructed heatmaps and exact equality
   of calibration/metric JSON values against the materialized preflight run;
8. measured archive size and projected full-matrix disk budget with a safety
   margin before any matrix launch.

If any numerical comparison differs, this design is rejected and the full
matrix remains paused pending either a corrected exact implementation or user
approval for additional storage.
