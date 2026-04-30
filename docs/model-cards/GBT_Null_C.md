# GBT_Null_C

**Null baseline** GBT for lens C. Single trivial tree, present only to match the I/O contract of [`GBT_C`](GBT_C.md) for the RS-40 stateful-vs-stateless comparison.

## Files

| File | Format | Size |
| --- | --- | --- |
| `GBT_Null_C.onnx` | ONNX (with `ai.onnx.ml`) | 79,214 B |
| `GBT_Null_C.lgb` | LightGBM native model dump | 71,987 B |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `OnnxMLTools 1.14.0` |
| IR version | 4 |
| Opsets | `ai.onnx.ml` v1, `ai.onnx` v8 |
| Input | `float_input` — shape `[N, 1000]`, dtype `float32` |
| Output | `variable` — shape `[N, 1]`, dtype `float32` |
| Op-type breakdown | TreeEnsembleRegressor ×1, Identity ×1 (2 nodes total) |

### Tree-ensemble attributes

| Attribute | Value |
| --- | --- |
| Trees | 1 |
| Total tree nodes | 2,001 |
| Post-transform | NONE |
| Base values | (none) |

## Architecture

A single LightGBM tree over the same 1,000-feature input as `GBT_C`. The 2,001-node count reflects a single deep tree (vs. 4,000 shallow boosted trees in `GBT_C`).

## Stateful classification

Stateless. The **stateless** (untrained / null) arm of the lens-C GBT pair.
