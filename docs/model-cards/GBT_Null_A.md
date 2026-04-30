# GBT_Null_A

**Null baseline** GBT for lens A. Single trivial tree, present only to match the I/O contract of [`GBT_A`](GBT_A.md) for the RS-40 stateful-vs-stateless comparison.

## Files

| File | Format | Size |
| --- | --- | --- |
| `GBT_Null_A.onnx` | ONNX (with `ai.onnx.ml`) | 5,030 B |
| `GBT_Null_A.lgb` | LightGBM native model dump | 4,474 B |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `OnnxMLTools 1.14.0` |
| IR version | 4 |
| Opsets | `ai.onnx.ml` v1, `ai.onnx` v8 |
| Input | `float_input` — shape `[N, 60]`, dtype `float32` |
| Output | `variable` — shape `[N, 1]`, dtype `float32` |
| Op-type breakdown | TreeEnsembleRegressor ×1, Identity ×1 (2 nodes total) |

### Tree-ensemble attributes

| Attribute | Value |
| --- | --- |
| Trees | 1 |
| Total tree nodes | 121 |
| Post-transform | NONE |
| Base values | (none) |

## Architecture

A single LightGBM tree over the same 60-feature input as `GBT_A`. With one tree and no boosting, this is a fixed null prediction — the GBT analogue of an untrained baseline.

## Stateful classification

Stateless. The **stateless** (untrained / null) arm of the lens-A GBT pair.

## Notes

- Same input/output shapes as [`GBT_A`](GBT_A.md), so harnesses can swap the two transparently.
