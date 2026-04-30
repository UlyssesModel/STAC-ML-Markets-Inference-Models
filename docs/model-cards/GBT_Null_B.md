# GBT_Null_B

**Null baseline** GBT for lens B. Single trivial tree, present only to match the I/O contract of [`GBT_B`](GBT_B.md) for the RS-40 stateful-vs-stateless comparison.

## Files

| File | Format | Size |
| --- | --- | --- |
| `GBT_Null_B.onnx` | ONNX (with `ai.onnx.ml`) | 10,084 B |
| `GBT_Null_B.lgb` | LightGBM native model dump | 8,979 B |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `OnnxMLTools 1.14.0` |
| IR version | 4 |
| Opsets | `ai.onnx.ml` v1, `ai.onnx` v8 |
| Input | `float_input` — shape `[N, 125]`, dtype `float32` |
| Output | `variable` — shape `[N, 1]`, dtype `float32` |
| Op-type breakdown | TreeEnsembleRegressor ×1, Identity ×1 (2 nodes total) |

### Tree-ensemble attributes

| Attribute | Value |
| --- | --- |
| Trees | 1 |
| Total tree nodes | 251 |
| Post-transform | NONE |
| Base values | (none) |

## Architecture

A single LightGBM tree over the same 125-feature input as `GBT_B`.

## Stateful classification

Stateless. The **stateless** (untrained / null) arm of the lens-B GBT pair.
