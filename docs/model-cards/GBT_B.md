# GBT_B

LightGBM gradient-boosted-tree regressor for **lens B** of the STAC-ML Markets inference suite.

## Files

| File | Format | Size |
| --- | --- | --- |
| `GBT_B.onnx` | ONNX (with `ai.onnx.ml`) | 7,093,657 B |
| `GBT_B.lgb` | LightGBM native model dump | 9,317,885 B |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `OnnxMLTools 1.14.0` |
| IR version | 4 |
| Opsets | `ai.onnx.ml` v1, `ai.onnx` v8 |
| Input | `float_input` — shape `[N, 125]`, dtype `float32` |
| Output | `variable` — shape `[N, 1]`, dtype `float32` |
| Op-type breakdown | TreeEnsembleRegressor ×1, Identity ×1 (2 nodes total) |
| Initializers | 0 |

### Tree-ensemble attributes

| Attribute | Value |
| --- | --- |
| Trees | 1,000 |
| Total tree nodes | 184,454 |
| Post-transform | NONE |
| Base values | (none) |

## Architecture

LightGBM gradient-boosted regression: 1,000 boosted trees over 125 tabular features.

## Stateful classification

Stateless tabular regressor. The **stateful** (trained) arm of the lens-B GBT pair (vs. [`GBT_Null_B`](GBT_Null_B.md)).

## Notes

- Tree count and feature count both grow ~4× from lens A.
