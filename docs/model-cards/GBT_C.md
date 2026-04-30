# GBT_C

LightGBM gradient-boosted-tree regressor for **lens C** of the STAC-ML Markets inference suite.

## Files

| File | Format | Size |
| --- | --- | --- |
| `GBT_C.onnx` | ONNX (with `ai.onnx.ml`) | 120,303,250 B |
| `GBT_C.lgb` | LightGBM native model dump | 146,820,612 B |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `OnnxMLTools 1.14.0` |
| IR version | 4 |
| Opsets | `ai.onnx.ml` v1, `ai.onnx` v8 |
| Input | `float_input` — shape `[N, 1000]`, dtype `float32` |
| Output | `variable` — shape `[N, 1]`, dtype `float32` |
| Op-type breakdown | TreeEnsembleRegressor ×1, Identity ×1 (2 nodes total) |
| Initializers | 0 |

### Tree-ensemble attributes

| Attribute | Value |
| --- | --- |
| Trees | 4,000 |
| Total tree nodes | 2,978,442 |
| Post-transform | NONE |
| Base values | (none) |

## Architecture

LightGBM gradient-boosted regression: 4,000 boosted trees over 1,000 tabular features. By a wide margin the heaviest GBT in the suite — file size is dominated by the encoded tree split structure.

## Stateful classification

Stateless tabular regressor. The **stateful** (trained) arm of the lens-C GBT pair (vs. [`GBT_Null_C`](GBT_Null_C.md)).

## Notes

- Feature count jumps to 1,000 (vs. 125 for lens B) and tree count quadruples — lens C is the high-capacity end of the GBT scaling curve.
