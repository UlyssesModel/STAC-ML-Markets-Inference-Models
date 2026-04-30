# GBT_A

LightGBM gradient-boosted-tree regressor for **lens A** of the STAC-ML Markets inference suite.

## Files

| File | Format | Size |
| --- | --- | --- |
| `GBT_A.onnx` | ONNX (with `ai.onnx.ml`) | 194,567 B |
| `GBT_A.lgb` | LightGBM native model dump | 302,461 B |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `OnnxMLTools 1.14.0` |
| IR version | 4 |
| Opsets | `ai.onnx.ml` v1, `ai.onnx` v8 |
| Input | `float_input` — shape `[N, 60]`, dtype `float32` |
| Output | `variable` — shape `[N, 1]`, dtype `float32` |
| Op-type breakdown | TreeEnsembleRegressor ×1, Identity ×1 (2 nodes total) |
| Initializers | 0 (all weights live inside `TreeEnsembleRegressor` attributes) |

### Tree-ensemble attributes

| Attribute | Value |
| --- | --- |
| Trees | 250 |
| Total tree nodes | 5,218 |
| Post-transform | NONE |
| Base values | (none) |

`n_targets = 1`; the leading `Identity` is just an output rename.

## Architecture

LightGBM gradient-boosted regression: 250 boosted trees over 60 tabular input features. `TreeEnsembleRegressor` from the `ai.onnx.ml` operator set evaluates the full ensemble in one node.

## Stateful classification

Stateless tabular regressor — does not maintain state across calls. Counts as the **stateful** (real, trained) arm of the lens-A GBT pair (vs. the untrained baseline [`GBT_Null_A`](GBT_Null_A.md)). Note that "stateful" here is shorthand for "the trained model in the pair", not for runtime statefulness.

## Notes

- The 60-feature tabular input is a separate view from the LSTM 50×100 sequence input — GBT and LSTM lenses share the lens label A/B/C but consume different feature representations.
