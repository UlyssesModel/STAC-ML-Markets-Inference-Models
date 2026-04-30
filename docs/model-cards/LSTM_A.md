# LSTM_A

Stateful (recurrent) LSTM regressor for **lens A** of the STAC-ML Markets inference suite.

## Files

| File | Format | Size |
| --- | --- | --- |
| `LSTM_A.onnx` | ONNX | 652,338 B |
| `LSTM_A.keras64/` | TensorFlow SavedModel (float64) | dir |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `tf2onnx 1.9.3` |
| IR version | 6 |
| Opset | `ai.onnx` v11 |
| Input | `lstm_input` — shape `[N, 50, 100]`, dtype `float32` |
| Output | `dense` — shape `[N, 1]`, dtype `float32` |
| Parameter count | 161,707 (14 initializers; 161,702 float + 4 int64 + 1 int32) |
| Op-type breakdown | LSTM ×2, Transpose ×3, Squeeze ×3, Slice ×3, Cast ×4, Shape ×2, Concat ×2, Expand ×2, Unsqueeze ×2, MatMul ×1, Add ×1 (25 nodes total) |

Two `LSTM` nodes correspond to the two stacked Keras LSTM layers; the surrounding `Transpose`/`Slice`/`Squeeze`/`Concat` boilerplate is tf2onnx's standard time-major rewriting.

## Keras SavedModel metadata

| Field | Value |
| --- | --- |
| Signature | `serving_default` |
| Input | `lstm_input` — shape `(None, 50, 100)`, dtype `float64` |
| Output | `dense` — shape `(None, 1)`, dtype `float64` |
| Trainable parameters | 160,901 |

### Variable shapes

| Variable | Shape | Dtype |
| --- | --- | --- |
| `lstm/lstm_cell_4/kernel` | `[100, 400]` | float64 |
| `lstm/lstm_cell_4/recurrent_kernel` | `[100, 400]` | float64 |
| `lstm/lstm_cell_4/bias` | `[400]` | float64 |
| `lstm_1/lstm_cell_5/kernel` | `[100, 400]` | float64 |
| `lstm_1/lstm_cell_5/recurrent_kernel` | `[100, 400]` | float64 |
| `lstm_1/lstm_cell_5/bias` | `[400]` | float64 |
| `dense/kernel` | `[100, 1]` | float64 |
| `dense/bias` | `[1]` | float64 |

## Architecture

```
Input (None, 50, 100)
  └─ LSTM(units=100, return_sequences=True)         80,400 params
     └─ LSTM(units=100)                             80,400 params
        └─ Dense(units=1)                              101 params
```

Gate count 4 (i, f, c, o) is implied by the kernel column dimension `4 × units = 400`.

## Stateful classification

Recurrent: maintains hidden / cell state across the 50 timesteps of one sequence. Counts as the **stateful** arm of the lens-A test pair (vs. the stateless [`LSTM_Null_A`](LSTM_Null_A.md)).

## Notes

- The Keras SavedModel is `float64` (hence the `.keras64` suffix). The ONNX export is `float32`; cast inputs accordingly.
- The 806-parameter delta between Keras (160,901) and ONNX (161,707) reflects ONNX-side helper constants (dimension/shape tensors) injected by tf2onnx, not extra learned weights.
