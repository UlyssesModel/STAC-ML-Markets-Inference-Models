# LSTM_B

Stateful (recurrent) LSTM regressor for **lens B** of the STAC-ML Markets inference suite.

## Files

| File | Format | Size |
| --- | --- | --- |
| `LSTM_B.onnx` | ONNX | 4,028,085 B |
| `LSTM_B.keras64/` | TensorFlow SavedModel (float64) | dir |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `tf2onnx 1.9.3` |
| IR version | 6 |
| Opset | `ai.onnx` v11 |
| Input | `lstm_2_input` — shape `[N, 100, 250]`, dtype `float32` |
| Output | `dense_1` — shape `[N, 1]`, dtype `float32` |
| Parameter count | 1,005,007 (17 initializers; 1,005,002 float + 4 int64 + 1 int32) |
| Op-type breakdown | LSTM ×3, Transpose ×5, Squeeze ×4, Slice ×4, Cast ×6, Shape ×3, Concat ×3, Expand ×3, Unsqueeze ×3, MatMul ×1, Add ×1 (36 nodes total) |

## Keras SavedModel metadata

| Field | Value |
| --- | --- |
| Signature | `serving_default` |
| Input | `lstm_2_input` — shape `(None, 100, 250)`, dtype `float64` |
| Output | `dense_1` — shape `(None, 1)`, dtype `float64` |
| Trainable parameters | 1,002,601 |

### Variable shapes

| Variable | Shape | Dtype |
| --- | --- | --- |
| `lstm_2/lstm_cell_12/kernel` | `[250, 800]` | float64 |
| `lstm_2/lstm_cell_12/recurrent_kernel` | `[200, 800]` | float64 |
| `lstm_2/lstm_cell_12/bias` | `[800]` | float64 |
| `lstm_3/lstm_cell_13/kernel` | `[200, 800]` | float64 |
| `lstm_3/lstm_cell_13/recurrent_kernel` | `[200, 800]` | float64 |
| `lstm_3/lstm_cell_13/bias` | `[800]` | float64 |
| `lstm_4/lstm_cell_14/kernel` | `[200, 800]` | float64 |
| `lstm_4/lstm_cell_14/recurrent_kernel` | `[200, 800]` | float64 |
| `lstm_4/lstm_cell_14/bias` | `[800]` | float64 |
| `dense_1/kernel` | `[200, 1]` | float64 |
| `dense_1/bias` | `[1]` | float64 |

## Architecture

```
Input (None, 100, 250)
  └─ LSTM(units=200, return_sequences=True)        360,800 params
     └─ LSTM(units=200, return_sequences=True)     320,800 params
        └─ LSTM(units=200)                         320,800 params
           └─ Dense(units=1)                           201 params
```

## Stateful classification

Recurrent: maintains hidden / cell state across the 100 timesteps of one sequence. Counts as the **stateful** arm of the lens-B test pair (vs. the stateless [`LSTM_Null_B`](LSTM_Null_B.md)).

## Notes

- The Keras SavedModel is `float64`. The ONNX export is `float32`.
- Three stacked LSTM layers (vs. two for lens A) — depth scales with lens.
