# LSTM_C

Stateful (recurrent) LSTM regressor for **lens C** of the STAC-ML Markets inference suite.

## Files

| File | Format | Size |
| --- | --- | --- |
| `LSTM_C.onnx` | ONNX | 120,142,435 B |
| `LSTM_C.keras64/` | TensorFlow SavedModel (float64) | dir |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `tf2onnx 1.9.3` |
| IR version | 6 |
| Opset | `ai.onnx` v11 |
| Input | `lstm_5_input` — shape `[N, 100, 500]`, dtype `float32` |
| Output | `dense_2` — shape `[N, 1]`, dtype `float32` |
| Parameter count | 30,033,007 (20 initializers; 30,033,002 float + 4 int64 + 1 int32) |
| Op-type breakdown | LSTM ×4, Transpose ×7, Squeeze ×5, Slice ×5, Cast ×8, Shape ×4, Concat ×4, Expand ×4, Unsqueeze ×4, MatMul ×1, Add ×1 (47 nodes total) |

## Keras SavedModel metadata

| Field | Value |
| --- | --- |
| Signature | `serving_default` |
| Input | `lstm_5_input` — shape `(None, 100, 500)`, dtype `float64` |
| Output | `dense_2` — shape `(None, 1)`, dtype `float64` |
| Trainable parameters | 30,017,001 |

### Variable shapes

| Variable | Shape | Dtype |
| --- | --- | --- |
| `lstm_5/lstm_cell_23/kernel` | `[500, 4000]` | float64 |
| `lstm_5/lstm_cell_23/recurrent_kernel` | `[1000, 4000]` | float64 |
| `lstm_5/lstm_cell_23/bias` | `[4000]` | float64 |
| `lstm_6/lstm_cell_24/kernel` | `[1000, 4000]` | float64 |
| `lstm_6/lstm_cell_24/recurrent_kernel` | `[1000, 4000]` | float64 |
| `lstm_6/lstm_cell_24/bias` | `[4000]` | float64 |
| `lstm_7/lstm_cell_25/kernel` | `[1000, 4000]` | float64 |
| `lstm_7/lstm_cell_25/recurrent_kernel` | `[1000, 4000]` | float64 |
| `lstm_7/lstm_cell_25/bias` | `[4000]` | float64 |
| `lstm_8/lstm_cell_26/kernel` | `[1000, 4000]` | float64 |
| `lstm_8/lstm_cell_26/recurrent_kernel` | `[1000, 4000]` | float64 |
| `lstm_8/lstm_cell_26/bias` | `[4000]` | float64 |
| `dense_2/kernel` | `[1000, 1]` | float64 |
| `dense_2/bias` | `[1]` | float64 |

## Architecture

```
Input (None, 100, 500)
  └─ LSTM(units=1000, return_sequences=True)     6,004,000 params
     └─ LSTM(units=1000, return_sequences=True)  8,004,000 params
        └─ LSTM(units=1000, return_sequences=True) 8,004,000 params
           └─ LSTM(units=1000)                   8,004,000 params
              └─ Dense(units=1)                      1,001 params
```

## Stateful classification

Recurrent: maintains hidden / cell state across the 100 timesteps of one sequence. Counts as the **stateful** arm of the lens-C test pair (vs. the stateless [`LSTM_Null_C`](LSTM_Null_C.md)).

## Notes

- The Keras SavedModel is `float64`. The ONNX export is `float32`.
- Four stacked LSTM layers, 1000 units each — the heaviest model in the suite at ≈30 M parameters.
- The on-disk ONNX size (≈120 MB) is dominated by the four LSTM kernel + recurrent_kernel + bias initializers.
