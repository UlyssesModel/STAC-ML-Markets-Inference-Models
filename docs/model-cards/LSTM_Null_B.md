# LSTM_Null_B

Stateless **null baseline** for lens B. Deterministic, untrained `Flatten → Dense(1)` with constant-init weights — matches the I/O contract of [`LSTM_B`](LSTM_B.md) for the RS-40 stateful-vs-stateless comparison. No recurrence, no learned signal.

## Files

| File | Format | Size |
| --- | --- | --- |
| `LSTM_Null_B.onnx` | ONNX | 100,657 B |
| `LSTM_Null_B.keras64/` | TensorFlow SavedModel (float64) | dir |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `tf2onnx 1.9.3` |
| IR version | 6 |
| Opset | `ai.onnx` v11 |
| Input | `flatten_1_input` — shape `[N, 100, 250]`, dtype `float32` |
| Output | `dense_1` — shape `[N, 1]`, dtype `float32` |
| Parameter count | 25,002 (2 initializers; 25,000 float kernel + 2 int64 reshape constants) |
| Op-type breakdown | Reshape ×1, MatMul ×1 (2 nodes total) |

## Keras SavedModel metadata

| Field | Value |
| --- | --- |
| Signature | `serving_default` |
| Input | `flatten_1_input` — shape `(None, 100, 250)`, dtype `float64` |
| Output | `dense_1` — shape `(None, 1)`, dtype `float64` |
| Trainable parameters | 25,000 |

### Variable shapes

| Variable | Shape | Dtype |
| --- | --- | --- |
| `dense_1/kernel` | `[25000, 1]` | float64 |

No bias term.

## Architecture

```
Input (None, 100, 250)
  └─ Flatten   →   (None, 25000)
     └─ Dense(units=1, use_bias=False)     25,000 params, all = 0.0021366863511502743
```

## Why "Null"

- **Untrained.** All 25,000 kernel weights share a single value (≈ `2.137e-3`); unique weight count is 1.
- **Deterministic and stateless.** Output is a fixed scalar projection of the flattened input. No recurrence, no hidden state, no nonlinearity.
- **Same I/O contract as `LSTM_B`.** Same input shape `(100, 250)` and output shape `(1,)` — exactly what the RS-40 stateless arm requires.

## Stateful classification

Stateless. The **stateless** arm of the lens-B test pair.

## Notes

- The Keras SavedModel is `float64`. The ONNX export is `float32`.
