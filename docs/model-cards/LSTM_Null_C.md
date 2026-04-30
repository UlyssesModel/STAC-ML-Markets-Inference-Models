# LSTM_Null_C

Stateless **null baseline** for lens C. Deterministic, untrained `Flatten → Dense(1)` with constant-init weights — matches the I/O contract of [`LSTM_C`](LSTM_C.md) for the RS-40 stateful-vs-stateless comparison. No recurrence, no learned signal.

## Files

| File | Format | Size |
| --- | --- | --- |
| `LSTM_Null_C.onnx` | ONNX | 200,657 B |
| `LSTM_Null_C.keras64/` | TensorFlow SavedModel (float64) | dir |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `tf2onnx 1.9.3` |
| IR version | 6 |
| Opset | `ai.onnx` v11 |
| Input | `flatten_2_input` — shape `[N, 100, 500]`, dtype `float32` |
| Output | `dense_2` — shape `[N, 1]`, dtype `float32` |
| Parameter count | 50,002 (2 initializers; 50,000 float kernel + 2 int64 reshape constants) |
| Op-type breakdown | Reshape ×1, MatMul ×1 (2 nodes total) |

## Keras SavedModel metadata

| Field | Value |
| --- | --- |
| Signature | `serving_default` |
| Input | `flatten_2_input` — shape `(None, 100, 500)`, dtype `float64` |
| Output | `dense_2` — shape `(None, 1)`, dtype `float64` |
| Trainable parameters | 50,000 |

### Variable shapes

| Variable | Shape | Dtype |
| --- | --- | --- |
| `dense_2/kernel` | `[50000, 1]` | float64 |

No bias term.

## Architecture

```
Input (None, 100, 500)
  └─ Flatten   →   (None, 50000)
     └─ Dense(units=1, use_bias=False)     50,000 params, all = 0.0015108654042705894
```

## Why "Null"

- **Untrained.** All 50,000 kernel weights share a single value (≈ `1.511e-3`); unique weight count is 1.
- **Deterministic and stateless.** Output is a fixed scalar projection of the flattened input. No recurrence, no hidden state, no nonlinearity.
- **Same I/O contract as `LSTM_C`.** Same input shape `(100, 500)` and output shape `(1,)` — exactly what the RS-40 stateless arm requires.

## Stateful classification

Stateless. The **stateless** arm of the lens-C test pair.

## Notes

- The Keras SavedModel is `float64`. The ONNX export is `float32`.
