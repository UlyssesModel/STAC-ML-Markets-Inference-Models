# LSTM_Null_A

Stateless **null baseline** for lens A. Deterministic, untrained `Flatten → Dense(1)` with constant-init weights — produced solely to match the input/output contract of [`LSTM_A`](LSTM_A.md) for the RS-40 stateful-vs-stateless comparison. There is no recurrence and no learned signal.

## Files

| File | Format | Size |
| --- | --- | --- |
| `LSTM_Null_A.onnx` | ONNX | 20,623 B |
| `LSTM_Null_A.keras64/` | TensorFlow SavedModel (float64) | dir |

## ONNX metadata

| Field | Value |
| --- | --- |
| Producer | `tf2onnx 1.9.3` |
| IR version | 6 |
| Opset | `ai.onnx` v11 |
| Input | `flatten_input` — shape `[N, 50, 100]`, dtype `float32` |
| Output | `dense` — shape `[N, 1]`, dtype `float32` |
| Parameter count | 5,002 (2 initializers; 5,000 float kernel + 2 int64 reshape constants) |
| Op-type breakdown | Reshape ×1, MatMul ×1 (2 nodes total) |

The two ops are precisely Flatten (encoded as Reshape) and Dense (encoded as MatMul, no bias).

## Keras SavedModel metadata

| Field | Value |
| --- | --- |
| Signature | `serving_default` |
| Input | `flatten_input` — shape `(None, 50, 100)`, dtype `float64` |
| Output | `dense` — shape `(None, 1)`, dtype `float64` |
| Trainable parameters | 5,000 |

### Variable shapes

| Variable | Shape | Dtype |
| --- | --- | --- |
| `dense/kernel` | `[5000, 1]` | float64 |

No bias term.

## Architecture

```
Input (None, 50, 100)
  └─ Flatten   →   (None, 5000)
     └─ Dense(units=1, use_bias=False)      5,000 params, all = 0.00477777561172843
```

## Why "Null"

- **Untrained.** All 5,000 kernel weights share a single value (≈ `4.778e-3`); the unique weight count is 1. This is a constant initialization, not the result of training — fitting against any non-trivial target would drive the weights apart.
- **Deterministic and stateless.** Output is a fixed scalar projection of the flattened input. There is no recurrence, no hidden state, no nonlinearity.
- **Same I/O contract as `LSTM_A`.** Same input shape `(50, 100)` and output shape `(1,)`, which is exactly what the RS-40 stateless arm requires.

## Stateful classification

Stateless. The **stateless** arm of the lens-A test pair.

## Notes

- The Keras SavedModel is `float64`. The ONNX export is `float32`.
- Useful as a runtime sanity check: if a harness can run `LSTM_Null_A` end-to-end it has at minimum a working data path; non-trivial behavior should be exercised against [`LSTM_A`](LSTM_A.md).
