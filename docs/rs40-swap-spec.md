# RS-40 Swap Spec: LSTM_A → Ulysses

## §1 Purpose

RS-40 specifies that Ulysses (Kirk) replaces only `LSTM_A`. This document
captures the precise input/output, dtype, and behavioural contract a
Ulysses-based replacement must satisfy to be drop-in compatible with whatever
harness drives the `LSTM_A` model. Other lenses (B, C) and other model classes
(`GBT_*`, all `_Null` variants) are out of scope.

### Naming note

`LSTM_A` in this document refers strictly to the model artifact stored in this
repo (`LSTM_A.onnx` / `LSTM_A.keras64/`), with input `(batch, 50, 100)` and
output `(batch, 1)`. Other documents in the broader project — notably the
bench harness's `LSTM_A_Swap_Plan.md` — discuss `LSTM_A` in an end-to-end
DeepLOB-pipeline sense (`(B, 1, 100, 40)` raw LOB → `(B, 2)` class). That
formulation includes an upstream CNN feature extractor and a downstream
classifier head, neither of which is in this artifact. The `(50, 100)` input
here is what the CNN's output looks like after time-pooling and channel
projection; the same model, viewed without its surrounding pipeline.

## §2 The contract LSTM_A defines

Verified by loading `LSTM_A.onnx` via `onnxruntime` and `LSTM_A.keras64/` via
`tf.saved_model.load`.

### Input tensor

| Path | Name | Declared shape | Dtype |
| --- | --- | --- | --- |
| ONNX | `lstm_input` | `[N, 50, 100]` (batch dim symbolic) | `float32` |
| Keras SavedModel | `lstm_input` | `(None, 50, 100)` | `float64` |

Layout is rank-3 `[batch, time, feature]`: 50 timesteps along axis 1, 100
features per timestep along axis 2. See §3 for what those 100 features
actually represent.

### Output tensor

| Path | Name | Declared shape | Dtype |
| --- | --- | --- | --- |
| ONNX | `dense` | `[N, 1]` | `float32` |
| Keras SavedModel | `dense` | `(None, 1)` | `float64` |

### Downstream readout

The model emits a real-valued scalar per batch element. Binary direction is
taken externally by sign-thresholding (RS-40 sign-of-imag). The model itself
does not apply a sigmoid, softmax, or any other squashing — the harness owns
the threshold step.

### Statefulness

Both Keras LSTM layers have `stateful=False`. Recurrence is within a single
forward pass over the 50 timesteps; no hidden/cell state is carried across
calls or across batch elements. Each call is independent given input.

### §2.1 Dtype note (not an open question, just a fact)

ONNX export is `float32`; Keras SavedModel is `float64`. A replacement
targeting the ONNX path must accept `float32`; a replacement targeting the
SavedModel path must accept `float64`. The harness picks one — the
replacement does not need to support both.

### §2.5 Architecture context (informative, not part of the contract)

```
Input (None, 50, 100)
  └─ LSTM(units=100, return_sequences=True)    80,400 params
     └─ LSTM(units=100)                        80,400 params
        └─ Dense(units=1)                         101 params
```

- Two stacked LSTM layers with 100 units each, then a `Dense(1)` readout.
- ONNX opset: `ai.onnx` v11. Producer: `tf2onnx 1.9.3`. IR version 6.
- Trainable parameters (Keras): 160,901.
- ONNX initializer count: 161,707 (the 806-parameter delta vs. Keras is
  tf2onnx-injected dimension/shape helper constants, not learned weights).
- Op-type breakdown in ONNX: `LSTM ×2`, `Transpose ×3`, `Squeeze ×3`,
  `Slice ×3`, `Cast ×4`, `Shape ×2`, `Concat ×2`, `Expand ×2`,
  `Unsqueeze ×2`, `MatMul ×1`, `Add ×1` (25 nodes total).

This subsection is informative only. A replacement is not obligated to
preserve the 2 × LSTM(100) + Dense(1) shape, the parameter count, the opset,
or the producer; it must only honour §2's I/O contract and §4's behavioural
contract.

## §3 What "100 features" means — open question

The model artifact tells us nothing about which 100 features the input
represents. The ONNX graph and the Keras SavedModel both expose the input as
an opaque rank-3 tensor `[batch, 50, 100]` with no per-channel naming or
metadata. Feature extraction lives upstream — in the bench harness or the
training pipeline — neither of which is in this repo.

**Open question (harness-side resolution):** what are the 100 features, in
what order, and on what scale? Without this information, a replacement model
cannot be retrained or fine-tuned, and a from-scratch Ulysses adapter cannot
be designed against meaningful priors.

## §4 Replacement contract

Any replacement model must:

1. Accept input `(batch, 50, 100)` `float32` (or `float64` if targeting the
   Keras path) with the same feature semantics as §3.
2. Return output `(batch, 1)` of matching dtype.
3. Produce a real-valued scalar that, after sign thresholding, gives the same
   direction labels `LSTM_A` would on the same data.
4. Be deterministic given input and frozen weights.
5. Honor the lens-2 p99 latency budget (10 µs). Lens-1 budget is presumably
   tighter but not formally specified — see open questions in §6.

## §5 Ulysses pipeline at the contract level

Sketch only — implementation is out of scope (see §7).

```
features (batch, 50, 100)  →  Hankel adapter  →  Kirk core (BF16 matmul)  →  scalar readout (batch, 1)
```

Stages:

### Stage 1: Hankel adapter

- **Inputs accepted:** `(batch, 50, 100)`, `float32` (ONNX path) or `float64`
  (Keras path), per §2.
- **Outputs produced:** Hankel-structured tensor in whatever shape and dtype
  Kirk consumes (concretely BF16, but exact rank/shape is a Kirk-side detail).
- **Stage-specific open questions:**
  - Hankel embedding dimension and stride: which row/column count does Kirk
    expect, and how is the 50-step window mapped onto it?
  - Where does the dtype downcast (float32/64 → BF16) happen — inside the
    adapter, or at the Kirk boundary? Numerical-equivalence implications for
    the §4(3) sign-agreement requirement.
  - Per-feature normalization or scaling: required, or assumed already done
    upstream of the `(50, 100)` input?

### Stage 2: Kirk core

Treat as a black box. No speculation on Kirk internals.

- **Inputs accepted:** whatever the Hankel adapter produces (BF16, shape TBD
  by Kirk).
- **Outputs produced:** a representation the readout consumes (shape and
  dtype TBD by Kirk).
- **Stage-specific open questions:**
  - Kirk's declared input/output contract — must be pinned before the
    Hankel-adapter and readout shapes can be finalized.
  - Determinism guarantees under BF16 matmul (relevant to §4(4)).

### Stage 3: Scalar readout

- **Inputs accepted:** Kirk's output (shape/dtype per Stage 2).
- **Outputs produced:** `(batch, 1)`, `float32` or `float64` to match the
  external dtype chosen in §2.1.
- **Stage-specific open questions:**
  - Does the readout need a learned linear projection (analogue of `LSTM_A`'s
    final `Dense(1)`) or a fixed reduction?
  - Upcast point from BF16 back to the contract dtype.

## §6 Cross-cutting open questions

Stage-specific open questions are inline in §5; this section lists
cross-cutting items only.

1. **What are the 100 input features?** Resolution lives on the
   harness/training side (see §3). Blocks any data-driven design choice in
   §5.
2. **Is the readout truly `sign(output)`** or does the harness apply a
   non-zero threshold or calibration (e.g. an offset learned on a holdout, or
   a deadband around zero)? Affects whether §4(3) is satisfied by sign
   agreement alone.
3. **Does the stateless test (`LSTM_Null_A` control) require its own
   Ulysses-shaped baseline,** or does the existing constant-init baseline
   suffice? If a Ulysses-shaped null is needed, it has to be specified
   separately and is outside this document's scope as written.
4. **Lens-1 latency budget.** Lens-2 is specified at p99 10 µs. Lens-1 is
   not formally documented in RS-40 and needs to be pinned before any
   replacement design can be validated against it.

## §7 Out of scope

- Implementation of the Ulysses replacement (any stage).
- Kirk internals.
- Training procedure for the replacement.
- The GBT family (`GBT_A`, `GBT_B`, `GBT_C`).
- The `_Null` family (`LSTM_Null_*`, `GBT_Null_*`).
- Lenses B and C (`LSTM_B`, `LSTM_C`, and their `_Null` counterparts).
