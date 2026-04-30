# STAC-ML Markets — Inference Models

Reference inference-model bundle for the STAC-ML Markets benchmark. STAC-ML Markets is a vendor-neutral benchmark from the Securities Technology Analysis Center (STAC) for measuring inference of trained ML models on financial-market workloads. RS-40 is the reference solution / model release that this repository ships: a fixed set of pre-trained models, their untrained null baselines, and the input/output contracts the test harness will drive.

This repo contains only the **inference artifacts** — `.onnx` for cross-runtime use and the original framework-native dumps (`.keras64` TensorFlow SavedModels, `.lgb` LightGBM dumps) for reference. Training code, datasets, and harness code live elsewhere.

## Contents

```
.
├── GBT_{A,B,C}.{onnx,lgb}            # trained gradient-boosted-tree regressors
├── GBT_Null_{A,B,C}.{onnx,lgb}       # 1-tree null baselines (matched I/O)
├── LSTM_{A,B,C}.{onnx,keras64/}      # trained stacked-LSTM regressors
├── LSTM_Null_{A,B,C}.{onnx,keras64/} # untrained Flatten→Dense(1) null baselines
├── checksums                         # md5sums of every artifact
├── docs/model-cards/                 # one card per model with full I/O metadata
└── scripts/verify_io.py              # synthetic-input shape verifier
```

Per-model details (input/output shapes and dtypes, opsets, parameter counts, op-type breakdowns, tree counts) are in [`docs/model-cards/`](docs/model-cards/).

## The three lenses (A, B, C)

The benchmark is parameterised by a **lens** — a fixed input shape and model size. The three lenses scale jointly across sequence length, feature width, and parameter count, giving small / medium / large workload points:

| Lens | LSTM input `(T, F)` | GBT feature width | LSTM stacked layers × units | LSTM trainable params | GBT trees |
| --- | --- | --- | --- | --- | --- |
| **A** | `50 × 100` | 60 | 2 × 100 | 160,901 | 250 |
| **B** | `100 × 250` | 125 | 3 × 200 | 1,002,601 | 1,000 |
| **C** | `100 × 500` | 1,000 | 4 × 1,000 | 30,017,001 | 4,000 |

Lens A → B → C scales sequence length up to 100, doubles or quadruples feature width, deepens the LSTM stack, and roughly 10× and 30× the parameter count at each step. The GBT lenses scale tabular feature width and tree count along the same A/B/C labelling. (LSTM and GBT share the lens labels but consume different feature representations of the same underlying data — flattening one to the other is not the intent.)

All models output a single scalar regression target (`shape [N, 1]`).

## Stateful vs. stateless arms

Every lens has a **paired baseline**. The pair is what RS-40 actually tests against: a real model and a null model with identical input/output shapes, run side-by-side.

### LSTM — stateful arm (`LSTM_A`, `LSTM_B`, `LSTM_C`)

Trained, stacked recurrent LSTMs (2/3/4 layers for A/B/C). Each maintains hidden and cell state across the timesteps of one input sequence — that is the "stateful" property the benchmark cares about. Stored as both `.onnx` (float32, opset 11, produced by `tf2onnx 1.9.3`) and `.keras64/` (TensorFlow SavedModel, **float64** — hence the `64` suffix).

### LSTM — stateless arm (`LSTM_Null_A`, `LSTM_Null_B`, `LSTM_Null_C`)

**Not trained.** Each Null variant is a deterministic `Flatten → Dense(1, use_bias=False)` graph with **constant-init weights** — the kernel is filled with a single floating-point value (one unique weight value across all 5,000 / 25,000 / 50,000 entries). They are *not* trained models; fitting against any non-trivial target would drive those weights apart. Their job is to:

- match the input/output contract of the corresponding `LSTM_*` model exactly, and
- exercise the test harness's data path with a model that has no recurrence and no learned signal.

Because there is no LSTM and no hidden state, these are stateless. The op-type breakdown is literally `Reshape ×1, MatMul ×1`.

### GBT — paired the same way

`GBT_{A,B,C}` are the trained boosted-tree regressors (250 / 1,000 / 4,000 trees). `GBT_Null_{A,B,C}` are 1-tree trivial baselines on the same input shape. Same idea: the Null files are I/O-shape-matched stand-ins, not learned models.

### "Stateful" terminology

For LSTM the term is literal — the recurrent cells carry state across timesteps. For GBT, "stateful" in the RS-40 grid is shorthand for **the trained member of the pair**: GBTs do not carry runtime state between calls. The benchmark uses one label across both model classes for grid uniformity.

## The RS-40 lens × stateful/stateless test grid

RS-40 specifies a 3 × 2 grid per model class: each lens is tested under both the trained ("stateful") and untrained ("stateless" / null) configuration, giving 6 model variants per class and 12 in total across LSTM and GBT.

|        | LSTM (recurrent)        | LSTM (null)             | GBT (trained)        | GBT (null)              |
| ------ | ----------------------- | ----------------------- | -------------------- | ----------------------- |
| **A**  | `LSTM_A`                | `LSTM_Null_A`           | `GBT_A`              | `GBT_Null_A`            |
| **B**  | `LSTM_B`                | `LSTM_Null_B`           | `GBT_B`              | `GBT_Null_B`            |
| **C**  | `LSTM_C`                | `LSTM_Null_C`           | `GBT_C`              | `GBT_Null_C`            |

For any one cell the test harness drives the model with synthetic or recorded inputs of the lens's declared shape and measures inference behaviour — latency, throughput, numerical agreement, etc. The Null column controls for harness-side overhead independent of model capacity: any difference in measured behaviour between a real model and its paired Null is attributable to the model itself, not to the test rig.

## File-format notes

- **ONNX.** Every model is exported to `.onnx`. Trees use the `ai.onnx.ml` v1 `TreeEnsembleRegressor` op (with `ai.onnx` v8 alongside, IR v4, producer `OnnxMLTools 1.14.0`). LSTMs are opset `ai.onnx` v11 (IR v6, producer `tf2onnx 1.9.3`). All ONNX inputs/outputs are `float32`.
- **`.keras64/`.** TensorFlow SavedModel directories with `float64` weights (signature: `serving_default`). Keras 3 cannot load these via `keras.models.load_model`; use `tf.saved_model.load(...)` or `keras.layers.TFSMLayer(..., call_endpoint="serving_default")`.
- **`.lgb`.** LightGBM native text dumps; cross-checks the ONNX export.
- **`checksums`.** md5 of every leaf artifact (including the SavedModel internals).

## Verification

Run `scripts/verify_io.py` to load every `.onnx` via onnxruntime and every `.keras64/` via TensorFlow, push a random tensor of the declared input shape through, and assert the output shape matches each model card.

```
python scripts/verify_io.py
```

Requires `onnx`, `onnxruntime`, `tensorflow`, and `numpy` — all installed in the bundled `.venv/`.

## Reproducing and extending

This repo ships **artifacts only** (model weights + ONNX exports). To run the RS-40 benchmark end to end, you also need the **bench harness** — driver code that produces `(50, 100)` features from raw market data, drives the model, and applies the sign-of-imag readout. The harness is not included here.

To **extend** the benchmark — train new models, swap in a Ulysses replacement, etc. — you also need the **training code** and the **labeled market data** the original LSTM/GBT models were trained on. Neither is in this repo.

A local latency baseline (`bench_results.json`, produced by `scripts/benchmark_io.py`) was measured on a `stac-claude-dev` `e2-standard-4` GCP VM (4 vCPU, no GPU, no AMX). These numbers are **not comparable** to RS-40 production hardware budgets; treat them as a relative baseline among the models in this repo, not as an absolute claim.
