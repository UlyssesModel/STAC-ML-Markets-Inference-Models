# Harness Relationship: STAC-ML-Markets-Bench-Harness ↔ this repo

## Purpose

This repo ships the RS-40 inference artifacts (`LSTM_{A,B,C}.{onnx,keras64}`,
`GBT_*`, `LSTM_Null_*`, `GBT_Null_*`). The sibling repo
`STAC-ML-Markets-Bench-Harness` ships a smoke-test harness built around
DeepLOB and a Kirk stub. Both repos use the label `LSTM_A`, but they mean
different things by it. This document pins down, file by file, how the
harness as written today relates to the `LSTM_A` artifact in this repo —
what it can drive, what it cannot, and what is reusable for the RS-40 swap
work that is tracked in `docs/rs40-swap-spec.md`.

All citations below refer to files in the sibling repo,
`~/work/STAC-ML-Markets-Bench-Harness/`, unless stated otherwise.

---

## §1 What the harness is built for

The harness is an end-to-end DeepLOB pipeline against the LOBSTER public LOB
schema. The full path is: LOBSTER `_orderbook_10.csv` → windowed tensor →
DeepLOB CNN+inception+LSTM → 2-class up/down logits → `argmax`.

### 1.1 Input data layout

`lobster_loader.py:32-34` declares the LOBSTER schema constants:

```python
LEVELS = 10
COLS_PER_LEVEL = 4  # ask_px, ask_vol, bid_px, bid_vol
TOTAL_COLS = LEVELS * COLS_PER_LEVEL  # 40
```

`load_orderbook` (`lobster_loader.py:37-51`) reads the CSV into a
`(T, 40)` `float32` array, asserts the column count, and preserves the
LOBSTER native interleaved order
`[ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, …]`. The
column ordering is load-bearing — the comment at `lobster_loader.py:42-44`
explicitly notes the 1×2 stride-2 conv in DeepLOB's first layer relies on
adjacent (price, volume) columns.

Windowing is in `make_windows` (`lobster_loader.py:65-76`) and
`load_window_dataset` (`lobster_loader.py:106-150`). The output is:

- `X`: `(N, 1, lookback, 40)` `float32` — channel-first, channel dim = 1
- `y`: `(N,)` `int64`, binary `{0, 1}` (after dropping unchanged-mid windows;
  `lobster_loader.py:149`)

Default `lookback=100`, default `horizon=10` (`lobster_loader.py:106-111`).
Labels are mid-price direction at `t+horizon` vs `t` (`make_labels`,
`lobster_loader.py:79-103`). Z-score normalization across the whole file is
the default (`normalize_book`, `lobster_loader.py:54-62`); it is keyed off
`normalize=True` in `load_window_dataset`.

### 1.2 Model architecture

`deeplob_model.py` is a faithful Zhang/Zohren/Roberts (2018) DeepLOB:

- Conv block 1 (`deeplob_model.py:32-39`): collapses (price, volume) pairs
  per level via `Conv2d(1, 16, kernel=(1,2), stride=(1,2))` plus two
  temporal `(4,1)` convs.
- Conv block 2 (`deeplob_model.py:42-49`): collapses (ask, bid) per level.
- Conv block 3 (`deeplob_model.py:52-59`): collapses across all 10 levels
  (`Conv2d(16, 16, kernel=(1,10))`).
- Inception module (`deeplob_model.py:62-78`): three parallel branches with
  temporal kernels 1/3/5 plus a max-pool branch, concatenated → 96
  channels.
- LSTM (`deeplob_model.py:82`): `nn.LSTM(input_size=96, hidden_size=64,
  batch_first=True)`.
- FC head (`deeplob_model.py:85`): `nn.Linear(64, num_classes)` with
  `num_classes=2` (`deeplob_model.py:28`).

`forward` (`deeplob_model.py:87-103`) takes the last LSTM timestep and
returns logits `(batch, num_classes)`.

### 1.3 End-to-end shape contract

Confirmed from code, not memory:

| Stage | Shape | Dtype | Source |
| --- | --- | --- | --- |
| File row | `(T, 40)` | `float32` | `lobster_loader.py:51` |
| Windowed batch | `(B, 1, 100, 40)` | `float32` | `lobster_loader.py:73`, `lobster_loader.py:139` |
| DeepLOB output | `(B, 2)` | `float32` | `deeplob_model.py:85`, `deeplob_model.py:103` |
| F1-eval prediction | `(B,)` `int64` | — | `run_smoke.py:73` (`logits.argmax(dim=1)`) |

So the harness's end-to-end contract — `(B, 1, 100, 40)` raw LOB → `(B, 2)`
class — matches the harness `README.md` ("DeepLOB and Kirk-stub on CPU
against the same windowed input"). Same contract holds in the trainer
(`train_deeplob.py:186` instantiates `DeepLOB(num_classes=2)`; loss is
`nn.CrossEntropyLoss` at `train_deeplob.py:196,199`; `argmax` at
`train_deeplob.py:129`).

### 1.4 Kirk stub contract

`kirk_stub.py:13-15`:

> Input:  numpy array (batch, 1, lookback=100, 40), float32
> Output: numpy array (batch,), int64 in {0, 1} for binary up/down

`KirkPredictor` (`kirk_stub.py:22-29`) is the interface; `KirkStub`
(`kirk_stub.py:32-59`) is a deterministic baseline that fits a linear slope
to the mid-price across the 100-step window and emits `(slope > 0)`. The
stub does not call into anything outside its own file. The header comment
at `kirk_stub.py:1-15` is explicit that real Kirk is "a Kavara-internal
Ulysses-architecture model" and that swapping it in means replacing
`KirkStub.predict` while keeping the I/O contract identical. There is no
network call, no model file, no library import that touches Kirk; from this
file alone the only thing we can say about Kirk is that it must conform to
the `(batch, 1, 100, 40) float32` → `(batch,) int64 {0,1}` contract.

---

## §2 Shape mismatch with this repo's `LSTM_A`

The harness's "LSTM_A" (the end-to-end DeepLOB pipeline) and this repo's
`LSTM_A.onnx` are not the same object. Naming aside, the I/O contracts
differ on every axis.

| Axis | Harness end-to-end ("LSTM_A" sense) | This repo's `LSTM_A.onnx` |
| --- | --- | --- |
| Input rank | 4 | 3 |
| Input shape | `(B, 1, 100, 40)` | `(B, 50, 100)` |
| Input semantics | Raw LOB rows: 1 channel × 100 timesteps × 40 LOBSTER columns | Pre-extracted features: 50 timesteps × 100 features per step |
| Input dtype | `float32` | `float32` (ONNX) / `float64` (Keras path) |
| Output rank | 2 | 2 |
| Output shape | `(B, 2)` | `(B, 1)` |
| Output semantics | Logits over `{down, up}`; harness applies `argmax` | Real-valued scalar; harness applies sign-of-imag externally |
| Output dtype | `float32` | `float32` (ONNX) / `float64` (Keras path) |
| Sequence length | 100 | 50 |
| Feature width | 40 | 100 |
| Channel dim | 1 (explicit) | none (rank-3) |
| Stateful flag | LSTM cell carries state within one forward pass; recurrence is internal | Same (Keras layers `stateful=False`; recurrence within the 50 steps only) |

Concrete consequences:

1. The harness cannot push its `(B, 1, 100, 40)` tensor into
   `LSTM_A.onnx` — onnxruntime will reject the rank.
2. Even after a `squeeze(1).transpose(...)` to rank-3, the time dimension
   is wrong (100 vs 50) and the feature dimension is wrong (40 vs 100).
3. The output post-processing is incompatible: DeepLOB logits go through
   `argmax`; LSTM_A's scalar goes through sign-thresholding (see
   `rs40-swap-spec.md` §2 "Downstream readout").
4. The 100-feature axis of LSTM_A is opaque — `rs40-swap-spec.md` §3 calls
   this out as an open question. The 40-column axis of the harness input
   is fully specified by the LOBSTER schema. They are not the same 100 vs
   40 numbers expressed differently; they are different layers of the
   pipeline. The most likely relationship is the one stated in
   `rs40-swap-spec.md` §1: LSTM_A's `(50, 100)` is what the CNN's output
   looks like after time-pooling and channel projection — i.e., the
   harness's DeepLOB conv stack would need to produce LSTM_A's input as
   an intermediate, which it does not currently do (its LSTM operates on
   `(T', 96)` features at `deeplob_model.py:82,100`, not `(50, 100)`).

---

## §3 What's reusable

The following pieces of the harness lift cleanly into RS-40 swap work
without modification. None of them are coupled to DeepLOB-specific shapes.

### 3.1 `latency_meter.py` — full file

- `LatencyStats` (`latency_meter.py:21-40`): a list-of-ns container with
  `record(ns)` and `percentiles()` returning `min/p50/p90/p99/max/mean` in
  µs. Generic; no model assumption.
- `measure_cpu` (`latency_meter.py:43-49`): `time.perf_counter_ns()`-based
  context manager. Drop-in for timing onnxruntime CPU inference of
  `LSTM_A.onnx` on this repo's `e2-standard-4` baseline rig.
- `measure_gpu_event` (`latency_meter.py:52-70`): CUDA-event timer that
  falls back to `measure_cpu` when CUDA is unavailable. Useful if/when the
  Ulysses replacement runs on GPU and a Kirk-shaped baseline needs timing.
- `warmup_iters` (`latency_meter.py:73-76`): generic warm-up loop. Worth
  using before the percentile loop in `scripts/benchmark_io.py` of this
  repo (which currently does its own ad-hoc warmup).

The header at `latency_meter.py:1-9` is explicit that this is smoke-grade,
not the production STAC rig — anyone using it for an RS-40 audit must
re-implement CPU pinning and exclusive-GPU disciplines on top.

### 3.2 `make_synthetic.py` — synthetic LOBSTER generator

`make_synthetic.py:45-120` writes a syntactically correct LOBSTER
`_orderbook_10.csv` with two embedded signals: AR(1) momentum on log mid
(`make_synthetic.py:69-78`) and order-flow imbalance correlated with future
direction (`make_synthetic.py:83-115`). It does not need to be modified to
be useful in swap work: it is a clean source of test data with a
non-trivial signal (the docstring at `make_synthetic.py:30-31` claims
DeepLOB reaches F1 ≈ 0.55-0.65 on it). For the RS-40 swap, it would
produce LOBSTER files that an upstream feature extractor (which does not
yet exist, see §4) could consume to manufacture `(50, 100)` LSTM_A inputs.

### 3.3 `lobster_loader.py` — schema parser and labeller

If RS-40 swap work needs to consume LOBSTER data at all, the loader
(`lobster_loader.py:37-51`), z-score normalizer
(`lobster_loader.py:54-62`), and label maker (`lobster_loader.py:79-103`)
are reusable as-is. The windowing function (`make_windows` and
`load_window_dataset`) is DeepLOB-shaped (`(N, 1, 100, 40)`); it would need
to be replaced — not reused — for an LSTM_A-shaped pipeline.

### 3.4 `run_smoke.py` orchestration scaffold

The structure of `run_smoke.py` — load data → run two predictors → record
F1 + latency percentiles → emit JSON — is generic. The relevant patterns:

- `_summary` (`run_smoke.py:35-37`): `{model, f1, latency: percentiles}`
  result dict. Same shape as this repo's `bench_results.json` would
  benefit from converging to.
- `_f1_binary` (`run_smoke.py:39-50`): self-contained binary F1, no
  sklearn dep. Useful if a Ulysses replacement is benchmarked against
  `LSTM_A` on a binary-direction proxy task.
- The split between F1 evaluation (batched, untimed; `run_smoke.py:67-74`)
  and latency measurement (single-sample, timed;
  `run_smoke.py:76-87`) is the right pattern and worth copying when
  benchmarking LSTM_A vs a Ulysses candidate. Lens-2 budget (10 µs) is
  per-call, so single-sample timing is the contract.

### 3.5 The KirkPredictor interface contract

`kirk_stub.py:22-29` — three-line abstract base class with `predict(x)`.
For an RS-40 Ulysses prototype, defining
`UlyssesPredictor.predict(x)` with the same shape contract (after
substituting `(B, 50, 100) → (B, 1)` for the `(B, 1, 100, 40) → (B,)`
contract here) gives a clean swap point in any harness. The pattern is
reusable; the specific shapes are not.

---

## §4 What is missing for an end-to-end RS-40 run with this `LSTM_A`

Honest gap inventory. None of the following is in either repo:

1. **Upstream feature extractor.** No code anywhere we can see produces a
   `(50, 100) float32` tensor from market data. The harness goes raw LOB →
   CNN-internal `(T', 96)` features → 64-d LSTM hidden → 2-d logits; it
   never materialises an exposed `(50, 100)` representation. The closest
   intermediate in the harness is the conv-stack output at
   `deeplob_model.py:91-99`, which has shape `(B, 96, T', 1)` after the
   inception concat and is then reshaped to `(B, T', 96)` at
   `deeplob_model.py:100` — different time dimension (`T'` ≠ 50) and
   different feature width (96 ≠ 100). Whatever produced LSTM_A's training
   inputs is not in either repo.
2. **Per-feature normalization for the `(50, 100)` axis.** Z-scoring at
   `lobster_loader.py:54-62` is column-wise on the 40 LOBSTER columns. It
   says nothing about what scaling LSTM_A's 100 features expect. This is
   the open question pinned in `rs40-swap-spec.md` §3.
3. **Training data for LSTM_A.** Neither LOBSTER files nor any other
   labelled `(50, 100)` dataset exists in either repo. This repo's README
   is explicit ("Training code, datasets, and harness code live elsewhere"
   — `README.md:5`).
4. **A harness driver for the `(B, 50, 100) → (B, 1)` shape.** The
   harness's `run_smoke.py:53-89` is hardwired to DeepLOB's PyTorch
   `nn.Module` interface; it does not load or drive any ONNX model. This
   repo has `scripts/benchmark_io.py` for ONNX latency on
   `LSTM_A.onnx`, but that is a microbenchmark with random inputs, not an
   end-to-end pipeline.
5. **The sign-of-imag readout step.** `rs40-swap-spec.md` §2 ("Downstream
   readout") describes external sign-thresholding of the scalar output.
   No code in either repo applies that readout to a real or synthetic
   data stream.
6. **Ground-truth labels for the `(50, 100)` axis.** The harness produces
   binary direction labels at `lobster_loader.py:79-103` against the
   40-col LOBSTER schema. There is no labelling code for whatever target
   LSTM_A was trained against (the model card at
   `docs/model-cards/LSTM_A.md` describes it as a regressor without
   pinning the target).
7. **A real Kirk implementation.** `kirk_stub.py:32-59` is explicitly a
   slope-of-mid placeholder. The real Ulysses model is not in either
   repo.

In short: the harness covers a different vertical slice of the problem.
Driving `LSTM_A.onnx` requires building the upstream feature pipeline that
the harness does not have and was not designed to expose.

---

## §5 Updates to `rs40-swap-spec.md` open questions

`rs40-swap-spec.md` §3 has one open question and §6 has four. Reading the
harness end to end:

### §3 — "What are the 100 input features?"

**Untouched.** The harness operates on a 40-column LOBSTER row layout
(`lobster_loader.py:32-34`), not on a 100-feature representation. The
DeepLOB conv stack produces internal feature maps of shape
`(B, 96, T', 1)` (`deeplob_model.py:91-99`) — not 100-wide, not 50-deep.
There is no place in the harness where a `(50, 100)` tensor is named,
materialised, or normalized. The harness gives no information about what
LSTM_A's 100 features are.

### §6.1 — "What are the 100 input features?" (cross-cutting echo of §3)

**Untouched.** Same reason as above.

### §6.2 — "Is the readout truly `sign(output)` or does the harness apply a non-zero threshold or calibration?"

**Untouched** for LSTM_A. The harness's only readout step is
`logits.argmax(dim=1)` at `run_smoke.py:73` and `train_deeplob.py:129`.
That is a 2-class arg-max over DeepLOB's `(B, 2)` output, not a
sign-threshold over a `(B, 1)` scalar. It tells us nothing about how an
external sign-of-imag step on LSTM_A would be applied — there is no such
step anywhere in the harness, with or without a deadband.

### §6.3 — "Does the stateless test (`LSTM_Null_A`) require its own Ulysses-shaped baseline?"

**Untouched.** No `_Null` model is referenced anywhere in the harness.
`KirkStub` (`kirk_stub.py:32-59`) is a CPU-side placeholder for *real*
Kirk, not a stateless control for LSTM_A. The harness's "two-model"
structure (DeepLOB vs Kirk) is "real model vs alternative real model",
not "real model vs null baseline".

### §6.4 — "Lens-1 latency budget"

**Partially informed.** The harness records single-sample forward latency
in nanoseconds and reports `p50/p90/p99` µs (`latency_meter.py:21-40`).
That confirms the harness is built around per-call latency at the same
percentile resolution RS-40 cares about — but the harness uses a
smoke-grade timer (`latency_meter.py:1-9` says so explicitly). It does not
state, document, or imply a specific lens-1 budget. The DeepLOB latency
expectation in the harness `README.md` is qualitative ("microseconds
range"). Lens-1's formal budget is still unspecified.

The `n_time` default of 200 samples (`run_smoke.py:53,127`) is well below
what is needed for stable p99 estimation; this is corroborating evidence
that the harness's timing rig is not the production rig.

---

## §6 Path forward — three options

The design tension is whether the swap point is at the raw-data boundary,
the feature boundary, or somewhere hybrid.

### Option A — Build a new harness driver for `(B, 50, 100) → (B, 1)`

Keep the existing harness as-is for the DeepLOB pipeline. Build a separate,
small harness in this repo (or a third repo) that drives `LSTM_A.onnx`
directly with synthetic or pre-computed `(50, 100)` features.

Work required:

- Define a synthetic-feature generator that emits `(N, 50, 100) float32`
  with some plausible structure (analogous to `make_synthetic.py` but at
  the feature level, not the LOBSTER level). Could be as simple as random
  Hankel-ish patterns or stationary AR processes per channel.
- Build a driver: load `LSTM_A.onnx` via onnxruntime, push batches, time
  per-call latency, apply a sign-threshold readout, optionally compare
  against `LSTM_Null_A.onnx`.
- Reuse the latency meter (`latency_meter.py` from the sibling repo) and
  the result-summary pattern (`run_smoke.py:35-37`).
- Add a Ulysses-replacement prototype that conforms to the same `(B, 50,
  100) → (B, 1)` ONNX contract; benchmark side-by-side.

Tradeoffs:

- Pro: minimal code, no dependency on resolving the §3 open question,
  fastest path to a swap-test rig for the artifact in this repo.
- Pro: cleanly mirrors the §1 framing of `rs40-swap-spec.md` —
  `LSTM_A` here is the model viewed without its surrounding pipeline.
- Con: synthetic features are decorrelated from any market signal —
  numerical-equivalence tests against the trained `LSTM_A` will reflect
  whatever the network does on noise, not whatever it does on real
  inputs. F1-style task fidelity cannot be measured.
- Con: doubles the harness count. Two pipelines to maintain.

### Option B — Refactor the existing harness to expose a feature-level swap point

Modify the sibling harness so that DeepLOB is split into two stages: a
frozen feature-extractor (the conv blocks + inception) producing some
exposed `(B, T_feat, F_feat)` tensor, and a swappable head (the LSTM +
Dense) that consumes it. Make the head pluggable so LSTM_A or a Ulysses
replacement can drop in at the head.

Work required:

- Refactor `deeplob_model.py:87-103` into two modules: `DeepLOBFeatures`
  (returns `(B, T', 96)` from `deeplob_model.py:99-100`) and
  `DeepLOBHead` (the LSTM + FC at `deeplob_model.py:101-103`).
- Add a projection adapter from `(B, T', 96)` to `(B, 50, 100)` — this is
  fundamentally a new design choice (time-pool, learned linear projection,
  or both).
- Train the projection + LSTM_A end-to-end on LOBSTER, validate F1 stays
  in the harness's expected range.
- Wire `LSTM_A.onnx` into the harness via an onnxruntime adapter behind
  the `KirkPredictor`-style interface.

Tradeoffs:

- Pro: gives an end-to-end testable pipeline that exercises real data
  through `LSTM_A`. F1 becomes meaningful.
- Pro: the projection adapter is a concrete answer to §3 (what 100
  features mean) — but only for *this* re-trained version, not the
  shipped `LSTM_A.onnx` weights.
- Con: the trained-`LSTM_A.onnx` weights in this repo were not trained
  through this projection. Using them with a freshly-designed projection
  is a different model from a numerical-equivalence standpoint.
- Con: significant new code in the sibling harness — touches
  `deeplob_model.py`, `train_deeplob.py`, and the model interface.

### Option C — Hybrid: keep DeepLOB end-to-end, extract features from a frozen DeepLOB CNN

Train DeepLOB on LOBSTER as the harness already supports. Freeze the conv
blocks. Tap into the inception output (`deeplob_model.py:97-99`, the
`(B, 96, T', 1)` tensor before the `squeeze + transpose`). Pool / project
to `(B, 50, 100)`. Use that as input to LSTM_A or a Ulysses replacement,
keeping the conv stack as a fixed feature extractor for both.

Work required:

- Train DeepLOB to a non-trivial F1 on LOBSTER (the trainer at
  `train_deeplob.py:134-235` already does this).
- Freeze the conv stack; fit the projection layer (`(T', 96) → (50, 100)`)
  with whatever objective LSTM_A expects (regression scalar, sign target).
- Either fine-tune or evaluate `LSTM_A.onnx`'s pre-trained weights against
  these features. (Same caveat as Option B: pre-trained weights weren't
  trained on these features.)
- Plug Ulysses behind the same projection, evaluate side by side.

Tradeoffs:

- Pro: real data signal flows all the way through. Latency and F1 can
  both be measured.
- Pro: separates the "feature" question from the "model" question. The
  conv stack is a concrete answer to "where do 100 features come from",
  even if it is not the answer that produced the shipped weights.
- Con: most work of the three options. Two training stages plus a
  projection.
- Con: the resulting numbers describe DeepLOB-features → LSTM_A, not the
  RS-40 production pipeline (which presumably has its own feature
  extractor outside both repos).
- Con: same numerical-equivalence caveat as Option B: shipped weights
  were not trained on DeepLOB-derived features.
