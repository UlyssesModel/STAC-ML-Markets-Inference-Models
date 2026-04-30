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

**Update 2026-04-30 (resolved at protocol level):** Verified against STAC's
reference implementation (STAC reference implementation, accessed via STAC
ML Track membership). The 100 features are *not* generated inside STAC's
reference driver; feature generation lives upstream in the test harness,
which materialises a pre-computed NumPy array on disk. The reference
driver then memory-maps that array at run start and slices windows from it
for each inference. The features themselves are simulated market-data
features produced by the harness's configuration-driven generator — a
component of the reference test rig, not of the inference path. The driver
reads the dtype straight from the on-disk array (so float32 vs. float64 is
data-driven, matching §2.1) and verifies the array's shape against the
configured samples × features layout before any inference runs. For
`LSTM_A`, that resolves to `features = 100`, `timesteps = 50`, and a
per-call window of `(1, 50, 100)` carved out of the contiguous 2D
on-disk array. A replacement model must therefore (a) accept whatever
dtype the harness's feature file ships in (per §2.1); (b) accept the
windowed `(batch, 50, 100)` shape unchanged — the driver does the slice
and reshape, the model does not see harness-side state; (c) treat the
features as already pre-scaled by the harness's generator (no in-model
normalisation is expected, and none is provided by STAC's published
spec). The exact per-feature schema is STAC-confidential and is not
reproduced here.

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

External-API black box; no speculation on Kirk internals. The bullets below
are *externally observable facts* about Kirk's compute model from Jarett
Artman's 2026-04-20 cross-platform affinity study (Kavara internal — see
§5.5), not internals. Inputs/outputs of Stage 2 in the contract sense
remain TBD by Kirk's external API; this is informative context, not a
redefinition of the contract.

The integration point on the *swap* side is formalized as the `KirkCore`
ABC in `scripts/ulysses_predictor.py` — see §5.4 for the contract a
real-Kirk subclass must satisfy.

- **Compute model:** complex N×N BF16 matmul, decomposed into 4 real BF16
  matmuls via `(A+iB)(C+iD) = (AC − BD) + i(AD + BC)`, dispatched through
  PyTorch oneDNN to AMX (Intel) or AVX-512 / NEON elsewhere.
- **Inference variants exposed:** five — `active_inference`,
  `active_inference_ent`, `active_inference_feat`, `inf_entropy`,
  `inf_features`. They have different latency/output characteristics; the
  swap may target one or several.
- **Production hardware target:** Granite Rapids under TDX (GNR+TDX),
  32-thread close OpenMP (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`).
- **Production N range:** `100 ≤ N ≤ 2000` per Hankel-embedding sizes used
  in deployed Ulysses pipelines.
- **Hard adapter requirement:** complex tensors must be `.contiguous()`
  *before* BF16 casting, or oneDNN takes a 7–12 ms/call reorder hit at
  N=1000. The replacement adapter must produce contiguous `ab::f0`-blocked
  BF16 inputs.

- **Inputs accepted:** whatever the Hankel adapter produces (BF16, exact
  rank/shape and the precise external API are TBD by Kirk).
- **Outputs produced:** a representation the readout consumes (shape and
  dtype TBD by Kirk).
- **Stage-specific open questions:**
  - Kirk's declared input/output contract — must be pinned before the
    Hankel-adapter and readout shapes can be finalized.
  - Determinism guarantees under BF16 matmul (relevant to §4(4)).
  - Which of the five inference variants is the RS-40 swap target.

### Stage 3: Scalar readout

- **Inputs accepted:** Kirk's output (shape/dtype per Stage 2).
- **Outputs produced:** `(batch, 1)`, `float32` or `float64` to match the
  external dtype chosen in §2.1.
- **Stage-specific open questions:**
  - Does the readout need a learned linear projection (analogue of `LSTM_A`'s
    final `Dense(1)`) or a fixed reduction?
  - Upcast point from BF16 back to the contract dtype.

### §5.4 Kirk integration point — `KirkCore` ABC

The Stage-2 swap point is formalized as the `KirkCore` ABC in
`scripts/ulysses_predictor.py`. To plug a real Kirk implementation
into the full pipeline:

1. Implement `KirkCore.transform(hankel_batch) -> projection_batch` —
   input is the Hankel adapter's output `(B, F, m, n)`; output is
   whatever the readout consumes (any rank, batch dim preserved).
2. Implement `KirkCore.output_shape(F, m, n) -> tuple[int, ...]` so
   the readout can be sized at `UlyssesPredictor` construction time.
3. Construct `UlyssesPredictor(kirk=YourKirk(...))` to plug it into
   the full `(B, 50, 100) -> (B, 1)` pipeline.

For a step-by-step guide to dropping a `KirkCore` implementation into
the pipeline, see `docs/kirk-integration-runbook.md`.

The Hankel adapter (Stage 1, `scripts/hankel_adapter.py`) and the
scalar readout (Stage 3, inside `UlyssesPredictor`) remain unchanged.
Two reference `KirkCore` subclasses ship in `ulysses_predictor.py` for
measurement: `IdentityKirk` (no-op Stage 2; collapses the pipeline to
Hankel + a single `(F·m·n) → 1` linear, exposing the Stage-1+3 floor)
and `LinearStubKirk` (deterministic random `(F·m·n) → K → 1`,
bounding a worst-case-ish Stage 2). Neither is real Kirk — both are
shape-and-dtype stand-ins for measuring end-to-end overhead. Latency
numbers under each mode appear in §5.7.

### §5.5 Empirical Kirk latency on production hardware (informative)

Source: Artman, *KIRK/ULYSSES Cross-Platform & Affinity Benchmark Study*,
Kavara internal, 2026-04-20.

Numbers are from the production GNR+TDX 32-thread close-OpenMP target
(`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), measured *after* the
`.contiguous()` adapter requirement called out in Stage 2 above is
honoured (without it, oneDNN incurs the 7–12 ms/call reorder cost at
N=1000):

| N | `active_inference` | `inf_features` |
| ---: | ---: | ---: |
| 5 | 0.86 ms | 0.30 ms |
| 20 | 1.45 ms | 0.40 ms |
| 100 | 17.6 ms | 2.3 ms |
| 200 | 57 ms | 6.7 ms |
| 500 | 192 ms | 13 ms |
| 1000 | 627 ms | 16.4 ms |
| 2000 | 3.26 s | 217 ms |

These are end-to-end pipeline-variant latencies (Stages 1 + 2 + 3 in the
production Kirk runtime), not isolated Stage-2 numbers — but the AMX
matmul cost dominates, so they upper-bound Kirk's contribution to the swap
latency at each N.

### §5.6 Comparison to STAC's reference driver (informative)

`scripts/stac_sumaco_driver.py` in this repo and STAC's reference driver
(STAC reference implementation, accessed via STAC ML Track membership)
share the same per-call latency contract: only the inference call is
timed, percentiles are reported across a population of timed calls, and
the inference engine sits behind a swappable predictor abstraction so
different model implementations can be benchmarked through the same
harness. Both drivers expose the two STAC-ML Markets per-call protocols
that STAC publishes publicly (Bishop Brock's 2024 STAC-ML Working Group
deck): **Sumaco** is the event-triggered single-inference suite — each
call gets an independent unique window of features — and **Tacana** is
the sliding-window streaming suite — each call rolls one or more new
timesteps into the window and rolls the same number out, so consecutive
windows overlap and a SUT is permitted to reuse computations on the
unchanged portion of the window. On our driver, the only difference
between the two protocols is how the input is supplied: Sumaco
pre-generates an `(n_timed, batch, T, F)` block of independent
fresh-random windows; Tacana pre-generates one `(batch, T + (n_timed - 1)
× stride, F)` fresh-random stream and slides a `(batch, T, F)` window
through it. The timed window itself, the latency-buffer pre-allocation,
the output-buffer pre-allocation, and the post-inference validation
phase are byte-identical across the two protocols. Real STAC SUTs *may*
exploit Tacana's overlap for protocol-specific optimisations
(KV-cache-equivalent state reuse, partial recomputation across
overlapping windows); our placeholder and ONNX predictors do not. In
particular, the `LSTM_*` artifacts in this repo ship with
`stateful=False` (per §2), so per-call recurrence is fully recomputed
inside one forward pass and the LSTM path pays the same per-call cost
under both protocols on this driver.

The drivers also differ in how the input window is sourced and in how
much surrounding methodology is applied. STAC's driver loads a fixed
configuration from a YAML file and orchestrates the per-suite (Sumaco /
Tacana) timing protocol around a swappable inference backend that
consumes a pre-materialised NumPy data file produced by the harness's
configuration-driven feature generator; ours synthesises a fresh random
tensor (Sumaco) or a fresh random stream (Tacana) per run from
`np.random.default_rng`. STAC's driver also layers on machinery our
driver does not: per-process CPU affinity and realtime scheduling for
hard-realtime SUTs, configurable parallel evaluation per model instance
with explicit out-of-order result handling, pre-allocated result storage
to suppress allocator-induced latency spikes, mid-call wall-clock
timestamps (Tsupply / Tresult) bracketing a narrow timing window with
sample-index computation and result-store deliberately *outside* it,
and a separate post-inference quality-check phase. Our driver is
therefore a *protocol-shaped approximation*: the protocol selection,
the timing window, and the predictor abstraction match STAC's, but the
input distribution and the precision of the surrounding measurement do
not. Numbers it produces are useful for relative comparison between
predictors and protocols on the same rig and are not directly comparable
to a STAC-audited result.

### §5.7 Timing methodology refinement (informative)

The Sumaco driver's timed window has been tightened to bring the
measurement closer to the discipline STAC's reference applies, without
copying any STAC code. Specifically: input tensors for the timed loop
are now pre-generated as a single `(n_timed, batch, T, F) float32`
array drawn from the seeded RNG before timing starts; latency samples
are written into a pre-allocated `np.empty(n_timed, dtype=np.int64)`
buffer via index-assignment; the timed block is restricted to
`t0 = perf_counter_ns(); predict(x); t1 = perf_counter_ns()` and one
index-assign — no RNG draws, no list growth, no per-iteration
allocation. Optional CPU pinning is now exposed via `--pin-cpu N`
(Linux-only via `os.sched_setaffinity`; on other platforms the flag
warns and continues without pinning).

Before/after on the same `e2-standard-4` rig, batch 1, no CPU pinning
(before from the loose-window driver at commit `56e9985`, 1000 timed;
after with the tight window, 100 warmup + 1000 timed):

| Config | Before p50 / p99 | After p50 / p99 |
| --- | ---: | ---: |
| `LSTM_A.onnx` | 593 / 843 µs | 611 / 947 µs |
| `ulysses_stub identity` (Stage 1+3 floor) | 96 / 233 µs | 173 / 339 µs |
| `ulysses_stub linear_stub` | 5079 / 9165 µs | 3336 / 8389 µs |

The qualitative picture is preserved — `LSTM_A.onnx` sits in the
high-hundreds-of-µs range, the identity floor is well under it, and
`linear_stub` is firmly in the multi-millisecond range — but the
individual deltas are within the run-to-run variance band of a shared
4-vCPU cloud VM and should not be read as causal effects of the
methodology change alone. The point of the change is the methodology
itself, not these specific numbers; relative comparisons taken under
the new methodology will be cleaner going forward.

A follow-on refinement closed one of the gaps: predictions are now
written into a pre-allocated `(n_timed, batch, 1) float32` buffer via
index-assignment outside the timed window, and a post-inference
validation phase verifies shape/dtype, asserts every sample is finite,
and records min / max / mean / std under `output_stats` in the result
JSON. This brings the driver one step closer to STAC's reference,
which uses the stored-results pattern for its post-inference
quality-check phase. The validation step adds a small fixed
per-iteration cost outside the timing window, so the latency numbers
above remain representative.

A further refinement adds *cross-predictor agreement* as the first
metric the driver produces beyond per-call latency. With
`--compare-with` set (to `ulysses_stub_identity`,
`ulysses_stub_linear_stub`, or an `.onnx` model path) the driver
runs a second predictor untimed on byte-identical input tensors and
computes sign-agreement percentage, Pearson r, Spearman rho, and
mean / max absolute difference in pure numpy (no scipy), surfacing
them under `agreement_stats` in the JSON together with summary stats
for both predictors' output distributions. With synthetic random
inputs (this driver's default) the sign-agreement number lands near
50% as expected, and the correlations are near zero — *meaningful*
agreement numbers require real STAC features (out of scope for this
commit).

Remaining gaps versus STAC's reference, called out explicitly: we
still do not implement separate Tsupply / Tresult timestamps (we
capture a single perf_counter_ns pair around the predict call rather
than splitting the boundary between data supply and result return);
we do not run NMI parallelism (multiple model instances inferring
concurrently); we do not implement out-of-order result handling for
parallel mode; and we do not invoke realtime scheduling. Closing any
of those is a separate step beyond this refinement.

## §6 Cross-cutting open questions

Stage-specific open questions are inline in §5; this section lists
cross-cutting items only.

1. **What are the 100 input features?** Resolution lives on the
   harness/training side (see §3). Blocks any data-driven design choice in
   §5.

   **Update 2026-04-30 (resolved at protocol level):** See §3 update. The
   features are STAC-generated simulated market-data features delivered to
   the inference path as a pre-materialised NumPy array; the reference
   driver memory-maps the file and windows into it. A replacement honours
   the contract by accepting the declared shape and dtype at the boundary
   — per-feature semantics are fixed by the STAC harness and are not a
   model-side design parameter.

2. **Is the readout truly `sign(output)`** or does the harness apply a
   non-zero threshold or calibration (e.g. an offset learned on a holdout, or
   a deadband around zero)? Affects whether §4(3) is satisfied by sign
   agreement alone.

   **Update 2026-04-30 (resolved at protocol level):** Verified against
   STAC's reference implementation (STAC reference implementation,
   accessed via STAC ML Track membership). In the Sumaco code path, the
   reference driver returns the raw scalar emerging from the inference
   engine — no sigmoid, no calibration offset, no deadband, no
   thresholding is applied inside the timed inference path. The Tsupply
   and Tresult wall-clock timestamps bracket the bare inference call, and
   the result is stored verbatim alongside the timestamps. Direction
   labels (or any other semantic interpretation) are derived later, in
   the post-inference quality-check phase that runs against the stored
   results array. The spec therefore mandates: the model emits a
   real-valued scalar; sign-thresholding (or any other readout
   convention) is harness post-processing, not part of the inference
   contract. §4(3) remains satisfiable by sign agreement under that
   convention. If a non-zero threshold or calibration is ever needed,
   it lives outside the inference path.

3. **Does the stateless test (`LSTM_Null_A` control) require its own
   Ulysses-shaped baseline,** or does the existing constant-init baseline
   suffice? If a Ulysses-shaped null is needed, it has to be specified
   separately and is outside this document's scope as written.

   **Update 2026-04-30 (Kavara design call, with STAC-side context):**
   STAC's reference treats Null and trained models symmetrically — both
   are first-class entries in the driver's per-model configuration, both
   are run through the identical driver path with identical timing
   methodology, and both are scored independently. Per STAC's audit
   conventions, Null configurations must mirror their non-Null
   counterparts exactly so the latency / quality comparison is clean at
   the SUT level; the reference does not score the Null *against* the
   trained model inside the driver — they are separate measured runs
   whose outputs are compared in post-processing. So the existing
   `LSTM_Null_A` control fits cleanly into STAC's symmetric-driver model.
   Whether RS-40's stateless arm needs a *Ulysses-shaped* null on top of
   these is a separate Kavara design call, unchanged by this resolution.

4. **Lens-1 latency budget.** Lens-2 is specified at p99 10 µs. Lens-1 is
   not formally documented in RS-40 and needs to be pinned before any
   replacement design can be validated against it.

   **Update 2026-04-30 (refined, not resolved):** Empirical Kirk
   `active_inference` latency on the production GNR+TDX 32t close target
   (per §5.5) is 17.6 ms at N=100 — the lowest end of the stated
   production range — versus the 10 µs lens-2 budget recorded in Kavara
   internal RS-40 design notes. The two figures are 1,760× apart. Even
   at N=5 (well below production range), Kirk's 0.86 ms is 86× over
   10 µs. Three reconciliations are possible: (a) RS-40's "Ulysses
   replaces LSTM_A" is a quality-vs-latency tradeoff demonstration, not
   a direct latency match; (b) RS-40's actual N for the swap is below 5;
   (c) the 10 µs figure applies under different operating assumptions
   than the ones the affinity study measures. Resolution is a
   Kavara-internal design call; the swap-spec captures the tension
   without picking.

   *STAC-side methodology refinement (informative).* STAC's reference
   implementation (STAC reference implementation, accessed via STAC ML
   Track membership) does not formalise a "lens-1" budget — that
   distinction is RS-40-internal — but it does pin the methodology
   around the timing measurement itself. Each inference is bracketed by
   two wall-clock timestamps (Tsupply just before the engine receives
   the window, Tresult immediately after the engine returns); per-call
   latency is the difference. Sample-index computation, result storage,
   and Python list-growth pauses are deliberately placed *outside* the
   timed window. Warmup is configurable and defaulted to a small number
   of seconds. For hard-realtime SUTs the reference recommends a tighter
   stack — pre-allocated result storage to suppress allocator-induced
   jitter, per-process CPU affinity / isolation, and the highest
   realtime scheduling priority — as precision-tightening conventions on
   the timing measurement, not as budget definitions. Reporting is
   percentile-based across a population of timed calls. Nothing in the
   reference contradicts the lens-2 10 µs target, but neither does it
   pin a lens-1 figure; the 1,760× tension above stands.

## §7 Out of scope

- Implementation of the Ulysses replacement (any stage).
- Kirk internals.
- Training procedure for the replacement.
- The GBT family (`GBT_A`, `GBT_B`, `GBT_C`).
- The `_Null` family (`LSTM_Null_*`, `GBT_Null_*`).
- Lenses B and C (`LSTM_B`, `LSTM_C`, and their `_Null` counterparts).
