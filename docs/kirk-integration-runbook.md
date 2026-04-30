# Kirk Integration Runbook

A step-by-step guide for dropping a real Kirk implementation into the Ulysses
pipeline at the Stage-2 swap point. Companion to `docs/rs40-swap-spec.md` §5.4.

## §1 TL;DR

1. Subclass `KirkCore` (defined in `scripts/ulysses_predictor.py`).
2. Implement `transform()` and `output_shape()`.
3. Instantiate `UlyssesPredictor(kirk=YourKirk(...))`.

Stage 1 (Hankel adapter) and Stage 3 (scalar readout) stay frozen.

## §2 The KirkCore contract

The `KirkCore` ABC lives in `scripts/ulysses_predictor.py`. It has exactly two
abstract methods, both required:

- `transform(hankel_batch) -> projection_batch` — runs whatever Kirk does on
  one batch of Hankel-embedded inputs.
- `output_shape(F, m, n) -> tuple[int, ...]` — declares the per-element output
  shape (excluding batch) so the readout can be sized at construction time.

### Input contract

`transform()` receives `hankel_batch` of shape `(B, F, m, n)`, dtype
`float32` — the output of Stage 1 (`scripts/hankel_adapter.py`).

- `B` is the runtime batch size (variable).
- `F` is the per-timestep feature count (100 for `LSTM_A`).
- `m` is the Hankel row count and `n = T - m + 1` is the column count, where
  `T` is the input timesteps (50 for `LSTM_A`). Default `m = T // 2 = 25`,
  giving `n = 26`.

### Output contract

`transform()` must return an array whose leading dim is the same `B` it
received and whose trailing per-element shape exactly matches what
`output_shape(F, m, n)` declared. Dtype must be whatever the readout will
accept — `float32` matches the existing readout; if your Kirk produces
something else, upcast inside `transform()` before returning.

The readout is a single `(prod(output_shape), 1)` linear projection,
materialised once at `UlyssesPredictor.__init__` time using the declared
shape. `output_shape()` is therefore load-bearing: the readout will be sized
wrong if it disagrees with what `transform()` actually returns, and you'll
get a matmul shape error at the first `predict()` call.

## §3 Implementation template

Minimum viable subclass:

```python
import numpy as np
from ulysses_predictor import KirkCore

class RealKirk(KirkCore):
    def __init__(self, ...config...):
        # any config the real implementation needs
        ...

    def output_shape(self, F: int, m: int, n: int) -> tuple[int, ...]:
        # declare what transform() will produce per batch element
        # usually rank-1: (K,) for some K, since the readout flattens anyway
        return (...,)

    def transform(self, hankel_batch: np.ndarray) -> np.ndarray:
        # hankel_batch: (B, F, m, n) float32
        # return: (B, *output_shape(F, m, n)), accepted by the readout
        ...
```

Then plug it in:

```python
from ulysses_predictor import UlyssesPredictor

predictor = UlyssesPredictor(kirk=RealKirk(...))
y = predictor.predict(x)  # x: (B, 50, 100) float32 -> y: (B, 1) float32
```

## §4 Verification steps

After implementing the subclass, run these three checks in order.

### 4.1 Shape contracts

Run the unit tests. They exercise `KirkCore` and `UlyssesPredictor` end-to-end
across batch sizes, dtypes, and the two reference subclasses, and will catch
shape/dtype-contract violations early:

```bash
python tests/test_ulysses_predictor.py
python tests/test_hankel.py
```

You can extend the tests with your own subclass instance to confirm the
declared `output_shape()` matches what `transform()` actually returns — the
existing tests already verify that pattern for `IdentityKirk` and
`LinearStubKirk`.

### 4.2 Sumaco smoke

Drive the new predictor through the Sumaco harness. This confirms the full
`(B, 50, 100) -> (B, 1)` pipeline runs without shape/dtype errors and gives
you a per-call latency profile under the same protocol used for the ONNX
baseline. Wire your subclass into the driver's `--predictor ulysses_stub`
path (extend `--ulysses-kirk-mode` or add a dedicated flag — see the
`PREDICTOR_CHOICES` block in `scripts/stac_sumaco_driver.py`):

```bash
python scripts/stac_sumaco_driver.py \
    --predictor ulysses_stub \
    --ulysses-kirk-mode <your_mode> \
    --n-warmup 50 --n-runs 1000
```

### 4.3 Cross-predictor agreement against `LSTM_A`

Compare the new predictor against `LSTM_A.onnx` to see the actual signal
relationship — this is the closest you can get to "does it produce the same
direction labels" without real STAC features:

```bash
python scripts/stac_sumaco_driver.py \
    --predictor onnx --model-path LSTM_A.onnx \
    --compare-with <your_predictor_spec> \
    --n-warmup 50 --n-runs 1000 \
    --output-json /tmp/agree.json
```

Look at `agreement_stats` in the output: `pearson_r` and `spearman_rho` are
scale-invariant; `mean_abs_diff` / `max_abs_diff` are scale-dependent (good
for diagnosing scale mismatch); `mean_abs_diff_z` / `max_abs_diff_z` are
scale-normalised and reflect rank/sign disagreement on synthetic inputs. With
the driver's default synthetic random inputs the absolute numbers will hover
near chance — meaningful agreement requires real STAC features. The metric
is still useful as a regression signal: any drift relative to a previous
real-Kirk run is a contract or determinism break.

## §5 Methodology notes

### Latency expectations

Per `docs/rs40-swap-spec.md` §5.5, empirical Kirk `active_inference` latency
on the production GNR+TDX 32-thread close target is **17.6 ms at N=100**, the
lowest end of the deployed range. At N=5 it is 0.86 ms; at N=1000 it is
627 ms. These are end-to-end pipeline latencies (Stages 1 + 2 + 3 in the
production Kirk runtime), with the AMX matmul cost dominating.

### The 1,760× tension

`docs/rs40-swap-spec.md` §6.4 records the unresolved tension between Kirk's
17.6 ms at N=100 and the lens-2 10 µs p99 budget — a 1,760× gap. Even at
N=5, Kirk's 0.86 ms is 86× over budget. The implication for an integrator: a
real Kirk is unlikely to match `LSTM_A`'s microsecond-scale latency on any
current hardware. Don't expect the swap to be a latency match; expect to be
demonstrating a quality-vs-latency tradeoff (or operating below the
production N range, or under different operating assumptions). The runbook
does not pick which.

## §6 What stays frozen

You should not need to touch any of the following to integrate real Kirk:

- **Stage 1** — `scripts/hankel_adapter.py`. Produces the
  `(B, F, m, n) float32` input to `transform()`.
- **Stage 3** — the scalar readout linear inside `UlyssesPredictor`. Sized
  at construction time from your `output_shape()` and applied as a single
  matmul to `(B, 1)`.
- **Test infrastructure** — `tests/test_ulysses_predictor.py`,
  `tests/test_hankel.py`. The contracts they enforce are the ones your
  subclass must satisfy; extend with subclass-specific cases rather than
  modifying the existing assertions.
- **Sumaco driver** — `scripts/stac_sumaco_driver.py`. The timing window,
  the agreement-metrics block, and the predictor abstraction stay as-is;
  only the construction call (`UlyssesPredictor(kirk=YourKirk(...))`)
  changes.

The only files that need changes are the new `KirkCore` subclass module and
whatever construction-site code instantiates `UlyssesPredictor`.
