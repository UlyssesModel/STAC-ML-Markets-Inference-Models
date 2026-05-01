#!/usr/bin/env python3
"""STAC-aligned latency driver (Sumaco + Tacana) with a pluggable model interface.

Implements the two STAC-ML Markets per-call latency protocols against any
model that conforms to the `(B, T, F) float32 -> (B, 1) float32` contract
shared by this repo's `LSTM_*` artifacts.

The driver is deliberately separated from the model behind a `STACPredictor`
ABC so a Ulysses-based replacement can drop in next to the ONNX baseline
without touching the timing harness.

Protocols
---------
**Sumaco** (event-triggered, default). Each inference receives an
independent fresh-random `(B, T, F)` window of input features. Inputs for
the whole timed loop are pre-generated as one `(n_timed, B, T, F)` array
and the i-th call uses `inputs[i]`. This matches the publicly-documented
shape of STAC's Sumaco suite — single-inference latency on a unique
window per call — at the protocol level.

**Tacana** (sliding-window streaming). Each inference advances by
`tacana_stride` timesteps along a longer fresh-random stream:
`stride = 1` rolls a single new timestep into the window per call,
`stride >= T` degenerates to non-overlapping windows. Inputs are
pre-generated as one `(B, T + (n_timed - 1) * stride, F)` stream and the
i-th call uses `stream[:, i*stride : i*stride + T, :]`. This mirrors the
shape of STAC's Tacana suite — sliding-window streaming inference where
each call's window overlaps with the previous — at the protocol level.

On this driver, the only difference between the two protocols is the
input distribution; the timed window, the latency / output buffers, the
post-inference validation phase, and the per-call timing methodology are
byte-identical. Real STAC SUTs *may* exploit the Tacana overlap for
optimisations our predictors do not (KV-cache-equivalent state reuse,
partial recomputation across calls). Our `LSTM_*.onnx` artifacts ship
with `stateful=False` (per docs/rs40-swap-spec.md §2), so the LSTMs in
particular pay the same per-call cost under both protocols.

NMI parallelism
---------------
The `--nmi N` flag runs N model instances in parallel through Python's
`multiprocessing` module on the `spawn` start method. Each child
process constructs its own predictor instance (so we never need to
pickle an ONNX session — onnxruntime sessions are not fork-safe and
not picklable in general), draws its own seed (`base_seed * 100 +
instance_id`), runs the chosen protocol independently, and returns its
latency array plus output stats to the parent. The parent then reports
both per-instance metrics (each child's p50 / p99 / n_timed) and an
aggregated block (concatenated latencies across all children, with
combined percentiles), plus a wall-clock-derived
`throughput_inf_per_sec = (N * n_timed) / (t_end - t_start)` where the
clock brackets only the parallel section.

`--nmi 1` skips the spawn machinery entirely and runs in-process; the
JSON output shape is unchanged from the single-instance default in
that case. NMI > 1 adds top-level `nmi`, `per_instance`, `aggregate`,
`wall_clock_s`, and `throughput_inf_per_sec` fields; `output_stats` is
computed across the concatenation of every child's outputs.

Per-instance latency typically degrades once N exceeds the rig's
physical core count — children contend for CPU, memory bandwidth, and
shared caches — and aggregate throughput therefore scales sub-linearly
with NMI. STAC's published vendor reports go up to 32 NMI (NVIDIA) and
48 NMI (Myrtle.ai); our 4-vCPU dev rig saturates well below that.

Timing methodology
------------------
The timed window is restricted to exactly the predictor's `predict()` call
between two `time.perf_counter_ns()` reads. Everything else — RNG draws,
tensor allocation, latency-buffer growth, prediction storage — is pushed
outside the window:

* Input tensors for the entire timed loop are pre-generated as a single
  `(n_timed, batch, T, F) float32` array, drawn from the seeded RNG
  before timing starts. Each timed iteration takes a `(batch, T, F)`
  slice — no `np.random` call inside the timed window.

* Latency samples are stored into a pre-allocated `np.empty(n_timed,
  dtype=np.int64)` buffer via index-assignment. No Python list growth
  inside the timed window.

* Predictions are stored into a pre-allocated `np.empty((n_timed, batch,
  1), dtype=np.float32)` buffer. The index-assignment is performed
  *outside* the timed window — between the latency write and the next
  iteration's `t0` read — so the only work inside the window remains
  the `predict()` call plus the int64 latency assign.

* CPU pinning is optional via `--pin-cpu N`, which calls
  `os.sched_setaffinity(0, {N})` once before warmup. Linux-only;
  non-Linux platforms print a warning and continue without pinning.

* After the timed loop, a post-inference validation phase verifies the
  output buffer's shape and dtype, asserts every prediction is finite,
  and computes summary stats (min / max / mean / std). Surfaced under
  `output_stats` in the result JSON. This brings us one step closer to
  STAC's reference, which uses the stored-results pattern for its
  post-inference quality-check phase.

This brings our driver methodologically closer to audit-grade harnesses
while remaining a *protocol-shaped approximation* of STAC's reference, per
docs/rs40-swap-spec.md §5.6. We still do not implement separate
Tsupply/Tresult timestamps, out-of-order result handling, or
hard-realtime scheduling. Numbers produced here remain relative; they
are not directly comparable to a STAC-audited result.

Cross-predictor agreement
-------------------------
The `--compare-with` flag enables an *untimed* secondary predictor that
runs on byte-identical input tensors to the primary. After both
predictors have produced their output buffers, the driver computes
agreement metrics in pure numpy (no scipy): sign-agreement percentage,
Pearson r, Spearman rho (rank correlation via `np.argsort(np.argsort(x))`),
and mean / max absolute difference, plus scale-normalised `_z` variants
of the absolute-difference metrics. Results are surfaced under
`agreement_stats` in the JSON, alongside summary stats for both
predictors' output distributions so it's clear whether the agreement
number is conditioned on similar or wildly different ranges.

Pearson r and Spearman rho are already scale-invariant, so they reflect
rank/sign-relevant agreement regardless of the two predictors' output
magnitudes. `mean_abs_diff` and `max_abs_diff` are scale-dependent and
therefore useful for diagnosing scale mismatch between predictors (e.g.
`LSTM_A` vs an unscaled Ulysses-stub readout, where one predictor's
output range dwarfs the other's and the raw diff is dominated by that
gap). The companion `mean_abs_diff_z` and `max_abs_diff_z` z-score each
predictor's output independently before differencing
(`(x - x.mean()) / (x.std() + eps)` with `eps = 1e-12` to handle a
degenerate constant predictor), so on synthetic-input runs the
agreement numbers reflect rank/sign disagreement rather than just
output-magnitude differences. Both raw and z-scored variants are kept
because each diagnoses a different failure mode.

With synthetic random inputs (this driver's default) the expected
sign-agreement is ~50% and the correlations are near zero — meaningful
agreement numbers require real STAC features. The CLI accepts:
`ulysses_stub_identity` and `ulysses_stub_linear_stub` (constructed
with the primary's `(T, F)` shape), or any other string treated as a
path to an `.onnx` model loaded via `ONNXPredictor`.
"""

from __future__ import annotations

import abc
import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np


class STACPredictor(abc.ABC):
    """Abstract predictor with the STAC LSTM `(B, T, F) -> (B, 1)` contract."""

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run one inference. `x` is `(B, T, F) float32`; returns `(B, 1) float32`."""

    @property
    @abc.abstractmethod
    def input_shape(self) -> tuple[int, int]:
        """Per-call input shape `(T, F)`, read from the loaded model."""


class ONNXPredictor(STACPredictor):
    """ONNX Runtime predictor. Loads any `LSTM_*.onnx` from this repo."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        # ONNX shape is [batch, T, F] with the batch dim symbolic ('unk__N' or similar).
        shape = inp.shape
        if len(shape) != 3:
            raise ValueError(
                f"expected rank-3 input, got shape {shape!r} from {model_path}"
            )
        self._T = int(shape[1])
        self._F = int(shape[2])

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self._T, self._F)

    def predict(self, x: np.ndarray) -> np.ndarray:
        (y,) = self._sess.run(None, {self._input_name: x})
        return y


def _percentile(sorted_us: list[float], q: float) -> float:
    n = len(sorted_us)
    return sorted_us[min(n - 1, int(round(q * n)) - 1)]


def _ranks(x: np.ndarray) -> np.ndarray:
    """Numerical ranks of a 1-D array. Ties broken by index (vanishingly rare for float32)."""
    return np.argsort(np.argsort(x))


def _compute_agreement_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute cross-predictor agreement metrics over two prediction buffers.

    `a` and `b` may be any shape — they are flattened identically before
    metrics are computed. All metrics are pure-numpy (no scipy).
    """
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    sign_a = a_flat >= 0
    sign_b = b_flat >= 0
    diff = np.abs(a_flat - b_flat)
    eps = 1e-12
    a_z = (a_flat - a_flat.mean()) / (a_flat.std() + eps)
    b_z = (b_flat - b_flat.mean()) / (b_flat.std() + eps)
    diff_z = np.abs(a_z - b_z)
    return {
        "n_pairs": int(a_flat.size),
        "sign_agreement_pct": float((sign_a == sign_b).mean() * 100.0),
        "pearson_r": float(np.corrcoef(a_flat, b_flat)[0, 1]),
        "spearman_rho": float(np.corrcoef(_ranks(a_flat), _ranks(b_flat))[0, 1]),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff_z": float(diff_z.mean()),
        "max_abs_diff_z": float(diff_z.max()),
        "primary_output_stats": {
            "min": float(a_flat.min()),
            "max": float(a_flat.max()),
            "mean": float(a_flat.mean()),
            "std": float(a_flat.std()),
        },
        "compared_output_stats": {
            "min": float(b_flat.min()),
            "max": float(b_flat.max()),
            "mean": float(b_flat.mean()),
            "std": float(b_flat.std()),
        },
    }


def _generate_inputs(
    protocol: str,
    rng: np.random.Generator,
    n_timed: int,
    batch: int,
    T: int,
    F: int,
    tacana_stride: int,
) -> "tuple[np.ndarray, callable]":
    """Pre-generate the timed-loop inputs and return (storage, indexer).

    `storage` holds whatever pre-generated tensor backs the protocol;
    `indexer(i)` returns the `(batch, T, F) float32` window for call `i`.
    Both protocols pre-generate all inputs before the timed loop starts so
    no RNG draws happen inside it.

    Sumaco: `storage` is `(n_timed, batch, T, F)`, indexer returns a fresh
    independent window each call.

    Tacana: `storage` is `(batch, T + (n_timed - 1) * stride, F)`, indexer
    returns a sliding view advanced by `tacana_stride` timesteps per call.
    """
    if protocol == "sumaco":
        storage = rng.standard_normal((n_timed, batch, T, F), dtype=np.float32)
        return storage, (lambda i: storage[i])
    if protocol == "tacana":
        if tacana_stride < 1:
            raise ValueError(f"tacana_stride must be >= 1, got {tacana_stride}")
        stream_len = T + (n_timed - 1) * tacana_stride
        storage = rng.standard_normal((batch, stream_len, F), dtype=np.float32)

        def _window(i: int) -> np.ndarray:
            start = i * tacana_stride
            view = storage[:, start : start + T, :]
            # Sliding views over a (B, stream_len, F) array along axis 1 are
            # already C-contiguous along the last axis, but the window itself
            # is a non-contiguous slice of the underlying buffer. Some
            # predictors (notably onnxruntime via DLPack) require contiguous
            # input — copy if needed.
            if not view.flags["C_CONTIGUOUS"]:
                view = np.ascontiguousarray(view)
            return view

        return storage, _window
    raise ValueError(f"unknown protocol: {protocol!r} (expected 'sumaco' or 'tacana')")


def run_protocol(
    predictor: STACPredictor,
    n_warmup: int,
    n_timed: int,
    batch: int,
    seed: int = 0,
    compare_with: STACPredictor | None = None,
    protocol: str = "sumaco",
    tacana_stride: int = 1,
) -> dict:
    """Drive `predictor` under the requested STAC-ML protocol and return a result dict.

    `protocol` selects the input-generation strategy: `"sumaco"`
    pre-generates one `(n_timed, batch, T, F)` array of independent
    fresh-random windows; `"tacana"` pre-generates a single
    `(batch, T + (n_timed - 1) * tacana_stride, F)` stream and slides a
    `(batch, T, F)` window through it. Per-call timing methodology is
    identical in both cases — the only difference is the input
    distribution. See the module docstring's *Protocols* section.

    Latency samples are written into a pre-allocated int64 buffer.
    Predictions are written into a pre-allocated `(n_timed, batch, 1)
    float32` buffer. The timed window contains only the `predict()` call
    and one int64 latency assignment; the prediction store happens outside
    it. After the timed loop a cheap validation phase computes
    prediction-distribution stats and confirms every sample is finite.

    If `compare_with` is provided, it is run (untimed) on the same input
    windows the primary saw, into a parallel pre-allocated output buffer;
    the result dict gains an `agreement_stats` block with cross-predictor
    metrics (sign agreement, Pearson, Spearman, mean / max abs diff and
    their z-scored variants). The two predictors must have matching
    `input_shape`. See module docstring for the full methodology.
    """
    T, F = predictor.input_shape
    rng = np.random.default_rng(seed)

    for _ in range(n_warmup):
        x = rng.standard_normal((batch, T, F), dtype=np.float32)
        predictor.predict(x)

    _input_storage, window_at = _generate_inputs(
        protocol, rng, n_timed, batch, T, F, tacana_stride
    )
    latencies_ns = np.empty(n_timed, dtype=np.int64)
    outputs = np.empty((n_timed, batch, 1), dtype=np.float32)

    for i in range(n_timed):
        x = window_at(i)
        t0 = time.perf_counter_ns()
        y = predictor.predict(x)
        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0
        outputs[i] = y       # outside the timing window

    if outputs.shape != (n_timed, batch, 1):
        raise AssertionError(
            f"output buffer shape mismatch: expected ({n_timed}, {batch}, 1), "
            f"got {outputs.shape}"
        )
    if outputs.dtype != np.float32:
        raise AssertionError(
            f"output buffer dtype mismatch: expected float32, got {outputs.dtype}"
        )
    flat = outputs.reshape(-1)
    all_finite = bool(np.isfinite(flat).all())
    if not all_finite:
        n_bad = int((~np.isfinite(flat)).sum())
        raise AssertionError(
            f"prediction buffer contains {n_bad} non-finite value(s) "
            f"out of {flat.size}; check predictor for NaN/Inf"
        )

    samples_us = sorted(latencies_ns.astype(np.float64) / 1000.0)
    result = {
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "protocol": protocol,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "batch_size": batch,
        "input_shape": [T, F],
        "p50_us": _percentile(samples_us, 0.50),
        "p90_us": _percentile(samples_us, 0.90),
        "p99_us": _percentile(samples_us, 0.99),
        "mean_us": statistics.fmean(samples_us),
        "std_us": statistics.pstdev(samples_us),
        "output_stats": {
            "n_predictions": int(flat.size),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "all_finite": all_finite,
        },
    }
    if protocol == "tacana":
        result["tacana_stride"] = tacana_stride

    if compare_with is not None:
        sec_T, sec_F = compare_with.input_shape
        if (sec_T, sec_F) != (T, F):
            raise ValueError(
                f"compare_with predictor input_shape ({sec_T}, {sec_F}) does "
                f"not match primary ({T}, {F}); shapes must match for "
                f"cross-comparison"
            )
        outputs_b = np.empty((n_timed, batch, 1), dtype=np.float32)
        for i in range(n_timed):
            outputs_b[i] = compare_with.predict(window_at(i))
        if not np.isfinite(outputs_b).all():
            n_bad = int((~np.isfinite(outputs_b)).sum())
            raise AssertionError(
                f"compare_with prediction buffer contains {n_bad} "
                f"non-finite value(s); cannot compute agreement stats"
            )
        result["agreement_stats"] = _compute_agreement_stats(outputs, outputs_b)

    return result


def run_sumaco(
    predictor: STACPredictor,
    n_warmup: int,
    n_timed: int,
    batch: int,
    seed: int = 0,
    compare_with: STACPredictor | None = None,
) -> dict:
    """Backward-compatible wrapper: drive `predictor` under the Sumaco protocol.

    Equivalent to `run_protocol(..., protocol="sumaco")`. Retained so
    existing callers (notably `tests/test_ulysses_predictor.py`) keep
    working unchanged.
    """
    return run_protocol(
        predictor,
        n_warmup=n_warmup,
        n_timed=n_timed,
        batch=batch,
        seed=seed,
        compare_with=compare_with,
        protocol="sumaco",
    )


def _build_predictor_from_args(args_dict: dict) -> STACPredictor:
    """Construct a primary predictor inside a (possibly child) process.

    The args dict carries the same fields the CLI assembles in `main()`
    (`predictor`, `model_path`, `ulysses_kirk_mode`, `ulysses_k`,
    `ulysses_m`, `seed`). Importing ulysses_predictor lazily keeps
    `--predictor onnx` runs (CI's path) free of the import cost.
    """
    p_type = args_dict["predictor"]
    if p_type == "onnx":
        return ONNXPredictor(args_dict["model_path"])
    if p_type == "ulysses_stub":
        from ulysses_predictor import (
            IdentityKirk,
            LinearStubKirk,
            UlyssesPredictor,
        )

        mode = args_dict["ulysses_kirk_mode"]
        if mode == "identity":
            kirk = IdentityKirk()
        elif mode == "linear_stub":
            kirk = LinearStubKirk(k=args_dict["ulysses_k"], seed=args_dict["seed"])
        else:
            raise ValueError(f"unknown kirk mode: {mode!r}")
        return UlyssesPredictor(
            m=args_dict["ulysses_m"],
            kirk=kirk,
            readout_seed=args_dict["seed"],
        )
    raise ValueError(f"unknown predictor: {p_type!r}")


def run_one_instance(args_dict: dict) -> dict:
    """Run one model instance end-to-end and return a serialisable result.

    Top-level so it is picklable for `multiprocessing.spawn`. Constructs
    its own predictor inside the process (predictor objects, particularly
    onnxruntime sessions, are not assumed to be picklable). Returns the
    full latency array, the per-call output buffer flattened, and the
    instance's `output_stats` block — all numpy arrays are converted to
    lists for IPC compactness on the parent side.

    Expected `args_dict` keys: `predictor`, `model_path`,
    `ulysses_kirk_mode`, `ulysses_k`, `ulysses_m`, `n_warmup`, `n_timed`,
    `batch`, `seed`, `protocol`, `tacana_stride`, plus `instance_id` for
    bookkeeping.
    """
    predictor = _build_predictor_from_args(args_dict)
    T, F = predictor.input_shape
    rng = np.random.default_rng(args_dict["seed"])

    for _ in range(args_dict["n_warmup"]):
        x = rng.standard_normal((args_dict["batch"], T, F), dtype=np.float32)
        predictor.predict(x)

    n_timed = args_dict["n_timed"]
    batch = args_dict["batch"]
    _input_storage, window_at = _generate_inputs(
        args_dict["protocol"],
        rng,
        n_timed,
        batch,
        T,
        F,
        args_dict["tacana_stride"],
    )
    latencies_ns = np.empty(n_timed, dtype=np.int64)
    outputs = np.empty((n_timed, batch, 1), dtype=np.float32)

    for i in range(n_timed):
        x = window_at(i)
        t0 = time.perf_counter_ns()
        y = predictor.predict(x)
        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0
        outputs[i] = y

    flat = outputs.reshape(-1)
    all_finite = bool(np.isfinite(flat).all())
    if not all_finite:
        n_bad = int((~np.isfinite(flat)).sum())
        raise AssertionError(
            f"instance {args_dict.get('instance_id', '?')}: prediction buffer "
            f"contains {n_bad} non-finite value(s) out of {flat.size}"
        )

    return {
        "instance_id": args_dict.get("instance_id", 0),
        "seed": args_dict["seed"],
        "n_timed": n_timed,
        "latencies_ns": latencies_ns.tolist(),
        "output_stats": {
            "n_predictions": int(flat.size),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "all_finite": all_finite,
        },
        "outputs_flat": flat.tolist(),
    }


def _apply_cpu_pin(cpu: int) -> bool:
    """Pin this process to CPU `cpu`. Returns True on success, False otherwise.

    Linux-only. On non-Linux platforms or if the call fails, prints a
    one-line warning to stderr and returns False — caller is expected to
    continue without pinning.
    """
    if not hasattr(os, "sched_setaffinity"):
        print(
            f"warning: --pin-cpu requested but os.sched_setaffinity is "
            f"unavailable on {sys.platform}; continuing without pinning",
            file=sys.stderr,
        )
        return False
    try:
        os.sched_setaffinity(0, {cpu})
        return True
    except OSError as e:
        print(
            f"warning: --pin-cpu {cpu} failed ({e}); continuing without pinning",
            file=sys.stderr,
        )
        return False


PREDICTOR_CHOICES = ("onnx", "ulysses_stub")


def _args_dict_for_child(args: argparse.Namespace, instance_id: int) -> dict:
    """Build the picklable dict `run_one_instance` consumes inside a child."""
    return {
        "instance_id": instance_id,
        "predictor": args.predictor,
        "model_path": str(args.model_path) if args.model_path else None,
        "ulysses_kirk_mode": args.ulysses_kirk_mode,
        "ulysses_k": args.ulysses_k,
        "ulysses_m": args.ulysses_m,
        "n_warmup": args.n_warmup,
        "n_timed": args.n_runs,
        "batch": args.batch,
        # Per-instance seed offset: keep 100-step gaps between instances so
        # accidental collisions with --seed defaults are extremely unlikely.
        "seed": args.seed * 100 + instance_id,
        "protocol": args.protocol,
        "tacana_stride": args.tacana_stride,
    }


def _run_nmi(args: argparse.Namespace, pin_cpu_applied: bool) -> int:
    """Spawn N model instances in parallel, aggregate, and emit the result JSON."""
    if args.predictor == "onnx" and not args.model_path:
        # Mirror the single-instance check; argparse can't enforce this
        # cross-flag dependency.
        print(
            "error: --model-path is required when --predictor onnx",
            file=sys.stderr,
        )
        return 2

    mp.set_start_method("spawn", force=True)
    children_args = [_args_dict_for_child(args, i) for i in range(args.nmi)]

    t_start = time.perf_counter()
    with mp.Pool(args.nmi) as pool:
        child_results = pool.map(run_one_instance, children_args)
    t_end = time.perf_counter()

    wall_clock_s = t_end - t_start
    all_latencies_us = np.concatenate(
        [np.asarray(r["latencies_ns"], dtype=np.int64) for r in child_results]
    ).astype(np.float64) / 1000.0
    all_outputs = np.concatenate(
        [np.asarray(r["outputs_flat"], dtype=np.float32) for r in child_results]
    )

    per_instance = []
    for r in child_results:
        per_us = sorted(np.asarray(r["latencies_ns"], dtype=np.int64).astype(np.float64) / 1000.0)
        per_instance.append({
            "instance_id": r["instance_id"],
            "seed": r["seed"],
            "n_timed": r["n_timed"],
            "p50_us": _percentile(per_us, 0.50),
            "p99_us": _percentile(per_us, 0.99),
        })

    agg_us = sorted(all_latencies_us.tolist())
    aggregate = {
        "p50_us": _percentile(agg_us, 0.50),
        "p90_us": _percentile(agg_us, 0.90),
        "p99_us": _percentile(agg_us, 0.99),
        "mean_us": statistics.fmean(agg_us),
        "std_us": statistics.pstdev(agg_us),
    }

    total_predictions = int(all_outputs.size)
    all_finite = bool(np.isfinite(all_outputs).all())

    result = {
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "protocol": args.protocol,
        "n_warmup": args.n_warmup,
        "n_timed": args.n_runs,
        "batch_size": args.batch,
        "nmi": args.nmi,
        "wall_clock_s": wall_clock_s,
        "throughput_inf_per_sec": (args.nmi * args.n_runs) / wall_clock_s,
        "per_instance": per_instance,
        "aggregate": aggregate,
        "output_stats": {
            "n_predictions": total_predictions,
            "min": float(all_outputs.min()),
            "max": float(all_outputs.max()),
            "mean": float(all_outputs.mean()),
            "std": float(all_outputs.std()),
            "all_finite": all_finite,
        },
        "model_path": str(args.model_path) if args.model_path else None,
        "predictor": args.predictor,
        "pin_cpu": args.pin_cpu if pin_cpu_applied else None,
    }
    if args.protocol == "tacana":
        result["tacana_stride"] = args.tacana_stride
    if args.predictor == "ulysses_stub":
        result["kirk_mode"] = args.ulysses_kirk_mode

    pretty = json.dumps(result, indent=2)
    print(pretty)
    if args.output_json:
        Path(args.output_json).write_text(pretty + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model file (required for --predictor onnx)",
    )
    parser.add_argument(
        "--predictor",
        choices=PREDICTOR_CHOICES,
        default="onnx",
        help="Predictor backend (default: onnx)",
    )
    parser.add_argument("--n-warmup", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ulysses-k",
        type=int,
        default=128,
        help="Kirk-stub projection width (only used with --predictor ulysses_stub)",
    )
    parser.add_argument(
        "--ulysses-m",
        type=int,
        default=None,
        help="Hankel rows m; defaults to T//2 (only used with --predictor ulysses_stub)",
    )
    parser.add_argument(
        "--ulysses-kirk-mode",
        choices=("linear_stub", "identity"),
        default="linear_stub",
        help="Stage-2 mode for ulysses_stub predictor (default: linear_stub)",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write result JSON")
    parser.add_argument(
        "--pin-cpu",
        type=int,
        default=None,
        help="Pin this process to the given CPU id before warmup (Linux only)",
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help=(
            "Run a second predictor (untimed) on the same inputs the primary "
            "saw and report cross-predictor agreement metrics. Accepts "
            "'ulysses_stub_identity', 'ulysses_stub_linear_stub', or a path "
            "to an .onnx model."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("sumaco", "tacana"),
        default="sumaco",
        help=(
            "STAC-ML protocol to drive: 'sumaco' (independent fresh-random "
            "window per call) or 'tacana' (sliding window over a longer "
            "fresh-random stream). See the module docstring for protocol "
            "shapes. Default: sumaco."
        ),
    )
    parser.add_argument(
        "--tacana-stride",
        type=int,
        default=1,
        help=(
            "Per-call stride (in timesteps) along the Tacana stream. "
            "stride=1 rolls one new timestep into the window per call; "
            "stride>=T degenerates to non-overlapping windows. Ignored "
            "when --protocol sumaco. Default: 1."
        ),
    )
    parser.add_argument(
        "--nmi",
        type=int,
        default=1,
        help=(
            "Number of Model Instances to run in parallel. nmi=1 (default) "
            "runs in-process and the JSON output shape is unchanged. "
            "nmi>1 spawns N child processes (multiprocessing.spawn), each "
            "running the chosen protocol with its own seed; the parent "
            "reports per-instance and aggregate latency plus throughput. "
            "Per-instance latency typically degrades once N exceeds the "
            "physical core count; aggregate throughput scales sub-linearly."
        ),
    )
    args = parser.parse_args()
    if args.tacana_stride < 1:
        parser.error(f"--tacana-stride must be >= 1, got {args.tacana_stride}")
    if args.nmi < 1:
        parser.error(f"--nmi must be >= 1, got {args.nmi}")
    if args.nmi > 1 and args.compare_with is not None:
        parser.error(
            "--compare-with is not supported with --nmi > 1; cross-predictor "
            "agreement is a single-instance metric."
        )

    pin_cpu_applied = False
    if args.pin_cpu is not None:
        pin_cpu_applied = _apply_cpu_pin(args.pin_cpu)

    if args.nmi > 1:
        return _run_nmi(args, pin_cpu_applied)

    if args.predictor == "onnx":
        if not args.model_path:
            parser.error("--model-path is required when --predictor onnx")
        predictor: STACPredictor = ONNXPredictor(args.model_path)
    elif args.predictor == "ulysses_stub":
        from ulysses_predictor import (
            IdentityKirk,
            LinearStubKirk,
            UlyssesPredictor,
        )

        # Translate the CLI mode string into a concrete KirkCore. Future Kirk
        # implementations can be plugged in by extending the KirkCore ABC and
        # routing a new mode value here (or, if the mode space gets unwieldy,
        # via a dedicated --ulysses-kirk-class flag — not added now to keep
        # the surface minimal).
        if args.ulysses_kirk_mode == "linear_stub":
            kirk = LinearStubKirk(k=args.ulysses_k, seed=args.seed)
        elif args.ulysses_kirk_mode == "identity":
            kirk = IdentityKirk()
        else:  # pragma: no cover — argparse choices guards this
            parser.error(f"unknown kirk mode: {args.ulysses_kirk_mode}")

        predictor = UlyssesPredictor(
            m=args.ulysses_m,
            kirk=kirk,
            readout_seed=args.seed,
        )
    else:  # pragma: no cover — argparse choices guards this
        parser.error(f"unknown predictor: {args.predictor}")

    compare_with: STACPredictor | None = None
    if args.compare_with is not None:
        primary_T, primary_F = predictor.input_shape
        spec = args.compare_with
        if spec == "ulysses_stub_identity":
            from ulysses_predictor import IdentityKirk, UlyssesPredictor

            compare_with = UlyssesPredictor(
                t=primary_T,
                f=primary_F,
                m=args.ulysses_m,
                kirk=IdentityKirk(),
                readout_seed=args.seed,
            )
        elif spec == "ulysses_stub_linear_stub":
            from ulysses_predictor import LinearStubKirk, UlyssesPredictor

            compare_with = UlyssesPredictor(
                t=primary_T,
                f=primary_F,
                m=args.ulysses_m,
                kirk=LinearStubKirk(k=args.ulysses_k, seed=args.seed),
                readout_seed=args.seed,
            )
        else:
            # Treat anything else as a path to an .onnx model.
            compare_with = ONNXPredictor(spec)

    result = run_protocol(
        predictor,
        n_warmup=args.n_warmup,
        n_timed=args.n_runs,
        batch=args.batch,
        seed=args.seed,
        compare_with=compare_with,
        protocol=args.protocol,
        tacana_stride=args.tacana_stride,
    )
    result["model_path"] = str(args.model_path) if args.model_path else None
    result["predictor"] = args.predictor
    result["pin_cpu"] = args.pin_cpu if pin_cpu_applied else None
    if args.predictor == "ulysses_stub":
        result["kirk_mode"] = args.ulysses_kirk_mode
    if args.compare_with is not None:
        result["compare_with"] = args.compare_with

    pretty = json.dumps(result, indent=2)
    print(pretty)
    if args.output_json:
        Path(args.output_json).write_text(pretty + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
