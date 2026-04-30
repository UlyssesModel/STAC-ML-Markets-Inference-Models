#!/usr/bin/env python3
"""STAC-aligned Sumaco-protocol latency driver with a pluggable model interface.

STAC's Sumaco suite is the event-triggered single-inference latency benchmark
in the STAC-ML Markets family: each inference receives a *fixed unique window*
of input features, and the timed quantity is per-call inference latency, not
streaming throughput. This driver implements that protocol against any model
that conforms to the `(B, T, F) float32 -> (B, 1) float32` contract shared by
this repo's `LSTM_*` artifacts.

The driver is deliberately separated from the model behind a `STACPredictor`
ABC so a Ulysses-based replacement can drop in next to the ONNX baseline
without touching the timing harness.

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
Tsupply/Tresult timestamps, NMI parallelism, out-of-order result handling,
or hard-realtime scheduling. Numbers produced here remain relative; they
are not directly comparable to a STAC-audited result.

Cross-predictor agreement
-------------------------
The `--compare-with` flag enables an *untimed* secondary predictor that
runs on byte-identical input tensors to the primary. After both
predictors have produced their output buffers, the driver computes
agreement metrics in pure numpy (no scipy): sign-agreement percentage,
Pearson r, Spearman rho (rank correlation via `np.argsort(np.argsort(x))`),
and mean / max absolute difference. Results are surfaced under
`agreement_stats` in the JSON, alongside summary stats for both
predictors' output distributions so it's clear whether the agreement
number is conditioned on similar or wildly different ranges.

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
    return {
        "n_pairs": int(a_flat.size),
        "sign_agreement_pct": float((sign_a == sign_b).mean() * 100.0),
        "pearson_r": float(np.corrcoef(a_flat, b_flat)[0, 1]),
        "spearman_rho": float(np.corrcoef(_ranks(a_flat), _ranks(b_flat))[0, 1]),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
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


def run_sumaco(
    predictor: STACPredictor,
    n_warmup: int,
    n_timed: int,
    batch: int,
    seed: int = 0,
    compare_with: STACPredictor | None = None,
) -> dict:
    """Drive `predictor` under the Sumaco protocol and return a result dict.

    Inputs for the timed loop are pre-generated as a single
    `(n_timed, batch, T, F) float32` array. Latency samples are written into
    a pre-allocated int64 buffer. Predictions are written into a
    pre-allocated `(n_timed, batch, 1) float32` buffer. The timed window
    contains only the `predict()` call and one int64 latency assignment;
    the prediction store happens outside it. After the timed loop a cheap
    validation phase computes prediction-distribution stats and confirms
    every sample is finite.

    If `compare_with` is provided, it is run (untimed) on the same
    pre-generated input tensors the primary saw, into a parallel
    pre-allocated output buffer; the result dict gains an
    `agreement_stats` block with cross-predictor metrics (sign agreement,
    Pearson, Spearman, mean / max abs diff). The two predictors must
    have matching `input_shape`. See module docstring for the full
    methodology.
    """
    T, F = predictor.input_shape
    rng = np.random.default_rng(seed)

    for _ in range(n_warmup):
        x = rng.standard_normal((batch, T, F), dtype=np.float32)
        predictor.predict(x)

    inputs = rng.standard_normal((n_timed, batch, T, F), dtype=np.float32)
    latencies_ns = np.empty(n_timed, dtype=np.int64)
    outputs = np.empty((n_timed, batch, 1), dtype=np.float32)

    for i in range(n_timed):
        x = inputs[i]
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
            outputs_b[i] = compare_with.predict(inputs[i])
        if not np.isfinite(outputs_b).all():
            n_bad = int((~np.isfinite(outputs_b)).sum())
            raise AssertionError(
                f"compare_with prediction buffer contains {n_bad} "
                f"non-finite value(s); cannot compute agreement stats"
            )
        result["agreement_stats"] = _compute_agreement_stats(outputs, outputs_b)

    return result


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
    args = parser.parse_args()

    pin_cpu_applied = False
    if args.pin_cpu is not None:
        pin_cpu_applied = _apply_cpu_pin(args.pin_cpu)

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

    result = run_sumaco(
        predictor,
        n_warmup=args.n_warmup,
        n_timed=args.n_runs,
        batch=args.batch,
        seed=args.seed,
        compare_with=compare_with,
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
