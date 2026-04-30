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

Note: STAC's own published vendor results use a similar but more sophisticated
protocol (CPU pinning, exclusive-GPU disciplines, formal warm-up regimes). The
numbers produced here are *relative*; they are not directly comparable to a
STAC-audited result.
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


def run_sumaco(
    predictor: STACPredictor,
    n_warmup: int,
    n_timed: int,
    batch: int,
    seed: int = 0,
) -> dict:
    """Drive `predictor` under the Sumaco protocol and return a result dict.

    Each call (warmup and timed) sees a fresh `(batch, T, F) float32` tensor
    drawn from the same RNG. Only the `predict` call is timed.
    """
    T, F = predictor.input_shape
    rng = np.random.default_rng(seed)

    for _ in range(n_warmup):
        x = rng.standard_normal((batch, T, F), dtype=np.float32)
        predictor.predict(x)

    samples_ns: list[int] = []
    for _ in range(n_timed):
        x = rng.standard_normal((batch, T, F), dtype=np.float32)
        t0 = time.perf_counter_ns()
        predictor.predict(x)
        samples_ns.append(time.perf_counter_ns() - t0)

    samples_us = sorted(s / 1000.0 for s in samples_ns)
    return {
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
    }


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
    args = parser.parse_args()

    if args.predictor == "onnx":
        if not args.model_path:
            parser.error("--model-path is required when --predictor onnx")
        predictor: STACPredictor = ONNXPredictor(args.model_path)
    elif args.predictor == "ulysses_stub":
        from ulysses_predictor import UlyssesPredictor

        predictor = UlyssesPredictor(
            k=args.ulysses_k,
            m=args.ulysses_m,
            seed=args.seed,
            kirk_mode=args.ulysses_kirk_mode,
        )
    else:  # pragma: no cover — argparse choices guards this
        parser.error(f"unknown predictor: {args.predictor}")

    result = run_sumaco(
        predictor,
        n_warmup=args.n_warmup,
        n_timed=args.n_runs,
        batch=args.batch,
        seed=args.seed,
    )
    result["model_path"] = str(args.model_path) if args.model_path else None
    result["predictor"] = args.predictor
    if args.predictor == "ulysses_stub":
        result["kirk_mode"] = args.ulysses_kirk_mode

    pretty = json.dumps(result, indent=2)
    print(pretty)
    if args.output_json:
        Path(args.output_json).write_text(pretty + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
