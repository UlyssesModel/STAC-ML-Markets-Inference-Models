#!/usr/bin/env python3
"""Single-call latency benchmark for the RS-40 model bundle.

For every `.onnx` file (loaded via onnxruntime, CPU provider) and every
`.keras64/` SavedModel (loaded via tf.saved_model.load + concrete signature)
in the repo root, push random batch-1 inputs of the declared shape through
100 warmup calls then 1000 timed calls, and report per-model p50, p99,
mean, and std of single-call latency in microseconds.

Also writes the full results to bench_results.json at the repo root.

Run from the repository root:

    python scripts/benchmark_io.py
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH = 1
WARMUP_RUNS = 100
TIMED_RUNS_ONNX = 1000
TIMED_RUNS_KERAS = 200

LSTM_INPUT_SHAPES: dict[str, tuple[int, int]] = {
    "LSTM_A": (50, 100),
    "LSTM_B": (100, 250),
    "LSTM_C": (100, 500),
    "LSTM_Null_A": (50, 100),
    "LSTM_Null_B": (100, 250),
    "LSTM_Null_C": (100, 500),
}
GBT_INPUT_FEATURES: dict[str, int] = {
    "GBT_A": 60,
    "GBT_B": 125,
    "GBT_C": 1000,
    "GBT_Null_A": 60,
    "GBT_Null_B": 125,
    "GBT_Null_C": 1000,
}

# Sort key: LSTM class first then GBT, lens A<B<C, trained before null.
def _sort_key(model_name: str) -> tuple[int, int, int]:
    base = model_name.split(".")[0]
    cls = 0 if base.startswith("LSTM") else 1
    is_null = 1 if "_Null_" in base else 0
    lens = base.rsplit("_", 1)[-1]
    lens_idx = {"A": 0, "B": 1, "C": 2}[lens]
    runtime_idx = 0 if model_name.endswith(".onnx") else 1
    return (cls, is_null, lens_idx, runtime_idx)


def _stats_us(samples_ns: list[int]) -> dict[str, float]:
    samples_us = [s / 1000.0 for s in samples_ns]
    samples_us_sorted = sorted(samples_us)
    n = len(samples_us_sorted)
    p50 = samples_us_sorted[n // 2]
    p99 = samples_us_sorted[min(n - 1, int(round(0.99 * n)) - 1)]
    return {
        "p50_us": p50,
        "p99_us": p99,
        "mean_us": statistics.fmean(samples_us),
        "std_us": statistics.pstdev(samples_us),
    }


def _input_for(name: str, dtype) -> np.ndarray:
    if name in LSTM_INPUT_SHAPES:
        T, F = LSTM_INPUT_SHAPES[name]
        return np.random.randn(BATCH, T, F).astype(dtype)
    if name in GBT_INPUT_FEATURES:
        F = GBT_INPUT_FEATURES[name]
        return np.random.randn(BATCH, F).astype(dtype)
    raise KeyError(f"unknown model name: {name}")


def bench_onnx() -> list[dict]:
    import onnxruntime as ort

    results: list[dict] = []
    for path in sorted(REPO_ROOT.glob("*.onnx")):
        name = path.stem
        if name not in LSTM_INPUT_SHAPES and name not in GBT_INPUT_FEATURES:
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        x = _input_for(name, np.float32)
        feed = {input_name: x}

        for _ in range(WARMUP_RUNS):
            sess.run(None, feed)

        samples_ns: list[int] = []
        for _ in range(TIMED_RUNS_ONNX):
            t0 = time.perf_counter_ns()
            sess.run(None, feed)
            samples_ns.append(time.perf_counter_ns() - t0)

        entry = {
            "name": name + ".onnx",
            "runtime": "onnxruntime",
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS_ONNX,
            **_stats_us(samples_ns),
        }
        results.append(entry)
    return results


def bench_keras64() -> list[dict]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    results: list[dict] = []
    sm_dirs = sorted(p for p in REPO_ROOT.iterdir() if p.is_dir() and p.suffix == ".keras64")
    for path in sm_dirs:
        name = path.stem
        if name not in LSTM_INPUT_SHAPES:
            continue
        sm = tf.saved_model.load(str(path))
        sig = sm.signatures["serving_default"]
        kw_inputs = sig.structured_input_signature[1]
        input_name, spec = next(iter(kw_inputs.items()))
        np_dtype = np.float64 if spec.dtype == tf.float64 else np.float32
        x_np = _input_for(name, np_dtype)
        x = tf.constant(x_np, dtype=spec.dtype)

        for _ in range(WARMUP_RUNS):
            sig(**{input_name: x})

        samples_ns: list[int] = []
        for _ in range(TIMED_RUNS_KERAS):
            t0 = time.perf_counter_ns()
            sig(**{input_name: x})
            samples_ns.append(time.perf_counter_ns() - t0)

        entry = {
            "name": name + ".keras64",
            "runtime": "tf.saved_model",
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS_KERAS,
            **_stats_us(samples_ns),
        }
        results.append(entry)
    return results


def main() -> int:
    np.random.seed(0)
    print(f"benchmarking models under {REPO_ROOT}")
    print(f"  batch={BATCH}, warmup={WARMUP_RUNS}, "
          f"timed_onnx={TIMED_RUNS_ONNX}, timed_keras={TIMED_RUNS_KERAS}")
    print()

    onnx_results = bench_onnx()
    keras_results = bench_keras64()
    all_results = sorted(onnx_results + keras_results, key=lambda r: _sort_key(r["name"]))

    header = f"{'model':28s} {'runtime':16s} {'p50_us':>10s} {'p99_us':>10s} {'mean_us':>10s} {'std_us':>10s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['name']:28s} {r['runtime']:16s} "
            f"{r['p50_us']:10.2f} {r['p99_us']:10.2f} "
            f"{r['mean_us']:10.2f} {r['std_us']:10.2f}"
        )

    payload = {
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "machine_type": "e2-standard-4",
        },
        "results": all_results,
    }
    out_path = REPO_ROOT / "bench_results.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
