#!/usr/bin/env python3
"""Synthetic-input shape verifier for the RS-40 model bundle.

For every `.onnx` file (loaded via onnxruntime) and every `.keras64/` SavedModel
(loaded via tf.saved_model.load) in the repo root, push a random tensor of the
declared input shape through and assert the output shape matches the metadata
recorded in the model cards.

Run from the repository root:

    python scripts/verify_io.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH = 2  # exercise the dynamic batch dimension with N>1

# Declared shapes per model card. Each entry maps model name -> (T, F) for LSTMs
# (input is rank 3 [N, T, F]) or (F,) for GBTs (input is rank 2 [N, F]).
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
EXPECTED_OUTPUT_SHAPE = (BATCH, 1)


def _onnx_input_meta(sess) -> tuple[str, str]:
    inp = sess.get_inputs()[0]
    return inp.name, inp.type


def verify_onnx() -> list[tuple[str, bool, str]]:
    import onnxruntime as ort

    results: list[tuple[str, bool, str]] = []
    onnx_files = sorted(REPO_ROOT.glob("*.onnx"))
    for path in onnx_files:
        name = path.stem
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            input_name, _ = _onnx_input_meta(sess)
            if name in LSTM_INPUT_SHAPES:
                T, F = LSTM_INPUT_SHAPES[name]
                x = np.random.randn(BATCH, T, F).astype(np.float32)
            elif name in GBT_INPUT_FEATURES:
                F = GBT_INPUT_FEATURES[name]
                x = np.random.randn(BATCH, F).astype(np.float32)
            else:
                results.append((name, False, f"unknown model name (no expected shape)"))
                continue

            (y,) = sess.run(None, {input_name: x})
            ok = tuple(y.shape) == EXPECTED_OUTPUT_SHAPE
            msg = f"in={x.shape} out={y.shape} expected={EXPECTED_OUTPUT_SHAPE}"
            results.append((name + ".onnx", ok, msg))
        except Exception as exc:
            results.append((name + ".onnx", False, f"FAILED: {exc!r}"))
    return results


def verify_keras64() -> list[tuple[str, bool, str]]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    results: list[tuple[str, bool, str]] = []
    sm_dirs = sorted(p for p in REPO_ROOT.iterdir() if p.is_dir() and p.suffix == ".keras64")
    for path in sm_dirs:
        name = path.stem
        try:
            sm = tf.saved_model.load(str(path))
            sig = sm.signatures["serving_default"]
            # The single positional kwarg is the model's input tensor name.
            (kw_inputs,) = (sig.structured_input_signature[1],)
            (input_name, spec) = next(iter(kw_inputs.items()))
            dtype = spec.dtype  # tf.float64 for these models

            if name not in LSTM_INPUT_SHAPES:
                results.append((name + ".keras64", False, "unknown model name (no expected shape)"))
                continue
            T, F = LSTM_INPUT_SHAPES[name]
            x = tf.constant(np.random.randn(BATCH, T, F), dtype=dtype)
            outputs = sig(**{input_name: x})
            (out_name, y) = next(iter(outputs.items()))
            ok = tuple(y.shape) == EXPECTED_OUTPUT_SHAPE
            msg = f"in={tuple(x.shape)} dtype={dtype.name} out={tuple(y.shape)} expected={EXPECTED_OUTPUT_SHAPE}"
            results.append((name + ".keras64", ok, msg))
        except Exception as exc:
            results.append((name + ".keras64", False, f"FAILED: {exc!r}"))
    return results


def main() -> int:
    print(f"verifying models under {REPO_ROOT}")
    print()
    print("=== ONNX (onnxruntime) ===")
    onnx_results = verify_onnx()
    for name, ok, msg in onnx_results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name:32s} {msg}")
    print()
    print("=== Keras SavedModel (tf.saved_model.load) ===")
    keras_results = verify_keras64()
    for name, ok, msg in keras_results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name:32s} {msg}")

    all_results = onnx_results + keras_results
    failed = [name for name, ok, _ in all_results if not ok]
    print()
    if failed:
        print(f"FAILED: {len(failed)} / {len(all_results)} — {failed}")
        return 1
    print(f"OK: {len(all_results)} / {len(all_results)} models match declared shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
