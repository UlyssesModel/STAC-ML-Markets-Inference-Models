#!/usr/bin/env python3
"""Assertion-based tests for `scripts.ulysses_predictor.UlyssesPredictor`.

Run directly:

    python tests/test_ulysses_predictor.py

No pytest dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ulysses_predictor import UlyssesPredictor  # noqa: E402


def test_predict_shape() -> None:
    p = UlyssesPredictor()
    rng = np.random.default_rng(0)
    for B in (1, 4):
        x = rng.standard_normal((B, 50, 100), dtype=np.float32)
        y = p.predict(x)
        assert y.shape == (B, 1), f"B={B}: got shape {y.shape}"
        assert y.dtype == np.float32, f"B={B}: got dtype {y.dtype}"


def test_determinism() -> None:
    p = UlyssesPredictor(seed=7)
    x = np.random.default_rng(0).standard_normal((2, 50, 100), dtype=np.float32)
    y1 = p.predict(x)
    y2 = p.predict(x)
    assert np.array_equal(y1, y2), "predict not deterministic across calls"


def test_seed_reproducibility() -> None:
    p_a = UlyssesPredictor(seed=42)
    p_b = UlyssesPredictor(seed=42)
    p_c = UlyssesPredictor(seed=43)
    x = np.random.default_rng(0).standard_normal((1, 50, 100), dtype=np.float32)
    y_a = p_a.predict(x)
    y_b = p_b.predict(x)
    y_c = p_c.predict(x)
    assert np.array_equal(y_a, y_b), "same seed must produce identical outputs"
    assert not np.array_equal(y_a, y_c), "different seeds must produce different outputs"


def main() -> int:
    test_predict_shape()
    test_determinism()
    test_seed_reproducibility()
    print("All UlyssesPredictor tests pass ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
