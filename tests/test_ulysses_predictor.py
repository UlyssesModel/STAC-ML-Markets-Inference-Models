#!/usr/bin/env python3
"""Assertion-based tests for `scripts.ulysses_predictor`.

Covers the `UlyssesPredictor` end-to-end as well as the `KirkCore` ABC and
its two reference subclasses (`IdentityKirk`, `LinearStubKirk`).

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

from stac_sumaco_driver import ONNXPredictor, run_sumaco  # noqa: E402
from ulysses_predictor import (  # noqa: E402
    IdentityKirk,
    KirkCore,
    LinearStubKirk,
    UlyssesPredictor,
)


def test_predict_shape() -> None:
    p = UlyssesPredictor()
    rng = np.random.default_rng(0)
    for B in (1, 4):
        x = rng.standard_normal((B, 50, 100), dtype=np.float32)
        y = p.predict(x)
        assert y.shape == (B, 1), f"B={B}: got shape {y.shape}"
        assert y.dtype == np.float32, f"B={B}: got dtype {y.dtype}"


def test_determinism() -> None:
    p = UlyssesPredictor(kirk=LinearStubKirk(seed=7), readout_seed=7)
    x = np.random.default_rng(0).standard_normal((2, 50, 100), dtype=np.float32)
    y1 = p.predict(x)
    y2 = p.predict(x)
    assert np.array_equal(y1, y2), "predict not deterministic across calls"


def test_seed_reproducibility() -> None:
    p_a = UlyssesPredictor(kirk=LinearStubKirk(seed=42), readout_seed=42)
    p_b = UlyssesPredictor(kirk=LinearStubKirk(seed=42), readout_seed=42)
    p_c = UlyssesPredictor(kirk=LinearStubKirk(seed=43), readout_seed=43)
    x = np.random.default_rng(0).standard_normal((1, 50, 100), dtype=np.float32)
    y_a = p_a.predict(x)
    y_b = p_b.predict(x)
    y_c = p_c.predict(x)
    assert np.array_equal(y_a, y_b), "same seed must produce identical outputs"
    assert not np.array_equal(y_a, y_c), "different seeds must produce different outputs"


def test_identity_mode() -> None:
    p = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=11)
    assert isinstance(p.kirk, IdentityKirk)
    rng = np.random.default_rng(0)
    for B in (1, 4):
        x = rng.standard_normal((B, 50, 100), dtype=np.float32)
        y = p.predict(x)
        assert y.shape == (B, 1), f"identity B={B}: got shape {y.shape}"
        assert y.dtype == np.float32, f"identity B={B}: got dtype {y.dtype}"

    p_det = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=11)
    x = np.random.default_rng(0).standard_normal((2, 50, 100), dtype=np.float32)
    y1 = p_det.predict(x)
    y2 = p_det.predict(x)
    assert np.array_equal(y1, y2), "identity mode must be deterministic across calls"


def test_kirkcore_abc_compliance() -> None:
    class IncompleteKirk(KirkCore):
        pass

    try:
        IncompleteKirk()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError(
            "instantiating a KirkCore subclass without transform/output_shape "
            "should raise TypeError"
        )

    class TransformOnly(KirkCore):
        def transform(self, hankel_batch: np.ndarray) -> np.ndarray:
            return hankel_batch

    try:
        TransformOnly()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError(
            "missing output_shape should also block instantiation"
        )


def test_identity_kirk() -> None:
    k = IdentityKirk()
    F, m, n = 100, 25, 26
    assert k.output_shape(F, m, n) == (F * m * n,)

    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, F, m, n), dtype=np.float32)
    z = k.transform(h)
    assert z.shape == (3, F * m * n), f"got {z.shape}"
    assert z.dtype == np.float32, f"got dtype {z.dtype}"


def _assert_output_stats(result: dict, expected_n: int) -> None:
    assert "output_stats" in result, "result missing output_stats"
    stats = result["output_stats"]
    expected_keys = {"n_predictions", "min", "max", "mean", "std", "all_finite"}
    assert set(stats.keys()) == expected_keys, (
        f"output_stats keys mismatch: got {set(stats.keys())}, "
        f"expected {expected_keys}"
    )
    assert stats["n_predictions"] == expected_n, (
        f"n_predictions={stats['n_predictions']}, expected {expected_n}"
    )
    assert stats["all_finite"] is True, "all_finite should be True"


def test_run_sumaco_output_stats() -> None:
    p_ulysses = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=0)
    result = run_sumaco(p_ulysses, n_warmup=10, n_timed=50, batch=2, seed=0)
    _assert_output_stats(result, expected_n=100)

    onnx_path = REPO_ROOT / "LSTM_A.onnx"
    p_onnx = ONNXPredictor(onnx_path)
    result_onnx = run_sumaco(p_onnx, n_warmup=10, n_timed=50, batch=2, seed=0)
    _assert_output_stats(result_onnx, expected_n=100)


def test_linear_stub_kirk() -> None:
    k = LinearStubKirk(k=64, seed=3)
    F, m, n = 100, 25, 26
    assert k.output_shape(F, m, n) == (64,)

    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, F, m, n), dtype=np.float32)
    z = k.transform(h)
    assert z.shape == (3, 64), f"got {z.shape}"
    assert z.dtype == np.float32, f"got dtype {z.dtype}"

    # Same seed -> same kernel -> same projection on the same input.
    k2 = LinearStubKirk(k=64, seed=3)
    z2 = k2.transform(h)
    assert np.array_equal(z, z2), "same-seed LinearStubKirk must be deterministic"


def main() -> int:
    test_predict_shape()
    test_determinism()
    test_seed_reproducibility()
    test_identity_mode()
    test_kirkcore_abc_compliance()
    test_identity_kirk()
    test_linear_stub_kirk()
    test_run_sumaco_output_stats()
    print("All UlyssesPredictor tests pass ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
