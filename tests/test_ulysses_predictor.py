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

from stac_sumaco_driver import (  # noqa: E402
    ONNXPredictor,
    run_one_instance,
    run_protocol,
    run_sumaco,
)
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


def test_run_sumaco_with_agreement() -> None:
    # Baseline: no compare_with — agreement_stats must NOT be present.
    p = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=0)
    result_plain = run_sumaco(p, n_warmup=5, n_timed=20, batch=2, seed=0)
    assert "agreement_stats" not in result_plain, (
        "agreement_stats must not be present when compare_with is None"
    )

    # Self-comparison: same predictor on both sides — every metric trivial.
    p2 = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=0)
    result_cmp = run_sumaco(
        p2, n_warmup=5, n_timed=20, batch=2, seed=0, compare_with=p2
    )
    assert "agreement_stats" in result_cmp
    a = result_cmp["agreement_stats"]
    expected_keys = {
        "n_pairs",
        "sign_agreement_pct",
        "pearson_r",
        "spearman_rho",
        "mean_abs_diff",
        "max_abs_diff",
        "mean_abs_diff_z",
        "max_abs_diff_z",
        "primary_output_stats",
        "compared_output_stats",
    }
    assert set(a.keys()) == expected_keys, (
        f"agreement_stats keys mismatch: got {set(a.keys())}, "
        f"expected {expected_keys}"
    )
    assert a["n_pairs"] == 40, f"n_pairs={a['n_pairs']}"
    assert a["sign_agreement_pct"] == 100.0, (
        f"self-comparison sign_agreement_pct={a['sign_agreement_pct']}"
    )
    assert a["pearson_r"] == 1.0, f"self-comparison pearson_r={a['pearson_r']}"
    assert a["spearman_rho"] == 1.0, (
        f"self-comparison spearman_rho={a['spearman_rho']}"
    )
    assert a["mean_abs_diff"] == 0.0, (
        f"self-comparison mean_abs_diff={a['mean_abs_diff']}"
    )
    assert a["max_abs_diff"] == 0.0, (
        f"self-comparison max_abs_diff={a['max_abs_diff']}"
    )
    assert a["mean_abs_diff_z"] == 0.0, (
        f"self-comparison mean_abs_diff_z={a['mean_abs_diff_z']}"
    )
    assert a["max_abs_diff_z"] == 0.0, (
        f"self-comparison max_abs_diff_z={a['max_abs_diff_z']}"
    )

    # Both nested output-stats blocks have the four expected keys.
    nested_keys = {"min", "max", "mean", "std"}
    assert set(a["primary_output_stats"].keys()) == nested_keys
    assert set(a["compared_output_stats"].keys()) == nested_keys


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


def test_run_protocol_sumaco() -> None:
    p = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=0)
    result = run_protocol(
        p, n_warmup=10, n_timed=50, batch=1, seed=0, protocol="sumaco"
    )
    assert result["protocol"] == "sumaco", f"got protocol={result['protocol']!r}"
    assert "tacana_stride" not in result, (
        "tacana_stride must not appear under the sumaco protocol"
    )
    assert result["output_stats"]["n_predictions"] == 50
    assert result["output_stats"]["all_finite"] is True


def test_run_protocol_tacana() -> None:
    p = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=0)
    result = run_protocol(
        p, n_warmup=10, n_timed=50, batch=1, seed=0, protocol="tacana"
    )
    assert result["protocol"] == "tacana", f"got protocol={result['protocol']!r}"
    assert result["tacana_stride"] == 1, (
        f"default tacana_stride should be 1, got {result['tacana_stride']}"
    )
    assert result["output_stats"]["all_finite"] is True
    assert result["output_stats"]["n_predictions"] == 50


def test_tacana_stride() -> None:
    p = UlyssesPredictor(kirk=IdentityKirk(), readout_seed=0)
    result = run_protocol(
        p,
        n_warmup=10,
        n_timed=50,
        batch=1,
        seed=0,
        protocol="tacana",
        tacana_stride=5,
    )
    assert result["protocol"] == "tacana"
    assert result["tacana_stride"] == 5
    assert result["output_stats"]["n_predictions"] == 50
    assert result["output_stats"]["all_finite"] is True


def test_run_one_instance_function_signature() -> None:
    """run_one_instance must be top-level (picklable) and accept the args-dict shape."""
    import stac_sumaco_driver as drv

    # Module-scope check: pickle uses qualname, which must equal the bare name.
    assert run_one_instance.__module__ == drv.__name__, (
        f"run_one_instance must live in {drv.__name__!r}, "
        f"got {run_one_instance.__module__!r}"
    )
    assert run_one_instance.__qualname__ == "run_one_instance", (
        "run_one_instance must be a module-scope function (not nested) "
        "so multiprocessing.spawn can pickle it"
    )

    args_dict = {
        "instance_id": 0,
        "predictor": "ulysses_stub",
        "model_path": None,
        "ulysses_kirk_mode": "identity",
        "ulysses_k": 128,
        "ulysses_m": None,
        "n_warmup": 2,
        "n_timed": 5,
        "batch": 1,
        "seed": 0,
        "protocol": "sumaco",
        "tacana_stride": 1,
    }
    r = run_one_instance(args_dict)
    for key in ("latencies_ns", "output_stats", "instance_id", "seed", "n_timed"):
        assert key in r, f"run_one_instance result missing {key!r}: keys={list(r)}"
    assert r["n_timed"] == 5
    assert len(r["latencies_ns"]) == 5
    assert r["output_stats"]["all_finite"] is True
    assert r["output_stats"]["n_predictions"] == 5


def test_nmi_smoke() -> None:
    """End-to-end CLI smoke at --nmi 2; verifies aggregate JSON shape."""
    import json
    import subprocess

    try:
        import multiprocessing
        multiprocessing.get_context("spawn")
    except (ImportError, ValueError):
        print("test_nmi_smoke: spawn context unavailable, skipping")
        return

    driver = REPO_ROOT / "scripts" / "stac_sumaco_driver.py"
    out_path = "/tmp/test_nmi_smoke.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--predictor", "ulysses_stub",
            "--ulysses-kirk-mode", "identity",
            "--nmi", "2",
            "--n-warmup", "5",
            "--n-runs", "20",
            "--output-json", out_path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"--nmi 2 driver exited {proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    with open(out_path) as f:
        result = json.load(f)
    assert result["nmi"] == 2, f"got nmi={result.get('nmi')}"
    assert isinstance(result["per_instance"], list)
    assert len(result["per_instance"]) == 2, (
        f"expected 2 per-instance entries, got {len(result['per_instance'])}"
    )
    assert "aggregate" in result, f"missing aggregate block: {list(result)}"
    for k in ("p50_us", "p90_us", "p99_us", "mean_us", "std_us"):
        assert k in result["aggregate"], f"aggregate missing {k}"
    assert "throughput_inf_per_sec" in result
    assert "wall_clock_s" in result
    assert result["output_stats"]["n_predictions"] == 2 * 20
    # Different seeds across instances: instance_id 0 -> seed 0, 1 -> seed 1
    seeds = [pi["seed"] for pi in result["per_instance"]]
    assert len(set(seeds)) == 2, f"per-instance seeds must differ, got {seeds}"


def main() -> int:
    test_predict_shape()
    test_determinism()
    test_seed_reproducibility()
    test_identity_mode()
    test_kirkcore_abc_compliance()
    test_identity_kirk()
    test_linear_stub_kirk()
    test_run_sumaco_output_stats()
    test_run_sumaco_with_agreement()
    test_run_protocol_sumaco()
    test_run_protocol_tacana()
    test_tacana_stride()
    test_run_one_instance_function_signature()
    test_nmi_smoke()
    print("All UlyssesPredictor tests pass ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
