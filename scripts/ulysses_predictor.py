#!/usr/bin/env python3
"""Placeholder-Kirk Ulysses predictor — stub for `rs40-swap-spec.md` §5 Stage 2.

This is **not real Ulysses** and **not real Kirk**. It wires the full
`(B, 50, 100) float32 → (B, 1) float32` contract end-to-end with:

  Stage 1 (Hankel adapter): `build_hankel` from `hankel_adapter.py`.
  Stage 2 (Kirk core):      random-init linear projection `(F·m·n) → K`.
  Stage 3 (Scalar readout): random-init linear projection `K → 1`.

The point of this stub is to **bound adapter + readout overhead**
independently of whatever real Kirk eventually does. Latency measured
against this predictor is "everything except the Kirk-specific compute" —
once Jarett's Kirk contract lands, only the Stage-2 matmul changes; the
Stage-1 Hankel shape and Stage-3 readout shape are already pinned by §4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hankel_adapter import build_hankel
from stac_sumaco_driver import STACPredictor


class UlyssesPredictor(STACPredictor):
    """Stages 1–3 wired with a placeholder Kirk linear projection."""

    def __init__(
        self,
        t: int = 50,
        f: int = 100,
        m: int | None = None,
        k: int = 128,
        seed: int = 0,
    ) -> None:
        self._t = t
        self._f = f
        self._m = m if m is not None else t // 2
        self._n = t - self._m + 1
        self._k = k

        rng = np.random.default_rng(seed)
        flat = f * self._m * self._n
        kirk_scale = np.float32(1.0 / np.sqrt(flat))
        readout_scale = np.float32(1.0 / np.sqrt(k))
        self._kirk_kernel = (
            rng.standard_normal((flat, k), dtype=np.float32) * kirk_scale
        )
        self._readout = (
            rng.standard_normal((k, 1), dtype=np.float32) * readout_scale
        )

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self._t, self._f)

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = build_hankel(x, m=self._m)              # (B, F, m, n)
        flat = h.reshape(h.shape[0], -1)            # (B, F*m*n)
        z = flat @ self._kirk_kernel                # (B, K)
        return z @ self._readout                    # (B, 1)


def _smoke() -> None:
    p = UlyssesPredictor()
    print(f"input_shape={p.input_shape}, k={p._k}, m={p._m}, n={p._n}")
    rng = np.random.default_rng(0)
    for B in (1, 4):
        x = rng.standard_normal((B, 50, 100), dtype=np.float32)
        y = p.predict(x)
        assert y.shape == (B, 1), f"B={B}: expected (B,1), got {y.shape}"
        assert y.dtype == np.float32, f"B={B}: expected float32, got {y.dtype}"
        print(f"  B={B}: input={x.shape} -> output={y.shape}, dtype={y.dtype}")
    print("UlyssesPredictor smoke OK")


if __name__ == "__main__":
    _smoke()
