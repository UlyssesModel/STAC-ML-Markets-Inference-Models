#!/usr/bin/env python3
"""Placeholder-Kirk Ulysses predictor — stub for `rs40-swap-spec.md` §5 Stage 2.

This is **not real Ulysses** and **not real Kirk**. It wires the full
`(B, 50, 100) float32 → (B, 1) float32` contract end-to-end with two
selectable Stage-2 modes:

  `kirk_mode="linear_stub"` (default):
    Stage 1: `build_hankel` (Hankel adapter).
    Stage 2: random-init linear projection `(F·m·n) → K`.
    Stage 3: random-init linear projection `K → 1`.
    A stand-in that exercises a brute-force matmul to bound a worst-case
    Stage 2 — useful as a conservative ceiling for adapter + Stage-2 cost.

  `kirk_mode="identity"`:
    Stage 1: `build_hankel` (Hankel adapter).
    Stage 2: identity (no-op; just the flatten that Stage 3 needs anyway).
    Stage 3: random-init linear projection `(F·m·n) → 1`.
    Collapses Stage 2 to a pass-through so we can read the floor cost of
    Stages 1 and 3 alone — every µs above this floor is Stage-2-attributable
    when real Kirk lands.

The point of either mode is to **bound adapter + readout overhead**
independently of whatever real Kirk eventually does. Once Jarett's Kirk
contract arrives, only the Stage-2 op needs to change; the Stage-1 Hankel
shape and Stage-3 readout shape are already pinned by §4.
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


VALID_KIRK_MODES = ("linear_stub", "identity")


class UlyssesPredictor(STACPredictor):
    """Stages 1–3 wired with a configurable placeholder Kirk core."""

    def __init__(
        self,
        t: int = 50,
        f: int = 100,
        m: int | None = None,
        k: int = 128,
        seed: int = 0,
        kirk_mode: str = "linear_stub",
    ) -> None:
        if kirk_mode not in VALID_KIRK_MODES:
            raise ValueError(
                f"kirk_mode must be one of {VALID_KIRK_MODES}, got {kirk_mode!r}"
            )
        self._t = t
        self._f = f
        self._m = m if m is not None else t // 2
        self._n = t - self._m + 1
        self._k = k
        self._kirk_mode = kirk_mode

        rng = np.random.default_rng(seed)
        flat = f * self._m * self._n

        if kirk_mode == "linear_stub":
            kirk_scale = np.float32(1.0 / np.sqrt(flat))
            readout_scale = np.float32(1.0 / np.sqrt(k))
            self._kirk_kernel = (
                rng.standard_normal((flat, k), dtype=np.float32) * kirk_scale
            )
            self._readout = (
                rng.standard_normal((k, 1), dtype=np.float32) * readout_scale
            )
        else:  # identity
            self._kirk_kernel = None
            readout_scale = np.float32(1.0 / np.sqrt(flat))
            self._readout = (
                rng.standard_normal((flat, 1), dtype=np.float32) * readout_scale
            )

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self._t, self._f)

    @property
    def kirk_mode(self) -> str:
        return self._kirk_mode

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = build_hankel(x, m=self._m)              # (B, F, m, n)
        flat = h.reshape(h.shape[0], -1)            # (B, F*m*n)
        if self._kirk_mode == "linear_stub":
            z = flat @ self._kirk_kernel            # (B, K)
        else:                                       # identity
            z = flat                                # (B, F*m*n)
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
