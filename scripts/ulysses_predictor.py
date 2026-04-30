#!/usr/bin/env python3
"""Ulysses predictor with a formal Stage-2 integration point.

Wires the full `(B, 50, 100) float32 -> (B, 1) float32` contract end-to-end
across three stages (per `docs/rs40-swap-spec.md` §5):

  Stage 1: `build_hankel` (Hankel adapter)              — pinned by §4.
  Stage 2: a `KirkCore` instance (the swap point)       — see ABC below.
  Stage 3: scalar readout linear projection `... -> 1`  — pinned by §4.

`KirkCore` is the **integration point** for any real Kirk implementation.
Swapping in real Kirk means implementing a `KirkCore` subclass whose
`transform()` does whatever real Kirk does, then constructing
`UlyssesPredictor(kirk=RealKirk(...))`. Stages 1 and 3 do not need to
change. Two reference subclasses ship in this module:

  `IdentityKirk` — Stage 2 is a no-op flatten. The readout becomes a
    `(F*m*n) -> 1` linear, exposing the floor cost of Stages 1 and 3.

  `LinearStubKirk` — Stage 2 is a deterministic random `(F*m*n) -> K`
    linear projection (the prior placeholder behaviour). Useful as a
    conservative ceiling for adapter + Stage-2 cost.

Neither is real Kirk; both are stand-ins for measurement and shape-wiring.
"""

from __future__ import annotations

import abc
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hankel_adapter import build_hankel
from stac_sumaco_driver import STACPredictor


class KirkCore(abc.ABC):
    """Stage-2 contract for the Ulysses pipeline.

    Implementations sit between the Hankel adapter and the scalar readout.
    The contract is intentionally minimal: declare the per-element output
    shape, and transform a Hankel batch into whatever the readout consumes.
    """

    @abc.abstractmethod
    def transform(self, hankel_batch: np.ndarray) -> np.ndarray:
        """Map a `(B, F, m, n)` Hankel batch to whatever the readout consumes.

        The leading batch dimension is preserved. The returned array's
        per-element shape must match `output_shape(F, m, n)`.
        """

    @abc.abstractmethod
    def output_shape(self, f: int, m: int, n: int) -> tuple[int, ...]:
        """Per-element output shape (excluding batch). Used to size Stage 3."""


class IdentityKirk(KirkCore):
    """No-op Stage 2. Flattens `(B, F, m, n)` to `(B, F*m*n)`."""

    def transform(self, hankel_batch: np.ndarray) -> np.ndarray:
        return hankel_batch.reshape(hankel_batch.shape[0], -1)

    def output_shape(self, f: int, m: int, n: int) -> tuple[int, ...]:
        return (f * m * n,)


class LinearStubKirk(KirkCore):
    """Deterministic random `(F*m*n) -> K` linear projection.

    The kernel is materialised lazily on the first `transform()` call,
    using the seeded RNG. This keeps the constructor signature minimal
    (no Hankel-shape coupling) while preserving determinism: same seed
    plus same input shape -> same kernel.
    """

    def __init__(self, k: int = 128, seed: int = 0) -> None:
        self._k = k
        self._seed = seed
        self._kernel: np.ndarray | None = None

    @property
    def k(self) -> int:
        return self._k

    def output_shape(self, f: int, m: int, n: int) -> tuple[int, ...]:
        return (self._k,)

    def transform(self, hankel_batch: np.ndarray) -> np.ndarray:
        flat = hankel_batch.reshape(hankel_batch.shape[0], -1)
        if self._kernel is None:
            flat_dim = flat.shape[1]
            scale = np.float32(1.0 / np.sqrt(flat_dim))
            rng = np.random.default_rng(self._seed)
            self._kernel = (
                rng.standard_normal((flat_dim, self._k), dtype=np.float32) * scale
            )
        return flat @ self._kernel


class UlyssesPredictor(STACPredictor):
    """Stages 1–3 wired with a swappable Stage-2 `KirkCore`."""

    def __init__(
        self,
        t: int = 50,
        f: int = 100,
        m: int | None = None,
        kirk: KirkCore | None = None,
        readout_seed: int = 0,
    ) -> None:
        self._t = t
        self._f = f
        self._m = m if m is not None else t // 2
        self._n = t - self._m + 1
        self._kirk: KirkCore = kirk if kirk is not None else LinearStubKirk(k=128, seed=0)

        kirk_out_shape = self._kirk.output_shape(self._f, self._m, self._n)
        readout_in = int(np.prod(kirk_out_shape))
        readout_scale = np.float32(1.0 / np.sqrt(readout_in))
        readout_rng = np.random.default_rng(readout_seed)
        self._readout = (
            readout_rng.standard_normal((readout_in, 1), dtype=np.float32) * readout_scale
        )

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self._t, self._f)

    @property
    def kirk(self) -> KirkCore:
        return self._kirk

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = build_hankel(x, m=self._m)              # (B, F, m, n)
        z = self._kirk.transform(h)                 # (B, *kirk_out_shape)
        z_flat = z.reshape(z.shape[0], -1)          # (B, prod(kirk_out_shape))
        return z_flat @ self._readout               # (B, 1)


def _smoke() -> None:
    p = UlyssesPredictor()
    print(
        f"input_shape={p.input_shape}, m={p._m}, n={p._n}, "
        f"kirk={type(p.kirk).__name__}"
    )
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
