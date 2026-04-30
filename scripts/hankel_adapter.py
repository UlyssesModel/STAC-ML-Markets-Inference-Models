#!/usr/bin/env python3
"""First sketch of Stage 1 of the Ulysses pipeline (`rs40-swap-spec.md` §5).

Builds a per-batch, per-feature Hankel matrix from a `(B, T, F)` time-series
tensor — the same input shape the STAC-aligned `LSTM_*` artifacts in this
repo consume. The output is `(B, F, m, n)` with `n = T - m + 1` and
`H[b, f, i, j] == x[b, i+j, f]`.

The exact `m` / `n` and output dtype (eventually BF16 for Kirk) are subject
to change once Kirk's input contract is pinned. This module is a generic
Hankel transform with sensible defaults so downstream wiring (Kirk core,
scalar readout) can be tested before that contract lands. Replace the
defaults — not the abstraction — when the contract arrives.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def build_hankel(x: np.ndarray, m: int | None = None) -> np.ndarray:
    """Build a Hankel tensor from a time-series batch.

    Args:
        x: input of shape `(B, T, F)`, any numeric dtype (commonly float32).
        m: number of Hankel rows. Defaults to `T // 2`.

    Returns:
        Array of shape `(B, F, m, n)` where `n = T - m + 1`, dtype matches
        `x.dtype`. Contiguous and owned (a `.copy()` of the strided view).
    """
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 input (B, T, F), got shape {x.shape}")
    B, T, F = x.shape
    if m is None:
        m = T // 2
    if not (1 <= m <= T):
        raise ValueError(f"m={m} must be in [1, T={T}]")
    n = T - m + 1

    # Move time to the last axis: (B, F, T). Then sliding_window_view of size n
    # along the last axis yields shape (B, F, m, n) with row i being x[i:i+n]
    # — exactly H[b, f, i, j] = x[b, i+j, f].
    transposed = np.transpose(x, (0, 2, 1))  # (B, F, T)
    view = sliding_window_view(transposed, window_shape=n, axis=-1)  # (B, F, m, n)
    return view.copy()


def _smoke() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 50, 100), dtype=np.float32)
    h = build_hankel(x)
    print(f"input shape:  {x.shape}, dtype={x.dtype}")
    print(f"output shape: {h.shape}, dtype={h.dtype}")
    print(f"default m=T//2={50 // 2}, n=T-m+1={50 - 25 + 1}")

    # Verify Hankel structure: anti-diagonals (constant i+j) carry the same
    # value, equal to x[b, i+j, f]. Sample a few cells.
    B, F, m, n = h.shape
    samples = [(0, 0, 0, 0), (0, 0, 5, 7), (1, 99, 12, 13), (1, 50, m - 1, n - 1)]
    for (b, f, i, j) in samples:
        got = h[b, f, i, j]
        want = x[b, i + j, f]
        ok = "PASS" if got == want else "FAIL"
        print(f"  [{ok}] h[{b},{f},{i},{j}]={got!r:>22}  x[{b},{i + j},{f}]={want!r}")


if __name__ == "__main__":
    _smoke()
