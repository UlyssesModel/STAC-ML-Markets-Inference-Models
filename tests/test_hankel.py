#!/usr/bin/env python3
"""Assertion-based tests for `scripts.hankel_adapter.build_hankel`.

Run directly:

    python tests/test_hankel.py

No pytest dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hankel_adapter import build_hankel  # noqa: E402


def test_shape() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 50, 100), dtype=np.float32)
    h = build_hankel(x, m=25)
    assert h.shape == (2, 100, 25, 26), f"got shape {h.shape}"


def test_default_m() -> None:
    rng = np.random.default_rng(1)
    T, F = 16, 4
    x = rng.standard_normal((3, T, F), dtype=np.float32)
    h = build_hankel(x)  # default m = T // 2 = 8, n = 9
    assert h.shape == (3, F, T // 2, T - T // 2 + 1), f"got shape {h.shape}"


def test_dtype() -> None:
    x = np.random.default_rng(2).standard_normal((1, 10, 3), dtype=np.float32)
    h = build_hankel(x, m=4)
    assert h.dtype == np.float32, f"got dtype {h.dtype}"


def test_hankel_structure() -> None:
    # Small explicit input so the assertion is easy to read.
    B, T, F = 2, 6, 3
    x = np.arange(B * T * F, dtype=np.float32).reshape(B, T, F)
    m = 3
    n = T - m + 1
    h = build_hankel(x, m=m)
    assert h.shape == (B, F, m, n)
    for b in range(B):
        for f in range(F):
            for i in range(m):
                for j in range(n):
                    expected = x[b, i + j, f]
                    actual = h[b, f, i, j]
                    assert actual == expected, (
                        f"h[{b},{f},{i},{j}]={actual} != x[{b},{i + j},{f}]={expected}"
                    )


def main() -> int:
    test_shape()
    test_default_m()
    test_dtype()
    test_hankel_structure()
    print("All Hankel tests pass ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
