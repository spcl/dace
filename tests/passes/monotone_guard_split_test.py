# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A loop whose carried dependence lives under an increasing guard (``i * 8192 < N``, ``i * i < N``) is split where the
guard flips: the guarded run stays a sequential loop and the rest becomes a map."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")


@dace.program
def affine_guard(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(1, N - 1):
        if i * 4 < N:
            a[i] = a[i - 1] + b[i]
        else:
            a[i] = c[i] + b[i]


@dace.program
def square_guard(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(1, N - 1):
        if i * i < N:
            a[i] = a[i - 1] + b[i]
        else:
            a[i] = c[i] + b[i]


@pytest.mark.parametrize(
    "program, flips", [(affine_guard, lambda i, n: i * 4 < n), (square_guard, lambda i, n: i * i < n)]
)
def test_a_loop_is_split_where_its_increasing_guard_flips(program, flips):
    sdfg = program.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    assert any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive()), "the unguarded run must map"
    compiled = sdfg.compile()
    for n in (3, 15, 16, 17, 64, 65, 1000):
        rng = np.random.default_rng(n)
        a, b, c = rng.random(n), rng.random(n), rng.random(n)
        expected = a.copy()
        for i in range(1, n - 1):
            expected[i] = expected[i - 1] + b[i] if flips(i, n) else c[i] + b[i]
        compiled(a=a, b=b, c=c, N=n)
        assert np.allclose(a, expected), n


if __name__ == "__main__":
    test_a_loop_is_split_where_its_increasing_guard_flips(affine_guard, lambda i, n: i * 4 < n)
    test_a_loop_is_split_where_its_increasing_guard_flips(square_guard, lambda i, n: i * i < n)
