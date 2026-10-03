# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Binding a nested ``@dc.program``'s shape symbols from the call site's arguments."""

import numpy as np
import pytest

import dace

M = dace.symbol('M', dtype=dace.int64, positive=True)
N = dace.symbol('N', dtype=dace.int64, positive=True)
K = dace.symbol('K', dtype=dace.int64, positive=True)


@dace.program
def twice(x: dace.float64[M], out: dace.float64[M]):
    out[:] = x * 2.0


@dace.program
def strided(x: dace.float64[M], out: dace.float64[(M - 7) // 2 + 1]):
    out[:] = x[0:(M - 7) // 2 + 1]


@dace.program
def prefix(x: dace.float64[M], out: dace.float64[M]):
    out[:] = 0.0
    for i in range(K):
        out[i] = x[i]


def test_a_callee_rebinds_its_shape_symbol_per_call_site():
    """One shape-generic callee serves two call sites of different extents."""

    @dace.program
    def caller(a: dace.float64[N], b: dace.float64[2 * N], oa: dace.float64[N], ob: dace.float64[2 * N]):
        twice(a, oa)
        twice(b, ob)

    n = 4
    a, b = np.arange(n, dtype=np.float64), np.arange(2 * n, dtype=np.float64)
    oa, ob = np.zeros(n), np.zeros(2 * n)
    caller(a, b, oa, ob, N=n)
    assert np.allclose(oa, a * 2.0)
    assert np.allclose(ob, b * 2.0)


def test_a_callee_extent_that_needs_inverting_an_int_floor():
    """The callee's OUT extent is ``(M - 7) // 2 + 1``, so binding ``M`` means solving through an
    ``int_floor``. That head is opaque to sympy, which answers ``NotImplementedError: equal
    function with more than 1 argument`` rather than declining."""

    @dace.program
    def caller(a: dace.float64[N], out: dace.float64[(N - 7) // 2 + 1]):
        strided(a, out)

    n = 32
    a = np.arange(n, dtype=np.float64)
    out = np.zeros((n - 7) // 2 + 1)
    caller(a, out, N=n)
    assert np.allclose(out, a[0:(n - 7) // 2 + 1])


def test_a_body_only_symbol_is_bound_by_keyword():
    """``K`` is in no parameter shape, so the caller passes it -- and only by keyword: positionally
    the frontend indexes its parameter-name list with the argument's position."""

    @dace.program
    def caller(a: dace.float64[N], oa: dace.float64[N], ob: dace.float64[N]):
        prefix(a, oa, K=N)
        prefix(a, ob, K=N - 1)

    n = 5
    a = np.arange(1.0, n + 1)
    oa, ob = np.zeros(n), np.zeros(n)
    caller(a, oa, ob, N=n)
    assert np.allclose(oa, a)
    assert np.allclose(ob, np.append(a[:n - 1], 0.0))


def test_a_shape_inferred_symbol_may_not_also_be_passed():
    """``M`` is solved from the argument's shape; naming it too is an error, not a redundancy."""

    @dace.program
    def caller(a: dace.float64[N], oa: dace.float64[N]):
        twice(a, oa, M=N)

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError, match='Invalid keyword argument "M"'):
        caller.to_sdfg(simplify=False)


if __name__ == '__main__':
    test_a_callee_rebinds_its_shape_symbol_per_call_site()
    test_a_callee_extent_that_needs_inverting_an_int_floor()
    test_a_body_only_symbol_is_bound_by_keyword()
    test_a_shape_inferred_symbol_may_not_also_be_passed()
