# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that free symbols which only appear inside the condition of an ``if`` statement (or ``while`` loop) are
registered as SDFG symbols, so that they are part of the SDFG's argument list.
"""
import numpy as np
import pytest
from typing import Optional

import dace

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K', dtype=dace.int64)


def test_symbol_only_in_if_condition():

    @dace.program
    def symbol_only_in_if_condition(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if i >= 1 and i < M:
                B[i] = A[i] * 2.0

    sdfg = symbol_only_in_if_condition.to_sdfg(simplify=True)
    assert 'M' in sdfg.symbols
    assert sdfg.symbols['M'] == M.dtype
    assert 'M' in sdfg.arglist()

    A = np.arange(16, dtype=np.float64)
    B = np.zeros(16, dtype=np.float64)
    sdfg(A=A, B=B, N=16, M=9)
    ref = np.zeros(16, dtype=np.float64)
    ref[1:9] = A[1:9] * 2.0
    assert np.allclose(B, ref)


def test_symbol_only_in_if_condition_dtype():
    """The registered symbol must keep the dtype it was declared with."""

    @dace.program
    def symbol_only_in_if_condition_dtype(A: dace.float64[N]):
        if K > 3:
            A[:] = 1.0
        else:
            A[:] = 2.0

    sdfg = symbol_only_in_if_condition_dtype.to_sdfg(simplify=True)
    assert 'K' in sdfg.symbols
    assert sdfg.symbols['K'] == dace.int64

    A = np.zeros(8, dtype=np.float64)
    sdfg(A=A, N=8, K=5)
    assert np.allclose(A, 1.0)
    sdfg(A=A, N=8, K=2)
    assert np.allclose(A, 2.0)


def test_symbol_only_in_elif_condition():

    @dace.program
    def symbol_only_in_elif_condition(A: dace.float64[N]):
        for i in range(N):
            if i < 2:
                A[i] = 0.0
            elif i < M:
                A[i] = 1.0
            else:
                A[i] = 2.0

    sdfg = symbol_only_in_elif_condition.to_sdfg(simplify=True)
    assert 'M' in sdfg.symbols

    A = np.zeros(10, dtype=np.float64)
    sdfg(A=A, N=10, M=6)
    ref = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float64)
    assert np.allclose(A, ref)


def test_symbol_only_in_while_condition():

    @dace.program
    def symbol_only_in_while_condition(A: dace.float64[N]):
        i = 0
        while i < M:
            A[i] = 1.0
            i += 1

    sdfg = symbol_only_in_while_condition.to_sdfg(simplify=True)
    assert 'M' in sdfg.symbols

    A = np.zeros(10, dtype=np.float64)
    sdfg(A=A, N=10, M=4)
    ref = np.zeros(10, dtype=np.float64)
    ref[:4] = 1.0
    assert np.allclose(A, ref)


def test_undefined_variable_in_if_condition():
    """Variables that are not defined anywhere must still be reported as errors."""

    @dace.program
    def undefined_variable_in_if_condition(A: dace.float64[N]):
        if undefined_name > 1:  # noqa: F821
            A[:] = 1.0

    with pytest.raises(dace.frontend.python.common.DaceSyntaxError):
        undefined_variable_in_if_condition.to_sdfg()


def test_none_comparison_in_if_condition():
    """``is None`` comparisons must not register the ``None`` placeholder as a symbol."""

    @dace.program
    def none_comparison_in_if_condition(A: dace.float64[N], B: Optional[dace.float64[N]] = None):
        if B is None:
            A[:] = 1.0
        else:
            A[:] = B

    sdfg = none_comparison_in_if_condition.to_sdfg(simplify=False)
    assert 'NoneSymbol' not in sdfg.symbols
    assert 'NoneSymbol' not in sdfg.arglist()


if __name__ == '__main__':
    test_symbol_only_in_if_condition()
    test_symbol_only_in_if_condition_dtype()
    test_symbol_only_in_elif_condition()
    test_symbol_only_in_while_condition()
    test_undefined_variable_in_if_condition()
    test_none_comparison_in_if_condition()
