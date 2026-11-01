# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Arguments of calls to nested DaCe programs: expressions that fold to constants, and symbols only the callee uses."""

import numpy as np

import dace

N = dace.symbol("N")
M = dace.symbol("M")
ONE = 1


@dace.program
def store(a, i: dace.int64, v: dace.int64):
    a[i] = v


@dace.program
def store_folded(a: dace.int64[N]):
    for i in dace.map[0:N]:
        store(a, i, i % ONE)  # Folds to the constant 0


def test_argument_folds_to_constant():
    a = np.ones(10, np.int64)
    store_folded(a)
    assert np.all(a == 0)


@dace.program
def fill_first(a):
    for i in dace.map[0:M]:  # A symbol the caller does not use
        a[i] = i


@dace.program
def call_with_callee_symbol(a: dace.int64[N]):
    fill_first(a)


def test_symbol_of_the_callee_only():
    a = np.full(10, -1, np.int64)
    call_with_callee_symbol(a, M=4)
    assert list(a) == [0, 1, 2, 3] + [-1] * 6


if __name__ == "__main__":
    test_argument_folds_to_constant()
    test_symbol_of_the_callee_only()
