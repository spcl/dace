# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests K Caching transformation."""

import numpy as np
import pytest
import copy
import dace
from dace.sdfg.state import LoopRegion, CodeBlock
from dace.transformation.interstate import KCaching


# Checks if KCaching applied N times and if memory footprint was reduced
def check_kcaching(kc_sdfg: dace.SDFG, N):
    orig_sdfg = copy.deepcopy(kc_sdfg)
    orig_sdfg.validate()
    assert kc_sdfg.apply_transformations_repeated(KCaching) == N
    kc_sdfg.validate()

    input_data_orig = {}
    input_data_kc = {}
    for argName, argType in kc_sdfg.arglist().items():
        arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
        arr[:] = np.random.rand(*argType.shape).astype(argType.dtype.type)
        input_data_orig[argName] = arr
        input_data_kc[argName] = copy.deepcopy(arr)

    orig_sdfg(**input_data_orig)
    kc_sdfg(**input_data_kc)

    # If memory footprint was reduced, N of the arrays should be different
    assert (
        sum(
            not np.array_equal(input_data_orig[argName], input_data_kc[argName])
            for argName, argType in kc_sdfg.arglist().items()
        )
        == N
    )


def test_simple():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 1)


def test_gaps():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(3, 32 - 1):
            b[i] = a[i - 1] + a[i - 3]
            a[i + 1] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 1)


def test_interleaved():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(3, 32 - 2):
            b[i] = a[i + 1] + a[i - 3]
            a[i + 2] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 1)


def test_nonlinear_index():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 5):
            b[i] = a[i - 1] + a[i - 2]
            a[i * i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_nonlinear_step():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)

    for node, p in sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion):
            node.update_statement = CodeBlock("i = i * 2")

    check_kcaching(sdfg, 0)


def test_nonconstant_step():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        step = 2
        for i in range(2, 32, step):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2
            step += 1

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_indirect_access():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            idx = int(b[i]) % 32
            a[idx] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_constant_index():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[0] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_larger_step():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32, 8):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_larger_index():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 4):
            b[i] = a[8 * i - 1] + a[8 * i - 2]
            a[8 * i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_reverse_step():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(32, 2, -1):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_reverse_index():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[-i - 1 + 33] + a[-i - 2 + 33]
            a[-i + 33] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_used_values():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2
        c[0] = a[0] + a[1]

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 0)


def test_used_values2():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        c[0] = a[0] + a[1]
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 1)


def test_used_values3():
    @dace.program
    def tester(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        for i in range(2, 32):
            b[i] = a[i - 1] + a[i - 2]
            a[i] = c[i] * 2
        a[0] = c[0]
        a[1] = c[1]
        c[0] = a[0] + a[1]

    sdfg = tester.to_sdfg(simplify=True)
    check_kcaching(sdfg, 1)


if __name__ == "__main__":
    test_simple()
    test_gaps()
    test_interleaved()
    test_nonlinear_index()
    test_nonlinear_step()
    test_nonconstant_step()
    test_indirect_access()
    test_constant_index()
    test_larger_step()
    test_larger_index()
    test_reverse_step()
    test_reverse_index()
    test_used_values()
    test_used_values2()
    test_used_values3()
