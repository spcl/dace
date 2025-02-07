# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests the ``dace.inline`` preprocessor call.
"""

import dace
from dace.frontend.python.common import DaceSyntaxError
import math
import numpy as np
import pytest


def _find_in_tasklet(sdfg: dace.SDFG, term: str) -> bool:
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.Tasklet) and term in n.code.as_string:
            return True
    return False


def _find_in_memlet(sdfg: dace.SDFG, term: str) -> bool:
    for e, _ in sdfg.all_edges_recursive():
        if isinstance(e.data, dace.Memlet) and term in str(e.data.subset):
            return True
    return False


def test_inlinepp_simple():

    def complex_function(a: int, b: float):
        c = np.random.rand()
        return int(c + ((math.ceil(b) + a) // 2) - c)

    N = 20

    @dace.program
    def tester(a):
        # a[11] = 13
        a[dace.inline(complex_function(N + 1, 0.4))] = dace.inline(complex_function(5, N) + 1)

    a = np.random.rand(N)
    tester(a)
    assert np.allclose(a[11], 13)

    sdfg = tester.to_sdfg(a)
    assert _find_in_tasklet(sdfg, '13'), 'Inlined expression not found in tasklets'
    assert _find_in_memlet(sdfg, '11'), 'Inlined expression not found in memlets'


def test_inlinepp_fail():

    def f(x):
        return x + 1

    @dace.program
    def tester(a):
        a[dace.inline(a[0])] = 1

    a = np.random.rand(20)
    with pytest.raises(DaceSyntaxError):
        tester(a)


def test_inlinepp_tuple_retval():

    def divmod(a, b):
        return a // b, a % b

    @dace.program
    def tester(a: dace.float64[20], b: dace.float64[20]):
        for i in dace.map[0:20]:
            d, m = dace.inline(divmod(4, 3))
            a[i] = d
            b[i] = m

    a = np.random.rand(20)
    b = np.random.rand(20)
    tester(a, b)
    d, m = divmod(4, 3)
    assert np.allclose(a, d)
    assert np.allclose(b, m)


def test_inlinepp_stateful():
    ctr = 11

    def stateful():
        nonlocal ctr
        ctr += 1
        return ctr

    @dace.program
    def tester(a: dace.float64[3]):
        a[0] = dace.inline(stateful())
        a[1] = dace.inline(stateful())
        a[2] = dace.inline(stateful() * 2)

    sdfg = tester.to_sdfg()
    assert _find_in_tasklet(sdfg, '12')
    assert _find_in_tasklet(sdfg, '13')
    assert _find_in_tasklet(sdfg, '28')

    a = np.random.rand(3)
    sdfg(a)
    assert np.allclose(a, np.array([12, 13, 28]))


def test_inlinepp_in_unroll():
    ctr = 11

    def stateful(i):
        nonlocal ctr
        ctr += 1
        return ctr + i

    @dace.program
    def tester(a: dace.float64[3]):
        for i in dace.unroll(range(3)):
            a[i] = dace.inline(stateful(i))

    sdfg = tester.to_sdfg()
    assert _find_in_tasklet(sdfg, '12')
    assert _find_in_tasklet(sdfg, '14')
    assert _find_in_tasklet(sdfg, '16')

    a = np.random.rand(3)
    sdfg(a)
    assert np.allclose(a, np.array([12, 14, 16]))


if __name__ == '__main__':
    test_inlinepp_simple()
    test_inlinepp_fail()
    test_inlinepp_tuple_retval()
    test_inlinepp_stateful()
    test_inlinepp_in_unroll()
