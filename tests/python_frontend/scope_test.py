# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests related to scopes and variable lifetime. """

import re
import dace
import numpy as np

from tests.python_frontend.string_test import test_bytes_literal


rng = np.random.default_rng(42)


def test_reassignment_simple():

    @dace.program
    def reassignment_simple(a: dace.float64[3, 3], b: dace.float64[3, 3]) -> dace.float64[3, 3]:
        out = a + b
        out = a - b
        return out
    
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    ref = reassignment_simple.f(A, B)

    sdfg = reassignment_simple.to_sdfg(simplify=False)
    func = sdfg.compile()
    val = func(a=A, b=B)
    assert(np.allclose(val, ref))

    val = reassignment_simple(A, B)
    assert(np.allclose(val, ref))


def test_reassignment_if():

    @dace.program
    def reassignment_if(a: dace.float64[3, 3], b: dace.float64[3, 3], c: dace.int64) -> dace.float64[3, 3]:
        if c >= 3:
            out = a + b
        else:
            out = a - b
        return out
    
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    ref0 = reassignment_if.f(A, B, 2)
    ref1 = reassignment_if.f(A, B, 4)

    sdfg = reassignment_if.to_sdfg(simplify=False)
    func = sdfg.compile()
    val0 = func(a=A, b=B, c=2)
    assert(np.allclose(val0, ref0))
    val1 = func(a=A, b=B, c=4)
    assert(np.allclose(val1, ref1))

    val0 = reassignment_if(A, B, 2)
    assert(np.allclose(val0, ref0))
    val1 = reassignment_if(A, B, 4)
    assert(np.allclose(val1, ref1))


def test_reassignment_for():

    @dace.program
    def reassignment_for(a: dace.float64[3, 3], b: dace.float64[3, 3]) -> dace.float64[3, 3]:
        out = np.copy(a)
        for _ in range(10):
            out = out - b
        return out
    
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    ref = reassignment_for.f(A, B)

    sdfg = reassignment_for.to_sdfg(simplify=False)
    func = sdfg.compile()
    val = func(a=A, b=B)
    assert(np.allclose(val, ref))

    sdfg = reassignment_for.to_sdfg(simplify=True)
    sdfg.view()
    func = sdfg.compile()
    val = func(a=A, b=B)
    assert(np.allclose(val, ref))

    val = reassignment_for(A, B)
    assert(np.allclose(val, ref))


def test_reassignment_while():

    @dace.program
    def reassignment_while(a: dace.float64[3, 3], b: dace.float64[3, 3]) -> dace.float64[3, 3]:
        out = np.copy(a)
        i = 0
        while (i < 10):
            out = out - b
            i += 1
        return out
    
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    ref = reassignment_while.f(A, B)

    sdfg = reassignment_while.to_sdfg(simplify=False)
    func = sdfg.compile()
    val = func(a=A, b=B)
    assert(np.allclose(val, ref))

    val = reassignment_while(A, B)
    assert(np.allclose(val, ref))


if __name__ == "__main__":

    test_reassignment_simple()
    test_reassignment_if()
    test_reassignment_for()
    test_reassignment_while()
