# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.interstate import InlineSDFG, StateFusion
from dace.transformation.dataflow import MergeSourceSinkArrays
from dace.libraries import blas
from dace.library import change_default
import numpy as np
import os
import pytest

W = dace.symbol('W')
H = dace.symbol('H')

def test_multistate_inline():

    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(5):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20]):
        nested(A)

    sdfg = outerprog.to_sdfg(simplify=False)

    A = np.random.rand(20)
    expected = np.copy(A)
    outerprog.f(expected)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    sdfg(A)
    assert np.allclose(A, expected)


def test_multistate_inline_samename():

    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(5):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20]):
        for i in range(5):
            nested(A)

    sdfg = outerprog.to_sdfg(simplify=False)

    A = np.random.rand(20)
    expected = np.copy(A)
    outerprog.f(expected)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    sdfg(A)
    assert np.allclose(A, expected)


def test_multistate_inline_outer_dependencies():

    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(1, 20):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20], B: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a >> A[i]
                b >> B[i]

                a = 0
                b = 1

        nested(A)

        for i in dace.map[0:20]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]

                b = 2 * a

    sdfg = outerprog.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(StateFusion)
    assert len(sdfg.states()) == 1

    A = np.random.rand(20)
    B = np.random.rand(20)
    expected_a = np.copy(A)
    expected_b = np.copy(B)
    outerprog.f(expected_a, expected_b)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 4
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 3

    sdfg(A, B)
    assert np.allclose(A, expected_a)
    assert np.allclose(B, expected_b)


def test_multistate_inline_concurrent_subgraphs():

    @dace.program
    def nested(A: dace.float64[10], B: dace.float64[10]):
        for i in range(1, 10):
            B[i] = A[i]

    @dace.program
    def outerprog(A: dace.float64[10], B: dace.float64[10], C: dace.float64[10]):
        nested(A, B)

        for i in dace.map[0:10]:
            with dace.tasklet:
                a << A[i]
                c >> C[i]

                c = 2 * a

    sdfg = outerprog.to_sdfg(simplify=False)

    dace.propagate_memlets_sdfg(sdfg)
    sdfg.apply_transformations_repeated(StateFusion)
    assert len(sdfg.states()) == 1
    assert len([node for node in sdfg.start_state.data_nodes()]) == 3

    A = np.random.rand(10)
    B = np.random.rand(10)
    C = np.random.rand(10)
    expected_a = np.copy(A)
    expected_b = np.copy(B)
    expected_c = np.copy(C)
    outerprog.f(expected_a, expected_b, expected_c)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 3
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 2

    sdfg(A, B, C)
    assert np.allclose(A, expected_a)
    assert np.allclose(B, expected_b)
    assert np.allclose(C, expected_c)

def test_multistate_inline_views():

    @dace.program
    def nested_squeezed(c: dace.int32[5], d: dace.int32[5]):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        nested_squeezed(A[1, :], B[:, 1])

    sdfg = inline_unsqueeze.to_sdfg(simplify=False)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    sdfg(A, B)
    for i in range(3):
        if i == 1:
            assert (np.array_equal(B[:, i], A[1, :]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_multistate_inline_views2():

    @dace.program
    def nested_squeezed(c, d):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        for i in range(2):
            nested_squeezed(A[i, :], B[:, 1 - i])

    sdfg = inline_unsqueeze.to_sdfg(simplify=False)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    sdfg(A, B)
    for i in range(3):
        if i < 2:
            assert (np.array_equal(B[:, 1 - i], A[i, :]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_multistate_inline_views3():

    @dace.program
    def nested_squeezed(c, d):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        for i in range(2):
            nested_squeezed(A[i, i:i + 2], B[i + 1:i + 3, 1 - i])

    sdfg = inline_unsqueeze.to_sdfg(simplify=False)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    sdfg(A, B)
    for i in range(3):
        if i < 2:
            assert (np.array_equal(B[i + 1:i + 3, 1 - i], A[i, i:i + 2]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_multistate_inline_views4():

    @dace.program
    def nested_squeezed(c, d):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        for i in range(2):
            nested_squeezed(A[i, i:2 * i + 2], B[i + 1:2 * i + 3, 1 - i])

    sdfg = inline_unsqueeze.to_sdfg(simplify=False)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)

    sdfg(A, B)

    for i in range(3):
        if i < 2:
            assert (np.array_equal(B[i + 1:2 * i + 3, 1 - i], A[i, i:2 * i + 2]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


@pytest.mark.skip
def test_inline_symbol_assignment():

    def nested(a, num):
        cat = num - 1
        last_step = (cat == 0)
        if last_step is True:
            return a + 1

        return a

    @dace.program
    def tester(a: dace.float64[20], b: dace.float64[10, 20]):
        for i in range(10):
            cat = nested(a, i)
            b[i] = cat

    sdfg = tester.to_sdfg(simplify=False)

    A = np.random.random(20).astype(np.float64)
    B = np.zeros((10, 20), dtype=np.float64)
    A_ = np.copy(A)
    B_ = np.copy(B)
    
    sdfg(A, B)

    from dace.transformation.interstate import InlineMultistateSDFG

    assert len(list(sdfg.all_sdfgs_recursive())) == 2
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert len(list(sdfg.all_sdfgs_recursive())) == 1

    sdfg(A_, B_)

    assert np.allclose(A, A_)
    assert np.allclose(B, B_)


if __name__ == "__main__":
    test_multistate_inline()
    test_multistate_inline_samename()
    test_multistate_inline_outer_dependencies()
    test_multistate_inline_concurrent_subgraphs()
    test_multistate_inline_views()
    test_multistate_inline_views2()
    test_multistate_inline_views3()
    test_multistate_inline_views4()
    test_inline_symbol_assignment()
