# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests loop overwrite elimination transformations."""

import numpy as np
import pytest
import dace
from dace.transformation.interstate import LoopOverwriteElimination
from dace.sdfg.state import LoopRegion
from copy import deepcopy


def _test_for_unchanged_behavior(prog, num_loops, num_eliminations):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly one loop
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == num_loops

    # Get ground truth values if we expect eliminations
    if num_eliminations > 0:
        input_data = {}
        for argName, argType in sdfg.arglist().items():
            arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
            arr[:] = np.random.rand(*argType.shape).astype(argType.dtype.type)
            input_data[argName] = arr
        ground_truth = deepcopy(input_data)
        sdfg(**ground_truth)

    # Apply the transformation
    sdfg.apply_transformations_repeated(LoopOverwriteElimination)
    sdfg.validate()

    # Check that the loop has been eliminated
    loop_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert len(loop_nodes) == num_loops - num_eliminations

    # Test if the behavior is unchanged if we expect eliminations
    if num_eliminations > 0:
        output_data = deepcopy(input_data)
        sdfg(**output_data)
        np.testing.assert_equal(output_data, ground_truth)


def test_overwrite_elimination_basic():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_basic2():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            A[i] = B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_nonzero():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            A[5] = B[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_reverse():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(9, -1, -1):
            A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_stride():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(0, 10, 2):
            A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_read():

    @dace.program
    def tester(A: dace.float32[10]):
        for i in range(10):
            A[0] = A[i] + 1.0

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_nested_loops():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10, 10]):
        for i in range(10):
            for j in range(10):
                A[0] = B[i, j]

    _test_for_unchanged_behavior(tester, 2, 2)


def test_overwrite_elimination_nested_loops2():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            A[0] = B[i]
            for j in range(5):
                A[j] = B[j] + 1.0

    _test_for_unchanged_behavior(tester, 2, 1)


def test_overwrite_elimination_reverse_nested():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[5, 5]):
        for i in range(4, -1, -1):
            for j in range(4, -1, -1):
                A[0] = B[i, j]

    _test_for_unchanged_behavior(tester, 2, 2)


def test_overwrite_elimination_strided_nested():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(0, 10, 2):
            for j in range(0, 10, 3):
                A[0] = B[j]

    _test_for_unchanged_behavior(tester, 2, 2)


def test_overwrite_elimination_nonlinear_access():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            idx = (i * 3) % 10
            A[0] = B[idx]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_with_break():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            if B[i] > 0.5:
                break
            A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_with_continue():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            if B[i] < 0.3:
                continue
            else:
                A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_with_return():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            if B[i] < 0.3:
                return
            A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_step_changing():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        i = 0
        while i < 10:
            A[0] = B[i]
            if B[i] > 0.5:
                i += 2
            else:
                i += 1

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_conditional_write():

    @dace.program
    def tester(A: dace.float32[20], B: dace.float32[20]):
        for i in range(10):
            if i % 3 == 0:
                A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_multiple_writes():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10], C: dace.float32[10]):
        for i in range(10):
            A[0] = B[i]
            A[0] = C[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_conditional_write2():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            if B[i] > 0.5:
                A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_read2():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(1, 10):
            A[0] = A[0] + B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_read_reverse():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(9, -1, -1):
            A[0] = A[i] + B[i]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_read_crossing():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(1, 10):
            A[5] = A[i] + B[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_triple_nested():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[5, 5, 5]):
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    A[0] = B[i, j, k]

    _test_for_unchanged_behavior(tester, 3, 3)


def test_overwrite_elimination_triple_nested_dependency():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[5, 5, 5]):
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    A[j] = B[i, j, k]

    _test_for_unchanged_behavior(tester, 3, 2)


def test_overwrite_elimination_reverse_strided():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(9, -1, -2):
            A[0] = B[i]

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_tmp():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            tmp = B[i]
            A[0] = tmp

    _test_for_unchanged_behavior(tester, 1, 1)


def test_overwrite_elimination_tmp2():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        tmp = dace.ndarray(shape=(2, ), dtype=dace.float32)
        for i in range(10):
            tmp[1] = B[i]
            A[0] = tmp[0]

    _test_for_unchanged_behavior(tester, 1, 0)


def test_overwrite_elimination_loop_dep():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            for j in range(i, 9, 1):
                A[j] = B[j]

    _test_for_unchanged_behavior(tester, 2, 0)


def test_overwrite_elimination_loop_dep2():

    @dace.program
    def tester(A: dace.float32[10], B: dace.float32[10]):
        for i in range(10):
            for j in range(i, 9, 1):
                A[0] = B[j]

    _test_for_unchanged_behavior(tester, 2, 2)


if __name__ == "__main__":
    test_overwrite_elimination_basic()
    test_overwrite_elimination_basic2()
    test_overwrite_elimination_nonzero()
    test_overwrite_elimination_reverse()
    test_overwrite_elimination_read()
    test_overwrite_elimination_nested_loops()
    test_overwrite_elimination_nested_loops2()
    test_overwrite_elimination_reverse_nested()
    test_overwrite_elimination_strided_nested()
    test_overwrite_elimination_nonlinear_access()
    test_overwrite_elimination_with_break()
    test_overwrite_elimination_with_continue()
    test_overwrite_elimination_with_return()
    test_overwrite_elimination_step_changing()
    test_overwrite_elimination_conditional_write()
    test_overwrite_elimination_multiple_writes()
    test_overwrite_elimination_conditional_write2()
    test_overwrite_elimination_read2()
    test_overwrite_elimination_read_reverse()
    test_overwrite_elimination_read_crossing()
    test_overwrite_elimination_triple_nested()
    test_overwrite_elimination_triple_nested_dependency()
    test_overwrite_elimination_reverse_strided()
    test_overwrite_elimination_tmp()
    test_overwrite_elimination_tmp2()
    test_overwrite_elimination_loop_dep()
    test_overwrite_elimination_loop_dep2()
