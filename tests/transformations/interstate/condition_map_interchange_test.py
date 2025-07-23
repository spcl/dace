# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests loop overwrite elimination transformations."""

import numpy as np
import pytest
import dace
from dace.transformation.interstate import ConditionMapInterchange
from dace.sdfg.state import LoopRegion
from copy import deepcopy


def _test_for_unchanged_behavior(prog, num_apps):
    sdfg: dace.SDFG = prog.to_sdfg(simplify=True)
    sdfg.validate()

    # Get ground truth values if we expect eliminations
    if num_apps > 0:
        input_data = {}
        for argName, argType in sdfg.arglist().items():
            arr = dace.ndarray(shape=argType.shape, dtype=argType.dtype)
            arr[:] = np.random.rand(*argType.shape).astype(argType.dtype.type)
            input_data[argName] = arr
        ground_truth = deepcopy(input_data)
        sdfg(**ground_truth)

    # Apply the transformation
    assert sdfg.apply_transformations_repeated(ConditionMapInterchange) == num_apps
    sdfg.validate()

    # Test if the behavior is unchanged if we expect eliminations
    if num_apps > 0:
        output_data = deepcopy(input_data)
        sdfg(**output_data)
        np.testing.assert_equal(output_data, ground_truth)


def test_condition_map_interchange_basic():

    @dace.program
    def tester(A: dace.float32[10], cond: dace.bool[1]):
        if cond[0]:
            for i in dace.map[0:10]:
                A[i] = A[i] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_nested():

    @dace.program
    def tester(A: dace.float32[10, 10], cond: dace.bool[1]):
        for i in dace.map[0:10]:
            if cond[0]:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_nested_dep():

    @dace.program
    def tester(A: dace.float32[10, 10]):
        for i in dace.map[0:10]:
            if A[i, 0] * A[i, 0] >= 0:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_multiple_conditions():

    @dace.program
    def tester(A: dace.float32[10, 10], cond1: dace.bool[1], cond2: dace.bool[1]):
        for i in dace.map[0:10]:
            if cond1[0] and cond2[0]:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_nested_multiple_conditions():

    @dace.program
    def tester(A: dace.float32[10, 10], cond1: dace.bool[1], cond2: dace.bool[1]):
        for i in dace.map[0:10]:
            if cond1[0]:
                for j in dace.map[0:10]:
                    if cond2[0]:
                        A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_conditional_write():

    @dace.program
    def tester(A: dace.float32[10, 10]):
        for i in dace.map[0:10]:
            if i % 2 == 0:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_multiple_writes():

    @dace.program
    def tester(A: dace.float32[10, 10], cond: dace.bool[1]):
        for i in dace.map[0:10]:
            if cond[0]:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1
                    A[i, j] = A[i, j] * 2

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_nested_dependency():

    @dace.program
    def tester(A: dace.float32[10, 10]):
        for i in dace.map[0:10]:
            if A[i, 0] > 0.5:
                for j in dace.map[1:10]:
                    A[i, j] = A[i, j] + A[i, 0]

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_no_condition():

    @dace.program
    def tester(A: dace.float32[10, 10]):
        for i in dace.map[0:10]:
            for j in dace.map[0:10]:
                A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 0)


def test_condition_map_interchange_complex_condition():

    @dace.program
    def tester(A: dace.float32[10, 10], cond1: dace.bool[1], cond2: dace.bool[1]):
        for i in dace.map[0:10]:
            if (cond1[0] or cond2[0]) and i % 2 == 0:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_dependent_condition():

    @dace.program
    def tester(A: dace.float32[10, 10]):
        for i in dace.map[0:10]:
            if A[i, 0] > 0:
                for j in dace.map[0:10]:
                    if A[i, j] < 0.5:
                        A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_multiple_nested_conditions():

    @dace.program
    def tester(A: dace.float32[10, 10], cond1: dace.bool[1], cond2: dace.bool[1]):
        for i in dace.map[0:10]:
            if cond1[0]:
                for j in dace.map[0:10]:
                    if cond2[0] and j % 2 == 0:
                        A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


def test_condition_map_interchange_non_affine_condition():

    @dace.program
    def tester(A: dace.float32[10, 10]):
        for i in dace.map[0:10]:
            if i * i < 50:
                for j in dace.map[0:10]:
                    A[i, j] = A[i, j] + 1

    _test_for_unchanged_behavior(tester, 1)


if __name__ == "__main__":
    test_condition_map_interchange_basic()
    test_condition_map_interchange_nested()
    test_condition_map_interchange_nested_dep()
    test_condition_map_interchange_multiple_conditions()
    test_condition_map_interchange_nested_multiple_conditions()
    test_condition_map_interchange_conditional_write()
    test_condition_map_interchange_multiple_writes()
    test_condition_map_interchange_nested_dependency()
    test_condition_map_interchange_no_condition()
    test_condition_map_interchange_complex_condition()
    test_condition_map_interchange_dependent_condition()
    test_condition_map_interchange_multiple_nested_conditions()
    test_condition_map_interchange_non_affine_condition()
