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
    sdfg.save("original.sdfg")
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
    sdfg.save("transformed.sdfg")
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


if __name__ == "__main__":
    test_condition_map_interchange_basic()
    test_condition_map_interchange_nested()
    test_condition_map_interchange_nested_dep()
