# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import dace
from dace import nodes
from dace.transformation.interstate import ConditionFusion
from dace.sdfg.state import ConditionalBlock


def test_consecutive_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        if a1 > 10:
            s += 1
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_nested_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            if a1 > 10:
                s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_dependent_consecutive_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        if s == 0:
            s += 1
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have not been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2


def test_dependent_consecutive_conditions2():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        a1 = a1 + 1
        if a1 > 10:
            s += 1
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have not been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2


if __name__ == "__main__":
    test_consecutive_conditions()
    test_nested_conditions()
    test_dependent_consecutive_conditions()
    test_dependent_consecutive_conditions2()
