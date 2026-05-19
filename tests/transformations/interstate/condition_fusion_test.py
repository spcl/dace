# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np

import dace
from dace.transformation.interstate import ConditionFusion
from dace.sdfg.state import ConditionalBlock


def _branch_conditions(sdfg):
    return [
        cnd.as_string for cb, _ in sdfg.all_nodes_recursive() if isinstance(cb, ConditionalBlock)
        for cnd, _ in cb.branches if cnd is not None
    ]


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


def test_consecutive_conditions2():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        else:
            s += 2
        if a1 > 10:
            s += 1
        else:
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


def test_deeply_nested_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 + 2 > 10:
            if a1 + 2 > 10:
                if a0 + 1 > 10:
                    if a1 + 1 > 10:
                        if a0 > 10:
                            if a1 > 10:
                                s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 6

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


def test_independent_consecutive_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        a3 = a[2] + 1
        if a1 > 10:
            s += 1
        a[2] = s + a3

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


def test_mixed_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 + 1 > 10:
            if a1 + 1 > 10:
                if a0 > 10:
                    if a1 > 10:
                        s += 2
        if a0 + 1 > 5:
            if a1 + 1 > 5:
                if a0 > 5:
                    if a1 > 5:
                        s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 8

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.simplify()  # To fuse empty states
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_identical_guards_no_duplicated_conjunct():
    """ConditionFusion of two identical guards must not emit a redundant
    ``(c) and (c)``: no fused branch condition repeats the predicate. The
    satisfiable branch is exactly the original predicate. Value-preserving."""

    @dace.program
    def tester(a: dace.float64[4]):
        c = a[0]
        s = 0.0
        if c > 10:
            s += 1.0
        if c > 10:
            s += 2.0
        a[3] = s

    base = tester.to_sdfg(simplify=True)
    sdfg = tester.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    cbs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cbs) == 1, f"guards not merged: {len(cbs)}"
    conds = _branch_conditions(sdfg)
    # ConditionFusion simplifies its own output: the redundant ``(c) and
    # (c)`` collapses and the unsatisfiable cross-terms are dropped, so no
    # surviving condition is a conjunction.
    assert all(' and ' not in c for c in conds), f"non-minimal condition: {conds}"

    for v in (5.0, 20.0):  # guard not-taken / taken
        ref, out = np.array([v, 0, 0, 0]), np.array([v, 0, 0, 0])
        copy.deepcopy(base)(a=ref)
        sdfg(a=out)
        assert np.allclose(out, ref), f"value mismatch at a[0]={v}: {out} vs {ref}"


def test_three_identical_guards_no_duplicated_conjunct():
    """Three chained identical guards: ConditionFusion still emits no
    branch with a duplicated predicate, and is value-preserving."""

    @dace.program
    def tester(a: dace.float64[5]):
        c = a[0]
        s = 0.0
        if c > 10:
            s += 1.0
        if c > 10:
            s += 2.0
        if c > 10:
            s += 4.0
        a[4] = s

    base = tester.to_sdfg(simplify=True)
    sdfg = tester.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    cbs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cbs) == 1, f"guards not merged: {len(cbs)}"
    conds = _branch_conditions(sdfg)
    assert all(' and ' not in c for c in conds), f"non-minimal condition: {conds}"

    for v in (5.0, 20.0):
        ref, out = np.array([v, 0, 0, 0, 0]), np.array([v, 0, 0, 0, 0])
        copy.deepcopy(base)(a=ref)
        sdfg(a=out)
        assert np.allclose(out, ref), f"value mismatch at a[0]={v}: {out} vs {ref}"


if __name__ == "__main__":
    test_consecutive_conditions()
    test_consecutive_conditions2()
    test_nested_conditions()
    test_deeply_nested_conditions()
    test_dependent_consecutive_conditions()
    test_dependent_consecutive_conditions2()
    test_independent_consecutive_conditions()
    test_mixed_conditions()
    test_identical_guards_no_duplicated_conjunct()
    test_three_identical_guards_no_duplicated_conjunct()
