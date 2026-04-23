# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the MoveIfIntoMap transformation.

These mirror the ICON ``_for_it_44`` motif: a conditional block that lives in
the body of an outer map and guards an inner map. The transformation pushes
the guard past the inner map so the two maps can be fused/collapsed by later
passes.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import MoveIfIntoMap


def _count_conditional_blocks(sdfg: dace.SDFG) -> int:
    total = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, ConditionalBlock):
            total += 1
    return total


def _inner_nsdfg_contains_conditional(outer_sdfg: dace.SDFG) -> bool:
    """After applying the transformation the conditional block should live
    inside an inner nested SDFG (the body of the innermost map)."""
    for node, _ in outer_sdfg.all_nodes_recursive():
        if not isinstance(node, NestedSDFG):
            continue
        # Look one level down: we want an NSDFG whose inner SDFG contains a
        # ConditionalBlock at its top level.
        inner = node.sdfg
        for block in inner.nodes():
            if isinstance(block, ConditionalBlock):
                return True
    return False


def _run_and_compare(prog, inputs, expected_apps: int, simplify: bool = False):
    sdfg: dace.SDFG = prog.to_sdfg(simplify=simplify)
    sdfg.validate()

    reference = copy.deepcopy(inputs)
    sdfg(**reference)

    sdfg2: dace.SDFG = prog.to_sdfg(simplify=simplify)
    applied = sdfg2.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == expected_apps, f"expected {expected_apps} applications, got {applied}"
    sdfg2.validate()

    transformed = copy.deepcopy(inputs)
    sdfg2(**transformed)

    for key in reference:
        np.testing.assert_allclose(transformed[key],
                                   reference[key],
                                   rtol=1e-5,
                                   atol=1e-6,
                                   err_msg=f"Mismatch for {key}")

    return sdfg2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_move_if_into_map_basic():
    """Core _for_it_44-style pattern: outer map, scalar guard, inner map."""

    N, M = 6, 5

    @dace.program
    def tester(A: dace.float64[N, M], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] + 1.0

    rng = np.random.default_rng(0)
    inputs_on = {
        "A": rng.random((N, M)).copy(),
        "cond": np.int32(1),
    }
    inputs_off = {
        "A": rng.random((N, M)).copy(),
        "cond": np.int32(0),
    }

    sdfg_on = _run_and_compare(tester, inputs_on, expected_apps=1)
    _run_and_compare(tester, inputs_off, expected_apps=1)

    # Structural assertion: the conditional now lives inside a nested SDFG.
    assert _count_conditional_blocks(sdfg_on) == 1
    assert _inner_nsdfg_contains_conditional(sdfg_on)


def test_move_if_into_map_symbolic_condition():
    """Condition uses a map parameter + outer symbol combo."""

    N, M = 8, 4

    @dace.program
    def tester(A: dace.float64[N, M], threshold: dace.int32):
        for i in dace.map[0:N]:
            if i < threshold:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] * 2.0

    rng = np.random.default_rng(1)
    inputs = {
        "A": rng.random((N, M)).copy(),
        "threshold": np.int32(5),
    }

    sdfg = _run_and_compare(tester, inputs, expected_apps=1)
    assert _inner_nsdfg_contains_conditional(sdfg)


def test_move_if_into_map_multiple_reads_and_writes():
    """Body has multiple access nodes both sides of the inner nested SDFG."""

    N, M = 4, 4

    @dace.program
    def tester(A: dace.float64[N, M], B: dace.float64[N, M], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] + B[i, j]

    rng = np.random.default_rng(2)
    inputs = {
        "A": rng.random((N, M)).copy(),
        "B": rng.random((N, M)).copy(),
        "cond": np.int32(1),
    }

    sdfg = _run_and_compare(tester, inputs, expected_apps=1)
    assert _inner_nsdfg_contains_conditional(sdfg)


def test_move_if_into_map_no_apply_missing_inner_map():
    """No inner map in the branch: should not match."""

    N = 6

    @dace.program
    def tester(A: dace.float64[N], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                A[i] = A[i] + 1.0

    sdfg: dace.SDFG = tester.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 0


def test_move_if_into_map_no_apply_else_branch():
    """Conditional has an else branch -> not supported."""

    N, M = 4, 4

    @dace.program
    def tester(A: dace.float64[N, M], cond: dace.int32):
        for i in dace.map[0:N]:
            if cond == 1:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] + 1.0
            else:
                for j in dace.map[0:M]:
                    A[i, j] = A[i, j] - 1.0

    sdfg: dace.SDFG = tester.to_sdfg(simplify=False)
    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 0


if __name__ == "__main__":
    test_move_if_into_map_basic()
    test_move_if_into_map_symbolic_condition()
    test_move_if_into_map_multiple_reads_and_writes()
    test_move_if_into_map_no_apply_missing_inner_map()
    test_move_if_into_map_no_apply_else_branch()
    print("All tests passed.")
