# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``NormalizeLoopsAndMaps``: every map range becomes ``0:trip:1``
    while the result stays identical. Covers offset, non-unit, negative and
    symbolic steps, and a mixed multi-dimensional map. Each test compares the
    SDFG end-to-end against a deep-copied pre-pass reference run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopsAndMaps
from dace.transformation.passes.insert_assign_tasklets_at_map_boundary import InsertAssignTaskletsAtMapBoundary
from dace.transformation.passes.insert_unit_copy_assign_tasklets import InsertAssignTaskletsForUnitCopies

N, S = dace.symbol('N'), dace.symbol('S')


@dace.program
def offset_stride(A: dace.float64[40], B: dace.float64[40]):
    for i in dace.map[3:31:4]:
        B[i] = A[i] * 2.0 + 1.0


@dace.program
def negative_step(A: dace.float64[40], B: dace.float64[40]):
    # ``range`` (LoopRegion) is the valid input shape for a reverse-iterating
    # loop. ``dace.map`` would build a Map with a negative step, which is
    # invalid -- Maps must use a positive step (canonicalize then normalizes
    # the LoopRegion to ``0:trip:1`` and converts it to a Map).
    for i in range(20, 1, -2):
        B[i] = A[i] - 3.0


@dace.program
def symbolic_step(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N:S]:
        B[i] = A[i] + 5.0


@dace.program
def mixed_2d(A: dace.float64[30, 30], B: dace.float64[30, 30]):
    for i, j in dace.map[2:27:3, 5:25:2]:
        B[i, j] = A[i, j] * 4.0


@dace.program
def loop_offset_stride(A: dace.float64[40], B: dace.float64[40]):
    for i in range(3, 31, 4):
        B[i] = A[i] * 2.0 + 1.0


@dace.program
def loop_negative(A: dace.float64[40], B: dace.float64[40]):
    for i in range(20, 1, -2):
        B[i] = A[i] - 3.0


@dace.program
def loop_symbolic(A: dace.float64[N], B: dace.float64[N]):
    for i in range(0, N, S):
        B[i] = A[i] + 5.0


@dace.program
def mixed_loop_and_map(A: dace.float64[40], B: dace.float64[40], C: dace.float64[40]):
    for i in range(5, 37, 4):
        B[i] = A[i] * 2.0
    for j in dace.map[2:38:3]:
        C[j] = A[j] - 1.0


@dace.program
def many_maps(A: dace.float64[64], B: dace.float64[64]):
    for i in dace.map[0:64:1]:
        B[i] = A[i]
    for i in dace.map[1:64:2]:
        B[i] = A[i] + 1.0
    for i in dace.map[7:60:5]:
        B[i] = A[i] * 3.0
    for i in dace.map[10:50:7]:
        B[i] = A[i] - 2.0


def _map_ranges(sdfg: dace.SDFG):
    out = []
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            out.extend((b, e, s) for b, e, s in n.map.range.ranges)
    return out


def _assert_canonical(sdfg: dace.SDFG):
    for b, _e, s in _map_ranges(sdfg):
        assert b == 0, f"map start not 0: {b}"
        assert s == 1, f"map step not 1: {s}"
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfg, LoopRegion):
            assert loop_analysis.get_init_assignment(cfg) == 0, "loop start not 0"
            assert loop_analysis.get_loop_stride(cfg) == 1, "loop step not 1"


def _run(program, simplify_inputs, **kw):
    sdfg = program.to_sdfg(simplify=False)
    ref = copy.deepcopy(sdfg)

    pre = {k: v.copy() for k, v in simplify_inputs.items()}
    ref(**pre, **kw)

    # Mirror the pipeline: the preparation cleanup removes ``other_subset``
    # copies before ``NormalizeLoopsAndMaps`` runs (the pass no longer does
    # this itself).
    InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})

    changed = NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    assert changed is not None, "pass did not normalize a non-canonical map"
    sdfg.validate()
    _assert_canonical(sdfg)

    post = {k: v.copy() for k, v in simplify_inputs.items()}
    sdfg(**post, **kw)
    for k in simplify_inputs:
        assert np.allclose(post[k], pre[k]), f"mismatch on {k}"
    return pre


def test_offset_stride():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    pre = _run(offset_stride, dict(A=a, B=b))
    exp = np.full(40, -1.0)
    for i in range(3, 31, 4):
        exp[i] = a[i] * 2.0 + 1.0
    assert np.allclose(pre['B'], exp)


def test_negative_step():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    pre = _run(negative_step, dict(A=a, B=b))
    exp = np.full(40, -1.0)
    for i in range(20, 1, -2):
        exp[i] = a[i] - 3.0
    assert np.allclose(pre['B'], exp)


def test_symbolic_step():
    n, s = 64, 5
    a = np.random.rand(n)
    b = np.full(n, -1.0)
    pre = _run(symbolic_step, dict(A=a, B=b), N=n, S=s)
    exp = np.full(n, -1.0)
    for i in range(0, n, s):
        exp[i] = a[i] + 5.0
    assert np.allclose(pre['B'], exp)


def test_mixed_2d():
    a = np.random.rand(30, 30)
    b = np.full((30, 30), -1.0)
    pre = _run(mixed_2d, dict(A=a, B=b))
    exp = np.full((30, 30), -1.0)
    for i in range(2, 27, 3):
        for j in range(5, 25, 2):
            exp[i, j] = a[i, j] * 4.0
    assert np.allclose(pre['B'], exp)


def test_loop_offset_stride():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    pre = _run(loop_offset_stride, dict(A=a, B=b))
    exp = np.full(40, -1.0)
    for i in range(3, 31, 4):
        exp[i] = a[i] * 2.0 + 1.0
    assert np.allclose(pre['B'], exp)


def test_loop_negative():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    pre = _run(loop_negative, dict(A=a, B=b))
    exp = np.full(40, -1.0)
    for i in range(20, 1, -2):
        exp[i] = a[i] - 3.0
    assert np.allclose(pre['B'], exp)


def test_loop_symbolic():
    n, s = 50, 6
    a = np.random.rand(n)
    b = np.full(n, -1.0)
    pre = _run(loop_symbolic, dict(A=a, B=b), N=n, S=s)
    exp = np.full(n, -1.0)
    for i in range(0, n, s):
        exp[i] = a[i] + 5.0
    assert np.allclose(pre['B'], exp)


def test_mixed_loop_and_map():
    a = np.random.rand(40)
    b = np.full(40, -1.0)
    c = np.full(40, -1.0)
    pre = _run(mixed_loop_and_map, dict(A=a, B=b, C=c))
    eb, ec = np.full(40, -1.0), np.full(40, -1.0)
    for i in range(5, 37, 4):
        eb[i] = a[i] * 2.0
    for j in range(2, 38, 3):
        ec[j] = a[j] - 1.0
    assert np.allclose(pre['B'], eb) and np.allclose(pre['C'], ec)


def test_many_maps_varied_steps():
    a = np.random.rand(64)
    b = np.full(64, -1.0)
    pre = _run(many_maps, dict(A=a, B=b))
    exp = np.full(64, -1.0)
    for i in range(0, 64, 1):
        exp[i] = a[i]
    for i in range(1, 64, 2):
        exp[i] = a[i] + 1.0
    for i in range(7, 60, 5):
        exp[i] = a[i] * 3.0
    for i in range(10, 50, 7):
        exp[i] = a[i] - 2.0
    assert np.allclose(pre['B'], exp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
