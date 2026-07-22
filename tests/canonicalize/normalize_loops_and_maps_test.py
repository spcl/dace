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


@dace.program
def _strided_and_offset_maps(A: dace.float64[64], B: dace.float64[64], C: dace.float64[64]):
    for i in dace.map[0:64:2]:  # non-unit step -> normalized (step folded into index)
        B[i] = A[i] * 2.0
    for i in dace.map[3:60:1]:  # unit step, non-zero base -> LEFT ALONE (strided-only)
        C[i] = A[i] + 1.0


def test_normalize_strided_maps_only_touches_strided_maps():
    """``NormalizeStridedMaps`` folds a non-unit map step into the index and
    leaves unit-step maps -- even with a non-zero base -- untouched (unlike the
    parent ``NormalizeLoopsAndMaps``, which also zero-bases them)."""
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeStridedMaps
    a = np.random.rand(64)
    sdfg = _strided_and_offset_maps.to_sdfg(simplify=False)
    ref = copy.deepcopy(sdfg)
    pre = dict(A=a.copy(), B=np.full(64, -1.0), C=np.full(64, -1.0))
    ref(**pre)

    InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})
    changed = NormalizeStridedMaps().apply_pass(sdfg, {})
    assert changed is not None, "did not normalize the strided map"
    sdfg.validate()

    ranges = _map_ranges(sdfg)
    assert all(str(s) == "1" for _b, _e, s in ranges), f"a non-unit map step survived: {ranges}"
    # The unit-step non-zero-base map keeps its base (only strided maps are touched).
    assert any(str(b) == "3" for b, _e, _s in ranges), f"a unit-step non-zero-base map was normalized: {ranges}"

    post = dict(A=a.copy(), B=np.full(64, -1.0), C=np.full(64, -1.0))
    sdfg(**post)
    assert np.allclose(post['B'], pre['B']) and np.allclose(post['C'], pre['C'])


def test_normalize_strided_maps_leaves_loops_untouched():
    """``NormalizeStridedMaps`` is map-only: a strided ``LoopRegion`` counter is
    left as-is (only maps are normalized)."""
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeStridedMaps
    sdfg = loop_offset_stride.to_sdfg(simplify=False)  # a strided for-loop, no maps
    changed = NormalizeStridedMaps().apply_pass(sdfg, {})
    assert changed is None, "strided LoopRegion was wrongly normalized by the map-only pass"
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfg, LoopRegion):
            assert loop_analysis.get_loop_stride(cfg) != 1 or loop_analysis.get_init_assignment(cfg) != 0, \
                "loop counter was normalized"


@dace.program
def loop_bounds_unit(A: dace.float64[N]):
    for i in range(2, N - 2):
        A[i] = A[i - 1] + 1.0  # sequential (recurrence): stays a loop


@dace.program
def loop_bounds_strided(A: dace.float64[N]):
    for i in range(1, N - 1, 2):
        A[i] = A[i] * 3.0


@dace.program
def two_offset_loops(A: dace.float64[N], B: dace.float64[N]):
    for i in range(0, N - 2):  # 0:N-2
        A[i + 1] = A[i] + B[i + 1]
    for j in range(1, N - 1):  # 1:N-1  -- same extent, offset by 1
        B[j] = B[j - 1] + A[j]


def test_normalize_loop_bounds_rebases_keeping_stride():
    """``NormalizeLoopBounds`` rebases a loop counter to 0 while KEEPING its stride
    (unlike ``NormalizeLoopsAndMaps``, which folds the step into the index)."""
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopBounds
    for prog, expect_stride in ((loop_bounds_unit, '1'), (loop_bounds_strided, '2')):
        sdfg = prog.to_sdfg(simplify=True)
        assert NormalizeLoopBounds().apply_pass(sdfg, {}) == 1
        loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
        assert len(loops) == 1
        assert str(loop_analysis.get_init_assignment(loops[0])) == '0'
        assert str(loop_analysis.get_loop_stride(loops[0])) == expect_stride


def test_normalize_loop_bounds_value_preserving():
    """The rebase (``i -> start + i``) is value-preserving end-to-end."""
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopBounds
    n = 12
    for prog in (loop_bounds_unit, loop_bounds_strided):
        rng = np.random.default_rng(4)
        a = rng.standard_normal(n)
        ref = prog.to_sdfg(simplify=True)
        b = a.copy()
        ref.compile()(A=b, N=n)
        sdfg = prog.to_sdfg(simplify=True)
        NormalizeLoopBounds().apply_pass(sdfg, {})
        got = a.copy()
        sdfg.compile()(A=got, N=n)
        assert np.allclose(got, b)


def test_normalize_loop_bounds_idempotent_and_equalizes_ranges():
    """Re-running is a no-op (already-0-based loops are skipped), and two loops of
    the same extent but different offset are rebased to the SAME range -- the
    enabler for same-range ``LoopFusion``."""
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopBounds
    sdfg = two_offset_loops.to_sdfg(simplify=True)
    # Only the ``1:N-1`` loop is rebased; the ``0:N-2`` loop is already 0-based.
    assert NormalizeLoopBounds().apply_pass(sdfg, {}) == 1
    assert NormalizeLoopBounds().apply_pass(sdfg, {}) is None  # idempotent
    loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    spans = {(str(loop_analysis.get_init_assignment(l)), str(loop_analysis.get_loop_end(l)),
              str(loop_analysis.get_loop_stride(l))) for l in loops}
    assert len(spans) == 1, f"ranges not equalized: {spans}"
    assert next(iter(spans))[0] == '0'


def test_normalize_loop_bounds_leaves_zero_based_untouched():
    """A loop already based at 0 is not rewritten (any stride)."""
    from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeLoopBounds

    @dace.program
    def already_zero(A: dace.float64[N]):
        for i in range(0, N):
            A[i] = A[i] + 1.0

    sdfg = already_zero.to_sdfg(simplify=True)
    assert NormalizeLoopBounds().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
