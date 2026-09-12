# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" End-to-end value-preservation tests for ``NormalizeLoopsAndMaps`` on
    indirect-gather kernels.

    These mirror the ICON ``z_v_grad_w`` neighbour-gather pattern
    (``out[i, j] = c1 * w[idx[i, 0], j] - c2 * w[idx[i, 1], j]``) where the
    map parameter appears inside an indirect subscript in a subset of the
    array dimensions only. The map ranges use non-trivial bounds: non-zero
    start, non-unit stride, negative (reverse) step, a mixed
    multi-dimensional combination, negative start/end bounds, and a
    ``dace.symbol`` (run-time) stride. The ``LoopRegion`` path is exercised
    too (negative-bound and symbolic-step ``range`` loops over the same
    gather body). Each test proves the rewrite is value-preserving (numeric
    equality vs a pure-numpy oracle and vs a deep-copied pre-pass SDFG run)
    and that every map range is canonicalized to ``0:trip:1`` (and every
    surviving ``LoopRegion`` to a ``0 : n : 1`` counter).
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

N, M, S = dace.symbol('N'), dace.symbol('M'), dace.symbol('S')


@dace.program
def gather_nonzero_start(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    for i, j in dace.map[2:N, 0:M]:
        b[i, j] = w[cidx[i, 0], j] + 1.0


@dace.program
def gather_nonunit_stride(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    for i, j in dace.map[0:N:3, 0:M]:
        b[i, j] = 2.0 * w[cidx[i, 0], j] - 0.5 * w[cidx[i, 1], j]


@dace.program
def gather_negative_step(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # Reverse iteration is expressed as a ``range`` (LoopRegion); a Map with a
    # negative step would be rejected at construction time. The
    # ``NormalizeLoopsAndMaps`` stage rewrites the loop to a positive-step
    # ``0:N:1`` Map by inverting the subscript.
    for i in range(N - 1, -1, -1):
        for j in range(0, M):
            b[i, j] = 3.0 * w[cidx[i, 0], j] - w[cidx[i, 1], j]


@dace.program
def gather_mixed_dims(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # Offset on dim 0, non-unit stride on dim 1.
    for i, j in dace.map[2:N, 0:M:2]:
        b[i, j] = w[cidx[i, 0], j] - 2.0 * w[cidx[i, 1], j]


@dace.program
def gather_negative_start(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # Map parameter range *starts and ends negative* (``-3 : N-3``). The
    # gather subscript shifts it back in-range (``cidx[i + 3, 0]``), so the
    # normalization (``i -> -3 + 1*i``) must compose correctly with the
    # already-present ``+ 3`` index arithmetic.
    for i, j in dace.map[-3:N - 3, 0:M]:
        b[i + 3, j] = w[cidx[i + 3, 0], j] + 1.0


@dace.program
def gather_negative_offset_step(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # Negative reverse step combined with a negative subscript offset:
    # the parameter walks ``N-1, N-3, ..., 1`` (odd values for odd ``N``,
    # all ``>= 1``) and the gather subscript reads ``cidx[i - 1, ...]`` (a
    # genuine negative offset that stays in ``[0, N)`` because ``i >= 1``).
    # Stresses a negative stride and a negative index offset together.
    # Reverse iteration is a ``range`` (LoopRegion); a Map with a negative
    # step would be invalid.
    for i in range(N - 1, 0, -2):
        for j in range(0, M):
            b[i, j] = 3.0 * w[cidx[i - 1, 0], j] - w[cidx[i - 1, 1], j]


@dace.program
def gather_symbolic_step(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # The stride is a free ``dace.symbol`` (``S``), only known at call time.
    # ``NormalizeLoopsAndMaps`` must form the symbolic trip count
    # ``floor((N-1-0)/S)+1`` and the substitution ``i -> 0 + S*i`` without
    # collapsing ``S`` to a literal.
    for i, j in dace.map[0:N:S, 0:M]:
        b[i, j] = 2.0 * w[cidx[i, 0], j] - 0.5 * w[cidx[i, 1], j]


@dace.program
def loop_gather_negative_start(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # Sequential ``range`` form (LoopRegion path) with a negative start/end.
    for i in range(-3, N - 3):
        for j in range(M):
            b[i + 3, j] = w[cidx[i + 3, 0], j] + 1.0


@dace.program
def loop_gather_symbolic_step(w: dace.float64[N, M], cidx: dace.int32[N, 2], b: dace.float64[N, M]):
    # Sequential ``range`` form with a ``dace.symbol`` stride over the
    # neighbour-gather body (exercises ``_normalize_loop``).
    for i in range(0, N, S):
        for j in range(M):
            b[i, j] = 2.0 * w[cidx[i, 0], j] - 0.5 * w[cidx[i, 1], j]


def _map_entries(sdfg: dace.SDFG):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _assert_canonical(sdfg: dace.SDFG):
    entries = _map_entries(sdfg)
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert entries or loops, "no MapEntry or LoopRegion found in SDFG"
    for me in entries:
        for b, _e, s in me.map.range.ranges:
            assert b == 0, f"map start not 0: {me.map.label} {b}"
            assert s == 1, f"map step not 1: {me.map.label} {s}"
    for loop in loops:
        assert loop_analysis.get_init_assignment(loop) == 0, f"loop start not 0: {loop.label}"
        assert loop_analysis.get_loop_stride(loop) == 1, f"loop step not 1: {loop.label}"


def _run(program, w, cidx, n, m, oracle, **symbols):
    sdfg = program.to_sdfg(simplify=True)

    # Reference: deep-copied SDFG run before the pass.
    ref = copy.deepcopy(sdfg)
    b_ref = np.full((n, m), -7.0)
    ref(w=w.copy(), cidx=cidx.copy(), b=b_ref, N=n, M=m, **symbols)
    assert np.allclose(b_ref, oracle), "pre-pass SDFG disagrees with numpy oracle"

    # Mirror the pipeline: the preparation cleanup removes ``other_subset``
    # copies before ``NormalizeLoopsAndMaps`` runs.
    InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})
    InsertAssignTaskletsForUnitCopies().apply_pass(sdfg, {})

    changed = NormalizeLoopsAndMaps().apply_pass(sdfg, {})
    assert changed is not None, "pass did not normalize a non-canonical map/loop"
    sdfg.validate()
    _assert_canonical(sdfg)

    b_post = np.full((n, m), -7.0)
    sdfg(w=w.copy(), cidx=cidx.copy(), b=b_post, N=n, M=m, **symbols)
    assert np.allclose(b_post, b_ref), "post-pass SDFG diverges from pre-pass reference"
    assert np.allclose(b_post, oracle), "post-pass SDFG disagrees with numpy oracle"


def test_gather_nonzero_start():
    rng = np.random.default_rng(1)
    n, m = 12, 7
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(2, n):
        for j in range(0, m):
            oracle[i, j] = w[cidx[i, 0], j] + 1.0
    _run(gather_nonzero_start, w, cidx, n, m, oracle)


def test_gather_nonunit_stride():
    rng = np.random.default_rng(2)
    n, m = 13, 6
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(0, n, 3):
        for j in range(0, m):
            oracle[i, j] = 2.0 * w[cidx[i, 0], j] - 0.5 * w[cidx[i, 1], j]
    _run(gather_nonunit_stride, w, cidx, n, m, oracle)


def test_gather_negative_step():
    rng = np.random.default_rng(3)
    n, m = 11, 5
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(n - 1, -1, -1):
        for j in range(0, m):
            oracle[i, j] = 3.0 * w[cidx[i, 0], j] - w[cidx[i, 1], j]
    _run(gather_negative_step, w, cidx, n, m, oracle)


def test_gather_mixed_dims():
    rng = np.random.default_rng(4)
    n, m = 14, 9
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(2, n):
        for j in range(0, m, 2):
            oracle[i, j] = w[cidx[i, 0], j] - 2.0 * w[cidx[i, 1], j]
    _run(gather_mixed_dims, w, cidx, n, m, oracle)


def test_gather_negative_start():
    rng = np.random.default_rng(5)
    n, m = 12, 7
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(-3, n - 3):
        for j in range(0, m):
            oracle[i + 3, j] = w[cidx[i + 3, 0], j] + 1.0
    _run(gather_negative_start, w, cidx, n, m, oracle)


def test_gather_negative_offset_step():
    rng = np.random.default_rng(6)
    n, m = 11, 8
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(n - 1, 0, -2):
        for j in range(0, m):
            oracle[i, j] = 3.0 * w[cidx[i - 1, 0], j] - w[cidx[i - 1, 1], j]
    _run(gather_negative_offset_step, w, cidx, n, m, oracle)


def test_gather_symbolic_step():
    rng = np.random.default_rng(7)
    n, m, s = 15, 6, 3
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(0, n, s):
        for j in range(0, m):
            oracle[i, j] = 2.0 * w[cidx[i, 0], j] - 0.5 * w[cidx[i, 1], j]
    _run(gather_symbolic_step, w, cidx, n, m, oracle, S=s)


def test_loop_gather_negative_start():
    rng = np.random.default_rng(8)
    n, m = 11, 5
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(-3, n - 3):
        for j in range(0, m):
            oracle[i + 3, j] = w[cidx[i + 3, 0], j] + 1.0
    _run(loop_gather_negative_start, w, cidx, n, m, oracle)


def test_loop_gather_symbolic_step():
    rng = np.random.default_rng(9)
    n, m, s = 16, 7, 4
    w = rng.random((n, m))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, m), -7.0)
    for i in range(0, n, s):
        for j in range(0, m):
            oracle[i, j] = 2.0 * w[cidx[i, 0], j] - 0.5 * w[cidx[i, 1], j]
    _run(loop_gather_symbolic_step, w, cidx, n, m, oracle, S=s)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
