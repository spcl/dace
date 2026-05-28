# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.promote_constant_index_access.\
PromoteConstantIndexAccess`. Each test is driven by a small ``@dace.program`` so the SDFG
shape exercises the actual pass entry point (no hand-built SDFG plumbing)."""
import sys

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.passes.promote_constant_index_access import PromoteConstantIndexAccess

N = dace.symbol('N')


def _num_maps(sdfg: dace.SDFG) -> int:
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _num_loops(sdfg: dace.SDFG) -> int:
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def _apply_loop_to_map(sdfg: dace.SDFG) -> int:
    """Run ``LoopToMap`` once over every match and return the number of maps it produced."""
    from dace.transformation.interstate.loop_to_map import LoopToMap
    return sdfg.apply_transformations_repeated([LoopToMap])


def test_unconditional_constant_index_promoted():
    """The classic shape: a shared array is unconditionally written and read at the same
    constant index every iteration. The pass promotes; LoopToMap then accepts."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl]

    sdfg = kern.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None
    sdfg.validate()
    assert _apply_loop_to_map(sdfg) >= 1
    assert _num_loops(sdfg) < loops_before

    n = 8
    rng = np.random.default_rng(0)
    arr = rng.random(5)
    scale = rng.random(n)
    out = np.zeros(n)
    sdfg(arr=arr, out=out, scale=scale, N=n)
    # The numpy oracle: each iteration writes arr[1] = 0.002 * scale[jl] then reads it back.
    expected = (0.002 * scale) * scale
    assert np.allclose(out, expected)


def test_conditional_write_with_external_live_in():
    """The cloudsc shape: the in-loop write is gated by a runtime flag carried in a
    length-1 array. When the flag is off the unconditional read must observe the external
    live-in value of ``arr[c]``."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], flag: dace.int64[1]):
        for jl in range(N):
            if flag[0] != 0:
                arr[2] = 0.5 * scale[jl]
            out[jl] = arr[2] * scale[jl]

    sdfg = kern.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, 'expected the conditional-write slot to be promoted'
    sdfg.validate()
    # The promoted loop is now parallelizable -- LoopToMap should convert it.
    assert _apply_loop_to_map(sdfg) >= 1
    assert _num_loops(sdfg) < loops_before

    n = 8
    rng = np.random.default_rng(1)
    scale = rng.random(n)

    # Flag off: every iteration's unconditional read sees the external ``arr[2]``.
    arr0 = rng.random(5)
    out = np.zeros(n)
    flag = np.array([0], dtype=np.int64)
    sdfg(arr=arr0.copy(), out=out, scale=scale, flag=flag, N=n)
    expected_off = arr0[2] * scale
    assert np.allclose(out, expected_off)

    # Flag on: every iteration writes then reads back ``arr[2] = 0.5*scale[jl]``.
    arr1 = rng.random(5)
    out = np.zeros(n)
    flag = np.array([1], dtype=np.int64)
    sdfg(arr=arr1.copy(), out=out, scale=scale, flag=flag, N=n)
    expected_on = (0.5 * scale) * scale
    assert np.allclose(out, expected_on)


def test_refuses_when_live_out_and_opted_out():
    """``allow_live_out=False`` preserves the original strict behavior: a live-out
    ``arr[c]`` read after the loop blocks promotion, so the loop stays untouched."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], post: dace.float64[1], scale: dace.float64[N]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl]
        post[0] = arr[1]  # live-out read

    sdfg = kern.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    res = PromoteConstantIndexAccess(allow_live_out=False).apply_pass(sdfg, {})
    assert res is None
    assert _num_loops(sdfg) == loops_before  # loop stays sequential


def test_refuses_when_array_indexed_by_loop_var_elsewhere():
    """A mix of ``arr[c]`` and ``arr[jl]`` accesses in the same loop disqualifies the
    constant-index pattern -- the pass must refuse."""

    @dace.program
    def kern(arr: dace.float64[N], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * arr[jl]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None


def test_refuses_when_arr_has_wcr():
    """A reduction (WCR) edge on the constant slot is a genuine accumulation, not a
    false dependence. The pass must refuse to privatize it."""
    sdfg = dace.SDFG('refuse_wcr')
    sdfg.add_array('arr', [5], dace.float64, transient=False)
    sdfg.add_array('out', [8], dace.float64, transient=False)
    sdfg.add_array('scale', [8], dace.float64, transient=False)
    loop = LoopRegion(label='lp',
                      condition_expr='jl < 8',
                      loop_var='jl',
                      initialize_expr='jl = 0',
                      update_expr='jl = jl + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state(label='body', is_start_block=True)
    s_r = body.add_read('scale')
    a_w = body.add_access('arr')
    o_w = body.add_access('out')
    t1 = body.add_tasklet('acc', {'s'}, {'a'}, 'a = 0.002 * s')
    t2 = body.add_tasklet('use', {'a', 's'}, {'o'}, 'o = a * s')
    body.add_edge(s_r, None, t1, 's', dace.Memlet(data='scale', subset='jl'))
    body.add_edge(t1, 'a', a_w, None, dace.Memlet(data='arr', subset='1', wcr='lambda x, y: x + y'))
    body.add_edge(a_w, None, t2, 'a', dace.Memlet(data='arr', subset='1'))
    body.add_edge(s_r, None, t2, 's', dace.Memlet(data='scale', subset='jl'))
    body.add_edge(t2, 'o', o_w, None, dace.Memlet(data='out', subset='jl'))

    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None


def test_idempotent():
    """A second application is a no-op: the privatized loop is already a map (or the
    structural candidate is gone), so the pass returns ``None``."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[3] = 0.002 * scale[jl]
            out[jl] = arr[3] * scale[jl]

    sdfg = kern.to_sdfg(simplify=True)
    first = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert first is not None
    second = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert second is None


def test_numeric_correctness_then_loop_to_map_maps():
    """End-to-end: the privatized loop maps under ``LoopToMap``, compiles, and matches the
    numpy reference at full IEEE precision."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl] + 1.5

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None
    maps_added = _apply_loop_to_map(sdfg)
    assert maps_added >= 1
    sdfg.validate()
    assert _num_maps(sdfg) >= 1

    n = 12
    rng = np.random.default_rng(7)
    arr = rng.random(5)
    scale = rng.random(n)
    out = np.zeros(n)
    sdfg(arr=arr, out=out, scale=scale, N=n)
    expected = (0.002 * scale) * scale + 1.5
    assert np.allclose(out, expected)


def test_live_out_refused_when_opt_out():
    """``allow_live_out=False`` recovers the strict pre-existing behavior (refuse the
    live-out case outright) -- this remains available for callers who don't want the
    permissive writeback semantics."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], sink: dace.float64[1]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl] + 1.5
        # ``arr`` is also read *after* the loop -- the value an external consumer sees
        # would have been the loop's last iteration value sequentially.
        sink[0] = arr[1]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess(allow_live_out=False).apply_pass(sdfg, {})
    assert res is None, '``allow_live_out=False`` must refuse the live-out case.'


def test_live_out_promotion_inserts_writeback_state():
    """With ``allow_live_out=True``, the same kernel privatizes the slot AND splices a
    writeback state after the loop that restores ``arr[1]`` to its final scalar value.
    The numeric result is identical to the un-promoted reference."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], sink: dace.float64[1]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl] + 1.5
        sink[0] = arr[1]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, 'Default settings (allow_live_out=True) must accept the live-out case.'
    sdfg.validate()
    # Writeback state's label tag uniquely identifies the epilogue we just inserted.
    writeback_states = [s for sd in sdfg.all_sdfgs_recursive() for s in sd.all_states() if 'writeback' in s.label]
    assert len(writeback_states) == 1, f'Expected exactly one writeback state, got {len(writeback_states)}'

    n = 12
    rng = np.random.default_rng(11)
    arr_in = rng.random(5)
    scale = rng.random(n)

    # Reference: pure Python with sequential last-iteration semantics.
    arr_ref = arr_in.copy()
    out_ref = np.zeros(n)
    sink_ref = np.zeros(1)
    for jl in range(n):
        arr_ref[1] = 0.002 * scale[jl]
        out_ref[jl] = arr_ref[1] * scale[jl] + 1.5
    sink_ref[0] = arr_ref[1]

    # SDFG run.
    arr_run = arr_in.copy()
    out_run = np.zeros(n)
    sink_run = np.zeros(1)
    sdfg(arr=arr_run, out=out_run, scale=scale, sink=sink_run, N=n)

    assert np.allclose(out_run, out_ref)
    assert np.allclose(arr_run, arr_ref), 'Writeback must restore arr[1] to last-iteration value.'
    assert np.allclose(sink_run, sink_ref)


def test_live_out_promotion_unblocks_loop_to_map():
    """With ``allow_live_out=True``, the privatized loop becomes a Map under ``LoopToMap``.
    Numerical equivalence under permissive last-writer-wins semantics.
    """

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], sink: dace.float64[1]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl] + 1.5
        sink[0] = arr[1]

    sdfg = kern.to_sdfg(simplify=True)
    PromoteConstantIndexAccess().apply_pass(sdfg, {})
    n_maps = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert n_maps >= 1, 'Live-out promotion should unblock LoopToMap on the body loop.'
    assert _num_loops(sdfg) == 0, 'Original LoopRegion must be gone (now a Map).'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
