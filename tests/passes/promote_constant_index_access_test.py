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


def test_refuses_when_same_slot_is_live_out():
    """A post-loop read of the *same* slot ``arr[c]`` makes the loop's writes observable
    externally; the scalar-form promotion would silently drop them, so the pass refuses."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], post: dace.float64[1], scale: dace.float64[N]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl]
        post[0] = arr[1]  # live-out read at the *same* slot ``arr[1]``

    sdfg = kern.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None
    assert _num_loops(sdfg) == loops_before


def test_promotes_when_only_other_slots_are_live_out():
    """The slot-precise live-out check allows promotion when a post-loop read targets a
    *different* slot of the same array -- the loop's writes to ``arr[1]`` are dead
    externally even though ``arr`` (at other slots) is live."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], post: dace.float64[1], scale: dace.float64[N]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl]
        post[0] = arr[2]  # different slot -- arr[1] is dead post-loop

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, 'Slot-precise live-out check should let promotion fire when arr[1] is dead.'
    sdfg.validate()


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


def test_promotes_and_lifts_when_other_slot_lives_out():
    """End-to-end: the slot-precise live-out gate accepts a kernel that reads a *different*
    slot post-loop, the resulting privatized loop maps under ``LoopToMap``, and the
    numeric result matches the un-promoted Python reference."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], sink: dace.float64[1]):
        for jl in range(N):
            arr[1] = 0.002 * scale[jl]
            out[jl] = arr[1] * scale[jl] + 1.5
        sink[0] = arr[2]  # different slot -- arr[1] is dead

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, 'Slot-precise live-out check should let promotion fire.'
    n_maps = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert n_maps >= 1
    assert _num_loops(sdfg) == 0

    n = 12
    rng = np.random.default_rng(11)
    arr_in = rng.random(5)
    scale = rng.random(n)

    # Reference: pure Python.
    arr_ref = arr_in.copy()
    out_ref = np.zeros(n)
    sink_ref = np.zeros(1)
    for jl in range(n):
        arr_ref[1] = 0.002 * scale[jl]
        out_ref[jl] = arr_ref[1] * scale[jl] + 1.5
    sink_ref[0] = arr_ref[2]

    arr_run = arr_in.copy()
    out_run = np.zeros(n)
    sink_run = np.zeros(1)
    sdfg(arr=arr_run, out=out_run, scale=scale, sink=sink_run, N=n)
    assert np.allclose(out_run, out_ref)
    assert np.allclose(sink_run, sink_ref)


def test_promotes_multiple_distinct_constant_slots_of_same_array():
    """Cloudsc ``zvqx[0]``/``zvqx[1]`` pattern (minimal). The loop body uses two distinct
    constant indices of the same (5,)-shape array, with no symbolic-index access to
    ``arr`` elsewhere. The slots are independent (writes to ``arr[0]`` don't alias reads
    from ``arr[1]``); the slot-precise live-out check applies to each independently.
    The pass should promote *both* slots to their own per-iteration private scalars and
    unblock ``LoopToMap``.

    A regression here means a kernel like the cloudsc 5x ``for_767`` species fall-speed
    setup loops -- each touching ``zvqx[0]`` and ``zvqx[1]`` of the (5,) array -- stays
    sequential.
    """

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[0] = 0.002 * scale[jl]
            arr[1] = 0.003 * scale[jl]
            out[jl] = arr[0] * arr[1] + 1.5

    sdfg = kern.to_sdfg(simplify=True)
    # Sanity: pre-fix, LoopToMap refuses this loop (the bug). The exact same refusal
    # shape blocks cloudsc for_767 species loops on zvqx[0]/zvqx[1].
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, ('Multi-slot promotion should fire: arr[0] and arr[1] are '
                             'independent constant points (no symbolic-index access to '
                             'arr in the loop), each safely privatizable.')
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1, 'After multi-slot promotion the loop should be LoopToMap-eligible.'
    assert _num_loops(sdfg) == 0

    n = 12
    rng = np.random.default_rng(13)
    arr_in = rng.random(5)
    scale = rng.random(n)

    # Reference: pure Python.
    arr_ref = arr_in.copy()
    out_ref = np.zeros(n)
    for jl in range(n):
        arr_ref[0] = 0.002 * scale[jl]
        arr_ref[1] = 0.003 * scale[jl]
        out_ref[jl] = arr_ref[0] * arr_ref[1] + 1.5

    arr_run = arr_in.copy()
    out_run = np.zeros(n)
    sdfg(arr=arr_run, out=out_run, scale=scale, N=n)
    assert np.allclose(
        out_run,
        out_ref), (f'Multi-slot promotion changed the numeric result. got={out_run[:4]}, expected={out_ref[:4]}')


def test_promotes_cloudsc_for767_species_pattern():
    """The 5-species cloudsc ``for_767`` pattern, faithful to the actual SDFG shape: a
    loop over horizontal index touches all 5 constant species slots of a (5,)-shape
    transient (``zvqx[0]`` through ``zvqx[4]``) -- each species is a write + a read in
    the same iteration. With multi-slot promotion, the loop becomes 5 independent per-
    iteration scalars and ``LoopToMap`` parallelizes the kfdia-kidia range.

    Cloudsc has 5 of these (unrolled across species) all sequentially blocked on the
    same PCIA refusal; this test is the minimal faithful reproducer."""

    @dace.program
    def kern(zvqx: dace.float64[5], pre_ice: dace.float64[N], zdtgdp: dace.float64[N], out: dace.float64[N]):
        for jl in range(N):
            zvqx[0] = 0.001 * pre_ice[jl] + zdtgdp[jl]
            zvqx[1] = 0.002 * pre_ice[jl] + zdtgdp[jl]
            zvqx[2] = 0.003 * pre_ice[jl] + zdtgdp[jl]
            zvqx[3] = 0.004 * pre_ice[jl] + zdtgdp[jl]
            zvqx[4] = 0.005 * pre_ice[jl] + zdtgdp[jl]
            out[jl] = zvqx[0] + zvqx[1] + zvqx[2] + zvqx[3] + zvqx[4]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, ('Cloudsc-shape 5-slot promotion should fire (one transient '
                             'scalar per species slot); without this, for_767 stays '
                             'sequential and the species fall-speed setup blocks 5 loops.')
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1, 'Post-promotion LoopToMap should accept the species loop.'
    assert _num_loops(sdfg) == 0

    n = 10
    rng = np.random.default_rng(19)
    zvqx_in = rng.random(5)
    pre_ice = rng.random(n)
    zdtgdp = rng.random(n)

    # Reference.
    zvqx_ref = zvqx_in.copy()
    out_ref = np.zeros(n)
    for jl in range(n):
        for s in range(5):
            zvqx_ref[s] = (0.001 * (s + 1)) * pre_ice[jl] + zdtgdp[jl]
        out_ref[jl] = zvqx_ref.sum()

    zvqx_run = zvqx_in.copy()
    out_run = np.zeros(n)
    sdfg(zvqx=zvqx_run, pre_ice=pre_ice, zdtgdp=zdtgdp, out=out_run, N=n)
    assert np.allclose(
        out_run,
        out_ref), (f'5-species multi-slot promotion changed the result. got={out_run[:3]} expected={out_ref[:3]}')


def test_multi_slot_refuses_if_symbolic_access_to_same_array():
    """Safety counter-test: the multi-slot relaxation must NOT activate if *any* memlet
    accesses the same array with a non-constant (loop-var-bearing) subset -- the writes
    to ``arr[c]`` could alias ``arr[i]`` reads. Mixed-access detection stays per-array
    (not per-slot) and refuses promotion."""

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[0] = 0.002 * scale[jl]
            arr[1] = 0.003 * scale[jl]
            # Symbolic-index read of arr: could alias the constant slots above.
            out[jl] = arr[jl % 5] * scale[jl]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None, ('Mixed constant/symbolic access on the same array must refuse '
                         'promotion -- the multi-slot relaxation only applies when '
                         'every access to that array is a constant point.')


def test_promotes_through_conditional_block_in_body():
    """The cloudsc ``for_767_X`` body shape: the iteration's species-specific slot is
    written inside an ``if yrecldp_laericesed`` conditional block. PCIA must
    promote ``arr[c]`` through the conditional structure -- the post-loop reads
    are at other slots only, the conditional governs only the write, and the
    body otherwise has only loop-invariant transient work.

    Failure mode before the extension: PCIA's slot-detection finds
    ``arr[c]`` but the speculative post-promote LoopToMap check still refuses
    because the conditional write reads as a loop-carried structure.
    """

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], flag: dace.int32,
             sink: dace.float64[1]):
        for jl in range(N):
            if flag > 0:
                arr[3] = 0.002 * scale[jl]
            out[jl] = arr[3] * scale[jl]
        # arr[3] is dead post-loop; reads only target a different slot.
        sink[0] = arr[2]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, ('PCIA must promote arr[3] through the conditional-write body; '
                             'the slot is dead post-loop and the conditional is the only barrier.')
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1, ('Post-promotion LoopToMap should accept the conditional-write loop -- the '
                             "conditional doesn't introduce a loop-carried dependence on its own.")


def test_promotes_outer_loop_with_multiple_inner_constant_indexed_writes():
    """The cloudsc ``for_430`` (outer klev loop) shape: many constant-indexed
    writes to the same array (``zvqx[0..3]``) spread across the body. PCIA's
    multi-slot relaxation should promote every distinct slot independently, and
    the post-promote LoopToMap should accept the parallelised level loop.

    Failure mode before the extension: PCIA correctly identifies the slot set
    but the post-promote check fails because the loop body has additional
    structural barriers (nested conditional, multiple writes per slot)
    that L2M independently rejects.
    """

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[0] = scale[jl] * 0.1
            arr[1] = scale[jl] * 0.2
            arr[2] = scale[jl] * 0.3
            arr[3] = scale[jl] * 0.4
            out[jl] = arr[0] + arr[1] + arr[2] + arr[3]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, ('PCIA must promote four independent slots arr[0..3]; the multi-slot '
                             'relaxation handles each as a separate per-iteration scalar.')
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1

    # Verify numerical correctness end-to-end.
    n = 16
    rng = np.random.default_rng(430)
    arr_in = rng.random(5)
    scale = rng.random(n)
    arr_ref = arr_in.copy()
    out_ref = np.zeros(n)
    for jl in range(n):
        arr_ref[0] = scale[jl] * 0.1
        arr_ref[1] = scale[jl] * 0.2
        arr_ref[2] = scale[jl] * 0.3
        arr_ref[3] = scale[jl] * 0.4
        out_ref[jl] = arr_ref[0] + arr_ref[1] + arr_ref[2] + arr_ref[3]
    arr_run = arr_in.copy()
    out_run = np.zeros(n)
    sdfg(arr=arr_run, out=out_run, scale=scale, N=n)
    assert np.allclose(out_run, out_ref)


def test_promotes_multi_dim_constant_slot():
    """A 2-D array with a fully-constant slot ``arr[3, 5]`` -- both axes are
    constant integers, so the slot is a single element that can be privatised
    to a scalar. Was refused before the multi-dim relaxation
    (``_is_constant_point_subset`` only accepted 1-D Ranges).
    """

    @dace.program
    def kern(arr: dace.float64[6, 8], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[3, 5] = 0.001 * scale[jl]
            out[jl] = arr[3, 5] * scale[jl]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, 'PCIA should promote the 2-D constant slot arr[3, 5].'
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1

    n = 12
    rng = np.random.default_rng(35)
    arr_in = rng.random((6, 8))
    scale = rng.random(n)
    arr_ref = arr_in.copy()
    out_ref = np.zeros(n)
    for jl in range(n):
        arr_ref[3, 5] = 0.001 * scale[jl]
        out_ref[jl] = arr_ref[3, 5] * scale[jl]
    arr_run = arr_in.copy()
    out_run = np.zeros(n)
    sdfg(arr=arr_run, out=out_run, scale=scale, N=n)
    assert np.allclose(out_run, out_ref)


def test_promotes_through_double_nested_conditionals():
    """The cloudsc ``for_767_X`` body has TWO nested ConditionalBlocks before
    the constant-index access. PCIA's verification must use per-loop
    ``LoopToMap.can_be_applied`` (not the whole-SDFG ``match_patterns`` walk
    that has a CFG-traversal coverage gap on nested regions).

    Failure mode before the per-loop fix: whole-SDFG ``match_patterns`` did not
    surface loops nested inside multiple ConditionalBlocks even when direct
    ``can_be_applied`` accepts them, so PCIA's speculate-promote-verify-keep
    pipeline saw "no transition to mappable" and undid every promotion.
    """

    @dace.program
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N], flag_a: dace.int32, flag_b: dace.int32,
             sink: dace.float64[1]):
        for jl in range(N):
            if flag_a > 0:
                if flag_b > 0:
                    arr[2] = 0.002 * scale[jl]
            out[jl] = arr[2] * scale[jl]
        sink[0] = arr[3]  # different slot post-loop

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, ('PCIA must promote arr[2] through two nested ConditionalBlocks; the '
                             "per-loop ``_l2m_accepts`` check is the path that surfaces this match.")
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1


def test_refuses_partial_multi_dim_slot():
    """``arr[jl, 5]`` mixes a loop-variable index with a constant -- not a fixed
    slot, the loop-variable axis varies per iteration. PCIA must refuse: this
    isn't a single privatisable point, it's a 1-D sweep at axis 0."""

    @dace.program
    def kern(arr: dace.float64[16, 8], out: dace.float64[N], scale: dace.float64[N]):
        for jl in range(N):
            arr[jl, 5] = 0.002 * scale[jl]
            out[jl] = arr[jl, 5] * scale[jl]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None, ('PCIA must refuse a slot whose subset has a loop-variable index on any axis -- '
                         "promoting ``arr[jl, 5]`` to a scalar would erase the per-iteration distinction.")


_RMW_N = dace.symbol('RMW_N')


@dace.program
def _rmw_accumulator(a: dace.float64[_RMW_N], sum_out: dace.float64[_RMW_N]):
    """Textbook reduction: ``sum_out[0] = 0; for i: sum_out[0] = sum_out[0] + a[i]``.
    The single accumulator slot ``sum_out[0]`` is both READ and WRITTEN in
    the loop body and the read feeds back into the write -- the canonical
    read-modify-write recurrence."""
    sum_out[0] = 0.0
    for i in range(_RMW_N):
        sum_out[0] = sum_out[0] + a[i]


def test_refuses_reduction_accumulator_rmw():
    """PCIA must refuse to promote a slot whose body forms a read-modify-write
    recurrence (the value at iteration ``i`` is computed from the value at
    iteration ``i-1`` via the SAME slot).

    Pre-fix PCIA promoted ``sum_out[0]`` to a transient scalar without an
    epilogue writeback. The in-loop accumulation landed in a dead scalar
    alias and the caller-visible ``sum_out[0]`` stayed at whatever the seed
    wrote -- a silent value corruption.

    Asserts (a) PCIA returns ``None`` (no promotions), (b) end-to-end the
    SDFG computes the right sum. Reduction-shape slots are intentionally
    left for the downstream reduction-to-WCR path
    (``AugAssignToWCR -> LoopToMap -> PrivatizeReductionAccumulator``)
    which DOES emit init + writeback."""
    n = 64
    sdfg = _rmw_accumulator.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    sdfg.validate()
    assert res is None, (f'PCIA must refuse a read-modify-write reduction-accumulator slot; '
                         f'got promotions: {res}')

    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    sum_out = np.zeros(n)
    sdfg(a=a.copy(), sum_out=sum_out, RMW_N=n)
    assert np.isclose(sum_out[0], a.sum()), (f'value mismatch: got {sum_out[0]}, expected {a.sum()}')


def test_refuses_multi_state_rmw_through_transient():
    """PCIA must refuse an RMW recurrence whose read-to-write dataflow path
    threads through a TRANSIENT intermediate across TWO body states.

    Shape: state A reads ``acc[0]`` into transient ``t``; state B reads ``t``
    and writes ``acc[0]``. The slot's value crosses iterations via ``t``.
    The earlier Phase 2.2 intra-state check missed this -- its BFS was
    confined to a single state -- and PCIA would promote, dropping the
    accumulation. The multi-state check follows the data through ``t``.
    """
    sdfg = dace.SDFG('multi_state_rmw')
    sdfg.add_array('a', [16], dace.float64)
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_transient('t', [1], dace.float64)

    loop = LoopRegion('outer', condition_expr='i < 16', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(sdfg.add_state('entry', is_start_block=True), loop, dace.InterstateEdge())
    sdfg.add_edge(loop, sdfg.add_state('exit'), dace.InterstateEdge())

    # State A: t = acc[0]
    sa = loop.add_state('read_into_t', is_start_block=True)
    acc_r1 = sa.add_read('acc')
    t_w1 = sa.add_write('t')
    tr1 = sa.add_tasklet('passthrough', {'x'}, {'y'}, 'y = x')
    sa.add_edge(acc_r1, None, tr1, 'x', dace.Memlet('acc[0]'))
    sa.add_edge(tr1, 'y', t_w1, None, dace.Memlet('t[0]'))

    # State B: acc[0] = t + a[i]
    sb = loop.add_state('write_back')
    t_r2 = sb.add_read('t')
    a_r2 = sb.add_read('a')
    acc_w2 = sb.add_write('acc')
    tr2 = sb.add_tasklet('accumulate', {'p', 'q'}, {'r'}, 'r = p + q')
    sb.add_edge(t_r2, None, tr2, 'p', dace.Memlet('t[0]'))
    sb.add_edge(a_r2, None, tr2, 'q', dace.Memlet('a[i]'))
    sb.add_edge(tr2, 'r', acc_w2, None, dace.Memlet('acc[0]'))

    loop.add_edge(sa, sb, dace.InterstateEdge())
    sdfg.validate()

    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None, (f'PCIA must refuse a multi-state RMW recurrence; got promotions: {res}')


def test_refuses_iedge_mediated_rmw():
    """PCIA must refuse when an interstate-edge assignment inside the loop
    body reads ``arr[c]`` and propagates the value into a downstream write
    of the same slot.

    Shape: an iedge in the body has ``{sym: arr[0] + 1.0}``; a later state
    writes ``arr[0]`` using ``sym`` as a source. The accumulation crosses
    iterations through the iedge's LHS symbol. A purely in-state BFS would
    miss the iedge's RHS reference; the iedge-aware analysis catches it.
    """
    sdfg = dace.SDFG('iedge_mediated_rmw')
    sdfg.add_array('arr', [1], dace.float64)
    sdfg.add_symbol('sym', dace.float64)

    loop = LoopRegion('outer', condition_expr='i < 16', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(sdfg.add_state('entry', is_start_block=True), loop, dace.InterstateEdge())
    sdfg.add_edge(loop, sdfg.add_state('exit'), dace.InterstateEdge())

    # State A is empty; the iedge A -> B reads arr[0] in its RHS and assigns sym.
    sa = loop.add_state('iedge_src', is_start_block=True)
    sb = loop.add_state('iedge_dst')
    loop.add_edge(sa, sb, dace.InterstateEdge(assignments={'sym': 'arr[0] + 1.0'}))

    # State B: arr[0] = sym (write the slot using the iedge-derived symbol).
    sym_to_arr = sb.add_tasklet('use_sym', {}, {'y'}, 'y = sym')
    arr_w = sb.add_write('arr')
    sb.add_edge(sym_to_arr, 'y', arr_w, None, dace.Memlet('arr[0]'))

    sdfg.validate()

    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is None, (f'PCIA must refuse iedge-mediated RMW; got promotions: {res}')


def test_undo_restores_all_arr_nodes_when_state_has_several():
    """Regression: a speculative promotion that is later reverted must restore EVERY
    ``arr`` AccessNode it removed, even when one state held more than one.

    The slot-precise rewrite reroutes all of a state's ``arr[c]`` edges onto a single
    per-state scalar and removes the now-orphaned ``arr`` nodes. The undo used to
    reconnect that scalar's edges to a single best-guess ``arr`` node found via
    ``next(...)``; when the state held several removed ``arr`` nodes (cloudsc's species
    bodies read the same constant slot from multiple AccessNodes), the siblings were
    left isolated and the SDFG failed validation with "Isolated node". The fix records
    each endpoint swap precisely and restores every edge to its own origin node.

    Shape: two ``arr[1]`` read nodes + one ``arr[1]`` write node in one body state
    (slot 1), a sibling ``arr[2]`` read (forces the slot-precise path), and an
    independent loop-carried ``acc[0]`` reduction so LoopToMap still refuses AFTER the
    ``arr[1]`` promotion -- which triggers the revert path under test.
    """
    sdfg = dace.SDFG('undo_multi_node')
    sdfg.add_array('arr', [4], dace.float64, transient=False)
    sdfg.add_array('out', [8], dace.float64, transient=False)
    sdfg.add_array('acc', [1], dace.float64, transient=False)
    loop = LoopRegion(label='lp', condition_expr='jl < 8', loop_var='jl',
                      initialize_expr='jl = 0', update_expr='jl = jl + 1')
    sdfg.add_node(loop, is_start_block=True)
    b = loop.add_state(label='body', is_start_block=True)

    # Write arr[1] (from sibling arr[2]) -- no in-body read of arr[1] feeds it, so it is
    # a privatizable write, not an RMW.
    ar2 = b.add_read('arr')
    tw = b.add_tasklet('w', {'a2'}, {'a1'}, 'a1 = a2 + 1.0')
    aw1 = b.add_access('arr')
    b.add_edge(ar2, None, tw, 'a2', dace.Memlet(data='arr', subset='2'))
    b.add_edge(tw, 'a1', aw1, None, dace.Memlet(data='arr', subset='1'))

    # Two SEPARATE arr[1] read nodes in the same state -- the multi-node case.
    ar1a = b.add_read('arr'); ar1b = b.add_read('arr')
    t1 = b.add_tasklet('u1', {'a'}, {'o'}, 'o = a * 2.0')
    t2 = b.add_tasklet('u2', {'a'}, {'o'}, 'o = a * 3.0')
    o1 = b.add_access('out'); o2 = b.add_access('out')
    b.add_edge(ar1a, None, t1, 'a', dace.Memlet(data='arr', subset='1'))
    b.add_edge(ar1b, None, t2, 'a', dace.Memlet(data='arr', subset='1'))
    b.add_edge(t1, 'o', o1, None, dace.Memlet(data='out', subset='jl'))
    b.add_edge(t2, 'o', o2, None, dace.Memlet(data='out', subset='jl'))

    # Loop-carried acc reduction: L2M refuses regardless of arr, so the arr promotion is
    # reverted (the path under test). PCIA itself refuses acc (in-body RMW).
    ra = b.add_read('acc'); ro = b.add_read('out'); aw = b.add_access('acc')
    tacc = b.add_tasklet('a', {'p', 'x'}, {'n'}, 'n = p + x')
    b.add_edge(ra, None, tacc, 'p', dace.Memlet(data='acc', subset='0'))
    b.add_edge(ro, None, tacc, 'x', dace.Memlet(data='out', subset='jl'))
    b.add_edge(tacc, 'n', aw, None, dace.Memlet(data='acc', subset='0'))

    PromoteConstantIndexAccess().apply_pass(sdfg, {})

    isolated = [n.data for sd in sdfg.all_sdfgs_recursive() for st in sd.states()
                for n in st.nodes() if isinstance(n, nodes.AccessNode) and st.degree(n) == 0]
    assert not isolated, f'reverted promotion left isolated nodes: {isolated}'
    sdfg.validate()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
