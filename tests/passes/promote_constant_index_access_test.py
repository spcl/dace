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
    assert np.allclose(out_run, out_ref), (
        f'Multi-slot promotion changed the numeric result. got={out_run[:4]}, expected={out_ref[:4]}')


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
    assert np.allclose(out_run, out_ref), (
        f'5-species multi-slot promotion changed the result. got={out_run[:3]} expected={out_ref[:3]}')


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
    def kern(arr: dace.float64[5], out: dace.float64[N], scale: dace.float64[N],
             flag: dace.int32, sink: dace.float64[1]):
        for jl in range(N):
            if flag > 0:
                arr[3] = 0.002 * scale[jl]
            out[jl] = arr[3] * scale[jl]
        # arr[3] is dead post-loop; reads only target a different slot.
        sink[0] = arr[2]

    sdfg = kern.to_sdfg(simplify=True)
    res = PromoteConstantIndexAccess().apply_pass(sdfg, {})
    assert res is not None, (
        'PCIA must promote arr[3] through the conditional-write body; '
        'the slot is dead post-loop and the conditional is the only barrier.')
    maps_added = _apply_loop_to_map(sdfg)
    sdfg.validate()
    assert maps_added >= 1, (
        'Post-promotion LoopToMap should accept the conditional-write loop -- the '
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
    assert res is not None, (
        'PCIA must promote four independent slots arr[0..3]; the multi-slot '
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
    assert res is None, (
        'PCIA must refuse a slot whose subset has a loop-variable index on any axis -- '
        "promoting ``arr[jl, 5]`` to a scalar would erase the per-iteration distinction.")


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
