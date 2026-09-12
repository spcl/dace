# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.buffer_expansion.BufferExpansion`.

The pass privatises a loop-local transient scratch buffer -- one that is fully (re)written and
then read on every iteration, so it carries no value across iterations -- by giving it an extra
loop-indexed dimension. The tests cover:

* ``_loop_index`` sizing (the new dimension must hold one slot *per iteration*, an off-by-one here
  under-sizes the axis and the last iteration accesses out of bounds);
* the soundness guard ``_defined_before_read`` -- it must credit a buffer defined before it is read
  each iteration (including a fill spread across several statements) and REFUSE a genuinely
  loop-carried buffer (accumulator / recurrence / reduction / partially-written);
* the ``_expand`` primitive -- expanding a scratch buffer that ``LoopToMap`` refuses turns the loop
  into a validated, bit-exact ``Map``;
* the pass as a whole -- it never grows an SDFG whose loop ``LoopToMap`` already accepts, and it is
  value-preserving.

SDFGs are built with the frontend where a ``@dace.program`` expresses the shape and by hand where a
specific edge topology (an accumulator's read-before-write, a WCR edge) is needed.
"""
import sys

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.buffer_expansion import BufferExpansion

N = dace.symbol('N')
M = dace.symbol('M')


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _num_maps(sdfg: dace.SDFG) -> int:
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _num_loops(sdfg: dace.SDFG) -> int:
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def _apply_loop_to_map(sdfg: dace.SDFG) -> int:
    from dace.transformation.interstate.loop_to_map import LoopToMap
    return sdfg.apply_transformations_repeated([LoopToMap])


def _single_loop(sdfg: dace.SDFG) -> LoopRegion:
    loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
    assert len(loops) == 1, f'expected exactly one loop, found {len(loops)}'
    return loops[0]


def _expand_buffer(sdfg: dace.SDFG, loop: LoopRegion, name: str):
    """Invoke the pass's expansion primitive on ``name`` exactly as ``apply_pass`` would."""
    pass_obj = BufferExpansion()
    index, size = BufferExpansion._loop_index(loop)
    order = BufferExpansion._ambient_order(sdfg)
    return pass_obj._expand(sdfg, loop, name, index, size, order)


def _scratch_sdfg(buf_transient: bool, ndim: int = 1, base: int = 0):
    """A loop that fully overwrites a scratch buffer, then reads it back into ``out[i]``.

    ``for i in range(N): buf[base:base+M(,base:base+M)] = 2*a[i,...]; out[i] = sum(buf[...])``.
    With ``buf_transient=False`` the buffer is a plain array, so ``LoopToMap`` refuses the loop (the
    buffer write is not indexed by ``i``); expanding it unblocks the map. ``ndim`` picks a 1-D or
    2-D buffer; ``base`` shifts the accessed slice off zero.
    """
    sdfg = dace.SDFG(f'scratch_{ndim}d_b{base}_{ "t" if buf_transient else "n" }')
    if ndim == 1:
        sdfg.add_array('a', [N, M], dace.float64)
        sdfg.add_array('buf', [M + base], dace.float64, transient=buf_transient)
        buf_r = f'buf[{base} + j]'  # live slice is buf[base:base+M]
        a_r = 'a[i, j]'
        wmap = rmap = dict(j='0:M')
    else:
        sdfg.add_array('a', [N, M, M], dace.float64)
        sdfg.add_array('buf', [M, M], dace.float64, transient=buf_transient)
        buf_r = 'buf[j, k]'
        a_r = 'a[i, j, k]'
        wmap = rmap = dict(j='0:M', k='0:M')
    sdfg.add_array('out', [N], dace.float64)

    loop = LoopRegion('lp', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    ws = loop.add_state('write', is_start_block=True)
    rs = loop.add_state('read')
    loop.add_edge(ws, rs, dace.InterstateEdge())

    ar = ws.add_read('a')
    bw = ws.add_access('buf')
    wme, wmx = ws.add_map('wmap', wmap)
    wt = ws.add_tasklet('w', {'x'}, {'y'}, 'y = 2.0 * x')
    ws.add_memlet_path(ar, wme, wt, dst_conn='x', memlet=dace.Memlet(a_r))
    ws.add_memlet_path(wt, wmx, bw, src_conn='y', memlet=dace.Memlet(buf_r))

    br = rs.add_read('buf')
    ow = rs.add_access('out')
    rme, rmx = rs.add_map('rmap', rmap)
    rt = rs.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    rs.add_memlet_path(br, rme, rt, dst_conn='x', memlet=dace.Memlet(buf_r))
    rs.add_memlet_path(rt, rmx, ow, src_conn='y', memlet=dace.Memlet('out[i]', wcr='lambda p, q: p + q'))
    sdfg.validate()
    return sdfg, loop


# ----------------------------------------------------------------------------------------------
# _loop_index -- the off-by-one guard
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lo,hi,count', [(0, 8, 8), (3, 11, 8), (0, 1, 1)])
def test_loop_index_reports_one_slot_per_iteration(lo, hi, count):
    """The private dimension must hold exactly one slot per iteration. ``get_loop_end`` returns the
    INCLUSIVE last value, so ``size`` is ``end - start + 1`` -- dropping the ``+1`` under-sizes the
    axis by one and the final iteration writes/reads out of bounds."""

    @dace.program
    def kern(a: dace.float64[64], b: dace.float64[64]):
        for i in range(lo, hi):
            b[i] = a[i] * 2.0

    sdfg = kern.to_sdfg(simplify=True)
    loop = _single_loop(sdfg)
    index, size = BufferExpansion._loop_index(loop)
    assert size == count, f'size {size} != iteration count {count} for range({lo},{hi})'
    # The index is 0-based (``i - start``) so it addresses [0, size).
    assert (index.subs(dace.symbol('i'), hi - 1) if hasattr(index, 'subs') else index) == count - 1


def test_loop_index_symbolic_bound_full_count():
    """``for i in range(N)`` needs ``N`` slots (indices 0..N-1), not ``N-1``."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0

    loop = _single_loop(kern.to_sdfg(simplify=True))
    _, size = BufferExpansion._loop_index(loop)
    assert (size - N).simplify() == 0, f'symbolic size {size} should equal N'


def test_loop_index_refuses_non_unit_step():
    """Only unit-step loops qualify (the private dimension is indexed by ``i - start``)."""

    @dace.program
    def kern(a: dace.float64[64], b: dace.float64[64]):
        for i in range(0, 64, 2):
            b[i] = a[i] * 2.0

    loop = _single_loop(kern.to_sdfg(simplify=True))
    assert BufferExpansion._loop_index(loop) is None


# ----------------------------------------------------------------------------------------------
# _expand -- expansion is bit-exact, correctly sized, and unblocks LoopToMap
# ----------------------------------------------------------------------------------------------
def test_expand_1d_scratch_unblocks_loop_to_map_bit_exact():
    """A 1-D scratch buffer that ``LoopToMap`` refuses (non-iter-indexed write) becomes a validated
    ``Map`` after expansion, and the run matches the sequential baseline at full precision."""
    from dace.transformation.interstate.loop_to_map import LoopToMap

    n, m = 7, 5
    rng = np.random.default_rng(0)
    a = rng.random((n, m))
    expected = (2.0 * a).sum(axis=1)

    baseline, loop = _scratch_sdfg(buf_transient=False)
    assert not LoopToMap.can_be_applied_to(baseline, loop=loop), 'baseline loop should be L2M-refused'
    out_base = np.zeros(n)
    baseline(a=a.copy(), buf=np.zeros(m), out=out_base, N=n, M=m)
    assert np.allclose(out_base, expected)

    sdfg, loop = _scratch_sdfg(buf_transient=False)
    _expand_buffer(sdfg, loop, 'buf')
    exp_shape = list(sdfg.arrays['buf'].shape)
    assert len(exp_shape) == 2 and (exp_shape[0] - N).simplify() == 0, f'expected [N, M], got {exp_shape}'
    assert _apply_loop_to_map(sdfg) >= 1, 'expansion should let LoopToMap fire'
    sdfg.validate()
    assert _num_maps(sdfg) >= 1 and _num_loops(sdfg) == 0
    out_map = np.zeros(n)
    sdfg(a=a.copy(), buf=np.zeros((n, m)), out=out_map, N=n, M=m)
    assert np.allclose(out_map, expected), f'expanded map result diverged: {out_map} vs {expected}'


def test_expand_places_axis_and_sizes_to_iteration_count():
    """C-order expansion prepends the new (slowest) axis, sized to the iteration count -- so the
    final iteration's slice is in bounds (the off-by-one regression)."""
    sdfg, loop = _scratch_sdfg(buf_transient=True)
    assert list(sdfg.arrays['buf'].shape) == [M]
    _expand_buffer(sdfg, loop, 'buf')
    shape = list(sdfg.arrays['buf'].shape)
    assert len(shape) == 2, f'buffer should gain one dimension, got {shape}'
    assert (shape[0] - N).simplify() == 0, f'new leading axis must be N slots, got {shape[0]}'
    assert shape[1] == M


def test_expand_multidim_scratch_bit_exact():
    """A 2-D scratch buffer expands to 3-D and stays bit-exact through LoopToMap."""
    n, m = 4, 3
    rng = np.random.default_rng(1)
    a = rng.random((n, m, m))
    expected = (2.0 * a).sum(axis=(1, 2))

    from dace.transformation.interstate.loop_to_map import LoopToMap
    baseline, bloop = _scratch_sdfg(buf_transient=False, ndim=2)
    assert not LoopToMap.can_be_applied_to(baseline, loop=bloop), 'baseline loop should be L2M-refused'
    out_base = np.zeros(n)
    baseline(a=a.copy(), buf=np.zeros((m, m)), out=out_base, N=n, M=m)
    assert np.allclose(out_base, expected)

    sdfg, loop = _scratch_sdfg(buf_transient=False, ndim=2)
    _expand_buffer(sdfg, loop, 'buf')
    assert len(sdfg.arrays['buf'].shape) == 3
    assert _apply_loop_to_map(sdfg) >= 1, 'expansion should let LoopToMap fire'
    sdfg.validate()
    out_map = np.zeros(n)
    sdfg(a=a.copy(), buf=np.zeros((n, m, m)), out=out_map, N=n, M=m)
    assert np.allclose(out_map, expected)


def test_expand_offset_base_subset_bit_exact():
    """A buffer whose live slice is offset off zero (``buf[2:2+M]``) expands and stays bit-exact --
    the new dimension is orthogonal to the existing (non-zero-based) access."""
    n, m, base = 6, 4, 2
    rng = np.random.default_rng(2)
    a = rng.random((n, m))
    expected = (2.0 * a).sum(axis=1)

    from dace.transformation.interstate.loop_to_map import LoopToMap
    baseline, bloop = _scratch_sdfg(buf_transient=False, base=base)
    assert not LoopToMap.can_be_applied_to(baseline, loop=bloop), 'baseline loop should be L2M-refused'
    out_base = np.zeros(n)
    baseline(a=a.copy(), buf=np.zeros(m + base), out=out_base, N=n, M=m)
    assert np.allclose(out_base, expected)

    sdfg, loop = _scratch_sdfg(buf_transient=False, base=base)
    _expand_buffer(sdfg, loop, 'buf')
    assert _apply_loop_to_map(sdfg) >= 1, 'expansion should let LoopToMap fire'
    sdfg.validate()
    out_map = np.zeros(n)
    sdfg(a=a.copy(), buf=np.zeros((n, m + base)), out=out_map, N=n, M=m)
    assert np.allclose(out_map, expected)


# ----------------------------------------------------------------------------------------------
# _defined_before_read / _privatizable_buffers -- the soundness guard
# ----------------------------------------------------------------------------------------------
def _fill(state, lo, hi, name='buf'):
    w = state.add_access(name)
    me, mx = state.add_map('wm', dict(j=f'{lo}:{hi}'))
    t = state.add_tasklet('t', {}, {'y'}, 'y = 1.0')
    state.add_edge(me, None, t, None, dace.Memlet())
    state.add_memlet_path(t, mx, w, src_conn='y', memlet=dace.Memlet(f'{name}[j]'))


def _read(state, lo, hi, name='buf'):
    r = state.add_read(name)
    ow = state.add_access('out')
    me, mx = state.add_map('rm', dict(j=f'{lo}:{hi}'))
    t = state.add_tasklet('t', {'x'}, {'y'}, 'y = x')
    state.add_memlet_path(r, me, t, dst_conn='x', memlet=dace.Memlet(f'{name}[j]'))
    state.add_memlet_path(t, mx, ow, src_conn='y', memlet=dace.Memlet('out[i]', wcr='lambda p, q: p + q'))


def _fill_read_loop(fills, read_hi):
    """Build a loop that ``_fill``s ``buf`` over the given ``(lo, hi)`` chunks (each in its own
    state), then reads ``buf[0:read_hi]``. Returns ``(sdfg, loop)``."""
    sdfg = dace.SDFG('fr')
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_transient('buf', [2 * M], dace.float64)
    loop = LoopRegion('lp', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    prev = None
    for idx, (lo, hi) in enumerate(fills):
        st = loop.add_state(f'w{idx}', is_start_block=(idx == 0))
        _fill(st, lo, hi)
        if prev is not None:
            loop.add_edge(prev, st, dace.InterstateEdge())
        prev = st
    rd = loop.add_state('rd')
    loop.add_edge(prev, rd, dace.InterstateEdge())
    _read(rd, 0, read_hi)
    sdfg.validate()
    return sdfg, loop


def test_guard_accepts_multi_statement_union_fill():
    """``buf[0:M] = ...; buf[M:2*M] = ...`` read back as ``buf[0:2*M]``: the two writes TOGETHER
    cover the read, so the buffer is defined-before-read and privatisable. This is the pass's
    ``_defined_before_read`` extension for multi-statement fills.

    Detection is widened, but the ``LoopToMap`` oracle still gates *expansion*: this buffer is a
    loop-local transient that ``LoopToMap`` already privatises, so the loop is already a Map
    candidate and the pass must NOT grow it (no needless expansion)."""
    sdfg, loop = _fill_read_loop([(0, 'M'), ('M', '2*M')], '2*M')
    assert BufferExpansion._defined_before_read(loop, 'buf') is True
    assert 'buf' in BufferExpansion()._privatizable_buffers(sdfg, loop)
    shape_before = list(sdfg.arrays['buf'].shape)
    assert BufferExpansion().apply_pass(sdfg, {}) is None, 'already-mappable loop must not be grown'
    assert list(sdfg.arrays['buf'].shape) == shape_before


def test_guard_refuses_partial_fill_uninitialized_read():
    """Only ``buf[0:M]`` is written but ``buf[0:2*M]`` is read -- the upper half is uninitialised
    (carried across iterations). The guard must refuse."""
    sdfg, loop = _fill_read_loop([(0, 'M')], '2*M')
    assert BufferExpansion._defined_before_read(loop, 'buf') is False
    assert 'buf' not in BufferExpansion()._privatizable_buffers(sdfg, loop)


def test_guard_refuses_gap_tiling():
    """``buf[0:M]`` and ``buf[M+1:2*M]`` leave a hole at index ``M``; the union does not cover
    ``buf[0:2*M]`` so the guard refuses (no inequality is allowed to paper over the gap)."""
    sdfg, loop = _fill_read_loop([(0, 'M'), ('M + 1', '2*M')], '2*M')
    assert BufferExpansion._defined_before_read(loop, 'buf') is False


def _rmw_accumulator_sdfg():
    """A read-before-write scalar accumulator: ``for i: t = s; s = t + a[i]`` -- ``s`` carries its
    value across iterations through transient ``t`` (two body states)."""
    sdfg = dace.SDFG('rmw')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_transient('s', [1], dace.float64)
    sdfg.add_transient('t', [1], dace.float64)
    loop = LoopRegion('lp', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    ra = loop.add_state('read_s', is_start_block=True)
    sr = ra.add_read('s')
    tw = ra.add_write('t')
    tr = ra.add_tasklet('mv', {'x'}, {'y'}, 'y = x')
    ra.add_edge(sr, None, tr, 'x', dace.Memlet('s[0]'))
    ra.add_edge(tr, 'y', tw, None, dace.Memlet('t[0]'))
    wb = loop.add_state('write_s')
    trr = wb.add_read('t')
    arr = wb.add_read('a')
    sw = wb.add_write('s')
    at = wb.add_tasklet('acc', {'p', 'q'}, {'r'}, 'r = p + q')
    wb.add_edge(trr, None, at, 'p', dace.Memlet('t[0]'))
    wb.add_edge(arr, None, at, 'q', dace.Memlet('a[i]'))
    wb.add_edge(at, 'r', sw, None, dace.Memlet('s[0]'))
    loop.add_edge(ra, wb, dace.InterstateEdge())
    sdfg.validate()
    return sdfg, loop


def test_guard_refuses_read_before_write_accumulator():
    """The canonical accumulator (``s`` read at the top of the iteration, before it is written)
    carries a value across iterations. The guard must refuse ``s`` -- privatising it would drop the
    accumulation -- while still recognising the genuinely-scratch transient ``t`` (written then read
    each iteration) as privatisable. Expanding ``t`` cannot make the ``s``-carried loop mappable, so
    the pass's oracle keeps no expansion and ``apply_pass`` returns ``None``."""
    sdfg, loop = _rmw_accumulator_sdfg()
    assert BufferExpansion._defined_before_read(loop, 's') is False, 'carried accumulator must be refused'
    assert BufferExpansion._defined_before_read(loop, 't') is True, 'written-then-read scratch is fine'
    assert BufferExpansion()._privatizable_buffers(sdfg, loop) == ['t']
    # The loop is not parallelisable (``s`` is carried); expanding ``t`` does not change that, so the
    # speculative expansion is reverted and nothing is kept.
    assert BufferExpansion().apply_pass(sdfg, {}) is None
    assert list(sdfg.arrays['t'].shape) == [1], 'reverted expansion must restore the buffer shape'


def test_guard_refuses_wcr_reduction_edge():
    """A write-conflict (WCR/reduction) edge on the buffer is a genuine accumulation, not a scratch
    fill; the guard refuses immediately."""
    sdfg = dace.SDFG('wcr')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_transient('acc', [1], dace.float64)
    loop = LoopRegion('lp', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('b', is_start_block=True)
    ar = body.add_read('a')
    aw = body.add_access('acc')
    ow = body.add_access('out')
    t = body.add_tasklet('r', {'x'}, {'y'}, 'y = x')
    body.add_edge(ar, None, t, 'x', dace.Memlet('a[i]'))
    body.add_edge(t, 'y', aw, None, dace.Memlet('acc[0]', wcr='lambda p, q: p + q'))
    t2 = body.add_tasklet('u', {'x'}, {'y'}, 'y = x')
    body.add_edge(aw, None, t2, 'x', dace.Memlet('acc[0]'))
    body.add_edge(t2, 'y', ow, None, dace.Memlet('out[i]'))
    sdfg.validate()
    assert BufferExpansion._defined_before_read(loop, 'acc') is False


def test_recurrence_a_i_from_a_i_minus_one_not_expanded_and_preserved():
    """A first-order recurrence ``a[i] = a[i-1] + b[i]`` carries a value across iterations through
    ``a``. The pass must not expand it, and the loop must stay sequential + value-preserving."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + b[i]

    n = 16
    rng = np.random.default_rng(3)
    a0 = rng.random(n)
    b = rng.random(n)
    ref = a0.copy()
    for i in range(1, n):
        ref[i] = ref[i - 1] + b[i]

    sdfg = kern.to_sdfg(simplify=True)
    loops_before = _num_loops(sdfg)
    assert BufferExpansion().apply_pass(sdfg, {}) is None, 'a recurrence must not be expanded'
    assert _num_loops(sdfg) == loops_before
    out = a0.copy()
    sdfg(a=out, b=b.copy(), N=n)
    assert np.allclose(out, ref)


# ----------------------------------------------------------------------------------------------
# pass-level behaviour -- no needless growth, value-preserving
# ----------------------------------------------------------------------------------------------
def test_no_growth_when_loop_to_map_already_accepts():
    """A loop-local *transient* scratch buffer is one ``LoopToMap`` already privatises on its own,
    so expanding it would only grow the SDFG. The pass must leave it untouched (return ``None`` and
    keep the buffer's shape)."""
    sdfg, loop = _scratch_sdfg(buf_transient=True)
    shape_before = list(sdfg.arrays['buf'].shape)
    assert BufferExpansion().apply_pass(sdfg, {}) is None
    assert list(sdfg.arrays['buf'].shape) == shape_before, 'buffer must not grow when it does not help'


def test_pass_is_value_preserving_end_to_end():
    """Running the pass (a no-op here, since ``LoopToMap`` accepts the transient scratch loop) then
    ``LoopToMap`` compiles to a map whose result matches the sequential baseline."""

    @dace.program
    def kern(a: dace.float64[N, M], out: dace.float64[N]):
        for i in range(N):
            buf = np.empty(M, dace.float64)
            for j in range(M):
                buf[j] = a[i, j] * 2.0
            acc = 0.0
            for j in range(M):
                acc += buf[j]
            out[i] = acc

    n, m = 5, 4
    rng = np.random.default_rng(4)
    a = rng.random((n, m))
    expected = (2.0 * a).sum(axis=1)

    sdfg = kern.to_sdfg(simplify=True)
    BufferExpansion().apply_pass(sdfg, {})
    sdfg.validate()
    _apply_loop_to_map(sdfg)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a.copy(), out=out, N=n, M=m)
    assert np.allclose(out, expected)


def test_view_operand_is_not_expanded():
    """A ``data.View`` (a reshape/alias of another array) must never be expanded -- widening the
    dimension it presents to its consumer corrupts the aliased shape."""

    @dace.program
    def kern(a: dace.float64[N, M], out: dace.float64[N]):
        for i in range(N):
            row = a[i]  # a view onto a[i, :]
            out[i] = np.sum(row * 2.0)

    sdfg = kern.to_sdfg(simplify=True)
    views = [name for name, d in sdfg.arrays.items() if isinstance(d, dace.data.View)]
    result = BufferExpansion().apply_pass(sdfg, {})
    # No view may appear in any expansion result.
    if result:
        for arrs in result.values():
            assert not (set(arrs) & set(views))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
