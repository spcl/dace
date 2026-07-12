# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fail-safe refuse-guards for the data-dependent scatter reduction / WCR-normalization path.

A data-dependent scatter accumulate ``hist[bin[i]] (op)= w[i]`` under a parallel map is
emitted by the frontend as a map-body ``NestedSDFG`` holding a single-element WCR write
into a write-only output connector, with a plain ``NestedSDFG -> MapExit -> accumulator``
edge chain. Three correctness guards are exercised here:

1. ``NormalizeWCR`` must NOT apply its drop-WCR / whole-buffer ``_nnr_out`` rewrite to such
   a sink: that rewrite writes only ONE element of a per-iteration whole-array buffer and
   reads the other ``n-1`` back uninitialised (garbage). The refusal holds for ANY op
   (``+`` / ``-`` / ``*`` / ``min`` / ``max`` ...) and ANY device, so an un-privatised
   scatter (GPU, or a non-reducible op like ``-``) falls back to the correct per-element
   atomic. ``NormalizeWCRSource`` widens its skip the same way.
2. The scatter is STILL optimised to the fast OpenMP array-section ``reduction(op:hist[0:n])``
   by ``PrivatizeScatterReduction`` -- but only when the map is PARALLEL. A sequential /
   nested map has no cross-thread contention, so it is left a plain serial WCR accumulate.
3. A self-referential accumulator read -- directly OR through a View / alias of the same
   array -- is refused (a whole-buffer privatisation would make those reads see the private
   identity copy, silently dropping the live contribution).

Every guard is asserted structurally AND, where the target is runnable, numerically
bit-exact / bit-close against numpy.
"""
import os

# Pin a deterministic single-threaded run before DaCe/OpenMP initialize.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import uuid

import numpy as np
import pytest

import dace
from dace import nodes
from dace.transformation.passes.normalize_wcr import NormalizeWCR
from dace.transformation.passes.normalize_wcr_source import NormalizeWCRSource
from dace.transformation.passes.privatize_scatter_reduction import (
    PrivatizeScatterReduction, surface_scatter_reduction, is_data_dependent_scatter_sink,
    data_dependent_scatter_wcr_edge, scatter_reduction_wcr_edge, map_is_parallel, resolve_root_data)
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, bins = (dace.symbol(s, dtype=dace.int64) for s in ('N', 'bins'))


# -- Frontend kernels ---------------------------------------------------------
@dace.program
def weighted_histogram(binidx: dace.int64[N], weights: dace.float64[N]):
    hist = np.ndarray((bins, ), dtype=np.float64)
    hist[:] = 0
    for i in dace.map[0:N]:
        hist[binidx[i]] += weights[i]
    return hist


@dace.program
def subtracted_histogram(binidx: dace.int64[N], weights: dace.float64[N]):
    hist = np.ndarray((bins, ), dtype=np.float64)
    hist[:] = 0
    for i in dace.map[0:N]:
        hist[binidx[i]] -= weights[i]
    return hist


# -- Helpers ------------------------------------------------------------------
def codegen_text(sdfg: dace.SDFG) -> str:
    return "\n".join(c.code for c in sdfg.generate_code())


def unique_build(sdfg: dace.SDFG, tag: str) -> None:
    """Unique build folder so parallel test workers do not race on one .dacecache dir."""
    sdfg.build_folder = os.path.join(sdfg.build_folder + f'_{tag}_{os.getpid()}_{uuid.uuid4().hex[:6]}')


def nnr_out_arrays(sdfg: dace.SDFG):
    """Every ``_nnr_out*`` transient -- NormalizeWCR's per-iteration whole-array scatter
    buffer, the marker of the unsound rewrite -- across the SDFG and its nested SDFGs."""
    return [nm for sd in sdfg.all_sdfgs_recursive() for nm in sd.arrays if nm.startswith('_nnr_out')]


def scatter_nsdfgs(sdfg: dace.SDFG):
    """Yield ``(state, nsdfg, oc)`` for every map-body NestedSDFG output connector."""
    for sd in sdfg.all_sdfgs_recursive():
        for st in sd.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.NestedSDFG) and isinstance(st.entry_node(n), nodes.MapEntry):
                    for oc in n.out_connectors:
                        yield st, n, oc


def build_scatter(schedule, wcr: str = 'lambda x, y: (x + y)') -> dace.SDFG:
    """A runnable ``acc[idx[i]] (wcr)= w[i]`` data-dependent scatter with a chosen map
    ``schedule`` (its body is a map-body NestedSDFG holding the single-element WCR into a
    write-only ``oc``, plain ``NestedSDFG -> MapExit -> acc`` edge chain)."""
    sdfg = dace.SDFG('scatter_' + uuid.uuid4().hex[:8])
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('bins', dace.int64)
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('w', [N], dace.float64)
    sdfg.add_array('acc', [bins], dace.float64)
    st = sdfg.add_state('main')

    body = dace.SDFG('body')
    body.add_scalar('b_in', dace.int64)
    body.add_scalar('w_in', dace.float64)
    body.add_array('oc', [bins], dace.float64)
    body.add_symbol('bsym', dace.int64)
    body.add_scalar('b_scal', dace.int64, transient=True)
    s0 = body.add_state('s0', is_start_block=True)
    r = s0.add_read('b_in')
    t0 = s0.add_tasklet('rd', {'__b'}, {'__o'}, '__o = __b')
    a0 = s0.add_access('b_scal')
    s0.add_edge(r, None, t0, '__b', dace.Memlet(data='b_in', subset='0'))
    s0.add_edge(t0, '__o', a0, None, dace.Memlet(data='b_scal', subset='0'))
    s1 = body.add_state('s1')
    body.add_edge(s0, s1, dace.InterstateEdge(assignments={'bsym': 'b_scal'}))
    tw = s1.add_read('w_in')
    t1 = s1.add_tasklet('acc', {'__w'}, {'__o'}, '__o = __w')
    ow = s1.add_write('oc')
    s1.add_edge(tw, None, t1, '__w', dace.Memlet(data='w_in', subset='0'))
    s1.add_edge(t1, '__o', ow, None, dace.Memlet(data='oc', subset='bsym', wcr=wcr))

    me, mx = st.add_map('scatter', {'i': '0:N'}, schedule=schedule)
    node = st.add_nested_sdfg(body, {'b_in', 'w_in'}, {'oc'}, symbol_mapping={'bins': 'bins'})
    st.add_memlet_path(st.add_read('idx'), me, node, dst_conn='b_in', memlet=dace.Memlet(data='idx', subset='i'))
    st.add_memlet_path(st.add_read('w'), me, node, dst_conn='w_in', memlet=dace.Memlet(data='w', subset='i'))
    mx.add_in_connector('IN_acc')
    mx.add_out_connector('OUT_acc')
    st.add_edge(node, 'oc', mx, 'IN_acc', dace.Memlet(data='acc', subset='0:bins'))
    st.add_edge(mx, 'OUT_acc', st.add_write('acc'), None, dace.Memlet(data='acc', subset='0:bins'))
    sdfg.validate()
    return sdfg


def build_self_ref_via_view() -> dace.SDFG:
    """``acc[idx[i]] += acc_view[i]`` where ``acc_view`` is a View of ``acc``'s upper half.
    ``acc`` has shape ``[2*bins]``: the scatter writes ``acc[0:bins]``; the weight is read
    from the disjoint, stable upper half ``acc[bins:2*bins]`` THROUGH the view -- a
    self-reference (same underlying array, different node name) a name-only guard misses."""
    sdfg = dace.SDFG('self_ref_view_' + uuid.uuid4().hex[:8])
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('bins', dace.int64)
    sdfg.add_array('idx', [N], dace.int64)
    sdfg.add_array('acc', [2 * bins], dace.float64)
    sdfg.add_view('acc_view', [bins], dace.float64)
    st = sdfg.add_state('main')

    body = dace.SDFG('body')
    body.add_scalar('b_in', dace.int64)
    body.add_scalar('w_in', dace.float64)
    body.add_array('oc', [bins], dace.float64)
    body.add_symbol('bsym', dace.int64)
    body.add_scalar('b_scal', dace.int64, transient=True)
    s0 = body.add_state('s0', is_start_block=True)
    r = s0.add_read('b_in')
    t0 = s0.add_tasklet('rd', {'__b'}, {'__o'}, '__o = __b')
    a0 = s0.add_access('b_scal')
    s0.add_edge(r, None, t0, '__b', dace.Memlet(data='b_in', subset='0'))
    s0.add_edge(t0, '__o', a0, None, dace.Memlet(data='b_scal', subset='0'))
    s1 = body.add_state('s1')
    body.add_edge(s0, s1, dace.InterstateEdge(assignments={'bsym': 'b_scal'}))
    tw = s1.add_read('w_in')
    t1 = s1.add_tasklet('acc', {'__w'}, {'__o'}, '__o = __w')
    ow = s1.add_write('oc')
    s1.add_edge(tw, None, t1, '__w', dace.Memlet(data='w_in', subset='0'))
    s1.add_edge(t1, '__o', ow, None, dace.Memlet(data='oc', subset='bsym', wcr='lambda x, y: (x + y)'))

    me, mx = st.add_map('scatter', {'i': '0:N'}, schedule=dace.ScheduleType.CPU_Multicore)
    node = st.add_nested_sdfg(body, {'b_in', 'w_in'}, {'oc'}, symbol_mapping={'bins': 'bins'})
    st.add_memlet_path(st.add_read('idx'), me, node, dst_conn='b_in', memlet=dace.Memlet(data='idx', subset='i'))
    acc_read = st.add_read('acc')
    acc_view = st.add_access('acc_view')
    st.add_edge(acc_read, None, acc_view, 'views', dace.Memlet(data='acc', subset='bins:2*bins', other_subset='0:bins'))
    st.add_memlet_path(acc_view, me, node, dst_conn='w_in', memlet=dace.Memlet(data='acc_view', subset='i'))
    mx.add_in_connector('IN_acc')
    mx.add_out_connector('OUT_acc')
    st.add_edge(node, 'oc', mx, 'IN_acc', dace.Memlet(data='acc', subset='0:bins'))
    st.add_edge(mx, 'OUT_acc', st.add_write('acc'), None, dace.Memlet(data='acc', subset='0:bins'))
    sdfg.validate()
    return sdfg


# -- Detection unit tests -----------------------------------------------------
def test_data_dependent_scatter_sink_is_op_agnostic():
    """``is_data_dependent_scatter_sink`` fires for ANY op (the fail-safe refuse predicate),
    while ``scatter_reduction_wcr_edge`` narrows to the OpenMP-reducible ops."""
    plus = weighted_histogram.to_sdfg(simplify=True)
    minus = subtracted_histogram.to_sdfg(simplify=True)
    for sdfg, reducible in ((plus, True), (minus, False)):
        found = [(n, oc) for _, n, oc in scatter_nsdfgs(sdfg) if is_data_dependent_scatter_sink(n, oc)]
        assert len(found) == 1, f'expected exactly one data-dependent scatter sink; got {len(found)}'
        n, oc = found[0]
        assert data_dependent_scatter_wcr_edge(n, oc) is not None
        # The reducible-op filter agrees only for the associative/commutative ops.
        assert (scatter_reduction_wcr_edge(n, oc) is not None) is reducible


def test_map_is_parallel_predicate():
    """``map_is_parallel``: top-level non-Sequential map -> parallel; explicit Sequential or
    a nested map -> not parallel."""
    par = build_scatter(dace.ScheduleType.CPU_Multicore)
    seq = build_scatter(dace.ScheduleType.Sequential)
    for sdfg, expected in ((par, True), (seq, False)):
        st, me = next((s, n) for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry))
        assert map_is_parallel(st, me) is expected


# -- NormalizeWCR / NormalizeWCRSource refuse (bugs 1 + 2) ---------------------
@pytest.mark.parametrize('prog', [weighted_histogram, subtracted_histogram], ids=['plus', 'minus'])
def test_normalize_wcr_refuses_data_dependent_scatter(prog):
    """``NormalizeWCR`` alone refuses the drop-WCR / whole-buffer rewrite for a
    data-dependent multi-element scatter sink (any op) -- no ``_nnr_out`` buffer, no rewrite."""
    sdfg = prog.to_sdfg(simplify=True)
    res = NormalizeWCR().apply_pass(sdfg, {})
    assert res is None, 'NormalizeWCR must not rewrite a data-dependent scatter sink'
    assert not nnr_out_arrays(sdfg), 'no per-iteration whole-array _nnr_out buffer may be inserted'


def test_normalize_wcr_source_skips_scatter_sink_regardless_of_op():
    """``NormalizeWCRSource`` skips a data-dependent scatter sink for a NON-reducible op too:
    even with a ``-`` WCR surfaced onto the ``NestedSDFG -> MapExit`` edge, the edge is left
    direct (no interposed whole-array ``_wcr_priv``) so codegen falls back to the atomic."""
    sdfg = build_scatter(dace.ScheduleType.CPU_Multicore, wcr='lambda x, y: (x - y)')
    st, nsdfg, _ = next(iter(scatter_nsdfgs(sdfg)))
    out_edge = next(e for e in st.out_edges(nsdfg) if e.src_conn == 'oc')
    # Simulate a surfaced WCR on the outer edge chain (what a hypothetical surfacing would do).
    for e in st.memlet_path(out_edge):
        e.data.wcr = 'lambda x, y: (x - y)'
    NormalizeWCRSource().apply_pass(sdfg, {})
    edge = next(e for e in st.out_edges(nsdfg) if e.src_conn == 'oc')
    assert isinstance(edge.dst, nodes.MapExit) and edge.data.wcr is not None, \
        'scatter WCR must stay on the direct NestedSDFG -> MapExit edge (no whole-array _wcr_priv)'


# -- CPU end-to-end: parallel privatizes, refuse cases fall back --------------
def test_cpu_parallel_reducible_scatter_privatized_and_bit_exact():
    """Regression guard (do not over-refuse): the parallel ``+`` histogram STILL privatizes to
    the fast ``reduction(+:hist[0:n])`` clause, with no ``_nnr_out`` buffer, and matches numpy."""
    sdfg = weighted_histogram.to_sdfg(simplify=True)
    unique_build(sdfg, 'par_plus')
    canonicalize(sdfg, validate=True, target='cpu')
    code = codegen_text(sdfg)
    assert 'reduction(+:' in code, 'a parallel reducible scatter must keep the fast OpenMP reduction'
    assert '_nnr_out' not in code, 'no per-iteration whole-array scatter buffer'

    rng = np.random.default_rng(0)
    n, nb = 20000, 37
    binidx = rng.integers(0, nb, n).astype(np.int64)
    weights = rng.random(n)
    res = sdfg(binidx=binidx, weights=weights, N=n, bins=nb)
    ref = np.zeros(nb)
    np.add.at(ref, binidx, weights)
    assert np.allclose(res, ref), f'maxerr={np.max(np.abs(res - ref))}'


def test_cpu_subtraction_scatter_refused_and_bit_exact():
    """Bug 2: a ``-`` scatter is non-reducible, so ``PrivatizeScatterReduction`` refuses it AND
    ``NormalizeWCR`` must NOT apply the unsound whole-buffer rewrite -- it falls back to the
    correct atomic. No ``_nnr_out``, no reduction clause, and matches numpy's sequential scatter."""
    sdfg = subtracted_histogram.to_sdfg(simplify=True)
    unique_build(sdfg, 'sub')
    canonicalize(sdfg, validate=True, target='cpu')
    code = codegen_text(sdfg)
    assert '_nnr_out' not in code, 'subtraction scatter must not get a whole-array _nnr_out buffer'
    assert 'reduction(-:' not in code, 'a non-reducible op must not emit an OpenMP reduction clause'

    rng = np.random.default_rng(2)
    n, nb = 20000, 31
    binidx = rng.integers(0, nb, n).astype(np.int64)
    weights = rng.random(n)
    res = sdfg(binidx=binidx, weights=weights, N=n, bins=nb)
    ref = np.zeros(nb)
    np.subtract.at(ref, binidx, weights)
    assert np.allclose(res, ref), f'maxerr={np.max(np.abs(res - ref))}'


def test_cpu_knob_off_scatter_refused_and_bit_exact():
    """Bug 1 (CPU face): with ``privatize_scatter_reductions=False`` (the GPU default) the
    reducible ``+`` scatter is NOT surfaced, so ``NormalizeWCR`` used to mangle it into a
    whole-buffer ``_nnr_out`` reduction -> garbage. The refuse-guard now leaves the correct
    atomic: no ``_nnr_out`` buffer, and the result matches numpy."""
    sdfg = weighted_histogram.to_sdfg(simplify=True)
    unique_build(sdfg, 'knoboff')
    canonicalize(sdfg, validate=True, target='cpu', privatize_scatter_reductions=False)
    code = codegen_text(sdfg)
    assert '_nnr_out' not in code, 'knob-off scatter must not be wrapped in a whole-array buffer'

    rng = np.random.default_rng(1)
    n, nb = 20000, 29
    binidx = rng.integers(0, nb, n).astype(np.int64)
    weights = rng.random(n)
    res = sdfg(binidx=binidx, weights=weights, N=n, bins=nb)
    ref = np.zeros(nb)
    np.add.at(ref, binidx, weights)
    assert np.allclose(res, ref), f'maxerr={np.max(np.abs(res - ref))}'


def test_gpu_target_scatter_not_whole_array_buffered():
    """Bug 1 (GPU face, structural): GPU-target canonicalize has ``privatize_scatter_reductions``
    off, so ``NormalizeWCR`` must refuse the scatter (leaving the inner WCR to lower to an
    atomicAdd) rather than insert a per-iteration whole-array ``_nnr_out`` buffer. GPU compile
    is not required in this env -- assert on the SDFG structure."""
    sdfg = weighted_histogram.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target='gpu')
    assert not nnr_out_arrays(sdfg), 'GPU scatter must not be wrapped in a whole-array _nnr_out buffer'


# -- Self-reference through a View (bug 3) ------------------------------------
def test_self_reference_via_view_refused_and_bit_exact():
    """Bug 3: the map READS the accumulator through a View (``acc_view`` of ``acc``). A
    name-only self-reference check misses it, surfacing a whole-buffer reduction whose private
    identity copies drop ``acc``'s live contribution. Resolving the View to its root refuses
    the privatisation; the correct atomic fallback matches the sequential reference."""
    sdfg = build_self_ref_via_view()
    # The shape IS an otherwise-eligible reducible scatter -- so the refusal is specifically
    # the View self-reference, not an unrelated mismatch.
    st, nsdfg, _ = next(iter(scatter_nsdfgs(sdfg)))
    assert scatter_reduction_wcr_edge(nsdfg, 'oc') is not None
    assert surface_scatter_reduction(st, nsdfg, 'oc') is False, 'self-reference through a View must be refused'
    assert PrivatizeScatterReduction().apply_pass(sdfg, {}) is None
    assert not any(e.data is not None and e.data.wcr is not None for s in sdfg.states() for e in s.edges()
                   if isinstance(e.dst, nodes.MapExit)), 'no reduction WCR may be surfaced onto the MapExit'

    sdfg = build_self_ref_via_view()
    unique_build(sdfg, 'selfrefview')
    canonicalize(sdfg, validate=True, target='cpu')
    code = codegen_text(sdfg)
    assert '_nnr_out' not in code and 'reduction(' not in code

    rng = np.random.default_rng(11)
    B, n = 24, 24
    idx = rng.integers(0, B, n).astype(np.int64)
    acc = np.zeros(2 * B, dtype=np.float64)
    acc[B:] = rng.random(B)  # stable upper half read through the view
    ref = acc.copy()
    for i in range(n):
        ref[idx[i]] += acc[B + i]
    accbuf = acc.copy()
    sdfg(idx=idx.copy(), acc=accbuf, N=n, bins=B)
    assert np.array_equal(accbuf, ref), f'maxerr={np.max(np.abs(accbuf - ref))}'


# -- Sequential scatter is not privatized (refinement) ------------------------
def test_sequential_scatter_not_privatized_and_bit_exact():
    """A SEQUENTIAL scatter has no cross-thread contention, so it is left a plain serial WCR
    accumulate -- ``PrivatizeScatterReduction`` does not fire and no OpenMP reduction clause is
    emitted -- yet it stays bit-exact (and unmangled: no ``_nnr_out`` whole-buffer rewrite)."""
    sdfg = build_scatter(dace.ScheduleType.Sequential)
    assert PrivatizeScatterReduction().apply_pass(sdfg, {}) is None, 'sequential scatter must not be privatized'

    sdfg = build_scatter(dace.ScheduleType.Sequential)
    unique_build(sdfg, 'seq')
    canonicalize(sdfg, validate=True, target='cpu')
    code = codegen_text(sdfg)
    assert '_nnr_out' not in code, 'sequential scatter must not get a whole-array buffer'
    assert 'reduction(+:' not in code, 'a sequential scatter needs no OpenMP array-reduction clause'

    rng = np.random.default_rng(5)
    n, nb = 4000, 23
    idx = rng.integers(0, nb, n).astype(np.int64)
    w = rng.random(n)
    accbuf = np.zeros(nb)
    sdfg(idx=idx.copy(), w=w.copy(), acc=accbuf, N=n, bins=nb)
    ref = np.zeros(nb)
    for i in range(n):
        ref[idx[i]] += w[i]
    assert np.allclose(accbuf, ref), f'maxerr={np.max(np.abs(accbuf - ref))}'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
