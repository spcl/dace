# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the CPU specialization ChunkAntiDependence (snapshot -> per-chunk seam buffer).

The canonical input is what ``BreakAntiDependence`` + ``LoopToMap`` leave behind: a window
snapshot copy plus a fully parallel map reading it at ``[i + 1]``.
"""
import contextlib
import os

import numpy as np
import sympy

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes import BreakAntiDependence
from dace.transformation.passes.cpu_specialization import ChunkAntiDependence

N = dace.symbol('N')

#: A compile-time-constant extent, short enough that the default chunk size would leave one chunk.
FIXED = 300


@dace.program
def _shift(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def _shift_two(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 2):
        a[i] = a[i + 2] + b[i]


@dace.program
def _shift_fixed(a: dace.float64[FIXED], b: dace.float64[FIXED]):
    for i in range(FIXED - 1):
        a[i] = a[i + 1] + b[i]


def _canonical(prog):
    """The device-neutral snapshot form: broken anti-dependence, mapped, states fused."""
    sdfg = prog.to_sdfg(simplify=True)
    BreakAntiDependence().apply_pass(sdfg, {})
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg.apply_transformations_repeated(LoopToMap)
        sdfg.simplify()
    return sdfg


def _copies_into(sdfg, suffix):
    return [
        e.data for st in sdfg.all_states() for e in st.edges() if isinstance(e.dst, nodes.AccessNode)
        and e.dst.data.endswith(suffix) and e.data is not None and not e.data.is_empty()
    ]


def _snapshot_copies(sdfg):
    return _copies_into(sdfg, '_antidep_snap')


def _seam_copies(sdfg):
    return _copies_into(sdfg, '_antidep_seam')


def _seam_buffer(sdfg):
    name, = [n for n in sdfg.arrays if n.endswith('_antidep_seam')]
    return sdfg.arrays[name]


def _chunk_maps(sdfg):
    """Every outer chunk map ``_tile`` introduced -- the body's and the seam iterations'."""
    return [
        n.map for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.params[0].startswith('antidep_chunk')
    ]


def _chunk_bodies(sdfg):
    """Per chunk map, what its scope holds: the ``LoopRegion`` labels, and the inner map params."""
    bodies = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not (isinstance(node, nodes.MapEntry) and node.map.params[0].startswith('antidep_chunk')):
                continue
            scope = state.scope_subgraph(node, include_entry=False, include_exit=False)
            loops = sorted(region.label for inner in scope.nodes() if isinstance(inner, nodes.NestedSDFG)
                           for region in inner.sdfg.all_control_flow_regions(recursive=True)
                           if isinstance(region, LoopRegion))
            maps = sorted(n.map.params[0] for n in scope.nodes() if isinstance(n, nodes.MapEntry))
            bodies.append((loops, maps))
    return sorted(bodies)


def _ref_shift(a, b):
    out = a.copy()
    for i in range(len(a) - 1):
        out[i] = out[i + 1] + b[i]
    return out


def test_the_in_chunk_sweep_is_a_loop_and_not_a_map():
    """A Map asserts that its iterations carry NO dependence. The in-chunk sweep carries one.

    Iteration ``i`` reads ``a[i + 1]``, which iteration ``i + 1`` overwrites, so reading the LIVE
    array inside a chunk is legal only in order -- which is the whole trade this pass makes. A
    ``Sequential`` schedule does not say that; it is a lowering hint, and consumers are entitled to
    act on the parallelism claim anyway. The tile vectorizer did: it widened the sweep and sank the
    ``a[i + 1]`` load past the ``a[i]`` store, so seven lanes in eight read values the same tile had
    just overwritten. Only the CHUNKS are parallel, so only they are a map.
    """
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    bodies = _chunk_bodies(sdfg)
    assert len(bodies) == 2, f'the body and the seam iterations each keep their chunk map: {bodies}'
    sweeps = [(loops, maps) for loops, maps in bodies if loops]
    assert len(sweeps) == 1, f'exactly one chunk map holds the in-chunk sweep: {bodies}'
    assert len(sweeps[0][0]) == 1 and not sweeps[0][1], f'the sweep is a loop, not a map: {bodies}'


def test_chunk_rewrite_is_bit_exact():
    """Sequential order inside a chunk satisfies the read-ahead; only the seam needs the snapshot."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    n = 40
    rng = np.random.default_rng(31)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    got = a.copy()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg(a=got, b=b.copy(), N=n)
    assert np.array_equal(got, _ref_shift(a, b))


def test_chunk_rewrite_is_bit_exact_when_the_chunk_size_does_not_divide():
    """The last chunk is short and its seam is the final read, which no boundary lands on."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1

    n = 33
    rng = np.random.default_rng(32)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    got = a.copy()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg(a=got, b=b.copy(), N=n)
    assert np.array_equal(got, _ref_shift(a, b))


def test_chunk_rewrite_copies_one_element_per_thread_plus_one():
    """The whole-window copy becomes a strided gather of one element per chunk, plus the trailing
    read. The point of the pass is the SIZE of that copy, so bind both symbols and count it."""
    sdfg = _canonical(_shift)
    before = _snapshot_copies(sdfg)
    assert len(before) == 1 and before[0].subset.num_elements() == dace.symbolic.pystr_to_symbolic('N - 1')

    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    after = _seam_copies(sdfg)
    assert len(after) == 2, 'the strided gather and the trailing read'

    threads = 8
    binding = {'N': 1000, dace.symbolic.NUM_THREADS_SYMBOL: threads}
    total = int(dace.symbolic.evaluate(sum(m.subset.num_elements() for m in after), binding))
    assert total == threads + 1, total  # one boundary element per thread, not N - 1


def test_seam_buffer_is_compact():
    """The buffer is indexed by chunk, so it holds one slot per thread, not one per iteration."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1

    threads = 8
    binding = {'N': 1000, dace.symbolic.NUM_THREADS_SYMBOL: threads}
    shape, = _seam_buffer(sdfg).shape
    assert int(dace.symbolic.evaluate(shape, binding)) == threads + 1, shape
    assert not any(n.endswith('_antidep_snap') for n in sdfg.arrays), 'the whole-window snapshot must be gone'

    # The gather is strided in the array and contiguous in the buffer; that pairing is the
    # whole point, and only ``other_subset`` expresses it.
    strided, = [m for m in _seam_copies(sdfg) if str(m.subset[0][2]) != '1']
    assert str(strided.other_subset[0][2]) == '1'
    assert strided.subset.num_elements() == strided.other_subset.num_elements()


def test_seam_reads_are_chunk_indexed():
    """The prologue reads slot 0; every seam iteration indexes off the outer chunk parameter."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    buf, = [n for n in sdfg.arrays if n.endswith('_antidep_seam')]

    reads = {}
    for st in sdfg.all_states():
        for e in st.edges():
            if e.data is not None and e.data.data == buf and isinstance(e.dst, nodes.AccessNode):
                reads[st.label] = e.data.subset

    pro, = [s for lbl, s in reads.items() if lbl.endswith('_prologue')]
    assert pro.num_elements() == 1 and str(pro[0][0]) == '0'

    tail, = [s for lbl, s in reads.items() if lbl.endswith('_seam_iters')]
    free = {str(s) for s in tail.free_symbols}
    assert any(s.startswith('antidep_chunk') for s in free), free
    assert 'i' not in free, 'a seam slot must not be recomputed from the array index'


def test_the_chunk_count_follows_the_thread_count():
    """The chunk stride is an even split of the extent over the threads, not a fixed size.

    Fixing the SIZE makes the seam grow with the array -- at a 4096-element chunk an XL extent of
    5.2e8 still buys 127,000 chunks and a 127,001-element copy. Fixing the COUNT at the thread
    count makes the seam thread-sized however long the array is: one boundary element per thread.
    """
    sdfg = _canonical(_shift_fixed)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    maps = _chunk_maps(sdfg)
    assert maps, 'the pass did not introduce a chunk map'
    for m in maps:
        stride = str(m.range[0][2])
        assert 'int_ceil' in stride and dace.symbolic.NUM_THREADS_SYMBOL in stride, stride
        assert 'ceil(' not in stride.replace('int_ceil(', ''), f'a raw ceiling truncates in C++: {stride}'


def test_the_seam_holds_exactly_one_slot_per_thread_plus_one():
    """``__dace_num_threads + 1``: one boundary element per thread, plus the trailing read.

    The count is the whole point of the rewrite -- the pass exists to avoid copying the window --
    so the shape is asserted literally. It must also stay a plain sum: a nested rounding here is
    what previously reached C++ as ``ceil()`` around a truncating integer division.
    """
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    shape = _seam_buffer(sdfg).shape
    assert len(shape) == 1, shape
    expected = dace.symbolic.pystr_to_symbolic(f'{dace.symbolic.NUM_THREADS_SYMBOL} + 1')
    assert dace.symbolic.equal(shape[0], expected) is not False, f'{shape[0]} != {expected}'
    assert not sympy.sympify(shape[0]).find(sympy.ceiling), f'a raw ceiling survived: {shape[0]}'


def test_short_constant_extent_stays_bit_exact():
    """The shrunk chunk still lands its seams where the reads cross, at a size that does not divide."""
    sdfg = _canonical(_shift_fixed)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    rng = np.random.default_rng(33)
    a, b = rng.standard_normal(FIXED), rng.standard_normal(FIXED)
    got = a.copy()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg(a=got, b=b.copy())
    assert np.array_equal(got, _ref_shift(a, b))


def test_chunk_rewrite_never_matches_a_gpu_scheduled_map():
    """Chunking bakes in CPU scheduling; a GPU map keeps the device-neutral snapshot."""
    sdfg = _canonical(_shift)
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            n.map.schedule = dtypes.ScheduleType.GPU_Device
    assert ChunkAntiDependence().apply_pass(sdfg, {}) is None
    assert len(_snapshot_copies(sdfg)) == 1


def test_chunk_rewrite_refuses_a_wider_read_ahead_offset():
    """``a[i] = a[i+2]`` needs a run of seam elements per chunk, not a point; keep the snapshot."""
    sdfg = _canonical(_shift_two)
    assert _snapshot_copies(sdfg), 'the canonical snapshot form is the precondition'
    assert ChunkAntiDependence().apply_pass(sdfg, {}) is None


def test_chunk_rewrite_is_idempotent():
    """After the rewrite no state holds a snapshot copy feeding a map, so nothing matches again."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence().apply_pass(sdfg, {}) == 1
    before = sdfg.hash_sdfg()
    assert ChunkAntiDependence().apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before


if __name__ == '__main__':
    test_the_in_chunk_sweep_is_a_loop_and_not_a_map()
    test_chunk_rewrite_is_bit_exact()
    test_chunk_rewrite_is_bit_exact_when_the_chunk_size_does_not_divide()
    test_chunk_rewrite_copies_one_element_per_thread_plus_one()
    test_seam_buffer_is_compact()
    test_seam_reads_are_chunk_indexed()
    test_the_chunk_count_follows_the_thread_count()
    test_the_seam_holds_exactly_one_slot_per_thread_plus_one()
    test_short_constant_extent_stays_bit_exact()
    test_chunk_rewrite_never_matches_a_gpu_scheduled_map()
    test_chunk_rewrite_refuses_a_wider_read_ahead_offset()
    test_chunk_rewrite_is_idempotent()


def test_the_full_pipeline_reaches_the_seam_rewrite():
    """Band placement is the point, so match through the real pipeline, not a hand-built graph.

    The pass used to run right after ``post_l2m`` on the argument that that band is the first to
    inline the map body and fuse the snapshot copy in. It is -- but not into the shape the matcher
    needs, and it matched nothing there. The fuse / collapse / terminal-``LoopToMap`` stages that
    follow are what put the copy and its consumer map in one state with the read spelled as a
    point, so the pass belongs in the CPU specialization stage, which runs after that whole
    pipeline. Asserting on the seam buffer here makes a future move that re-breaks the match fail
    loudly instead of quietly costing the traffic back.
    """
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.cpu_specialization import cpu_specialize

    sdfg = canonicalize(_shift.to_sdfg(simplify=True), validate=True, validate_all=False, target='cpu')
    cpu_specialize(sdfg)
    assert [n for n in sdfg.arrays if n.endswith('_antidep_seam')], 'ChunkAntiDependence did not fire'
    assert not _snapshot_copies(sdfg), 'the whole-window snapshot copy survived the rewrite'

    n = 4096
    rng = np.random.default_rng(17)
    a = rng.random(n)
    b = rng.random(n)
    got = a.copy()
    sdfg(a=got, b=b, N=n)
    assert np.array_equal(got, _ref_shift(a, b))
