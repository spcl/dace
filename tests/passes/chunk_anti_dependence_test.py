# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the CPU specialization ChunkAntiDependence (snapshot -> per-chunk seam buffer).

The canonical input is what ``BreakAntiDependence`` + ``LoopToMap`` leave behind: a window
snapshot copy plus a fully parallel map reading it at ``[i + 1]``.
"""
import contextlib
import os

import numpy as np

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes import BreakAntiDependence
from dace.transformation.passes.cpu_specialization import ChunkAntiDependence

N = dace.symbol('N')


@dace.program
def _shift(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i] = a[i + 1] + b[i]


@dace.program
def _shift_two(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 2):
        a[i] = a[i + 2] + b[i]


def _canonical(prog):
    """The device-neutral snapshot form: broken anti-dependence, mapped, states fused."""
    sdfg = prog.to_sdfg(simplify=True)
    BreakAntiDependence().apply_pass(sdfg, {})
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg.apply_transformations_repeated(LoopToMap)
        sdfg.simplify()
    return sdfg


def _snapshot_copies(sdfg):
    return [
        e.data for st in sdfg.all_states() for e in st.edges() if isinstance(e.dst, nodes.AccessNode)
        and '_snap' in e.dst.data and e.data is not None and not e.data.is_empty()
    ]


def _ref_shift(a, b):
    out = a.copy()
    for i in range(len(a) - 1):
        out[i] = out[i + 1] + b[i]
    return out


def test_chunk_rewrite_is_bit_exact():
    """Sequential order inside a chunk satisfies the read-ahead; only the seam needs the snapshot."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence(chunk_size=8).apply_pass(sdfg, {}) == 1
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
    assert ChunkAntiDependence(chunk_size=7).apply_pass(sdfg, {}) == 1

    n = 33
    rng = np.random.default_rng(32)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    got = a.copy()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg(a=got, b=b.copy(), N=n)
    assert np.array_equal(got, _ref_shift(a, b))


def test_chunk_rewrite_copies_one_element_per_chunk():
    """The whole-window copy becomes a stride-C copy plus the one trailing read."""
    sdfg = _canonical(_shift)
    before = _snapshot_copies(sdfg)
    assert len(before) == 1 and before[0].subset.num_elements() == dace.symbolic.pystr_to_symbolic('N - 1')

    assert ChunkAntiDependence(chunk_size=8).apply_pass(sdfg, {}) == 1
    after = _snapshot_copies(sdfg)
    assert len(after) == 2
    steps = sorted(str(m.subset[0][2]) for m in after)
    assert steps == ['1', '8'], steps
    total = sum(m.subset.num_elements() for m in after)
    at_n = int(dace.symbolic.evaluate(total, {'N': 1000}))
    assert at_n == 126, at_n  # (N - 2) / 8 chunks + the trailing read, not N - 1


def test_chunk_rewrite_never_matches_a_gpu_scheduled_map():
    """Chunking bakes in CPU scheduling; a GPU map keeps the device-neutral snapshot."""
    sdfg = _canonical(_shift)
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            n.map.schedule = dtypes.ScheduleType.GPU_Device
    assert ChunkAntiDependence(chunk_size=8).apply_pass(sdfg, {}) is None
    assert len(_snapshot_copies(sdfg)) == 1


def test_chunk_rewrite_refuses_a_wider_read_ahead_offset():
    """``a[i] = a[i+2]`` needs a run of seam elements per chunk, not a point; keep the snapshot."""
    sdfg = _canonical(_shift_two)
    assert _snapshot_copies(sdfg), 'the canonical snapshot form is the precondition'
    assert ChunkAntiDependence(chunk_size=8).apply_pass(sdfg, {}) is None


def test_chunk_rewrite_is_idempotent():
    """After the rewrite no state holds a snapshot copy feeding a map, so nothing matches again."""
    sdfg = _canonical(_shift)
    assert ChunkAntiDependence(chunk_size=8).apply_pass(sdfg, {}) == 1
    before = sdfg.hash_sdfg()
    assert ChunkAntiDependence(chunk_size=8).apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before


if __name__ == '__main__':
    test_chunk_rewrite_is_bit_exact()
    test_chunk_rewrite_is_bit_exact_when_the_chunk_size_does_not_divide()
    test_chunk_rewrite_copies_one_element_per_chunk()
    test_chunk_rewrite_never_matches_a_gpu_scheduled_map()
    test_chunk_rewrite_refuses_a_wider_read_ahead_offset()
    test_chunk_rewrite_is_idempotent()
