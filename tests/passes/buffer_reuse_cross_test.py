# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for buffer_reuse_cross_pass — cross-size, cross-dtype sub-allocation reuse.

buffer_reuse_cross_pass extends buffer_reuse_same_pass to cross-size donors: a
larger just-freed buffer can serve multiple smaller upcoming allocations at
non-overlapping byte offsets (strict bump allocator). Donor selection is
best-fit by remaining capacity.
"""
from __future__ import annotations

import numpy as np
import pytest

import dace
from dace.sdfg import SDFG, InterstateEdge
from dace.libraries.allocation import make_explicit, buffer_reuse_same_pass
from dace.libraries.allocation.reuse import (
    _AllocEntry,
    _FreeEntry,
    _greedy_cross_size_scan,
    _extract_liveness,
    buffer_reuse_cross_pass,
)


def _two_size_sequential_sdfg(name: str, big: int = 16, small: int = 8) -> SDFG:
    """Sequential SDFG with separated A-read and B-write states so A's
    last-use block and B's first-use block are distinct.

    init -> write_A -> read_A -> write_B -> read_B -> done

    A is (big,) float64; B is (small,) float64.  ``make_explicit`` places
    A's free and B's alloc on the *same* edge (read_A → write_B) which
    satisfies the strict edge-order safety check in buffer_reuse_cross_pass
    A's last-use block (read_A) is strictly before B's first-use block
    (write_B) in topological order.
    """
    sdfg = SDFG(name)
    sdfg.add_array('A', [big], dace.float64, transient=True)
    sdfg.add_array('B', [small], dace.float64, transient=True)
    sdfg.add_array('out', [small], dace.float64, transient=False)

    init = sdfg.add_state('init', is_start_block=True)
    wA = sdfg.add_state('write_A')
    rA = sdfg.add_state('read_A')
    wB = sdfg.add_state('write_B')
    rB = sdfg.add_state('read_B')
    done = sdfg.add_state('done')
    sdfg.add_edge(init, wA, InterstateEdge())
    sdfg.add_edge(wA, rA, InterstateEdge())
    sdfg.add_edge(rA, wB, InterstateEdge())
    sdfg.add_edge(wB, rB, InterstateEdge())
    sdfg.add_edge(rB, done, InterstateEdge())

    # write_A: A[i] = i for i in [0, big)
    m1, x1 = wA.add_map('wA', {'i': f'0:{big}'})
    t1 = wA.add_tasklet('wA', {}, {'a'}, 'a = (double)i;', language=dace.Language.CPP)
    aw = wA.add_write('A')
    wA.add_edge(m1, None, t1, None, dace.Memlet())
    wA.add_memlet_path(t1, x1, aw, src_conn='a', memlet=dace.Memlet('A[i]'))

    # read_A: out[i] = A[i] * 2.0 for i in [0, small)  (uses A; result later overwritten)
    m2, x2 = rA.add_map('rA', {'i': f'0:{small}'})
    t2 = rA.add_tasklet('rA', {'a'}, {'o'}, 'o = a * 2.0;', language=dace.Language.CPP)
    ar = rA.add_read('A')
    ow2 = rA.add_write('out')
    rA.add_memlet_path(ar, m2, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    rA.add_memlet_path(t2, x2, ow2, src_conn='o', memlet=dace.Memlet('out[i]'))

    # write_B: B[i] = (double)i for i in [0, small)
    m3, x3 = wB.add_map('wB', {'i': f'0:{small}'})
    t3 = wB.add_tasklet('wB', {}, {'b'}, 'b = (double)i;', language=dace.Language.CPP)
    bw = wB.add_write('B')
    wB.add_edge(m3, None, t3, None, dace.Memlet())
    wB.add_memlet_path(t3, x3, bw, src_conn='b', memlet=dace.Memlet('B[i]'))

    # read_B: out[i] = B[i] + 1.0 for i in [0, small)
    m4, x4 = rB.add_map('rB', {'i': f'0:{small}'})
    t4 = rB.add_tasklet('rB', {'b'}, {'o'}, 'o = b + 1.0;', language=dace.Language.CPP)
    br = rB.add_read('B')
    ow4 = rB.add_write('out')
    rB.add_memlet_path(br, m4, t4, dst_conn='b', memlet=dace.Memlet('B[i]'))
    rB.add_memlet_path(t4, x4, ow4, src_conn='o', memlet=dace.Memlet('out[i]'))

    return sdfg


# ---------------------------------------------------------------------------
# Pure-function tests for _greedy_cross_size_scan
# ---------------------------------------------------------------------------

class TestGreedyCrossSizeScan:

    def test_chooses_larger_donor_for_smaller_consumer(self):
        f64 = dace.float64
        liveness = [
            _AllocEntry('A', 128, f64),
            _FreeEntry('A', 128, f64),
            _AllocEntry('B', 64, f64),
            _FreeEntry('B', 64, f64),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == [('B', 'A', 0)]

    def test_best_fit_prefers_smaller_donor_when_multiple_fit(self):
        f64 = dace.float64
        liveness = [
            _AllocEntry('big', 256, f64),
            _AllocEntry('mid', 128, f64),
            _FreeEntry('big', 256, f64),
            _FreeEntry('mid', 128, f64),
            _AllocEntry('small', 64, f64),
            _FreeEntry('small', 64, f64),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == [('small', 'mid', 0)]

    def test_picks_cross_dtype_donor_when_donor_wider(self):
        """A float64 donor (8-byte alignment) can satisfy a float32
        consumer (4-byte alignment) — alignof(donor) >= alignof(consumer)."""
        liveness = [
            _AllocEntry('A_d', 128, dace.float64),
            _FreeEntry('A_d', 128, dace.float64),
            _AllocEntry('B_f', 64, dace.float32),
            _FreeEntry('B_f', 64, dace.float32),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == [('B_f', 'A_d', 0)]

    def test_skips_narrower_donor_for_wider_consumer(self):
        """A float32 donor cannot satisfy a float64 consumer even when
        the donor's size_bytes is sufficient: the donor's heap block is
        only float-aligned, not double-aligned."""
        liveness = [
            _AllocEntry('A_f', 256, dace.float32),
            _FreeEntry('A_f', 256, dace.float32),
            _AllocEntry('B_d', 64, dace.float64),
            _FreeEntry('B_d', 64, dace.float64),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == []

    def test_skips_donors_too_small(self):
        f64 = dace.float64
        liveness = [
            _AllocEntry('A', 64, f64),
            _FreeEntry('A', 64, f64),
            _AllocEntry('B', 128, f64),
            _FreeEntry('B', 128, f64),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == []  # A is too small to donate to B

    def test_returns_empty_when_lifetimes_overlap(self):
        f64 = dace.float64
        liveness = [
            _AllocEntry('A', 128, f64),
            _AllocEntry('B', 64, f64),  # B alloc'd before A freed → no chain
            _FreeEntry('A', 128, f64),
            _FreeEntry('B', 64, f64),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == []

    def test_two_consumers_share_one_donor(self):
        """D[160 B] freed; B[80 B] gets offset 0, C[64 B] gets offset 80."""
        f64 = dace.float64
        liveness = [
            _FreeEntry('D', 160, f64),
            _AllocEntry('B', 80, f64),
            _AllocEntry('C', 64, f64),
        ]
        plan = _greedy_cross_size_scan(liveness)
        assert plan == [('B', 'D', 0), ('C', 'D', 80)]


def _three_array_sdfg(name: str, d_size: int = 20,
                      b_size: int = 10, c_size: int = 8) -> SDFG:
    """Sequential SDFG: D used then freed, then B used, then C used.

    init -> write_D -> read_D -> write_B -> read_B -> write_C -> read_C -> done

    D[d_size], B[b_size], C[c_size] all float64. D's last access (read_D)
    precedes B's first access (write_B) and C's first access (write_C) in
    topological order, so both B and C can sub-allocate from D.
    """
    sdfg = SDFG(name)
    sdfg.add_array('D', [d_size], dace.float64, transient=True)
    sdfg.add_array('B', [b_size], dace.float64, transient=True)
    sdfg.add_array('C', [c_size], dace.float64, transient=True)
    sdfg.add_array('out', [1], dace.float64, transient=False)

    states = [sdfg.add_state(s, is_start_block=(i == 0))
              for i, s in enumerate(
                  ['init', 'write_D', 'read_D', 'write_B', 'read_B',
                   'write_C', 'read_C', 'done'])]
    for a, b in zip(states, states[1:]):
        sdfg.add_edge(a, b, dace.InterstateEdge())

    init, wD, rD, wB, rB, wC, rC, done = states

    def _write(st, arr, sz):
        m, x = st.add_map(f'w{arr}', {'i': f'0:{sz}'})
        t = st.add_tasklet(f'w{arr}', {}, {'v'},
                           'v = (double)i;', language=dace.Language.CPP)
        w = st.add_write(arr)
        st.add_edge(m, None, t, None, dace.Memlet())
        st.add_memlet_path(t, x, w, src_conn='v',
                           memlet=dace.Memlet(f'{arr}[i]'))

    def _read(st, arr):
        m, x = st.add_map(f'r{arr}', {'i': '0:1'})
        t = st.add_tasklet(f'r{arr}', {'v'}, {'o'},
                           'o = v;', language=dace.Language.CPP)
        r = st.add_read(arr)
        ow = st.add_write('out')
        st.add_memlet_path(r, m, t, dst_conn='v',
                           memlet=dace.Memlet(f'{arr}[0]'))
        st.add_memlet_path(t, x, ow, src_conn='o',
                           memlet=dace.Memlet('out[0]'))

    _write(wD, 'D', d_size)
    _read(rD, 'D')
    _write(wB, 'B', b_size)
    _read(rB, 'B')
    _write(wC, 'C', c_size)
    _read(rC, 'C')
    return sdfg


# ---------------------------------------------------------------------------
# Integration: buffer_reuse_cross_pass on a sequential cross-size SDFG
# ---------------------------------------------------------------------------

class TestBufferReuseCrossPassIntegration:

    def test_skips_when_donor_too_small(self):
        """Swap sizes: A is the smaller, B is the larger. No chain pair
        should be applied because A cannot donate to B."""
        sdfg = _two_size_sequential_sdfg('test_arena_too_small', big=8, small=16)
        applied = buffer_reuse_cross_pass(sdfg, {}, {})
        assert applied == [], f"Expected no pairs; got {applied}"

    def test_cross_size_pair_found(self):
        sdfg = _two_size_sequential_sdfg('test_arena_pair_found', big=16, small=8)
        applied = buffer_reuse_cross_pass(sdfg, {}, {})
        assert ('B', 'A') in applied, f"Expected (B,A); got {applied}"

    def test_end_to_end_correctness(self):
        big, small = 32, 16
        sdfg_base  = _two_size_sequential_sdfg('test_arena_e2e_base',  big=big, small=small)
        sdfg_arena = _two_size_sequential_sdfg('test_arena_e2e_arena', big=big, small=small)

        applied = buffer_reuse_cross_pass(sdfg_arena, {}, {})
        assert ('B', 'A') in applied, f"expected (B,A); got {applied}"

        out_base  = np.zeros(small, dtype=np.float64)
        out_arena = np.zeros(small, dtype=np.float64)
        sdfg_base(out=out_base)
        sdfg_arena(out=out_arena)
        assert np.array_equal(out_base, out_arena), (
            f"diverged: base={out_base}, arena={out_arena}"
        )

    def test_two_consumers_sub_allocated_from_one_donor(self):
        """D[20 × float64 = 160 B] is large enough for both B[10 × 8 = 80 B]
        and C[8 × 8 = 64 B] (80 + 64 = 144 ≤ 160).  After the pass both B and
        C should have reuse entries pointing to D, with C at a non-zero offset."""
        sdfg = _three_array_sdfg('test_suballoc', d_size=20, b_size=10, c_size=8)

        applied = buffer_reuse_cross_pass(sdfg, {}, {})

        consumers = {c for c, _ in applied}
        assert 'B' in consumers, f"B not applied; applied={applied}"
        assert 'C' in consumers, f"C not applied; applied={applied}"

        reuse_entries = [r for e in sdfg.all_interstate_edges(recursive=True)
                         for r in e.data.reuse]
        b_entry = next((r for r in reuse_entries if r[0] == 'B'), None)
        c_entry = next((r for r in reuse_entries if r[0] == 'C'), None)

        assert b_entry is not None, "no reuse entry for B"
        assert c_entry is not None, "no reuse entry for C"
        assert b_entry[1] == 'D', f"B should reuse D; got {b_entry[1]}"
        assert c_entry[1] == 'D', f"C should reuse D; got {c_entry[1]}"
        assert int(b_entry[2]) == 0,  f"B offset should be 0; got {b_entry[2]}"
        assert int(c_entry[2]) == 80, f"C offset should be 80 (B uses 10×8 B); got {c_entry[2]}"

        print("\n  Applied pairs:", applied)
        print("  Edge annotations:")
        for e in sdfg.all_interstate_edges(recursive=True):
            d = e.data
            if d.alloc or d.free or d.reuse:
                print(f"    {e.src.label} -> {e.dst.label}:"
                      f"  alloc={d.alloc}  free={d.free}  reuse={d.reuse}")
