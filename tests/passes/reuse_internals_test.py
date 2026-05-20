# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for internal helpers in reuse.py and quantitative constructed
examples for every reuse pass.

Internal helpers covered:
  - _greedy_same_size_scan: empty, no-donor, size mismatch, dtype mismatch, LIFO
  - _collect_scopes: region-id sets, cross-scope detection
  - _edge_order_safe: safe (strictly ordered), unsafe (different regions)
  - _resolve_donor_root: no chain, 2-tuple chain, 3-tuple chain with offset
  - _apply_arena_reuse: 3-tuple emitted, donor free moved, consumer free removed

Quantitative examples (alloc count and footprint reduction):
  - _apply_reuse:      2 arrays → 1 alloc (-1 count, -50% bytes)
  - buffer_reuse_cross_pass: large→small cross-size (-1 count, -small_bytes footprint)
"""

import pytest
import dace
from dace import dtypes
from dace.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import LoopRegion
from dace.libraries.allocation import make_explicit
from dace.libraries.allocation.reuse import (
    _AllocEntry,
    _FreeEntry,
    _greedy_same_size_scan,
    _collect_scopes,
    _edge_order_safe,
    _resolve_donor_root,
    _apply_arena_reuse,
    _apply_reuse,
    buffer_reuse_cross_pass,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _count_allocs(sdfg: SDFG) -> int:
    """Number of array names across all alloc lists in the SDFG."""
    return sum(len(e.data.alloc)
               for e in sdfg.all_interstate_edges(recursive=True))


def _alloc_footprint_bytes(sdfg: SDFG) -> int:
    """Sum of bytes for every array that has at least one alloc annotation."""
    names = set()
    for e in sdfg.all_interstate_edges(recursive=True):
        names.update(e.data.alloc)
    return sum(
        sdfg.arrays[n].total_size * sdfg.arrays[n].dtype.bytes
        for n in names if n in sdfg.arrays
    )


# ---------------------------------------------------------------------------
# Shared SDFG factory helpers
# ---------------------------------------------------------------------------

def _sequential_sdfg(name: str, big: int, small: int,
                     dtype_big=dace.float64, dtype_small=dace.float64) -> SDFG:
    """init → use_A → use_B → done with A[big] and B[small].

    A and B are used in separate states so make_explicit places their
    alloc/free edges strictly around each state (no overlap).
    """
    sdfg = SDFG(name)
    sdfg.add_array('A', [big],   dtype_big,   transient=True)
    sdfg.add_array('B', [small], dtype_small, transient=True)

    init  = sdfg.add_state('init',  is_start_block=True)
    use_A = sdfg.add_state('use_A')
    use_B = sdfg.add_state('use_B')
    done  = sdfg.add_state('done')
    sdfg.add_edge(init,  use_A, InterstateEdge())
    sdfg.add_edge(use_A, use_B, InterstateEdge())
    sdfg.add_edge(use_B, done,  InterstateEdge())

    t_A = use_A.add_tasklet('a', {}, {'o'}, 'o = 1.0')
    use_A.add_edge(t_A, 'o', use_A.add_write('A'), None, dace.Memlet('A[0]'))
    t_B = use_B.add_tasklet('b', {}, {'o'}, 'o = 1.0')
    use_B.add_edge(t_B, 'o', use_B.add_write('B'), None, dace.Memlet('B[0]'))

    make_explicit(sdfg, ['A', 'B'])
    return sdfg


# ---------------------------------------------------------------------------
# _greedy_same_size_scan unit tests
# ---------------------------------------------------------------------------

class TestGreedySameSizeScan:

    def test_empty_liveness_returns_empty(self):
        assert _greedy_same_size_scan([]) == []

    def test_alloc_only_no_donors(self):
        events = [_AllocEntry('A', 100, dace.float64)]
        assert _greedy_same_size_scan(events) == []

    def test_free_only_no_alloc(self):
        events = [_FreeEntry('A', 100, dace.float64)]
        assert _greedy_same_size_scan(events) == []

    def test_size_mismatch_no_pair(self):
        events = [
            _FreeEntry('A', 100, dace.float64),
            _AllocEntry('B', 200, dace.float64),
        ]
        assert _greedy_same_size_scan(events) == []

    def test_dtype_mismatch_no_pair(self):
        events = [
            _FreeEntry('A', 100, dace.float64),
            _AllocEntry('B', 100, dace.float32),
        ]
        assert _greedy_same_size_scan(events) == []

    def test_matching_pair_produces_reuse(self):
        events = [
            _FreeEntry('A', 100, dace.float64),
            _AllocEntry('B', 100, dace.float64),
        ]
        assert _greedy_same_size_scan(events) == [('B', 'A')]

    def test_lifo_within_same_bucket(self):
        """A freed first, B freed second; LIFO picks B for next alloc of C."""
        events = [
            _FreeEntry('A', 100, dace.float64),
            _FreeEntry('B', 100, dace.float64),
            _AllocEntry('C', 100, dace.float64),
        ]
        assert _greedy_same_size_scan(events) == [('C', 'B')]

    def test_two_sequential_pairs(self):
        events = [
            _FreeEntry('A', 100, dace.float64),
            _AllocEntry('B', 100, dace.float64),
            _FreeEntry('B', 100, dace.float64),
            _AllocEntry('C', 100, dace.float64),
        ]
        pairs = _greedy_same_size_scan(events)
        assert ('B', 'A') in pairs
        assert ('C', 'B') in pairs


# ---------------------------------------------------------------------------
# _collect_scopes unit tests
# ---------------------------------------------------------------------------

class TestCollectScopes:

    def test_returns_region_ids_for_alloc_and_free(self):
        sdfg = SDFG('scopes_basic')
        sdfg.add_array('A', [10], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Explicit)
        init = sdfg.add_state('init', is_start_block=True)
        use  = sdfg.add_state('use')
        done = sdfg.add_state('done')
        e0 = sdfg.add_edge(init, use,  InterstateEdge())
        e1 = sdfg.add_edge(use,  done, InterstateEdge())
        e0.data.alloc.append('A')
        e1.data.free.append('A')

        scopes = _collect_scopes(sdfg, {'A'})

        assert 'A' in scopes
        alloc_regions, free_regions = scopes['A']
        assert len(alloc_regions) == 1
        assert len(free_regions) == 1
        assert alloc_regions == free_regions  # both on same SDFG

    def test_cross_scope_gives_different_region_sets(self):
        """Alloc at SDFG level, free inside a LoopRegion → mismatched sets."""
        sdfg = SDFG('scopes_cross')
        sdfg.add_array('A', [10], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Explicit)

        init = sdfg.add_state('init', is_start_block=True)
        loop = LoopRegion('myloop', 'i < 5', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop)
        done = sdfg.add_state('done')
        e_before = sdfg.add_edge(init, loop, InterstateEdge())
        sdfg.add_edge(loop, done, InterstateEdge())

        body     = loop.add_state('body', is_start_block=True)
        body_end = loop.add_state('body_end')
        e_inner  = loop.add_edge(body, body_end, InterstateEdge())

        e_before.data.alloc.append('A')
        e_inner.data.free.append('A')

        scopes = _collect_scopes(sdfg, {'A'})
        alloc_regions, free_regions = scopes['A']
        assert alloc_regions != free_regions


# ---------------------------------------------------------------------------
# _edge_order_safe unit tests
# ---------------------------------------------------------------------------

class TestEdgeOrderSafe:

    def test_returns_true_when_strictly_ordered(self):
        """B reuses A: A's last-use block (use_A) is strictly before B's
        first-use block (use_B) → safe."""
        sdfg = _sequential_sdfg('eos_true', big=10, small=10)
        assert _edge_order_safe(sdfg, 'B', 'A') is True

    def test_returns_false_when_different_regions(self):
        """A allocated at SDFG level, B allocated inside LoopRegion → not safe."""
        sdfg = SDFG('eos_diff')
        sdfg.add_array('A', [10], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Explicit)
        sdfg.add_array('B', [10], dace.float64, transient=True,
                       lifetime=dtypes.AllocationLifetime.Explicit)

        init = sdfg.add_state('init', is_start_block=True)
        loop = LoopRegion('myloop', 'i < 5', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop)
        done = sdfg.add_state('done')
        e0 = sdfg.add_edge(init, loop, InterstateEdge())
        e1 = sdfg.add_edge(loop, done, InterstateEdge())

        body     = loop.add_state('body', is_start_block=True)
        body_end = loop.add_state('body_end')
        e_body   = loop.add_edge(body, body_end, InterstateEdge())

        # A at SDFG level, B inside loop → different parent regions
        e0.data.alloc.append('A')
        e1.data.free.append('A')
        e_body.data.alloc.append('B')
        e_body.data.free.append('B')

        assert _edge_order_safe(sdfg, 'B', 'A') is False

    def test_returns_false_when_multiple_alloc_edges(self):
        """If new_arr has more than one alloc edge the check returns False
        (ambiguous ordering)."""
        sdfg = _sequential_sdfg('eos_multi', big=10, small=10)
        # Inject a second alloc edge for B
        extra = sdfg.add_state('extra')
        use_B = next(s for s in sdfg.states() if s.label == 'use_B')
        e_extra = sdfg.add_edge(extra, use_B, InterstateEdge())
        e_extra.data.alloc.append('B')

        assert _edge_order_safe(sdfg, 'B', 'A') is False


# ---------------------------------------------------------------------------
# _resolve_donor_root unit tests
# ---------------------------------------------------------------------------

class TestResolveDonorRoot:

    def test_non_reuse_consumer_returns_self(self):
        sdfg = SDFG('resolve_base')
        sdfg.add_array('A', [10], dace.float64, transient=True)
        sdfg.add_state('s0', is_start_block=True)

        root, off = _resolve_donor_root(sdfg, 'A')
        assert root == 'A'
        assert off == 0

    def test_2tuple_chain_returns_root_zero_offset(self):
        """B → A (2-tuple): root of B is A at offset 0."""
        sdfg = SDFG('resolve_2t')
        sdfg.add_array('A', [20], dace.float64, transient=True)
        sdfg.add_array('B', [10], dace.float64, transient=True)
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        e  = sdfg.add_edge(s0, s1, InterstateEdge())
        e.data.reuse.append(['B', 'A'])

        root, off = _resolve_donor_root(sdfg, 'B')
        assert root == 'A'
        assert off == 0

    def test_3tuple_chain_accumulates_offset(self):
        """C → B at 8; B → A at 16 → root of C is A at offset 24."""
        sdfg = SDFG('resolve_3t')
        sdfg.add_array('A', [40], dace.float64, transient=True)
        sdfg.add_array('B', [20], dace.float64, transient=True)
        sdfg.add_array('C', [10], dace.float64, transient=True)
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        e  = sdfg.add_edge(s0, s1, InterstateEdge())
        e.data.reuse.append(['B', 'A', 16])
        e.data.reuse.append(['C', 'B', 8])

        root, off = _resolve_donor_root(sdfg, 'C')
        assert root == 'A'
        assert off == 24


# ---------------------------------------------------------------------------
# _apply_arena_reuse unit tests
# ---------------------------------------------------------------------------

class TestApplyArenaReuse:

    def test_consumer_alloc_replaced_with_3tuple(self):
        sdfg = _sequential_sdfg('arena_3t', big=20, small=10)
        _apply_arena_reuse(sdfg, 'B', 'A', offset_bytes=0)

        all_allocs = [n for e in sdfg.all_interstate_edges(recursive=True)
                      for n in e.data.alloc]
        all_reuses = [r for e in sdfg.all_interstate_edges(recursive=True)
                      for r in e.data.reuse]

        assert 'B' not in all_allocs
        assert any(r[0] == 'B' and r[1] == 'A' and int(r[2]) == 0
                   for r in all_reuses)

    def test_consumer_free_removed(self):
        sdfg = _sequential_sdfg('arena_nofree', big=20, small=10)
        _apply_arena_reuse(sdfg, 'B', 'A', offset_bytes=0)

        all_frees = [n for e in sdfg.all_interstate_edges(recursive=True)
                     for n in e.data.free]
        assert 'B' not in all_frees

    def test_donor_free_moved_to_consumer_old_free_site(self):
        sdfg = _sequential_sdfg('arena_moved', big=20, small=10)
        # Capture B's free edge before the call
        b_free_edge = next(
            e for e in sdfg.all_interstate_edges(recursive=True)
            if 'B' in e.data.free
        )

        _apply_arena_reuse(sdfg, 'B', 'A', offset_bytes=0)

        assert 'A' in b_free_edge.data.free

    def test_donor_alloc_retained(self):
        sdfg = _sequential_sdfg('arena_alloc', big=20, small=10)
        _apply_arena_reuse(sdfg, 'B', 'A', offset_bytes=0)

        all_allocs = [n for e in sdfg.all_interstate_edges(recursive=True)
                      for n in e.data.alloc]
        assert 'A' in all_allocs

    def test_nonzero_offset_stored(self):
        sdfg = _sequential_sdfg('arena_off', big=20, small=10)
        _apply_arena_reuse(sdfg, 'B', 'A', offset_bytes=32)

        all_reuses = [r for e in sdfg.all_interstate_edges(recursive=True)
                      for r in e.data.reuse]
        assert any(r[0] == 'B' and int(r[2]) == 32 for r in all_reuses)

    def test_error_unknown_array(self):
        sdfg = _sequential_sdfg('arena_err', big=20, small=10)
        with pytest.raises(ValueError, match='not found'):
            _apply_arena_reuse(sdfg, 'NOPE', 'A')

    def test_sdfg_validates_after_arena_reuse(self):
        sdfg = _sequential_sdfg('arena_valid', big=20, small=10)
        _apply_arena_reuse(sdfg, 'B', 'A', offset_bytes=0)
        sdfg.validate()


# ---------------------------------------------------------------------------
# Quantitative constructed examples
# ---------------------------------------------------------------------------

class TestApplyReuseQuantitative:
    """_apply_reuse on a two-array same-size SDFG: verify alloc count and
    footprint decrease by the expected amounts."""

    N = 10  # elements per array; dtype = float64 → 8 bytes each

    def _make(self, name):
        return _sequential_sdfg(name, big=self.N, small=self.N)

    def test_alloc_count_decreases_by_one(self):
        sdfg = self._make('reuse_count')
        count_before = _count_allocs(sdfg)  # 2: one for A, one for B

        _apply_reuse(sdfg, 'B', 'A')

        count_after = _count_allocs(sdfg)   # 1: only A
        assert count_after == count_before - 1, (
            f"Expected alloc count {count_before - 1}, got {count_after}"
        )

    def test_footprint_decreases_by_B_size(self):
        sdfg = self._make('reuse_footprint')
        b_bytes = self.N * dace.float64.bytes
        footprint_before = _alloc_footprint_bytes(sdfg)  # 2*N*8

        _apply_reuse(sdfg, 'B', 'A')

        footprint_after = _alloc_footprint_bytes(sdfg)   # N*8
        assert footprint_after == footprint_before - b_bytes, (
            f"Footprint should decrease by {b_bytes}; "
            f"was {footprint_before}, now {footprint_after}"
        )

    def test_reuse_entry_created_for_B(self):
        sdfg = self._make('reuse_entry')
        _apply_reuse(sdfg, 'B', 'A')
        reuse_entries = [r for e in sdfg.all_interstate_edges(recursive=True)
                         for r in e.data.reuse]
        assert any(r[0] == 'B' and r[1] == 'A' for r in reuse_entries)


class TestBufferArenaPassQuantitative:
    """buffer_reuse_cross_pass on a large-then-small SDFG: verify alloc count and
    footprint after the pass match expected values (only A's memory survives
    as an alloc entry; B's allocation is replaced by a reuse into A)."""

    BIG   = 16
    SMALL = 8

    def _make_sequential_cross_size(self, name):
        """Sequential: A[big] used, then B[small] used — non-overlapping."""
        sdfg = SDFG(name)
        sdfg.add_array('A', [self.BIG],   dace.float64, transient=True)
        sdfg.add_array('B', [self.SMALL], dace.float64, transient=True)
        sdfg.add_array('out', [self.SMALL], dace.float64, transient=False)

        init = sdfg.add_state('init', is_start_block=True)
        wA   = sdfg.add_state('write_A')
        rA   = sdfg.add_state('read_A')
        wB   = sdfg.add_state('write_B')
        rB   = sdfg.add_state('read_B')
        done = sdfg.add_state('done')
        sdfg.add_edge(init, wA, InterstateEdge())
        sdfg.add_edge(wA,   rA, InterstateEdge())
        sdfg.add_edge(rA,   wB, InterstateEdge())
        sdfg.add_edge(wB,   rB, InterstateEdge())
        sdfg.add_edge(rB, done, InterstateEdge())

        big, small = self.BIG, self.SMALL

        m1, x1 = wA.add_map('wA', {'i': f'0:{big}'})
        t1 = wA.add_tasklet('wA', {}, {'a'}, 'a = (double)i;',
                             language=dace.Language.CPP)
        aw = wA.add_write('A')
        wA.add_edge(m1, None, t1, None, dace.Memlet())
        wA.add_memlet_path(t1, x1, aw, src_conn='a', memlet=dace.Memlet('A[i]'))

        m2, x2 = rA.add_map('rA', {'i': f'0:{small}'})
        t2 = rA.add_tasklet('rA', {'a'}, {'o'}, 'o = a;',
                             language=dace.Language.CPP)
        ar  = rA.add_read('A')
        ow2 = rA.add_write('out')
        rA.add_memlet_path(ar, m2, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
        rA.add_memlet_path(t2, x2, ow2, src_conn='o', memlet=dace.Memlet('out[i]'))

        m3, x3 = wB.add_map('wB', {'i': f'0:{small}'})
        t3 = wB.add_tasklet('wB', {}, {'b'}, 'b = (double)i;',
                             language=dace.Language.CPP)
        bw = wB.add_write('B')
        wB.add_edge(m3, None, t3, None, dace.Memlet())
        wB.add_memlet_path(t3, x3, bw, src_conn='b', memlet=dace.Memlet('B[i]'))

        m4, x4 = rB.add_map('rB', {'i': f'0:{small}'})
        t4 = rB.add_tasklet('rB', {'b'}, {'o'}, 'o = b + 1.0;',
                             language=dace.Language.CPP)
        br  = rB.add_read('B')
        ow4 = rB.add_write('out')
        rB.add_memlet_path(br, m4, t4, dst_conn='b', memlet=dace.Memlet('B[i]'))
        rB.add_memlet_path(t4, x4, ow4, src_conn='o', memlet=dace.Memlet('out[i]'))

        return sdfg

    def test_alloc_count_decreases_by_one(self):
        """Before the pass both arrays carry explicit alloc annotations (2 total).
        After, B's alloc is replaced by a reuse entry — only A's alloc survives."""
        sdfg = self._make_sequential_cross_size('arena_q_count')
        make_explicit(sdfg, ['A', 'B'])
        count_before = _count_allocs(sdfg)  # 2

        applied = buffer_reuse_cross_pass(sdfg, {}, {})
        assert len(applied) == 1, f"Expected 1 pair applied; got {applied}"

        count_after = _count_allocs(sdfg)
        assert count_after == count_before - 1, (
            f"Expected alloc count {count_before - 1}, got {count_after}"
        )

    def test_footprint_decreases_by_small_array_bytes(self):
        """Before: A_bytes + B_bytes allocated; after: only A_bytes (B reuses A)."""
        sdfg = self._make_sequential_cross_size('arena_q_fp')
        make_explicit(sdfg, ['A', 'B'])
        b_bytes = self.SMALL * dace.float64.bytes
        footprint_before = _alloc_footprint_bytes(sdfg)

        buffer_reuse_cross_pass(sdfg, {}, {})

        footprint_after = _alloc_footprint_bytes(sdfg)
        assert footprint_after == footprint_before - b_bytes, (
            f"Footprint should decrease by {b_bytes}; "
            f"was {footprint_before}, now {footprint_after}"
        )

    def test_pair_applied_is_b_reuses_a(self):
        sdfg = self._make_sequential_cross_size('arena_q_pair')
        applied = buffer_reuse_cross_pass(sdfg, {}, {})
        assert ('B', 'A') in applied
