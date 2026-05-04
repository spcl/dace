# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for buffer_reuse_same_pass_ua — U/A-ratio variant of buffer_reuse_same_pass.

Difference from buffer_reuse_same_pass: the effective free point of each array is
its last CATS DataAccessEvent rather than its DeallocationEvent. This matters
whenever an array stays live past its last real use — most importantly for
AllocationLifetime.Explicit arrays, which CATS conservatively treats as
scope-wide (see determine_allocation_lifetime fallback).
"""
import copy
from typing import List, Tuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG, InterstateEdge
from dace.libraries.allocation import make_explicit, buffer_reuse_same_pass
from dace.libraries.allocation.reuse import (
    _extract_liveness,
    _AllocEntry,
    _FreeEntry,
    _greedy_same_size_scan,
    _greedy_cross_size_scan,
    _ua_greedy_same_size_scan,
    buffer_reuse_same_pass_ua,
)

from dace.libraries.allocation.reuse import _apply_reuse

# ---------------------------------------------------------------------------
# Minimal 4-stage image-pipeline used by integration tests
# ---------------------------------------------------------------------------

_N = dace.symbol('N')


@dace.program
def _blur(src: dace.float64[_N, _N], dst: dace.float64[_N, _N]):
    for i, j in dace.map[1:_N-1, 1:_N-1]:
        dst[i, j] = (
            src[i-1, j-1] * 1.0 + src[i-1, j] * 2.0 + src[i-1, j+1] * 1.0 +
            src[i,   j-1] * 2.0 + src[i,   j] * 4.0 + src[i,   j+1] * 2.0 +
            src[i+1, j-1] * 1.0 + src[i+1, j] * 2.0 + src[i+1, j+1] * 1.0
        ) / 16.0


@dace.program
def _sobel(src: dace.float64[_N, _N], dst: dace.float64[_N, _N]):
    for i, j in dace.map[1:_N-1, 1:_N-1]:
        gx = (
            -1.0 * src[i-1, j-1] + 1.0 * src[i-1, j+1] +
            -2.0 * src[i,   j-1] + 2.0 * src[i,   j+1] +
            -1.0 * src[i+1, j-1] + 1.0 * src[i+1, j+1]
        )
        gy = (
            -1.0 * src[i-1, j-1] + -2.0 * src[i-1, j] + -1.0 * src[i-1, j+1] +
             1.0 * src[i+1, j-1] +  2.0 * src[i+1, j] +  1.0 * src[i+1, j+1]
        )
        dst[i, j] = (gx * gx + gy * gy) ** 0.5


@dace.program
def _smooth(src: dace.float64[_N, _N], dst: dace.float64[_N, _N]):
    for i, j in dace.map[1:_N-1, 1:_N-1]:
        dst[i, j] = (
            src[i-1, j-1] * 1.0 + src[i-1, j] * 2.0 + src[i-1, j+1] * 1.0 +
            src[i,   j-1] * 2.0 + src[i,   j] * 4.0 + src[i,   j+1] * 2.0 +
            src[i+1, j-1] * 1.0 + src[i+1, j] * 2.0 + src[i+1, j+1] * 1.0
        ) / 16.0


@dace.program
def _threshold(src: dace.float64[_N, _N], dst: dace.float64[_N, _N]):
    for i, j in dace.map[0:_N, 0:_N]:
        dst[i, j] = float(src[i, j] > 0.5)


@dace.program
def _pipeline(img: dace.float64[_N, _N], out: dace.float64[_N, _N]):
    tmp1 = np.zeros((_N, _N), dtype=np.float64)
    _blur(img, tmp1)
    tmp2 = np.zeros((_N, _N), dtype=np.float64)
    _sobel(tmp1, tmp2)
    tmp3 = np.zeros((_N, _N), dtype=np.float64)
    _smooth(tmp2, tmp3)
    _threshold(tmp3, out)


def _make_pipeline_sdfg(mode: str) -> dace.SDFG:
    """Return an image-pipeline SDFG in baseline, explicit, or reused mode."""
    sdfg = _pipeline.to_sdfg(simplify=False)
    if mode == 'baseline':
        return sdfg
    make_explicit(sdfg, ['tmp1', 'tmp2', 'tmp3'])
    if mode == 'explicit':
        return sdfg
    if mode == 'reused':
        _apply_reuse(sdfg, new_arr='tmp3', donor_arr='tmp1')
        return sdfg
    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Helpers for constructing small hand-built SDFGs
# ---------------------------------------------------------------------------

def _sequential_two_array_sdfg(name: str, n: int = 10) -> SDFG:
    """SDFG with two same-size Explicit arrays whose *real* uses are disjoint.

    Flow:
      init -> write_A -> read_A_write_B -> read_B -> done

    A is explicit-allocated at entry and explicit-freed at exit, even though
    its last access is in read_A_write_B. Likewise for B. So CATS sees both
    live SDFG-wide, but they are in fact disjoint after the middle state.
    """
    sdfg = SDFG(name)
    sdfg.add_array('A', [n], dace.float64, transient=True)
    sdfg.add_array('B', [n], dace.float64, transient=True)
    sdfg.add_array('out', [n], dace.float64, transient=False)

    init = sdfg.add_state('init', is_start_block=True)
    wA = sdfg.add_state('write_A')
    rAwB = sdfg.add_state('read_A_write_B')
    rB = sdfg.add_state('read_B')
    done = sdfg.add_state('done')
    sdfg.add_edge(init, wA, InterstateEdge())
    sdfg.add_edge(wA, rAwB, InterstateEdge())
    sdfg.add_edge(rAwB, rB, InterstateEdge())
    sdfg.add_edge(rB, done, InterstateEdge())

    # write_A: A[i] = i
    m1, x1 = wA.add_map('wA', {'i': f'0:{n}'})
    t1 = wA.add_tasklet('wA', {}, {'a'}, 'a = (double)i;', language=dace.Language.CPP)
    aw = wA.add_write('A')
    wA.add_edge(m1, None, t1, None, dace.Memlet())
    wA.add_memlet_path(t1, x1, aw, src_conn='a', memlet=dace.Memlet('A[i]'))

    # read_A_write_B: B[i] = A[i] * 2
    m2, x2 = rAwB.add_map('rAwB', {'i': f'0:{n}'})
    t2 = rAwB.add_tasklet('rAwB', {'a'}, {'b'}, 'b = a * 2.0;', language=dace.Language.CPP)
    ar = rAwB.add_read('A')
    bw = rAwB.add_write('B')
    rAwB.add_memlet_path(ar, m2, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    rAwB.add_memlet_path(t2, x2, bw, src_conn='b', memlet=dace.Memlet('B[i]'))

    # read_B: out[i] = B[i] + 1
    m3, x3 = rB.add_map('rB', {'i': f'0:{n}'})
    t3 = rB.add_tasklet('rB', {'b'}, {'o'}, 'o = b + 1.0;', language=dace.Language.CPP)
    br = rB.add_read('B')
    ow = rB.add_write('out')
    rB.add_memlet_path(br, m3, t3, dst_conn='b', memlet=dace.Memlet('B[i]'))
    rB.add_memlet_path(t3, x3, ow, src_conn='o', memlet=dace.Memlet('out[i]'))

    return sdfg


# ---------------------------------------------------------------------------
# Pure-function tests for _extract_liveness (tightened access-window variant)
# ---------------------------------------------------------------------------

class TestExtractLivenessUA:

    def test_returns_alloc_and_free_for_every_explicit_array(self):
        sdfg = _sequential_two_array_sdfg('test_ua_sizes', n=16)
        make_explicit(sdfg, ['A', 'B'])
        events = _extract_liveness(sdfg, {}, {})
        allocs = [e for e in events if isinstance(e, _AllocEntry)]
        frees = [e for e in events if isinstance(e, _FreeEntry)]
        assert {a.array_name for a in allocs} == {'A', 'B'}
        assert {f.array_name for f in frees} == {'A', 'B'}

    def test_free_of_A_precedes_alloc_of_B(self):
        sdfg = _sequential_two_array_sdfg('test_ua_order', n=16)
        make_explicit(sdfg, ['A', 'B'])
        events = _extract_liveness(sdfg, {}, {})
        def _first(name, cls):
            return next(i for i, e in enumerate(events)
                        if isinstance(e, cls) and e.array_name == name)
        free_A = _first('A', _FreeEntry)
        alloc_B = _first('B', _AllocEntry)
        assert free_A < alloc_B, (
            f"Expected free(A) before alloc(B); events={events}"
        )


# ---------------------------------------------------------------------------
# Integration tests — compare UA vs greedy on scenarios where CATS misreports.
# ---------------------------------------------------------------------------

class TestUAvsGreedy:

    def test_unified_liveness_finds_disjoint_pair(self):
        """After unification, _extract_liveness already tightens windows so
        the plain greedy scan finds the disjoint pair — no separate UA liveness
        function needed."""
        sdfg = _sequential_two_array_sdfg('test_unified_finds_pair', n=16)
        make_explicit(sdfg, ['A', 'B'])
        pairs = _greedy_same_size_scan(_extract_liveness(sdfg, {}, {}))
        assert pairs == [('B', 'A')], f"Expected [('B','A')], got {pairs}"

    def test_both_passes_find_tmp3_tmp1_in_pipeline(self):
        """Both buffer_reuse_same_pass and buffer_reuse_same_pass_ua use tightened liveness
        now, so both should find (tmp3, tmp1) in the image pipeline."""
        n = 32
        sdfg_g = _make_pipeline_sdfg('explicit')
        greedy_pairs = buffer_reuse_same_pass(sdfg_g, {'N': n}, {'N': n})
        assert ('tmp3', 'tmp1') in set(greedy_pairs), (
            f"buffer_reuse_same_pass did not find (tmp3, tmp1); got {greedy_pairs}"
        )

        sdfg_u = _make_pipeline_sdfg('explicit')
        ua_pairs = buffer_reuse_same_pass_ua(sdfg_u, {'N': n}, {'N': n})
        assert ('tmp3', 'tmp1') in set(ua_pairs), (
            f"buffer_reuse_same_pass_ua did not find (tmp3, tmp1); got {ua_pairs}"
        )


# ---------------------------------------------------------------------------
# Unit tests for _FreeEntry.ua_ratio field
# ---------------------------------------------------------------------------

class TestFreeEntryRatio:

    def test_free_entry_accepts_ua_ratio(self):
        fe = _FreeEntry('x', 100, dace.float64, ua_ratio=0.3)
        assert fe.ua_ratio == 0.3

    def test_free_entry_ua_ratio_defaults_to_one(self):
        fe = _FreeEntry('y', 50, dace.float64)
        assert fe.ua_ratio == 1.0


# ---------------------------------------------------------------------------
# Unit tests for _ua_greedy_same_size_scan
# ---------------------------------------------------------------------------

class TestUAGreedySameSize:

    def test_ua_picks_lowest_ratio_over_lifo(self):
        """Two same-size donors: A freed first (ratio 0.1), B freed second (ratio 0.9).
        LIFO picks B (last freed); ua greedy picks A (lowest ratio)."""
        events = [
            _FreeEntry('A', 100, dace.float64, ua_ratio=0.1),
            _FreeEntry('B', 100, dace.float64, ua_ratio=0.9),
            _AllocEntry('C', 100, dace.float64),
        ]
        lifo_pairs = _greedy_same_size_scan(events)
        ua_pairs = _ua_greedy_same_size_scan(events)
        assert lifo_pairs == [('C', 'B')], f"LIFO expected B: {lifo_pairs}"
        assert ua_pairs == [('C', 'A')], f"UA expected A: {ua_pairs}"

    def test_ua_same_size_no_donor_available(self):
        """No donors: no pairs produced."""
        events = [_AllocEntry('X', 100, dace.float64)]
        assert _ua_greedy_same_size_scan(events) == []

    def test_ua_same_size_single_donor(self):
        """Single donor: same result as LIFO."""
        events = [
            _FreeEntry('A', 100, dace.float64, ua_ratio=0.5),
            _AllocEntry('B', 100, dace.float64),
        ]
        assert _ua_greedy_same_size_scan(events) == [('B', 'A')]


# ---------------------------------------------------------------------------
# New unified liveness tests (TDD: written before implementation)
# ---------------------------------------------------------------------------

class TestExtractLivenessUnified:

    def test_tightens_to_access_window(self):
        """free(A) must precede alloc(B) — A's last access is before B's first."""
        sdfg = _sequential_two_array_sdfg('test_unified_tight', n=16)
        make_explicit(sdfg, ['A', 'B'])
        events = _extract_liveness(sdfg, {}, {})
        def _first(name, cls):
            return next(i for i, e in enumerate(events)
                        if isinstance(e, cls) and e.array_name == name)
        assert _first('A', _FreeEntry) < _first('B', _AllocEntry)

    def test_free_entries_carry_ua_ratio(self):
        sdfg = _sequential_two_array_sdfg('test_unified_ratio', n=16)
        make_explicit(sdfg, ['A', 'B'])
        events = _extract_liveness(sdfg, {}, {})
        free_events = [e for e in events if isinstance(e, _FreeEntry)]
        assert len(free_events) == 2
        for fe in free_events:
            assert 0.0 <= fe.ua_ratio <= 1.0


# ---------------------------------------------------------------------------
# Wiring test: buffer_reuse_same_pass_ua uses _ua_greedy_same_size_scan
# ---------------------------------------------------------------------------

class TestUAPassWiring:

    def test_ua_end_to_end_pipeline_correct(self):
        """Run baseline and UA on the image pipeline; outputs must match."""
        n = 32
        rng = np.random.default_rng(7)
        img = rng.random((n, n))

        out_base = np.zeros((n, n), dtype=np.float64)
        sdfg_base = _make_pipeline_sdfg('baseline')
        sdfg_base(img=img, out=out_base, N=n)

        out_ua = np.zeros((n, n), dtype=np.float64)
        sdfg_ua = _make_pipeline_sdfg('explicit')
        applied = buffer_reuse_same_pass_ua(sdfg_ua, {'N': n}, {'N': n})
        sdfg_ua(img=img, out=out_ua, N=n)

        assert len(applied) >= 1, f"UA applied no pairs"
        assert np.array_equal(out_base, out_ua), (
            f"UA output diverges; max diff={np.abs(out_base - out_ua).max()}"
        )
