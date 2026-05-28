# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.scan_to_reduction.ScanToReduction`.

Covers the canonical forward prefix-scan shapes the pass rewrites, the
disqualifying patterns it must refuse, and idempotence on a fixed SDFG.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.scan_to_reduction import ScanToReduction


def _count_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _count_map_entries(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _has_delta_transient(sdfg: dace.SDFG) -> bool:
    return any(name.startswith('_scan_delta_') for name in sdfg.arrays)


def test_additive_scan_1d_value_preserving():
    """``out[i+1] = out[i] + src[i]`` over a 1-D array.

    After the pass:
      * a fresh ``_scan_delta_out`` transient exists,
      * the SDFG still validates and runs,
      * the result matches the numpy prefix-sum oracle,
      * the new elementwise (delta) loop maps cleanly under LoopToMap.
    """

    @dace.program
    def scan1d(out: dace.float64[9], src: dace.float64[8]):
        for i in range(8):
            out[i + 1] = out[i] + src[i]

    sdfg = scan1d.to_sdfg(simplify=True)
    assert _count_loops(sdfg) == 1

    rewritten = ScanToReduction().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten and len(rewritten) == 1
    assert _has_delta_transient(sdfg)
    # Two sequential LoopRegions remain (delta + prefix).
    assert _count_loops(sdfg) == 2

    # Numerical equivalence with the numpy oracle.
    src = np.random.default_rng(0).random(8)
    out = np.zeros(9)
    out[0] = 1.5
    ref = out.copy()
    for i in range(8):
        ref[i + 1] = ref[i] + src[i]

    sdfg(out=out, src=src)
    assert np.allclose(out, ref)

    # The elementwise stage parallelizes; the prefix stage stays sequential.
    maps_before = _count_map_entries(sdfg)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _count_map_entries(sdfg) > maps_before


def test_additive_scan_2d_per_column():
    """The cloudsc ``for_1133`` shape: ``out[i+1, j] = out[i, j] + src[i, j]``.

    The scan axis is ``i``; the column axis ``j`` is a passive co-index. After
    the pass, the per-``j`` elementwise stage becomes parallel under LoopToMap.
    """

    @dace.program
    def scan2d(out: dace.float64[9, 4], src: dace.float64[8, 4]):
        for i in range(8):
            for j in range(4):
                out[i + 1, j] = out[i, j] + src[i, j]

    sdfg = scan2d.to_sdfg(simplify=True)
    # The outer ``i`` loop is the scan; the inner ``j`` is a parallel column loop.
    assert _count_loops(sdfg) >= 1

    rewritten = ScanToReduction().apply_pass(sdfg, {})
    sdfg.validate()
    assert rewritten and len(rewritten) == 1
    assert _has_delta_transient(sdfg)

    src = np.random.default_rng(1).random((8, 4))
    out = np.zeros((9, 4))
    out[0] = np.arange(4, dtype=np.float64)
    ref = out.copy()
    for i in range(8):
        for j in range(4):
            ref[i + 1, j] = ref[i, j] + src[i, j]

    sdfg(out=out, src=src)
    assert np.allclose(out, ref)

    # LoopToMap must turn the delta stage into a parallel map.
    maps_before = _count_map_entries(sdfg)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _count_map_entries(sdfg) > maps_before


def test_refuses_non_adjacent_stride():
    """``out[i+2] = out[i] + src[i]`` is a stride-2 scan, not adjacent; refuse."""

    @dace.program
    def stride2(out: dace.float64[10], src: dace.float64[8]):
        for i in range(8):
            out[i + 2] = out[i] + src[i]

    sdfg = stride2.to_sdfg(simplify=True)
    loops_before = _count_loops(sdfg)

    rewritten = ScanToReduction().apply_pass(sdfg, {})
    assert rewritten is None
    # The original loop is untouched.
    assert _count_loops(sdfg) == loops_before
    assert not _has_delta_transient(sdfg)


def test_refuses_non_scan_carry():
    """A loop with a scan carry *and* an unrelated accumulator must be refused.

    The body writes both ``out[i+1] = out[i] + src[i]`` and ``acc[0] += src[i]``;
    the second carried dependency (``acc``) disqualifies the loop.
    """

    @dace.program
    def two_carries(out: dace.float64[9], acc: dace.float64[1], src: dace.float64[8]):
        for i in range(8):
            out[i + 1] = out[i] + src[i]
            acc[0] = acc[0] + src[i]

    sdfg = two_carries.to_sdfg(simplify=True)
    loops_before = _count_loops(sdfg)

    rewritten = ScanToReduction().apply_pass(sdfg, {})
    assert rewritten is None
    assert _count_loops(sdfg) == loops_before
    assert not _has_delta_transient(sdfg)


def test_refuses_unknown_operator():
    """v1 supports only ``+``; subtraction (not associative for a prefix scan) is refused."""

    @dace.program
    def scan_minus(out: dace.float64[9], src: dace.float64[8]):
        for i in range(8):
            out[i + 1] = out[i] - src[i]

    sdfg = scan_minus.to_sdfg(simplify=True)
    loops_before = _count_loops(sdfg)

    rewritten = ScanToReduction().apply_pass(sdfg, {})
    assert rewritten is None
    assert _count_loops(sdfg) == loops_before
    assert not _has_delta_transient(sdfg)


def test_idempotent():
    """A second apply on the already-rewritten SDFG is a no-op."""

    @dace.program
    def scan1d_idem(out: dace.float64[9], src: dace.float64[8]):
        for i in range(8):
            out[i + 1] = out[i] + src[i]

    sdfg = scan1d_idem.to_sdfg(simplify=True)

    first = ScanToReduction().apply_pass(sdfg, {})
    sdfg.validate()
    assert first and len(first) == 1

    arrays_after_first = set(sdfg.arrays)
    loops_after_first = _count_loops(sdfg)

    second = ScanToReduction().apply_pass(sdfg, {})
    sdfg.validate()
    assert second is None  # nothing left to rewrite
    assert set(sdfg.arrays) == arrays_after_first
    assert _count_loops(sdfg) == loops_after_first


if __name__ == '__main__':
    test_additive_scan_1d_value_preserving()
    test_additive_scan_2d_per_column()
    test_refuses_non_adjacent_stride()
    test_refuses_non_scan_carry()
    test_refuses_unknown_operator()
    test_idempotent()
