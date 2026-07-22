# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the compound-body reduction pattern (TSVC s319 shape).

A "compound" reduction is a loop that mixes:

  * a scalar accumulator update (``s = s + a[i]``), and
  * per-element side writes (``a[i] = c[i] + d[i]``).

These should split via ``LoopFission`` into one map (parallel side writes)
plus one reduction (the scalar accumulator). Today neither ``LoopFission``
nor ``LoopToReduce`` catches the shape, and the loop stays sequential. The
tests below pin the expected behavior so the fix lands with a contract.
"""
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes
from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol("N")


def _stats(sdfg: dace.SDFG):
    """Returns (loops, maps, reduces, wcr_edges)."""
    return (
        sum(1 for c in sdfg.all_control_flow_regions() if isinstance(c, LoopRegion)),
        sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)),
        sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)),
        sum(1 for st in sdfg.all_states() for e in st.edges() if e.data is not None and e.data.wcr is not None),
    )


# ---------------------------------------------------------------------------
# TSVC s319 SHAPE: ``a[i]=c+d; s += a[i]; b[i]=c+e; s += b[i]; b[0]=s``
#
# Body interleaves two pure-side-write ops (a[i], b[i]) with two accumulator
# updates (s += a[i]; s += b[i]). LoopFission should split into:
#   loop A: a[i]=c+d; b[i]=c+e          (pure parallel)
#   loop B: s += a[i] + b[i]            (reduction)
# Then LoopToMap parallelizes loop A and LoopToReduce / AugAssignToWCR
# handles loop B.
# ---------------------------------------------------------------------------


@dace.program
def _compound_reduction(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N],
                        e: dace.float64[N]):
    s = 0.0
    for i in range(N):
        a[i] = c[i] + d[i]
        s = s + a[i]
        b[i] = c[i] + e[i]
        s = s + b[i]
    b[0] = s


@pytest.mark.xfail(reason="TSVC s319: LoopFission does not currently split this 'compound body' shape "
                   "(reduction + per-element side writes). Once fission catches it, the two "
                   "resulting loops -- one pure side writes (parallel map) and one pure "
                   "accumulator (reduction) -- become independently parallelizable.",
                   strict=True)
def test_compound_body_splits_via_loop_fission():
    """LoopFission should produce two loops from the compound body."""
    sdfg = _compound_reduction.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    loops_before, _, _, _ = _stats(sdfg)
    assert loops_before == 1

    applied = LoopFission().apply_pass(sdfg, {})
    assert applied is not None and applied >= 1
    loops_after, _, _, _ = _stats(sdfg)
    assert loops_after == 2


@pytest.mark.xfail(reason="TSVC s319: after fission lands (see test above), the side-writes "
                   "loop should parallelize via LoopToMap and the accumulator loop should "
                   "either lift to a Reduce libnode (LoopToReduce) or to a WCR-map "
                   "(AugAssignToWCR + LoopToMap). The final state should contain at least "
                   "one parallel map AND a reduction (Reduce libnode or WCR edge).",
                   strict=True)
def test_compound_body_parallelizes_end_to_end():
    """End-to-end: fission then parallelize both halves."""
    sdfg = _compound_reduction.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    LoopFission().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(AugAssignToWCR, validate=False, validate_all=False, permissive=True)
    sdfg.apply_transformations_repeated(LoopToMap, validate=False, validate_all=False)
    loops, maps, reduces, wcr = _stats(sdfg)

    # At least one parallel map (the side-writes half) AND a reduction (Reduce
    # libnode OR a WCR edge from the accumulator half).
    assert maps >= 1
    assert reduces + wcr >= 1
    assert loops < 2  # at least one of the two fissioned loops parallelized


# ---------------------------------------------------------------------------
# Smaller "delta + accumulate" probe -- the minimal s319 family shape.
# ---------------------------------------------------------------------------


@dace.program
def _delta_and_accumulate(a: dace.float64[N], b: dace.float64[N], s_out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        a[i] = b[i] + 1.0
        s = s + a[i]
    s_out[0] = s


@pytest.mark.xfail(reason="Same as test_compound_body_splits_via_loop_fission: LoopFission "
                   "should split the per-element write from the accumulator update.",
                   strict=True)
def test_minimal_delta_then_accumulate_splits():
    sdfg = _delta_and_accumulate.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
    applied = LoopFission().apply_pass(sdfg, {})
    assert applied is not None
    loops, _, _, _ = _stats(sdfg)
    assert loops == 2


if __name__ == "__main__":
    test_compound_body_splits_via_loop_fission()
    test_compound_body_parallelizes_end_to_end()
    test_minimal_delta_then_accumulate_splits()
