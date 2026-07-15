# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout transforms must stay correct across WCR reduction edges.

The one WCR form the layout suite supports is a reduction into a single element or subset,
carried on the edge ``AccessNode(subset) <- MapExit <- ... <- Tasklet`` (the map exit may be
nested). A layout change (Permute, Block, Unblock, Shuffle) that touches such an array must
rewrite the WCR edge's subset like any other memlet AND keep its ``wcr`` -- rebuilding the memlet
with only the subset would silently drop the reduction, turning ``+=`` into last-writer-wins.

Every case runs through the documented front door ``prepare_for_layout`` (which parallelizes and
widens nested-SDFG boundary memlets) and checks the result is bit-exact with the numpy reduction.
Both the collapsed (single map exit) and nested (parallel outer map, reduce inner map -> WCR
through two map exits + a nested SDFG) forms are covered.
"""
import numpy
import pytest
import dace

from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.unblock_dimensions import UnblockDimensions
from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout
from dace.transformation.layout.prepare import prepare_for_layout
from dace.sdfg.core_dialect import CoreDialectCompliant

I, J, K = (dace.symbol(s) for s in ("I", "J", "K"))


def _has_wcr(sdfg) -> bool:
    return len(CoreDialectCompliant.offenders_wcr_edges(sdfg)) > 0


# ---- reduction programs -------------------------------------------------------------------------
@dace.program
def reduce_collapsed(A: dace.float64[I, J, K], out: dace.float64[I, K]):
    # out[i, k] = sum_j A[i, j, k]; single collapsed (i, j, k) map -> one WCR edge out[i, k]
    for i, j, k in dace.map[0:I, 0:J, 0:K]:
        out[i, k] += A[i, j, k]


@dace.program
def reduce_nested(A: dace.float64[I, J, K], out: dace.float64[I, K]):
    # parallel outer (i, k), reduce inner j -> WCR out[i, k] propagates through both map exits
    for i, k in dace.map[0:I, 0:K]:
        for j in dace.map[0:J]:
            out[i, k] += A[i, j, k]


@dace.program
def reduce_scalar(A: dace.float64[I, J], out: dace.float64[1]):
    # full reduction into a single element
    for i, j in dace.map[0:I, 0:J]:
        out[0] += A[i, j]


II, JJ, KK = 8, 5, 6
A3 = numpy.random.default_rng(0).random((II, JJ, KK))
A2 = numpy.random.default_rng(1).random((II, JJ))
REF3 = A3.sum(axis=1)  # [I, K]


def _prepared(program):
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    return sdfg


def _run3(sdfg):
    o = numpy.zeros((II, KK))
    sdfg(A=A3.copy(), out=o, I=II, J=JJ, K=KK)
    return o


@pytest.mark.parametrize("program", [reduce_collapsed, reduce_nested])
def test_permute_wcr_target_out(program):
    """Permuting the reduction TARGET out[i,k] -> out[k,i] rewrites the WCR subset and keeps wcr."""
    sdfg = _prepared(program)
    assert _has_wcr(sdfg)
    PermuteDimensions(permute_map={"out": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    assert _has_wcr(sdfg), "permute dropped the WCR"
    assert numpy.allclose(_run3(sdfg), REF3)


@pytest.mark.parametrize("program", [reduce_collapsed, reduce_nested])
def test_permute_reduction_input(program):
    """Permuting the reduction INPUT A leaves the WCR untouched and stays bit-exact."""
    sdfg = _prepared(program)
    PermuteDimensions(permute_map={"A": [2, 0, 1]}, add_permute_maps=True).apply_pass(sdfg, {})
    assert _has_wcr(sdfg)
    assert numpy.allclose(_run3(sdfg), REF3)


@pytest.mark.parametrize("program", [reduce_collapsed, reduce_nested])
@pytest.mark.parametrize("factor", [2, 4])
def test_block_wcr_target_out(program, factor):
    """Blocking the reduction TARGET out on dim 0 keeps the wcr (the regression: a rebuilt memlet
    that drops wcr turns the sum into last-writer-wins). out phys becomes [I/f, K, f]."""
    sdfg = _prepared(program)
    assert _has_wcr(sdfg)
    SplitDimensions(split_map={"out": ([True, False], [factor, 1])}).apply_pass(sdfg, {})
    normalize_schedule_for_layout(sdfg)
    assert _has_wcr(sdfg), "block dropped the WCR (reduction lost)"
    oshape = tuple(int(dace.symbolic.evaluate(x, {I: II, K: KK})) for x in sdfg.arrays["out"].shape)
    o = numpy.zeros(oshape)
    sdfg(A=A3.copy(), out=o, I=II, J=JJ, K=KK)
    got = o.transpose(0, 2, 1).reshape(II, KK)  # [I/f, K, f] -> [I/f, f, K] -> [I, K]
    assert numpy.allclose(got, REF3)


def test_block_then_unblock_wcr_roundtrip():
    """Block then Unblock the reduction target: the round trip restores [I, K] and the wcr survives
    both rebuilds, so the reduction is still a sum."""
    factor = 4
    sdfg = _prepared(reduce_collapsed)
    SplitDimensions(split_map={"out": ([True, False], [factor, 1])}).apply_pass(sdfg, {})
    assert _has_wcr(sdfg)
    # unblock uses the SAME masks/factors as block (length = original unblocked rank)
    UnblockDimensions(unblock_map={"out": ([True, False], [factor, 1])}).apply_pass(sdfg, {})
    assert _has_wcr(sdfg), "unblock dropped the WCR"
    assert tuple(int(dace.symbolic.evaluate(x, {I: II, K: KK})) for x in sdfg.arrays["out"].shape) == (II, KK)
    assert numpy.allclose(_run3(sdfg), REF3)


def test_permute_scalar_reduction_input():
    """Full reduction into out[0]; permuting the 2D input keeps the scalar WCR correct."""
    sdfg = reduce_scalar.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    assert _has_wcr(sdfg)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    assert _has_wcr(sdfg)
    o = numpy.zeros(1)
    sdfg(A=A2.copy(), out=o, I=II, J=JJ)
    assert numpy.allclose(o, A2.sum())


if __name__ == "__main__":
    for p in (reduce_collapsed, reduce_nested):
        test_permute_wcr_target_out(p)
        test_permute_reduction_input(p)
        for f in (2, 4):
            test_block_wcr_target_out(p, f)
    test_block_then_unblock_wcr_roundtrip()
    test_permute_scalar_reduction_input()
    print("WCR layout tests PASS")
