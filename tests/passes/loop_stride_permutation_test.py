# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``LoopStridePermutation`` canonicalization pass.

The pass interchanges a perfect two-level loop nest so a unit-stride DOALL loop
becomes the innermost loop -- the ``LoopRegion`` analogue of
``MinimizeStridePermutation``. It is sound because it only moves a loop that is
data-parallel once innermost (a DOALL loop is freely interchangeable), verified
via the ``LoopToMap`` oracle; it rejects triangular nests (non-rectangular
iteration space) and leaves already-canonical nests untouched.

Each scenario builds a raw loop nest with the dace Python frontend (``for``
loops -> ``LoopRegion`` nest under ``simplify=True``), runs it to capture a
reference, applies the pass, and re-runs to confirm the interchange preserved
the result.
"""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_stride_permutation import LoopStridePermutation

N = dace.symbol('N')


def _loop_nest_order(sdfg: dace.SDFG):
    """Loop variables from outermost to innermost across the (single) nest."""
    order = []

    def walk(region):
        for blk in region.nodes():
            if isinstance(blk, LoopRegion):
                order.append(blk.loop_variable)
                walk(blk)

    walk(sdfg)
    return order


# ---------------------------------------------------------------------------
#  Rectangular recurrence: i is the unit-stride parallel axis (outer), j carries
#  the recurrence (inner). Interchange must move i innermost.
# ---------------------------------------------------------------------------
@dace.program
def recurrence_unit_stride_outer(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for i in range(N):
        for j in range(1, N):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def test_interchange_moves_unit_stride_loop_innermost():
    sdfg = recurrence_unit_stride_outer.to_sdfg(simplify=True)
    assert _loop_nest_order(sdfg) == ['i', 'j']

    rng = np.random.default_rng(0)
    aa0 = rng.standard_normal((16, 16))
    bb = rng.standard_normal((16, 16))
    ref_aa = aa0.copy()
    sdfg(aa=ref_aa, bb=bb.copy(), N=16)

    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert applied == 1, "the unit-stride DOALL loop i should be interchanged innermost"
    assert _loop_nest_order(sdfg) == ['j', 'i'], "j (recurrence) outer, i (unit-stride) inner"

    got_aa = aa0.copy()
    sdfg(aa=got_aa, bb=bb.copy(), N=16)
    assert np.allclose(got_aa, ref_aa), "interchange must preserve the result"


# ---------------------------------------------------------------------------
#  Triangular nest: the inner bound references the outer loop var, so a metadata
#  swap would change the iteration set. Must be rejected.
# ---------------------------------------------------------------------------
@dace.program
def triangular(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for j in range(N):
        for i in range(0, j + 1):
            aa[j, i] = aa[j, i] + bb[j, i]


def test_reject_triangular_nest():
    sdfg = triangular.to_sdfg(simplify=True)
    before = _loop_nest_order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "triangular (non-rectangular) nest must not be interchanged"
    assert _loop_nest_order(sdfg) == before


# ---------------------------------------------------------------------------
#  Already canonical: the unit-stride axis (i) is already innermost -> no-op.
# ---------------------------------------------------------------------------
@dace.program
def unit_stride_already_inner(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for j in range(1, N):
        for i in range(N):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def test_noop_when_unit_stride_already_innermost():
    sdfg = unit_stride_already_inner.to_sdfg(simplify=True)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "no interchange needed when the unit-stride loop is already innermost"
    assert _loop_nest_order(sdfg) == ['j', 'i']


# ---------------------------------------------------------------------------
#  Unit-stride loop is NOT DOALL: the outer (unit-stride) loop carries the
#  recurrence, so moving it inward is not provably legal -> rejected.
# ---------------------------------------------------------------------------
@dace.program
def unit_stride_outer_carries_recurrence(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(N):
            aa[j, i] = aa[j, i - 1] + bb[j, i]


def test_reject_when_unit_stride_loop_not_doall():
    sdfg = unit_stride_outer_carries_recurrence.to_sdfg(simplify=True)
    before = _loop_nest_order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "a non-DOALL unit-stride loop must not be moved (not provably legal)"
    assert _loop_nest_order(sdfg) == before


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
