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

import re
import time

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


def _loop_control_tokens(sdfg: dace.SDFG, variable: str):
    """Identifiers appearing anywhere in the loop control of the loop owning ``variable``."""
    found = []

    def walk(region):
        for blk in region.nodes():
            if isinstance(blk, LoopRegion):
                if blk.loop_variable == variable:
                    text = ' '.join(code.as_string
                                    for code in (blk.init_statement, blk.loop_condition, blk.update_statement)
                                    if code is not None)
                    found.append(set(re.findall(r'\b[A-Za-z_]\w*\b', text)))
                walk(blk)

    walk(sdfg)
    assert len(found) == 1, f'expected exactly one loop over {variable}, found {len(found)}'
    return found[0]


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
#  swap would change the iteration set. The lower-bound (trapezoidal) form is
#  rebuilt; the upper-bound form is refused.
# ---------------------------------------------------------------------------
@dace.program
def trapezoid_lower_bound(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for j in range(N):
        for i in range(8 * j, N):
            aa[i, j] = aa[i, j] + bb[i, j]


def test_trapezoidal_nest_is_interchanged_by_bound_rewrite():
    """TSVC s1232: the inner loop STARTS at 8*j, so the unit-stride axis (j) is the outer one.
       A metadata swap alone would change the iteration set; the bounds are rebuilt instead."""
    sdfg = trapezoid_lower_bound.to_sdfg(simplify=True)
    assert _loop_nest_order(sdfg) == ['j', 'i']
    assert 'j' in _loop_control_tokens(sdfg, 'i'), 'the i bound must start out referencing j'

    assert LoopStridePermutation().apply_pass(sdfg, {}) == 1
    assert _loop_nest_order(sdfg) == ['i', 'j']
    # The trapezoid REWRITE, asserted on the bounds rather than on a stopwatch: the dependence
    # between the two bounds has to change direction with the loops. A metadata swap that left
    # `i in range(8 * j, N)` under a j loop would enumerate a different iteration set.
    assert 'j' not in _loop_control_tokens(sdfg, 'i'), 'the i bound must no longer reference j'
    assert 'i' in _loop_control_tokens(sdfg, 'j'), 'the rebuilt j bound must be capped by i'
    sdfg.validate()

    n = 64
    aa = np.arange(n * n, dtype=np.float64).reshape(n, n).copy()
    bb = np.arange(n * n, dtype=np.float64).reshape(n, n).copy() * 0.5
    expected = aa.copy()
    for j in range(n):
        for i in range(8 * j, n):
            expected[i, j] = aa[i, j] + bb[i, j]
    sdfg(aa=aa, bb=bb, N=n)
    assert np.allclose(aa, expected)


@dace.program
def triangular_upper_bound(aa: dace.float64[N, N], bb: dace.float64[N, N]):
    for j in range(N):
        for i in range(0, j + 1):
            aa[i, j] = aa[i, j] + bb[i, j]


def test_reject_triangular_upper_bound():
    """The outer variable in the inner CONDITION is outside the rewrite: the interchanged upper
       bound is not an affine function of the new outer iterate."""
    sdfg = triangular_upper_bound.to_sdfg(simplify=True)
    before = _loop_nest_order(sdfg)
    assert not LoopStridePermutation().apply_pass(sdfg, {})
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


# ---------------------------------------------------------------------------
#  N-level (3-deep) nest: the unit-stride DOALL axis (i) is outermost and must
#  bubble all the way to innermost, past the two recurrence-carrying loops j, k.
# ---------------------------------------------------------------------------
@dace.program
def three_level_unit_stride_outer(aa: dace.float64[N, N, N], bb: dace.float64[N, N, N]):
    for i in range(N):
        for j in range(1, N):
            for k in range(1, N):
                aa[k, j, i] = aa[k - 1, j, i] + aa[k, j - 1, i] + bb[k, j, i]


def test_interchange_bubbles_unit_stride_through_three_levels():
    sdfg = three_level_unit_stride_outer.to_sdfg(simplify=True)
    assert _loop_nest_order(sdfg) == ['i', 'j', 'k']

    rng = np.random.default_rng(1)
    aa0 = rng.standard_normal((10, 10, 10))
    bb = rng.standard_normal((10, 10, 10))
    ref_aa = aa0.copy()
    sdfg(aa=ref_aa, bb=bb.copy(), N=10)

    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert applied == 1, "the unit-stride DOALL axis i must bubble to innermost"
    order = _loop_nest_order(sdfg)
    assert order[-1] == 'i', f"i (unit-stride) must be innermost, got {order}"
    assert set(order) == {'i', 'j', 'k'}

    got_aa = aa0.copy()
    sdfg(aa=got_aa, bb=bb.copy(), N=10)
    assert np.allclose(got_aa, ref_aa), "N-level interchange must preserve the result"


# ---------------------------------------------------------------------------
#  N-level: the unit-stride DOALL axis sits in the MIDDLE of a 3-deep nest and
#  must bubble one level inward (past the inner recurrence loop) to innermost.
# ---------------------------------------------------------------------------
@dace.program
def three_level_unit_stride_middle(bb: dace.float64[N, N, N], cc: dace.float64[N, N, N]):
    for i in range(1, N):
        for j in range(N):
            for k in range(1, N):
                bb[i, k, j] = bb[i - 1, k, j] + bb[i, k - 1, j] + cc[i, k, j]


def test_interchange_bubbles_middle_axis_to_innermost():
    sdfg = three_level_unit_stride_middle.to_sdfg(simplify=True)
    assert _loop_nest_order(sdfg) == ['i', 'j', 'k']

    rng = np.random.default_rng(2)
    bb0 = rng.standard_normal((10, 10, 10))
    cc = rng.standard_normal((10, 10, 10))
    ref = bb0.copy()
    sdfg(bb=ref, cc=cc.copy(), N=10)

    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert applied == 1, "the unit-stride DOALL middle axis j must bubble to innermost"
    assert _loop_nest_order(sdfg) == ['i', 'k', 'j'], "j (unit-stride) innermost, k stays a sequential loop"

    got = bb0.copy()
    sdfg(bb=got, cc=cc.copy(), N=10)
    assert np.allclose(got, ref), "middle-axis interchange must preserve the result"


# ---------------------------------------------------------------------------
#  N-level reject: the unit-stride axis (i, outermost) itself carries a
#  recurrence (aa[k, j, i-1]), so it is not DOALL once innermost -> whole bubble
#  reverts, nest untouched.
# ---------------------------------------------------------------------------
@dace.program
def three_level_unit_stride_outer_carries(aa: dace.float64[N, N, N], bb: dace.float64[N, N, N]):
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                aa[k, j, i] = aa[k, j, i - 1] + aa[k - 1, j, i] + bb[k, j, i]


def test_reject_three_level_when_moved_axis_not_doall():
    sdfg = three_level_unit_stride_outer_carries.to_sdfg(simplify=True)
    before = _loop_nest_order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "a unit-stride axis that carries a dependence must not be moved"
    assert _loop_nest_order(sdfg) == before


# ---------------------------------------------------------------------------
#  Imperfect nest (TSVC s2233 shape): the outer body holds TWO sibling inner
#  loops, so it is not a perfect nest -> LoopStridePermutation is a no-op here
#  (this shape needs LoopFission first, handled later in the pipeline).
# ---------------------------------------------------------------------------
@dace.program
def imperfect_two_sibling_inner(aa: dace.float64[N, N], bb: dace.float64[N, N], cc: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(1, N):
            bb[i, j] = bb[i, j - 1] + cc[i, j]


def test_noop_on_imperfect_nest_two_sibling_inner_loops():
    sdfg = imperfect_two_sibling_inner.to_sdfg(simplify=True)
    before = _loop_nest_order(sdfg)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "an imperfect nest (sibling inner loops) is not a perfect-nest chain"
    assert _loop_nest_order(sdfg) == before


# ---------------------------------------------------------------------------
#  2D wavefront (TSVC s2111 shape): the unit-stride axis i is ALREADY innermost
#  (and carries a dependence). No axis to move inward -> no-op (skewing, not
#  permutation, is what unlocks this one).
# ---------------------------------------------------------------------------
@dace.program
def wavefront_unit_stride_inner(aa: dace.float64[N, N]):
    for j in range(1, N):
        for i in range(1, N):
            aa[j, i] = (aa[j, i - 1] + aa[j - 1, i]) / 1.9


def test_noop_on_wavefront_unit_stride_already_inner():
    sdfg = wavefront_unit_stride_inner.to_sdfg(simplify=True)
    applied = LoopStridePermutation().apply_pass(sdfg, {})
    assert not applied, "unit-stride already innermost -> nothing to interchange"
    assert _loop_nest_order(sdfg) == ['j', 'i']


if __name__ == '__main__':
    pytest.main([__file__, '-q'])


# ---------------------------------------------------------------------------
#  A/B: the interchange is the whole difference between the two binaries, so the
#  measurement isolates access order (both nests stay sequential, same work).
# ---------------------------------------------------------------------------
@dace.program
def s1232(aa: dace.float64[N, N], bb: dace.float64[N, N], cc: dace.float64[N, N]):
    for j in range(N):
        for i in range(8 * j, N):
            aa[i, j] = bb[i, j] + cc[i, j]


@pytest.mark.perf
def test_trapezoid_interchange_is_faster():
    """Same iteration set, same arithmetic, opposite access order: the strided nest touches a new
       cache line per iteration. Measured 16x (N=1024) and 28x (N=2048) here; assert only 2x, which
       no cache hierarchy inverts."""
    n = 1024
    rng = np.random.default_rng(0)
    bb = rng.random((n, n))
    cc = rng.random((n, n))

    def build(permute: bool):
        sdfg = s1232.to_sdfg(simplify=True)
        if permute:
            assert LoopStridePermutation().apply_pass(sdfg, {}) == 1
        return sdfg.compile()

    def best(csdfg) -> tuple:
        aa = np.zeros((n, n))
        csdfg(aa=aa, bb=bb, cc=cc, N=n)  # warm the caches and the first-call overhead
        times = []
        for _ in range(5):
            start = time.perf_counter()
            csdfg(aa=aa, bb=bb, cc=cc, N=n)
            times.append(time.perf_counter() - start)
        return min(times), aa

    strided, aa_strided = best(build(False))
    unit, aa_unit = best(build(True))
    assert np.allclose(aa_strided, aa_unit)
    assert unit * 2 < strided, f'unit-stride {unit:.4f}s not faster than strided {strided:.4f}s'
