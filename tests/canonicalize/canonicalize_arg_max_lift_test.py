# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.transformation.passes.canonicalize.arg_max_lift.ArgMaxLift`.

Covers TSVC s314 (max), s316 (min), and refusals on the v1 out-of-scope shapes
(s3113 -- unary transform on the gather; s315 -- index-tracking variant).
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import Reduce
from dace.transformation.passes.canonicalize.arg_max_lift import ArgMaxLift


N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_reduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))


# -----------------------------------------------------------------------------
# Positive: TSVC s314 (max) and s316 (min).
# -----------------------------------------------------------------------------

def test_tsvc_s314_max_value_only():
    """``x = a[0]; for i in range(1, N): if a[i] > x: x = a[i]`` lifts to a
    ``Reduce(Max)`` libnode. The pre-loop init ``x = a[0]`` is preserved as
    the seed via the libnode's ``identity=None`` semantics (WCR-Max folds the
    output's existing value into the reduction).
    """

    @dace.program
    def s314(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = s314.to_sdfg(simplify=True)
    assert _num_loops(sdfg) == 1
    res = ArgMaxLift().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1
    assert _num_loops(sdfg) == 0
    assert _num_reduces(sdfg) == 1

    n = 16
    rng = np.random.default_rng(314)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.max(a)), f"got {out[0]}, expected {np.max(a)}"


def test_tsvc_s316_min_value_only():
    """``<`` instead of ``>`` → ``Reduce(Min)``."""

    @dace.program
    def s316(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] < x:
                x = a[i]
        result[0] = x

    sdfg = s316.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    sdfg.validate()
    assert res == 1

    n = 12
    rng = np.random.default_rng(316)
    a = rng.standard_normal(n)
    out = np.zeros(1)
    sdfg(a=a, result=out, N=n)
    assert np.isclose(out[0], np.min(a)), f"got {out[0]}, expected {np.min(a)}"


def test_max_corner_first_element_is_max():
    """``a[0]`` is the maximum -- the libnode's pre-existing-output seed picks
    it up even though the input slice ``a[1:N]`` excludes index 0."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    ArgMaxLift().apply_pass(sdfg, {})
    sdfg.validate()
    a = np.array([100.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    out = np.zeros(1)
    sdfg(a=a, result=out, N=6)
    assert np.isclose(out[0], 100.0)


# -----------------------------------------------------------------------------
# Refusals: v1 out-of-scope shapes.
# -----------------------------------------------------------------------------

def test_refuses_unary_transform_on_gather_s3113():
    """TSVC s3113: ``av = abs(a[i]); if av > maxv: maxv = av``. The gather is
    transformed by ``abs`` before the comparison; v1 only recognises direct
    array reads. Refused; the loop stays sequential."""

    @dace.program
    def s3113(a: dace.float64[N], b: dace.float64[2]):
        maxv = abs(a[0])
        for i in range(N):
            av = abs(a[i])
            if av > maxv:
                maxv = av
        b[0] = maxv

    sdfg = s3113.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res is None, "abs(a[i]) transform should be refused in v1"


def test_refuses_index_tracking_s315():
    """TSVC s315: ``if a[i] > x: x = a[i]; index = i``. The true-branch writes
    BOTH the value carrier and an index; v1 only handles the value carrier.
    The ``index = i`` write lives on an interstate edge inside the true-branch
    -- the matcher checks for any such edge assignment and refuses."""

    @dace.program
    def s315(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        index = 0
        for i in range(N):
            if a[i] > x:
                x = a[i]
                index = i
        result[0] = x + float(index)

    sdfg = s315.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res is None, "index-tracking variant should be refused in v1"


def test_refuses_non_comparison_condition():
    """The body's condition must be a single ``a OP b`` comparison; bitwise/
    boolean operators are out of scope."""

    @dace.program
    def kernel(a: dace.float64[N], b: dace.int64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if b[i] != 0 and a[i] > x:
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    # The compound condition is wrapped in a chain of iedges that the matcher
    # can't trace back to a single ``Compare`` AST node. Refuse.
    assert res is None


def test_refuses_subtraction_op():
    """``Sub`` is not in :data:`_CMP_AST_TO_RTYPE`; only ``>``, ``<``, ``>=``, ``<=``."""

    @dace.program
    def kernel(a: dace.float64[N], result: dace.float64[1]):
        x = a[0]
        for i in range(1, N):
            if a[i] != x:    # ``!=`` not in the set
                x = a[i]
        result[0] = x

    sdfg = kernel.to_sdfg(simplify=True)
    res = ArgMaxLift().apply_pass(sdfg, {})
    assert res is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
