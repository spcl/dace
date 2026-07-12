# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Refuse-guard correctness tests for
:class:`~dace.transformation.passes.canonicalize.early_exit_to_find_index.EarlyExitToFindIndex`.

The find-first lift reproduces every predicate array read at exactly
``name[loop_var]`` (the per-iteration point read the phi Map can emit). Any other
subscript must be REFUSED so the loop stays a correct sequential ``LoopRegion``:

* an offset ``a[i-1]`` / ``a[i+1]`` or coefficient -- otherwise the two reads
  collapse to one ``a[loop_var]`` (a silent miscompile of the exit index);
* a gather / nested subscript ``a[idx[i]]`` -- otherwise the read rewrite emits a
  dangling ``__r_a]`` (a ``SyntaxError`` crash);
* a multi-dim ``a[i, j]`` -- otherwise a 1-D point memlet lands on an N-D array
  (a validate error).

The genuine TSVC find-first shapes (a single ``name[i]`` read predicate) MUST
still lift and optimize -- the refuse-guard must be precise, not blanket.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import Reduce
from dace.transformation.passes.canonicalize.early_exit_to_find_index import EarlyExitToFindIndex
from dace.transformation.interstate import LoopToMap
from tests.corpus.tsvc_2_5.tsvc_2_5 import (ext_break_find_first, ext_break_post_body, ext_break_capture, LEN_1D, K)

N = dace.symbol('N')


def _num_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _num_reduces(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))


# ---------------------------------------------------------------------------
# Broken predicate shapes: each MUST refuse (stay a sequential LoopRegion) and
# still run bit-exact vs the sequential numpy reference. Before the guard these
# either miscompiled (offset reads collapsed) or crashed (gather / multi-dim).
# ---------------------------------------------------------------------------


@dace.program
def pred_offset_prev(a: dace.float64[N], out: dace.float64[N]):
    for i in range(1, N):
        if a[i] > a[i - 1]:  # reads a at i AND i-1 -- not a single name[i] read
            break
        out[i] = a[i] * 2.0


@dace.program
def pred_offset_next(a: dace.float64[N + 1], out: dace.float64[N]):
    for i in range(N):
        if a[i] != a[i + 1]:  # reads a at i AND i+1
            break
        out[i] = a[i] * 2.0


@dace.program
def pred_gather(a: dace.float64[N], idx: dace.int64[N], out: dace.float64[N]):
    for i in range(N):
        if a[idx[i]] < 0.0:  # gather / nested subscript
            break
        out[i] = a[i] * 2.0


@dace.program
def pred_multidim(a: dace.float64[N, N], out: dace.float64[N]):
    for i in range(N):
        if a[i, 0] < 0.0:  # multi-dim subscript
            break
        out[i] = a[i, i] * 2.0


def _assert_refuses_and_runs(sdfg):
    """Run the pass; assert it is a NO-OP (0 matches, loop count unchanged and
    >= 1, no crash) and the SDFG still validates. Returns the loop count."""
    before = _num_loops(sdfg)
    assert before >= 1, 'kernel must start with at least one LoopRegion'
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res is None, f'unsupported predicate must refuse (0 matches), got res={res}'
    after = _num_loops(sdfg)
    assert after == before >= 1, f'loop must stay sequential (before={before}, after={after})'
    sdfg.validate()
    return after


def test_refuses_offset_prev_read():
    """``a[i] > a[i-1]``: the phi Map would collapse both reads to ``a[i]`` and
    compute the wrong exit index. Must refuse and stay bit-exact."""
    sdfg = pred_offset_prev.to_sdfg(simplify=True)
    _assert_refuses_and_runs(sdfg)

    n = 8
    a = np.array([5.0, 4.0, 3.0, 4.0, 9.0, 1.0, 2.0, 3.0])  # non-increasing then a[3] > a[2]
    ref = np.zeros(n)
    for i in range(1, n):
        if a[i] > a[i - 1]:
            break
        ref[i] = a[i] * 2.0
    got = np.zeros(n)
    sdfg(a=a.copy(), out=got, N=n)
    assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


def test_refuses_offset_next_read():
    """``a[i] != a[i+1]``: two reads of ``a`` at different offsets. Must refuse."""
    sdfg = pred_offset_next.to_sdfg(simplify=True)
    _assert_refuses_and_runs(sdfg)

    n = 6
    a = np.array([3.0, 3.0, 3.0, 7.0, 8.0, 9.0, 10.0])  # equal run then a[3] differs
    ref = np.zeros(n)
    for i in range(n):
        if a[i] != a[i + 1]:
            break
        ref[i] = a[i] * 2.0
    got = np.zeros(n)
    sdfg(a=a.copy(), out=got, N=n)
    assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


def test_refuses_gather_read():
    """``a[idx[i]] < 0``: a gather predicate. Before the guard the read rewrite
    produced a dangling ``__r_a]`` -> SyntaxError. Must refuse cleanly."""
    sdfg = pred_gather.to_sdfg(simplify=True)
    _assert_refuses_and_runs(sdfg)

    n = 6
    a = np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0])
    idx = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    ref = np.zeros(n)
    for i in range(n):
        if a[idx[i]] < 0.0:
            break
        ref[i] = a[i] * 2.0
    got = np.zeros(n)
    sdfg(a=a.copy(), idx=idx.copy(), out=got, N=n)
    assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


def test_refuses_multidim_read():
    """``a[i, 0] < 0``: a 2-D predicate. Before the guard a 1-D point memlet was
    attached to a 2-D array -> validate/dimensionality error. Must refuse."""
    sdfg = pred_multidim.to_sdfg(simplify=True)
    _assert_refuses_and_runs(sdfg)

    n = 5
    a = np.array([[1.0, 0.5, 0.0, 0.0, 0.0], [2.0, 0.0, 0.5, 0.0, 0.0], [-1.0, 0.0, 0.0, 3.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0, 6.0]])  # a[2,0] < 0 fires at i=2
    ref = np.zeros(n)
    for i in range(n):
        if a[i, 0] < 0.0:
            break
        ref[i] = a[i, i] * 2.0
    got = np.zeros(n)
    sdfg(a=a.copy(), out=got, N=n)
    assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


# ---------------------------------------------------------------------------
# Precision: a single ``name[i]`` read predicate is the intended target and MUST
# still lift (do not over-refuse).
# ---------------------------------------------------------------------------


@dace.program
def pred_single_threshold(a: dace.float64[N], out: dace.float64[N], threshold: dace.float64):
    for i in range(N):
        if a[i] > threshold:  # single name[i] read + loop-invariant scalar -- MUST lift
            break
        out[i] = a[i] * 2.0


def test_single_read_predicate_still_lifts():
    """``a[i] > threshold`` -- the TSVC find-first shape. The guard must NOT
    refuse it: the pass fires (1 match), a find-first Reduce(Min) appears, and the
    result is bit-exact (fire + no-fire)."""
    sdfg = pred_single_threshold.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res == 1, 'single name[i] read predicate must still lift'
    sdfg.validate()
    assert _num_reduces(sdfg) == 1, 'find-first Reduce(Min) must be present'

    n = 8
    for a in (
            np.array([1.0, 2.0, 3.0, 50.0, 5.0, 6.0, 7.0, 8.0]),  # fires at i=3
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])):  # never fires (all <= 10)
        ref = np.zeros(n)
        for i in range(n):
            if a[i] > 10.0:
                break
            ref[i] = a[i] * 2.0
        got = np.zeros(n)
        sdfg(a=a.copy(), out=got, threshold=10.0, N=n)
        assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


# ---------------------------------------------------------------------------
# The ACTUAL TSVC early-exit / find-first corpus kernels (s481 / s482 / s332)
# must STILL lift and stay bit-exact after the guard -- the guard is precise.
# ---------------------------------------------------------------------------


def test_tsvc_ext_break_find_first_still_lifts():
    """TSVC ``s481`` (``ext_break_find_first``): ``if d[i] < 0: break`` then body.
    Single ``d[i]`` read -> lifts to find-first Reduce + parallel body."""
    sdfg = ext_break_find_first.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res == 1, 'ext_break_find_first (s481) must lift'
    sdfg.validate()
    assert _num_reduces(sdfg) == 1
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()
    assert _num_loops(sdfg) == 0, 'find-first body must fully parallelize'

    n = 12
    rng = np.random.default_rng(481)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    for d in (np.linspace(1.0, -1.0, n), np.ones(n)):  # fire + no-fire
        ref = a.copy()
        for i in range(n):
            if d[i] < 0.0:
                break
            ref[i] = ref[i] + b[i] * c[i]
        got = a.copy()
        sdfg(a=got, b=b.copy(), c=c.copy(), d=d.copy(), LEN_1D=n)
        assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


def test_tsvc_ext_break_post_body_still_lifts():
    """TSVC ``s482`` (``ext_break_post_body``): body then ``if c[i] > b[i]: break``.
    Two DIFFERENT arrays each at ``[i]`` -- both are point reads, so it lifts."""
    sdfg = ext_break_post_body.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res == 1, 'ext_break_post_body (s482) must lift'
    sdfg.validate()
    assert _num_reduces(sdfg) == 1

    n = 12
    rng = np.random.default_rng(482)
    b = rng.standard_normal(n)
    for c in (b + np.linspace(-1.0, 1.0, n), b - 1.0):  # fire + no-fire
        a = rng.standard_normal(n)
        ref = a.copy()
        for i in range(n):
            ref[i] = ref[i] + b[i] * c[i]
            if c[i] > b[i]:
                break
        got = a.copy()
        sdfg(a=got, b=b.copy(), c=c.copy(), LEN_1D=n)
        assert np.allclose(got, ref, equal_nan=True), f'got {got} ref {ref}'


def test_tsvc_ext_break_capture_still_lifts():
    """TSVC ``s332`` (``ext_break_capture``): find first ``a[i] > K``, capture
    index + value, break. Single ``a[i]`` read -> lifts (with Phase-3 rebind)."""
    sdfg = ext_break_capture.to_sdfg(simplify=True)
    res = EarlyExitToFindIndex().apply_pass(sdfg, {})
    assert res == 1, 'ext_break_capture (s332) must lift'
    sdfg.validate()
    assert _num_reduces(sdfg) == 1

    n = 8
    for a, kthr in (
        (np.array([1.0, 2.0, 3.0, 50.0, 5.0, 6.0, 7.0, 8.0]), 10),  # fires at i=3
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), 100)):  # never fires
        idx_ref, val_ref = -1, -1.0
        for i in range(n):
            if a[i] > kthr:
                idx_ref, val_ref = i, a[i]
                break
        out_index = np.zeros(1, dtype=np.int64)
        out_value = np.zeros(1)
        sdfg(a=a.copy(), out_index=out_index, out_value=out_value, LEN_1D=n, K=kthr)
        assert out_index[0] == idx_ref, f'index got {out_index[0]} ref {idx_ref}'
        assert np.allclose(out_value[0], val_ref, equal_nan=True), f'value got {out_value[0]} ref {val_ref}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
