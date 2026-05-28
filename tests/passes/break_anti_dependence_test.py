# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the optional BreakAntiDependence pass (snapshot-rename to break a
loop-carried WAR so LoopToMap can parallelize). SDFGs via the Python frontend."""
import contextlib
import os

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes import BreakAntiDependence

N = dace.symbol('N')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return len([r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable])


def _l2m(sdfg):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        sdfg.apply_transformations_repeated(LoopToMap)


def test_break_anti_dependence_read_ahead_parallelizes():
    """``a[i] = a[i+1] + b[i]`` is a read-ahead WAR: renaming snapshots ``a`` so the
    loop maps, value-preserving (TSVC s121)."""

    @dace.program
    def s121(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1):
            a[i] = a[i + 1] + b[i]

    base = s121.to_sdfg(simplify=True)
    _l2m(base)
    assert _nmaps(base) == 0  # LoopToMap alone refuses the anti-dependence

    sdfg = s121.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    _l2m(sdfg)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0

    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(7):
        ref[i] = a[i + 1] + b[i]  # reads ORIGINAL a (read-ahead)
    out = a.copy()
    sdfg(a=out, b=b.copy(), N=8)
    assert np.allclose(out, ref)


def test_break_anti_dependence_read_behind_refused():
    """``a[i] = a[i-1] + b[i]`` is a read-behind RAW recurrence: renaming would be
    unsound, so the pass refuses and the loop stays sequential (TSVC s112)."""

    @dace.program
    def s112(a: dace.float64[N], b: dace.float64[N]):
        for i in range(1, N):
            a[i] = a[i - 1] + b[i]

    sdfg = s112.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) is None  # refused
    _l2m(sdfg)
    assert _nloops(sdfg) >= 1  # recurrence stays a sequential loop

    a = np.arange(1, 9, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(1, 8):
        ref[i] = ref[i - 1] + b[i]
    out = a.copy()
    sdfg(a=out, b=b.copy(), N=8)
    assert np.allclose(out, ref)


def test_break_anti_dependence_out_of_place_noop():
    """An out-of-place shifted read ``c[i] = a[i+1] + b[i]`` is already parallel
    (distinct arrays); the pass leaves it alone."""

    @dace.program
    def shift(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in range(N - 1):
            c[i] = a[i + 1] + b[i]

    sdfg = shift.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) is None  # nothing to break
    sdfg.validate()


def test_break_anti_dependence_symbolic_positive_offset():
    """``a[i] = a[i + inc] + b[i]`` with ``inc`` a free symbol -- carried offset is
    ``+inc`` which is non-numeric. The pass renames under the assumption ``inc > 0``
    AND inserts a runtime ``__builtin_trap`` guard on ``inc <= 0``. This mirrors
    TSVC s175 (forward-read parallel with symbolic stride)."""
    inc = dace.symbol('inc')

    @dace.program
    def s175_like(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - inc):
            a[i] = a[i + inc] + b[i]

    sdfg = s175_like.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    _l2m(sdfg)
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0

    # The pass must have planted a guard tasklet whose code asserts ``inc > 0``
    # (CPP language, no connectors -- pure side effect on a free symbol).
    guards = []
    for st in sdfg.all_states():
        for n in st.nodes():
            if (isinstance(n, nodes.Tasklet) and n.label.startswith('_break_antidep_guard')):
                guards.append(n)
    assert len(guards) == 1, [g.label for g in guards]
    g = guards[0]
    assert g.code.language == dace.dtypes.Language.CPP
    assert not g.in_connectors and not g.out_connectors
    # The guard's expression should contain the offset symbol.
    assert 'inc' in g.code.as_string and '__builtin_trap' in g.code.as_string

    # Numerical correctness (with inc=1, equivalent to the constant-offset case s121).
    rng = np.random.default_rng(0)
    a = rng.random(50)
    b = rng.random(50)
    ref_a = a.copy()
    for i in range(50 - 1):
        ref_a[i] = ref_a[i + 1] + b[i]  # numpy serial reference (inc=1 case)
    a_run = a.copy()
    sdfg(a=a_run, b=b, N=50, inc=1)
    assert np.allclose(a_run, ref_a)


def test_break_anti_dependence_data_indirected_offset_via_runtime_check():
    """``a[i + idx[i]] -> a[i]`` -- the SDFG splits the index computation into a
    separate state binding a free symbol (e.g. ``__sym_i_plus_idx_slice := i +
    idx[i]``); the read subset of ``a`` is that symbol. The carried offset
    ``__sym - i`` is resolved by walking back through interstate-edge
    assignments + the producing tasklet to the array read ``idx[i]``. Renaming
    is sound iff every element of ``idx`` is positive, so the pass plants a
    per-element ``__builtin_trap`` guard tasklet (one input edge reading
    ``idx`` whole, CPP body with a tight ``for`` loop checking each slot)."""
    idx_dtype = dace.int32

    @dace.program
    def indirect(a: dace.float64[N], b: dace.float64[N], idx: idx_dtype[N]):
        # Bound at N-1 so a[i + idx[i]] with idx[i] == 1 stays in range.
        for i in range(N - 1):
            a[i] = a[i + idx[i]] + b[i]

    sdfg = indirect.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    _l2m(sdfg)
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0

    # The pass must have planted an ARRAY guard tasklet over ``idx`` (CPP,
    # one input connector for ``idx``, no outputs, body asserts each slot > 0).
    array_guards = [
        n for st in sdfg.all_states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.label.startswith('_break_antidep_array_guard_')
    ]
    assert len(array_guards) == 1, [g.label for g in array_guards]
    g = array_guards[0]
    assert g.code.language == dace.dtypes.Language.CPP
    assert len(g.in_connectors) == 1 and not g.out_connectors
    assert 'idx' in g.code.as_string and '__builtin_trap' in g.code.as_string

    # Numerical correctness with a permutation that satisfies idx[i] > 0
    # for the in-range positions.
    n = 16
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b = rng.random(n)
    idx = np.array([1] * n, dtype=np.int32)
    ref = a.copy()
    for i in range(n - 1):
        ref[i] = a[i + idx[i]] + b[i]
    out = a.copy()
    sdfg(a=out, b=b.copy(), idx=idx.copy(), N=n)
    assert np.allclose(out, ref)


def test_break_anti_dependence_symbolic_offset_uses_iter_var_refused():
    """A carried offset that DEPENDS on the iteration variable (``a[2*i+1]``) is
    not a simple positive constant; the pass must refuse it."""

    @dace.program
    def bad(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N // 2):
            a[i] = a[2 * i + 1] + b[i]

    sdfg = bad.to_sdfg(simplify=True)
    # Either refused outright OR classified as complex -- in both cases nothing is renamed.
    assert BreakAntiDependence().apply_pass(sdfg, {}) is None


if __name__ == '__main__':
    test_break_anti_dependence_read_ahead_parallelizes()
    test_break_anti_dependence_read_behind_refused()
    test_break_anti_dependence_out_of_place_noop()
    test_break_anti_dependence_symbolic_positive_offset()
    test_break_anti_dependence_symbolic_offset_uses_iter_var_refused()
