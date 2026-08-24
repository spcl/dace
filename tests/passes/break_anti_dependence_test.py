# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the optional BreakAntiDependence pass (snapshot-rename to break a
loop-carried WAR so LoopToMap can parallelize). SDFGs via the Python frontend."""
import contextlib
import os

import numpy as np

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
    AND inserts a runtime ``std::abort`` guard on ``inc <= 0``. This mirrors
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
    assert 'inc' in g.code.as_string and 'std::abort' in g.code.as_string

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


def test_break_anti_dependence_symbolic_guard_survives_full_canonicalize():
    """The runtime positive guard for ``a[i] = a[i + inc] + b[i]`` (TSVC s175) must
    survive the FULL canonicalize pipeline -- a connector-less side-effect tasklet
    is otherwise pruned by dead-code elimination, silently restoring the unsound
    assume-nonneg parallelization. Runs with a valid ``inc > 0`` (the trap must not
    fire) and checks the snapshot-renamed parallel result."""
    from dace.transformation.passes.canonicalize import canonicalize
    inc = dace.symbol('inc')

    @dace.program
    def s175_like(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - inc):
            a[i] = a[i + inc] + b[i]

    sdfg = s175_like.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)

    guards = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.Tasklet) and 'std::abort' in (n.code.as_string or '')
    ]
    assert len(guards) >= 1, 'the positive-offset guard must survive full canonicalize'
    assert all(g.side_effects for g in guards), 'guard must be side-effecting so DCE keeps it'
    assert any('inc' in g.code.as_string for g in guards)
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0, 'the read-ahead loop should be parallelized'

    n = 50
    rng = np.random.default_rng(1)
    a0, b = rng.standard_normal(n), rng.standard_normal(n)
    got = a0.copy()
    sdfg(a=got, b=b, N=n, inc=1)  # valid inc>0: trap does not fire
    exp = a0.copy()
    for i in range(n - 1):
        exp[i] = a0[i + 1] + b[i]  # snapshot read of the ORIGINAL a (read-ahead)
    assert np.allclose(got, exp)


def test_break_anti_dependence_refuses_symbolic_difference_offset():
    """``a[i] = a[i + K - M] + b[i]`` -- the carried offset ``K - M`` is a difference
    of two (nonnegative) symbols whose sign is undecidable even under the
    canonicalization assumption that symbols are nonnegative. It must be refused
    (left sequential), NOT snapshot-renamed: at runtime ``K < M`` makes this a
    read-behind RAW recurrence, and renaming it produced wrong results (the old
    ``could_extract_minus_sign`` test let ``K - M`` through as a guarded WAR while
    refusing the algebraically equivalent ``M - K`` -- a canonical-ordering
    artifact -- emitting an unsatisfiable ``> 0`` guard that trapped or, once
    DCE'd, silently corrupted the output)."""
    K, M = dace.symbol('K'), dace.symbol('M')

    @dace.program
    def diff_offset(a: dace.float64[N], b: dace.float64[N]):
        for i in range(M, N):
            a[i] = a[i + K - M] + b[i]

    sdfg = diff_offset.to_sdfg(simplify=True)
    # Refused: no snapshot transient, no rename.
    assert not BreakAntiDependence().apply_pass(sdfg, {})
    assert not any(name.endswith('_antidep_snap') for name in sdfg.arrays), list(sdfg.arrays)
    _l2m(sdfg)
    # The read-behind RAW cannot be mapped -> stays sequential.
    assert _nloops(sdfg) == 1 and _nmaps(sdfg) == 0
    sdfg.validate()

    # Numerical correctness with nonnegative symbol values K=2 < M=5 (read-behind).
    n, k, m = 16, 2, 5
    rng = np.random.default_rng(7)
    a = rng.random(n)
    b = rng.random(n)
    ref_a = a.copy()
    for i in range(m, n):
        ref_a[i] = ref_a[i + k - m] + b[i]  # sequential oracle (offset k-m = -3)
    a_run = a.copy()
    sdfg(a=a_run, b=b, N=n, K=k, M=m)
    assert np.allclose(a_run, ref_a), f'max-diff {np.abs(a_run - ref_a).max()}'


def test_break_anti_dependence_sum_of_symbols_offset_renames():
    """``a[i] = a[i + K + P] + b[i]`` -- the carried offset ``K + P`` is a sum of
    nonnegative symbols, hence provably ``>= 0`` (the soundness condition for the
    snapshot rename). It must still be renamed and parallelized, confirming the
    nonnegative-difference refusal does not over-reject genuine read-ahead WARs."""
    K, P = dace.symbol('K'), dace.symbol('P')

    @dace.program
    def sum_offset(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - K - P):
            a[i] = a[i + K + P] + b[i]

    sdfg = sum_offset.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    assert any(name.endswith('_antidep_snap') for name in sdfg.arrays), list(sdfg.arrays)
    _l2m(sdfg)
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0
    sdfg.validate()

    # Numerical correctness with nonnegative symbol values K=1, P=2 (read-ahead).
    n, k, p = 32, 1, 2
    rng = np.random.default_rng(11)
    a = rng.random(n)
    b = rng.random(n)
    ref_a = a.copy()
    for i in range(n - k - p):
        ref_a[i] = ref_a[i + k + p] + b[i]
    a_run = a.copy()
    sdfg(a=a_run, b=b, N=n, K=k, P=p)
    assert np.allclose(a_run, ref_a), f'max-diff {np.abs(a_run - ref_a).max()}'


def test_break_anti_dependence_data_indirected_offset_via_runtime_check():
    """``a[i + idx[i]] -> a[i]`` -- the SDFG splits the index computation into a
    separate state binding a free symbol (e.g. ``__sym_i_plus_idx_slice := i +
    idx[i]``); the read subset of ``a`` is that symbol. The carried offset
    ``__sym - i`` is resolved by walking back through interstate-edge
    assignments + the producing tasklet to the array read ``idx[i]``. Renaming
    is sound iff every element of ``idx`` is positive, so the pass plants a
    per-element ``std::abort`` guard tasklet (one input edge reading
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
    assert 'idx' in g.code.as_string and 'std::abort' in g.code.as_string
    # PARALLEL, not a serial scan: the guard sits right in front of the loop the snapshot exists
    # to parallelize, so it delegates to the omp/simd min-reduction in dace/runtime/include/dace/
    # detect.h instead of aborting on the first violation inside a loop of its own.
    assert 'dace::detect_all_positive' in g.code.as_string
    assert 'for (' not in g.code.as_string, f'guard grew a serial loop back: {g.code.as_string!r}'

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


def test_break_anti_dependence_post_normalize_negative_stride_reverse_scan():
    """Reverse-stride scan (TSVC s112-reversed: ``for i in range(N-2, -1, -1):
    a[i+1] = a[i] + b[i]``) is a WAR pattern in iteration time -- iter 0 reads
    ``a[N-2]`` before iter 1 writes ``a[N-2]``.

    Pipeline: ``NormalizeNegativeStride`` rewrites the loop to positive stride
    with an iedge ``i := LEN_1D - _loop_pos_0 - 2`` and the loop variable
    becomes ``_loop_pos_0``. After NNS the write expression in terms of the
    new iterator ``k`` is ``a[N-1-k]`` (coefficient ``-1``).

    BAD must (a) inline the iedge binding into the memlet subsets so the
    matcher sees ``wb = N-1-k``, and (b) recognise the ``alpha = -1`` write-
    coefficient case and flip the iteration-time direction of
    ``carried_offset`` accordingly. The snapshot-and-redirect rewrite is
    direction-agnostic; only the matcher needed the extension. Numerics
    match the sequential reverse-scan oracle.
    """
    from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride

    @dace.program
    def rev_scan(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 2, -1, -1):
            a[i + 1] = a[i] + b[i]

    sdfg = rev_scan.to_sdfg(simplify=True)
    # NNS first -- BAD operates on the positive-stride form.
    nns = NormalizeNegativeStride().apply_pass(sdfg, {})
    assert nns == 1, "NormalizeNegativeStride must rewrite the negative-stride loop"
    # BAD now recognises the alpha=-1 WAR.
    bad = BreakAntiDependence().apply_pass(sdfg, {})
    assert bad == 1, "BAD must recognise the reverse-scan WAR post-NNS"
    _l2m(sdfg)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0, ("after NNS + BAD + L2M the reverse scan must be lifted")

    # Numerics: reverse-iteration scan produces a SPECIFIC value pattern
    # (different from the forward-iteration scan because of the WAR).
    n = 8
    a = np.arange(1, n + 1, dtype=np.float64)
    b = np.arange(n, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(n - 2, -1, -1):
        ref[i + 1] = ref[i] + b[i]  # reads ORIGINAL a[i] thanks to reverse iteration
    out = a.copy()
    sdfg(a=out, b=b.copy(), N=n)
    assert np.allclose(out, ref), f"got {out}, expected {ref}"


def test_break_anti_dependence_alpha_minus_one_with_larger_offset():
    """``alpha = -1`` with carried offset of magnitude > 1: the post-NNS
    shape of ``a[i+3] = a[i] + b[i]`` iterating in reverse.

    Sequential: at iter ``i``, writes ``a[i+3]``, reads ``a[i]``. Reverse
    iteration: iter k=0 (= original i=N-4) reads ``a[N-4]`` and writes
    ``a[N-1]``; iter k=1 (= original i=N-5) reads ``a[N-5]`` and writes
    ``a[N-2]``; ... Iter 0's read at a[N-4] doesn't overlap with iter k>0's
    writes (which go to ``a[N-1-k]`` for k=0..N-4, never reaching N-4).
    Actually a[N-4] is reached by k=3 (write at N-1-3 = N-4). So iter 0 reads
    a[N-4], iter 3 writes a[N-4] -- WAR with offset > 1 in iteration time.

    Post-NNS, BAD must classify this as WAR with offset = +3 (after the
    alpha=-1 sign flip), not RAW. Snapshot + redirect + parallelize.
    """
    from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride

    @dace.program
    def kernel(a: dace.float64[N + 3], b: dace.float64[N]):
        for i in range(N - 1, -1, -1):
            a[i + 3] = a[i] + b[i]

    sdfg = kernel.to_sdfg(simplify=True)
    NormalizeNegativeStride().apply_pass(sdfg, {})
    bad = BreakAntiDependence().apply_pass(sdfg, {})
    assert bad == 1
    _l2m(sdfg)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0

    n = 8
    a = np.arange(1, n + 4, dtype=np.float64)
    b = np.arange(n, dtype=np.float64) * 0.5
    ref = a.copy()
    for i in range(n - 1, -1, -1):
        ref[i + 3] = ref[i] + b[i]
    out = a.copy()
    sdfg(a=out, b=b.copy(), N=n)
    assert np.allclose(out, ref)


def test_break_anti_dependence_cast_wrapped_iterator_in_indirected_chain():
    """Regression: the frontend emits the index tasklet as ``__out = dace.int32(i) + idx_index``
    -- the iterator ``i`` is wrapped in a type-cast CALL, not a bare ``Name``. The indirected
    recogniser (:meth:`_try_recognize_indirected`) must STRIP the cast to match ``i``; otherwise
    ``a[i + idx[i]]`` mis-classifies as ``complex`` and the WAR is never broken (the loop stays
    sequential). This asserts the whole chain resolves through the cast to the ``idx`` array."""

    @dace.program
    def cast_indirect(a: dace.float64[N], b: dace.float64[N], idx: dace.int32[N]):
        for i in range(N - 1):
            a[i] = a[i + idx[i]] + b[i]

    sdfg = cast_indirect.to_sdfg(simplify=True)
    # Must recognise the indirection (renamed == 1) despite the ``dace.int32(i)`` cast.
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    assert any(name.endswith('_antidep_snap') for name in sdfg.arrays), list(sdfg.arrays)
    # A per-element array guard over ``idx`` must be planted (idx[i] > 0 soundness).
    guards = [
        n for st in sdfg.all_states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.label.startswith('_break_antidep_array_guard_')
    ]
    assert len(guards) == 1 and 'idx' in guards[0].code.as_string
    _l2m(sdfg)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0

    n = 16
    rng = np.random.default_rng(2)
    a, b = rng.random(n), rng.random(n)
    idx = np.ones(n, dtype=np.int32)  # idx[i] == 1 > 0, in range for i < n-1
    ref = a.copy()
    for i in range(n - 1):
        ref[i] = a[i + idx[i]] + b[i]
    out = a.copy()
    sdfg(a=out, b=b.copy(), idx=idx.copy(), N=n)
    assert np.allclose(out, ref)


def test_break_anti_dependence_loop_invariant_array_offset_refused():
    """``a[i + idx[0]]`` -- a LOOP-INVARIANT offset read from an ARRAY (``idx[0]``, not the
    iterator). The value is not known at compile time and, being array-sourced, is NOT assumed
    nonnegative (only pure symbols get that assumption); it also is not the ``idx[i]`` per-element
    indirection shape. So the pass conservatively REFUSES it (leaves the loop sequential) rather
    than renaming under an unchecked assumption -- ``idx[0] < 0`` would be a read-behind RAW
    recurrence. (An opt-in guarded rename -- trap unless ``idx[0] > 0`` -- is a possible future
    enhancement; today's contract is safe-refuse.)"""

    @dace.program
    def inv_array(a: dace.float64[N], b: dace.float64[N], idx: dace.int32[N]):
        for i in range(N - 1):
            a[i] = a[i + idx[0]] + b[i]

    sdfg = inv_array.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) is None  # refused, no rename
    assert not any(name.endswith('_antidep_snap') for name in sdfg.arrays), list(sdfg.arrays)
    _l2m(sdfg)
    assert _nloops(sdfg) >= 1 and _nmaps(sdfg) == 0  # stays a sequential loop

    # Numerical correctness with idx[0] = 1 (an in-bounds read-ahead the pass STILL refuses,
    # because it cannot prove the array-sourced offset is nonnegative). The refused loop runs
    # sequentially and reads the ORIGINAL a[i+1] (not yet written at iteration i).
    n = 12
    rng = np.random.default_rng(5)
    a, b = rng.random(n), rng.random(n)
    idx = np.ones(n, dtype=np.int32)  # idx[0] = 1 -> a[i+1], in-bounds read-ahead
    ref = a.copy()
    for i in range(n - 1):
        ref[i] = a[i + 1] + b[i]  # sequential reads ORIGINAL a[i+1]
    out = a.copy()
    sdfg(a=out, b=b.copy(), idx=idx.copy(), N=n)
    assert np.allclose(out, ref), f'max-diff {np.abs(out - ref).max()}'


def test_break_anti_dependence_pure_positive_subs_doesnt_break_indirected():
    """Regression for the iedge-substitution fix. The indirected-gather case
    (``a[i + idx[i]] = ...``) MUST still be recognised as ``WAR_indirected``
    even with the new substitution path active. The ``_collect_iedge_substitutions``
    helper specifically refuses to inline bindings whose RHS does not
    reference the loop iterator (the indirected chain
    ``__sym := i_plus_idx_slice`` would otherwise erase the chain that
    ``_try_recognize_indirected`` walks to identify the indirection array).
    """

    @dace.program
    def indirect(a: dace.float64[N], b: dace.float64[N], idx: dace.int32[N]):
        for i in range(N - 1):
            a[i] = a[i + idx[i]] + b[i]

    sdfg = indirect.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    _l2m(sdfg)
    sdfg.validate()
    assert _nmaps(sdfg) >= 1 and _nloops(sdfg) == 0


# ===========================================================================
# forward_reads: break ONE read-ahead edge of an array whose other reads are true
# dependences. The loop stays sequential; what it buys is that the read-ahead no
# longer binds two otherwise-independent statements, so fission can distribute them.
# ===========================================================================
K = dace.symbol('K')


@dace.program
def _mixed_carry_and_read_ahead(a: dace.float64[N], d: dace.float64[N], x: dace.float64[N]):
    """``a`` carries a recurrence AND is read ahead by a second, independent statement."""
    for i in range(1, N - 1):
        a[i] = a[i - 1] + x[i]
        d[i] = a[i + 1]


def _snaps(sdfg):
    return sorted(nm for nm in sdfg.arrays if '_split_snap' in nm)


def test_forward_reads_breaks_what_the_whole_array_policy_refuses():
    """``a[i]=a[i-1]+x[i]; d[i]=a[i+1]``: one read of ``a`` is a true recurrence, the other is a
    read-ahead. The whole-array policy disqualifies the array on the first RAW pair, because its
    goal is a loop that maps. ``forward_reads`` breaks the read-ahead edge alone."""
    whole = _mixed_carry_and_read_ahead.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(whole, {}) is None
    assert not _snaps(whole)

    sdfg = _mixed_carry_and_read_ahead.to_sdfg(simplify=True)
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    assert len(_snaps(sdfg)) == 1
    sdfg.validate()


def test_forward_reads_preserves_values():
    n = 12
    rng = np.random.default_rng(11)
    a0, x = rng.random(n), rng.random(n)

    ref_a, ref_d = a0.copy(), np.zeros(n)
    _mixed_carry_and_read_ahead.to_sdfg(simplify=True)(a=ref_a, d=ref_d, x=x.copy(), N=n)

    sdfg = _mixed_carry_and_read_ahead.to_sdfg(simplify=True)
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    got_a, got_d = a0.copy(), np.zeros(n)
    sdfg(a=got_a, d=got_d, x=x.copy(), N=n)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_d, ref_d)


def test_forward_reads_is_idempotent():
    """Once the read-ahead edge sits on the snapshot there is no read-ahead of the live array
    left, so a second application finds nothing and adds no second copy."""
    sdfg = _mixed_carry_and_read_ahead.to_sdfg(simplify=True)
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    before = sdfg.hash_sdfg()
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) is None
    assert sdfg.hash_sdfg() == before
    sdfg.validate()


def test_forward_reads_leaves_a_same_index_read_on_the_live_array():
    """``a[i]=x[i]; d[i]=a[i]+a[i+1]`` -- only ``a[i+1]`` moves.

    ``a[i]`` is the value the SIBLING statement just wrote this iteration; redirecting it to the
    pre-loop snapshot would read the stale original.
    """

    @dace.program
    def s1244(a: dace.float64[N], d: dace.float64[N], x: dace.float64[N]):
        for i in range(N - 1):
            a[i] = x[i] * 2.0
            d[i] = a[i] + a[i + 1]

    n = 12
    rng = np.random.default_rng(12)
    x = rng.random(n)
    ref_a, ref_d = np.zeros(n), np.zeros(n)
    s1244.to_sdfg(simplify=True)(a=ref_a, d=ref_d, x=x.copy(), N=n)

    sdfg = s1244.to_sdfg(simplify=True)
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    got_a, got_d = np.zeros(n), np.zeros(n)
    sdfg(a=got_a, d=got_d, x=x.copy(), N=n)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_d, ref_d)


def test_forward_reads_symbolic_offset_guard_is_strictly_positive():
    """The mixed shape needs ``offset >= 1``, not the ``offset >= 0`` the whole-array policy emits.

    A sibling statement writes ``a[i]`` earlier in the SAME iteration, so an offset that turns out
    to be 0 at runtime aliases that just-written live value; redirecting it to the stale snapshot
    would be a silent miscompile, and the guard has to trap instead.
    """

    @dace.program
    def sym_mixed(a: dace.float64[N], d: dace.float64[N], x: dace.float64[N]):
        for i in range(N - K - 1):
            a[i] = x[i] * 2.0
            d[i] = a[i] + a[i + K]

    sdfg = sym_mixed.to_sdfg(simplify=True)
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    guards = [
        n.code.as_string for st in sdfg.all_states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.name.startswith('_break_antidep_guard_')
    ]
    assert guards, 'a symbolic offset must carry a runtime guard'
    strict = dace.symbolic.symstr(dace.symbolic.pystr_to_symbolic('K') - 1)
    assert any(strict in g for g in guards), guards


# ===========================================================================
# Snapshot window: the copy covers the elements the redirected reads touch, not
# the whole array. Proportional, so it only shows where the loop sweeps a slice.
# ===========================================================================
B = dace.symbol('B')
NB = dace.symbol('NB')


def _snapshot_copies(sdfg):
    """Every ``name -> snap`` copy memlet the pass planted, in state order."""
    return [
        e.data for st in sdfg.all_states() for e in st.edges() if isinstance(e.dst, nodes.AccessNode)
        and '_snap' in e.dst.data and e.data is not None and not e.data.is_empty()
    ]


def test_snapshot_copies_only_the_block_a_nested_loop_reads():
    """A blocked inner sweep ``a[i] = a[i+1] + b[i]`` over ``a[t*B : (t+1)*B]``.

    The snapshot state sits inside the outer loop, so a whole-array copy is paid once per
    outer iteration -- O(NB x N) of memcpy for an O(N) sweep. Only the block is ever read
    off the snapshot, so only the block is copied and ``N`` does not appear in the copy."""

    @dace.program
    def blocked(a: dace.float64[N], b: dace.float64[N]):
        for t in range(NB):
            for i in range(t * B, (t + 1) * B - 1):
                a[i] = a[i + 1] + b[i]

    sdfg = blocked.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    copies = _snapshot_copies(sdfg)
    assert len(copies) == 1
    assert 'N' not in {str(s) for s in copies[0].subset.free_symbols}, copies[0].subset

    n, b_, nb = 16, 4, 4
    rng = np.random.default_rng(21)
    a0, b0 = rng.random(n), rng.random(n)
    ref = a0.copy()
    for t in range(nb):
        for i in range(t * b_, (t + 1) * b_ - 1):
            ref[i] = ref[i + 1] + b0[i]
    got = a0.copy()
    sdfg(a=got, b=b0.copy(), N=n, B=b_, NB=nb)
    assert np.array_equal(got, ref)


def test_forward_reads_snapshot_copies_only_the_swept_window():
    """The per-edge policy shrinks the same way: a sweep of ``[4, N-4)`` reading ``a[i+1]``
    copies the ``N - 8`` elements it reads, not the array."""

    @dace.program
    def windowed(a: dace.float64[N], d: dace.float64[N], x: dace.float64[N]):
        for i in range(4, N - 4):
            a[i] = x[i] * 2.0
            d[i] = a[i] + a[i + 1]

    sdfg = windowed.to_sdfg(simplify=True)
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    sdfg.validate()
    copies = _snapshot_copies(sdfg)
    assert len(copies) == 1
    assert copies[0].subset.num_elements() == dace.symbolic.pystr_to_symbolic('N - 8')

    n = 16
    rng = np.random.default_rng(22)
    x = rng.random(n)
    ref_a, ref_d = np.zeros(n), np.zeros(n)
    windowed.to_sdfg(simplify=True)(a=ref_a, d=ref_d, x=x.copy(), N=n)
    got_a, got_d = np.zeros(n), np.zeros(n)
    sdfg(a=got_a, d=got_d, x=x.copy(), N=n)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_d, ref_d)


if __name__ == '__main__':
    test_snapshot_copies_only_the_block_a_nested_loop_reads()
    test_forward_reads_snapshot_copies_only_the_swept_window()
    test_forward_reads_breaks_what_the_whole_array_policy_refuses()
    test_forward_reads_preserves_values()
    test_forward_reads_is_idempotent()
    test_forward_reads_leaves_a_same_index_read_on_the_live_array()
    test_forward_reads_symbolic_offset_guard_is_strictly_positive()
    test_break_anti_dependence_read_ahead_parallelizes()
    test_break_anti_dependence_read_behind_refused()
    test_break_anti_dependence_out_of_place_noop()
    test_break_anti_dependence_symbolic_positive_offset()
    test_break_anti_dependence_symbolic_offset_uses_iter_var_refused()
    test_break_anti_dependence_post_normalize_negative_stride_reverse_scan()
    test_break_anti_dependence_alpha_minus_one_with_larger_offset()
    test_break_anti_dependence_pure_positive_subs_doesnt_break_indirected()


def test_smt_fallback_breaks_a_guarded_indirected_read_ahead():
    """``A[i] = A[IDX[i] * IDX[i-1]] + 1`` under the guard ``IDX[i] * IDX[i-1] > i``.

    The product is non-linear and the guard lives in a branch, so the affine matcher gets no
    carried offset at all and returns ``complex``. The guard is nevertheless decisive: it forces
    every read strictly above the reading iteration, so no iteration up to and including this one
    has written the element -- a pure anti-dependence, and a pre-loop snapshot of ``A`` breaks it.

    The frontend is what makes this hard: it materializes ``IDX[i]`` once for the condition and
    again for the subscript, under different names, so the guard and the read only line up after
    their interstate bindings are expanded."""

    @dace.program
    def guarded_indirect(A: dace.float64[N], IDX: dace.int64[N]):
        for i in range(1, N):
            if IDX[i] * IDX[i - 1] > i:
                A[i] = A[IDX[i] * IDX[i - 1]] + 1.0
            else:
                A[i] = 0.0

    base = guarded_indirect.to_sdfg(simplify=True)
    _l2m(base)
    assert _nmaps(base) == 0, 'LoopToMap alone must refuse the guarded indirection'

    sdfg = guarded_indirect.to_sdfg(simplify=True)
    assert BreakAntiDependence().apply_pass(sdfg, {}) == 1, 'the SMT fallback should break exactly one array'
    assert any(name.endswith('_antidep_snap') or '_antidep_snap' in name
               for name in sdfg.arrays), f'no snapshot transient was added: {sorted(sdfg.arrays)}'

    n = 16
    rng = np.random.default_rng(1)
    a = rng.random(n)
    idx = np.full(n, 3, dtype=np.int64)
    expected = a.copy()
    for i in range(1, n):
        p = int(idx[i]) * int(idx[i - 1])
        expected[i] = expected[p] + 1.0 if p > i else 0.0

    got = a.copy()
    sdfg(A=got, IDX=idx.copy(), N=n)
    assert np.allclose(got, expected), 'breaking the guarded anti-dependence must preserve values'


def test_smt_fallback_refuses_the_same_shape_without_the_guard():
    """Fail-closed twin of the test above: drop the branch and the read can land anywhere,
    including on an element an EARLIER iteration wrote. That is a true recurrence, and the
    snapshot would feed it stale values -- so the pass must decline, guard or no solver."""

    @dace.program
    def unguarded_indirect(A: dace.float64[N], IDX: dace.int64[N]):
        for i in range(1, N):
            A[i] = A[IDX[i] * IDX[i - 1]] + 1.0

    sdfg = unguarded_indirect.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert BreakAntiDependence().apply_pass(sdfg, {}) is None, 'an unguarded indirect read is not read-ahead'
    assert sdfg.to_json() == before, 'a refusing pass must leave the SDFG bit-identical'


def test_smt_fallback_refuses_when_the_indirection_array_is_written_in_the_loop():
    """The oracle models ``IDX`` as one immutable array, so the same guard means nothing once the
    loop writes ``IDX`` -- ``IDX[i]`` at the reading iteration is not ``IDX[i]`` at the writing
    one. The gate is in the pass, not in the solver, because only the pass knows the loop."""

    @dace.program
    def mutating_indirect(A: dace.float64[N], IDX: dace.int64[N]):
        for i in range(1, N):
            if IDX[i] * IDX[i - 1] > i:
                A[i] = A[IDX[i] * IDX[i - 1]] + 1.0
            else:
                A[i] = 0.0
            IDX[i - 1] = IDX[i] + 1

    sdfg = mutating_indirect.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert BreakAntiDependence().apply_pass(sdfg, {}) is None, 'a loop-written indirection array must refuse'
    assert sdfg.to_json() == before, 'a refusing pass must leave the SDFG bit-identical'
