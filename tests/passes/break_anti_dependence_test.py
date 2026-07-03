# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the optional BreakAntiDependence pass (snapshot-rename to break a
loop-carried WAR so LoopToMap can parallelize). SDFGs via the Python frontend."""
import contextlib
import os

import numpy as np

import dace
from dace.sdfg.state import LoopRegion, SDFGState
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
        if isinstance(n, nodes.Tasklet) and '__builtin_trap' in (n.code.as_string or '')
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
# Per-edge MIXED forward-read break (``_break_mixed_forward_reads``): an array
# written at ``a[i]`` and read at BOTH ``a[i]`` (same-index RAW) and ``a[i+1]``
# (forward WAR) off the same node -- the whole-array rename skips it, so only the
# read-ahead edge is redirected while the RAW read stays live. Hand-built
# single-compute-state loops so the pass fires directly (the frontend leaves slice
# states that would return no single compute state).
# ===========================================================================
def _mixed_loop(name):
    """A ``for i in range(N - 1)`` loop with one (empty) body state."""
    sdfg = dace.SDFG(name)
    loop = LoopRegion('loop', 'i < N - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    return sdfg, loop, body


def _pre_state(sdfg, loop):
    """The pre-loop state the mixed break inserts before ``loop`` (or None)."""
    for e in loop.parent_graph.in_edges(loop):
        if isinstance(e.src, SDFGState):
            return e.src
    return None


def test_mixed_forward_read_redirected_to_snapshot():
    """s1244 ``d[i] = a[i] + a[i+1]``: only ``a[i+1]`` (WAR) moves to the snapshot;
    ``a[i]`` (RAW, same index) keeps its live-array value."""
    sdfg, loop, body = _mixed_loop('mixed_s1244')
    for nm in ('A', 'B', 'D'):
        sdfg.add_array(nm, [N], dace.float64)
    rB = body.add_read('B')
    tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
    wA = body.add_access('A')
    body.add_edge(rB, None, tw, 'b', dace.Memlet('B[i]'))
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    tr = body.add_tasklet('r', {'a0', 'a1'}, {'d'}, 'd = a0 + a1')
    body.add_edge(wA, None, tr, 'a0', dace.Memlet('A[i]'))
    body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + 1]'))
    wD = body.add_write('D')
    body.add_edge(tr, 'd', wD, None, dace.Memlet('D[i]'))

    assert BreakAntiDependence()._break_mixed_forward_reads(loop, sdfg) == 1
    snap = next(n for n in sdfg.arrays if n.startswith('A_fwd_snap'))
    assert sdfg.arrays[snap].transient
    pre = _pre_state(sdfg, loop)
    assert pre is not None and any(n.data == snap for n in pre.data_nodes())
    a1 = next(e for e in body.in_edges(tr) if e.dst_conn == 'a1')
    a0 = next(e for e in body.in_edges(tr) if e.dst_conn == 'a0')
    assert a1.src.data == snap, "forward read a[i+1] moved to the snapshot"
    assert a0.src.data == 'A', "same-index read a[i] stays on the live array"


def test_mixed_copy_forward_read_preserves_destination_subset():
    """``bout[i] = a[i+1]`` is an access-node copy: the redirect must keep the
    destination subset ``bout[i]`` (not drop it to a whole-array write)."""
    sdfg, loop, body = _mixed_loop('mixed_copy')
    for nm in ('A', 'Bout'):
        sdfg.add_array(nm, [N], dace.float64)
    rAb = body.add_read('A')
    th = body.add_tasklet('h', {'x'}, {'y'}, 'y = x * 0.5')
    wA = body.add_access('A')
    body.add_edge(rAb, None, th, 'x', dace.Memlet('A[i - 1]'))
    body.add_edge(th, 'y', wA, None, dace.Memlet('A[i]'))
    wBout = body.add_write('Bout')
    body.add_edge(wA, None, wBout, None, dace.Memlet(data='A', subset='i + 1', other_subset='i'))

    assert BreakAntiDependence()._break_mixed_forward_reads(loop, sdfg) == 1
    snap = next(n for n in sdfg.arrays if n.startswith('A_fwd_snap'))
    in_e = next(e for e in body.in_edges(wBout) if e.data is not None)
    assert in_e.src.data == snap
    dst = in_e.data.get_dst_subset(in_e, body)
    assert dst is not None and str(dst) in ('i', 'i:i + 1'), "Bout[i] destination subset must survive"
    # The read-behind recurrence a[i-1] stays on the live array (RAW, not moved).
    assert next(e for e in body.out_edges(rAb) if e.data is not None).data.data == 'A'


def test_mixed_symbolic_forward_offset_snapshots_with_guard():
    """``d[i] = a[i] + a[i + K]`` (K a positive symbol): snapshot AND plant a
    runtime ``K >= 0`` guard so the rename is sound."""
    K = dace.symbol('K')
    sdfg, loop, body = _mixed_loop('mixed_symK')
    for nm in ('A', 'B', 'D'):
        sdfg.add_array(nm, [N], dace.float64)
    rB = body.add_read('B')
    tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
    wA = body.add_access('A')
    body.add_edge(rB, None, tw, 'b', dace.Memlet('B[i]'))
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    tr = body.add_tasklet('r', {'a0', 'a1'}, {'d'}, 'd = a0 + a1')
    body.add_edge(wA, None, tr, 'a0', dace.Memlet('A[i]'))
    body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + K]'))
    wD = body.add_write('D')
    body.add_edge(tr, 'd', wD, None, dace.Memlet('D[i]'))

    assert BreakAntiDependence()._break_mixed_forward_reads(loop, sdfg) == 1
    snap = next(n for n in sdfg.arrays if n.startswith('A_fwd_snap'))
    a1 = next(e for e in body.in_edges(tr) if e.dst_conn == 'a1')
    assert a1.src.data == snap
    pre = _pre_state(sdfg, loop)
    assert any(isinstance(n, nodes.Tasklet) and 'guard' in n.label for n in pre.nodes()), \
        "a symbolic offset must plant a sym>=0 soundness guard"


def test_mixed_symbolic_behind_offset_is_recurrence_noop():
    """``a[i - K]`` (K positive) is a read-behind RAW recurrence, not an
    anti-dependence -> no snapshot."""
    K = dace.symbol('K')
    sdfg, loop, body = _mixed_loop('mixed_symK_behind')
    sdfg.add_array('A', [N], dace.float64)
    rAb = body.add_read('A')
    th = body.add_tasklet('h', {'x'}, {'y'}, 'y = x * 0.5')
    wA = body.add_access('A')
    body.add_edge(rAb, None, th, 'x', dace.Memlet('A[i - K]'))
    body.add_edge(th, 'y', wA, None, dace.Memlet('A[i]'))
    assert BreakAntiDependence()._break_mixed_forward_reads(loop, sdfg) == 0
    assert not any(n.startswith('A_fwd_snap') for n in sdfg.arrays)


def test_mixed_same_index_only_is_noop():
    """``c[i] = a[i]`` (offset 0, RAW) is not a forward read -> no snapshot."""
    sdfg, loop, body = _mixed_loop('mixed_same')
    for nm in ('A', 'B', 'C'):
        sdfg.add_array(nm, [N], dace.float64)
    rB = body.add_read('B')
    tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
    wA = body.add_access('A')
    body.add_edge(rB, None, tw, 'b', dace.Memlet('B[i]'))
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    wC = body.add_write('C')
    body.add_edge(wA, None, wC, None, dace.Memlet(data='A', subset='i', other_subset='i'))
    assert BreakAntiDependence()._break_mixed_forward_reads(loop, sdfg) == 0
    assert not any(n.startswith('A_fwd_snap') for n in sdfg.arrays)


def test_mixed_transient_write_not_snapshotted():
    """Only non-transient (global) arrays are snapshot candidates."""
    sdfg, loop, body = _mixed_loop('mixed_transient')
    sdfg.add_array('A', [N], dace.float64, transient=True)
    sdfg.add_array('D', [N], dace.float64)
    tw = body.add_tasklet('w', {}, {'a'}, 'a = 1.0')
    wA = body.add_access('A')
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    tr = body.add_tasklet('r', {'a1'}, {'d'}, 'd = a1')
    body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + 1]'))
    wD = body.add_write('D')
    body.add_edge(tr, 'd', wD, None, dace.Memlet('D[i]'))
    assert BreakAntiDependence()._break_mixed_forward_reads(loop, sdfg) == 0


if __name__ == '__main__':
    test_break_anti_dependence_read_ahead_parallelizes()
    test_break_anti_dependence_read_behind_refused()
    test_break_anti_dependence_out_of_place_noop()
    test_break_anti_dependence_symbolic_positive_offset()
    test_break_anti_dependence_symbolic_offset_uses_iter_var_refused()
    test_break_anti_dependence_post_normalize_negative_stride_reverse_scan()
    test_break_anti_dependence_alpha_minus_one_with_larger_offset()
    test_break_anti_dependence_pure_positive_subs_doesnt_break_indirected()
    test_mixed_forward_read_redirected_to_snapshot()
    test_mixed_copy_forward_read_preserves_destination_subset()
    test_mixed_symbolic_forward_offset_snapshots_with_guard()
    test_mixed_symbolic_behind_offset_is_recurrence_noop()
    test_mixed_same_index_only_is_noop()
    test_mixed_transient_write_not_snapshotted()
