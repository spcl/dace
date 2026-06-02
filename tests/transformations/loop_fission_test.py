# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for LoopFission (loop distribution), the LoopRegion equivalent of
    MapFission. Mirrors the map-fission frontend kernels with ``dace.map``
    replaced by ``range`` so the frontend emits loops. Every test checks
    numerical equivalence against a deep-copied pre-pass run; loop counts are
    asserted where the independent-group partition is deterministic.

    LoopFission only distributes a single-body-state loop; data-dependent
    statements (and bodies with control flow / nested loops) stay in one
    loop -- those mirror as value-preserving no-ops.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_fission import LoopFission

N = dace.symbol('N')
START, STOP, STEP = (dace.symbol(s) for s in ('START', 'STOP', 'STEP'))


@dace.program
def loop_two(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_three(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(N):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0
        C[i] = a[i] - 3.0


@dace.program
def loop_single(a: dace.float64[N], A: dace.float64[N]):
    for i in range(N):
        A[i] = a[i] + 1.0


@dace.program
def loop_dependent(a: dace.float64[N], T: dace.float64[N], B: dace.float64[N]):
    for i in range(N):
        T[i] = a[i] + 1.0
        B[i] = T[i] * 2.0


@dace.program
def loop_carried(A: dace.float64[N]):
    for i in range(1, N):
        A[i] = A[i - 1] + 1.0


@dace.program
def loop_strided(a: dace.float64[40], A: dace.float64[40], B: dace.float64[40]):
    for i in range(0, 9, 2):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_offset_strided(a: dace.float64[40], A: dace.float64[40], B: dace.float64[40]):
    for i in range(10, 29, 3):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_symbolic_strided(a: dace.float64[64], A: dace.float64[64], B: dace.float64[64]):
    for i in range(START, STOP, STEP):
        A[i] = a[i] + 1.0
        B[i] = a[i] * 2.0


@dace.program
def loop_five_set_five_cpy(s0: dace.float64[N], s1: dace.float64[N], s2: dace.float64[N], s3: dace.float64[N],
                           s4: dace.float64[N], a0: dace.float64[N], a1: dace.float64[N], a2: dace.float64[N],
                           a3: dace.float64[N], a4: dace.float64[N], c0: dace.float64[N], c1: dace.float64[N],
                           c2: dace.float64[N], c3: dace.float64[N], c4: dace.float64[N]):
    for i in range(N):
        s0[i] = 0.0
        s1[i] = 1.0
        s2[i] = 2.0
        s3[i] = 3.0
        s4[i] = 4.0
        c0[i] = a0[i]
        c1[i] = a1[i]
        c2[i] = a2[i]
        c3[i] = a3[i]
        c4[i] = a4[i]


@dace.program
def loop_nested_two(x: dace.float64[N, N], y: dace.float64[N, N]):
    for j in range(N):
        for i in range(N):
            x[i, j] = 1.0
        for i in range(N):
            y[i, j] = 2.0


@dace.program
def loop_nested_dependent(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for j in range(N):
        for i in range(N):
            B[i, j] = A[i, j] + 1.0
        for i in range(N):
            C[i, j] = B[i, j] * 2.0


@dace.program
def loop_nested_cycle(A: dace.float64[N, N], B: dace.float64[N, N]):
    for j in range(1, N):
        for i in range(N):
            A[i, j] = B[i, j - 1] + 1.0
        for i in range(N):
            B[i, j] = A[i, j] * 2.0


# --- TSVC loop-carried-dependence patterns: fission must be refused ---


@dace.program
def tsvc_s211(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N]):
    for i in range(1, N - 1):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = b[i + 1] - e[i] * d[i]


@dace.program
def tsvc_s1213(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N - 1):
        a[i] = b[i - 1] + c[i]
        b[i] = a[i + 1] * d[i]


@dace.program
def tsvc_s221(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i] + c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]


@dace.program
def tsvc_s222(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], e: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i] + b[i] * c[i]
        e[i] = e[i - 1] * e[i - 1]
        a[i] = a[i] - b[i] * c[i]


@dace.program
def tsvc_s111(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N, 2):
        a[i] = a[i - 1] + b[i]


@dace.program
def tsvc_recurrence_plus_independent(a: dace.float64[N], b: dace.float64[N], x: dace.float64[N], y: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + b[i]
        x[i] = y[i] * 2.0


@dace.program
def loop_conditional(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], c: dace.int32[1]):
    for i in range(N):
        if c[0] > 0:
            A[i] = a[i] + 1.0
            B[i] = a[i] * 2.0


def _loop_count(sdfg: dace.SDFG) -> int:
    return sum(1 for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion))


def _run(prog, args, kw, expect_loops):
    """Apply LoopFission, validate, assert loop count and e2e numerics.

    :param prog: The dace program.
    :param args: Keyword arrays compared before/after.
    :param kw: Extra scalar kwargs (symbols) for the calls.
    :param expect_loops: Expected LoopRegion count after fission.
    :returns: The transformed SDFG.
    """
    sdfg = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(sdfg)(**ref, **kw)

    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    assert _loop_count(sdfg) == expect_loops, f"expected {expect_loops} loops, got {_loop_count(sdfg)}"

    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, **kw)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"
    return sdfg


def test_loop_fission_two():
    n = 16
    a = np.random.rand(n)
    _run(loop_two, dict(a=a, A=np.zeros(n), B=np.zeros(n)), dict(N=n), 2)


def test_loop_fission_three():
    n = 12
    a = np.random.rand(n)
    _run(loop_three, dict(a=a, A=np.zeros(n), B=np.zeros(n), C=np.zeros(n)), dict(N=n), 3)


def test_loop_fission_single_is_noop():
    n = 8
    a = np.random.rand(n)
    sdfg = loop_single.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    A = np.zeros(n)
    sdfg(a=a.copy(), A=A, N=n)
    assert np.allclose(A, a + 1.0)


def test_loop_fission_dependent_kept_together_modified_splits_per_iter():
    """``T[i] = a[i] + 1; B[i] = T[i] * 2`` -- ``T`` is written by stmt 1 and
    read by stmt 2 at the *same* iteration index. Sequential loop fission
    preserves the value (loop 1 fills T completely; loop 2 then reads the
    just-written T), so the per-iter dependency is fissionable. (Earlier
    contract refused this case; updated when multi-statement bodies for the
    TSVC s221/s222 family became supported.)"""
    n = 10
    a = np.random.rand(n)
    sdfg = loop_dependent.to_sdfg(simplify=True)
    res = LoopFission().apply_pass(sdfg, {})
    assert res is not None and res >= 1
    sdfg.validate()
    T, B = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), T=T, B=B, N=n)
    assert np.allclose(T, a + 1.0) and np.allclose(B, (a + 1.0) * 2.0)


def test_loop_fission_loop_carried_is_noop():
    n = 9
    sdfg = loop_carried.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    A = np.zeros(n)
    A[0] = 5.0
    ref = A.copy()
    for i in range(1, n):
        ref[i] = ref[i - 1] + 1.0
    sdfg(A=A, N=n)
    assert np.allclose(A, ref)


def test_loop_fission_strided():
    a = np.random.rand(40)
    _run(loop_strided, dict(a=a, A=np.full(40, -1.0), B=np.full(40, -1.0)), {}, 2)


def test_loop_fission_offset_strided():
    a = np.random.rand(40)
    _run(loop_offset_strided, dict(a=a, A=np.full(40, -1.0), B=np.full(40, -1.0)), {}, 2)


def test_loop_fission_symbolic_strided():
    a = np.random.rand(64)
    _run(loop_symbolic_strided, dict(a=a, A=np.full(64, -1.0), B=np.full(64, -1.0)), dict(START=0, STOP=64, STEP=5), 2)


def test_loop_fission_many_set_cpy():
    n = 8
    arrs = {f'a{i}': np.random.rand(n) for i in range(5)}
    arrs.update({f's{i}': np.zeros(n) for i in range(5)})
    arrs.update({f'c{i}': np.zeros(n) for i in range(5)})
    _run(loop_five_set_five_cpy, arrs, dict(N=n), 10)


def test_loop_fission_perfect_nesting():
    """Parent loop with 2 independent inner loops -> 2 parent loops, each
    wrapping one inner loop (perfect-loop-nesting for loops)."""
    n = 5
    base = loop_nested_two.to_sdfg(simplify=True)
    x0, y0 = np.zeros((n, n)), np.zeros((n, n))
    copy.deepcopy(base)(x=x0, y=y0, N=n)

    sdfg = loop_nested_two.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    top = [c for c in sdfg.nodes() if isinstance(c, LoopRegion)]
    assert len(top) == 2, f"expected 2 parent loops, got {len(top)}"
    assert _loop_count(sdfg) == 4  # 2 parent + 2 inner

    x1, y1 = np.zeros((n, n)), np.zeros((n, n))
    sdfg(x=x1, y=y1, N=n)
    assert np.allclose(x1, x0) and np.allclose(y1, y0)
    assert np.allclose(x1, 1.0) and np.allclose(y1, 2.0)


def test_loop_fission_nested_dependent_inner_loops_not_split():
    """Loop-in-loop where inner loop 2 reads what inner loop 1 writes
    (RAW across blocks): perfect-nesting fission must NOT distribute them."""
    n = 6
    A = np.random.rand(n, n)
    sdfg = loop_nested_dependent.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None  # dependent -> kept together
    sdfg.validate()
    assert sum(1 for c in sdfg.nodes() if isinstance(c, LoopRegion)) == 1, \
        "dependent inner loops were illegally distributed"
    B, C = np.zeros((n, n)), np.zeros((n, n))
    sdfg(A=A.copy(), B=B, C=C, N=n)
    assert np.allclose(B, A + 1.0) and np.allclose(C, (A + 1.0) * 2.0)


def test_loop_fission_nested_cycle_not_split():
    """Two inner loops with a cross/loop-carried cycle (loop1 writes A reads
    B; loop2 writes B reads A): fission is illegal -> kept together, valid."""
    n = 5
    base = loop_nested_cycle.to_sdfg(simplify=True)
    A0 = np.random.rand(n, n)
    B0 = np.random.rand(n, n)
    refA, refB = A0.copy(), B0.copy()
    copy.deepcopy(base)(A=refA, B=refB, N=n)

    sdfg = loop_nested_cycle.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None
    sdfg.validate()
    A1, B1 = A0.copy(), B0.copy()
    sdfg(A=A1, B=B1, N=n)
    assert np.allclose(A1, refA) and np.allclose(B1, refB)


def _assert_noop_numeric(prog, args, n):
    """LoopFission must be a no-op (illegal to distribute) and value-preserving.

    :param prog: The dace program.
    :param args: Keyword arrays.
    :param n: Value bound to ``N``.
    """
    base = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = prog.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is None, "illegal fission was applied"
    sdfg.validate()
    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


@pytest.mark.parametrize(
    "prog,arrs",
    [
        (tsvc_s211, "abcde"),
        (tsvc_s1213, "abcd"),
        # ``tsvc_s221`` previously listed here under the old "carried-dep cannot
        # split" contract -- its multi-statement body is now fissioned per the
        # per-iter-shared-container relaxation; see test_loop_fission_tsvc_s221_*.
        (tsvc_s111, "ab"),
    ])
def test_loop_fission_tsvc_carried_dep_not_split(prog, arrs):
    """TSVC s21*/s22* kernels with a genuine cross-statement (or single-
    statement recurrence) dependence: distributing is illegal -- LoopFission
    must refuse and stay numerically identical to the un-fissioned loop."""
    n = 32
    args = {k: np.random.rand(n) for k in arrs}
    _assert_noop_numeric(prog, args, n)


def test_loop_fission_tsvc_s222_correct_split():
    """TSVC s222 distributes legally into the ``a``-group loop and the ``e``
    recurrence loop. Rigorous oracle: a controlled *growing* ``e`` seed
    (e[i]=e[i-1]^2 stays distinct & order-sensitive, so a wrong split or a
    reordered recurrence is detected), an independent analytic reference for
    ``e``, the net-zero invariant on ``a``, and an explicit check that the
    two resulting loops partition the data (one touches only ``e``, the
    other only ``a/b/c``)."""
    n = 8
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.random.rand(n)
    e = np.zeros(n)
    e[0] = 1.1  # >1 -> recurrence grows, all e[i] distinct (no underflow)

    base = tsvc_s222.to_sdfg(simplify=True)
    ref = dict(a=a.copy(), b=b.copy(), c=c.copy(), e=e.copy())
    copy.deepcopy(base)(**ref, N=n)

    sdfg = tsvc_s222.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    loops = [cfg for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion)]
    # Under the per-iter-shared relaxation, statements 1 and 3 (both write
    # ``a`` at the same iteration index) also fission apart; the loop count is
    # ``3`` (stmt 1 a-add, stmt 2 e-recurrence, stmt 3 a-sub) rather than the
    # earlier "2 (a-group + e)". Either count is semantically equivalent
    # (stmts 1 and 3 cancel net on ``a``); the maximally-fissioned shape is the
    # one downstream lifts care about.
    assert 2 <= len(loops) <= 3, f"expected 2 or 3 loops, got {len(loops)}"

    # Partition check: exactly one loop's body is the ``e`` recurrence; the
    # remaining loops touch only a/b/c.
    def touched(loop):
        return {nd.data for st in loop.all_states() for nd in st.nodes() if isinstance(nd, nodes.AccessNode)}

    e_loops = [L for L in loops if 'e' in touched(L)]
    a_loops = [L for L in loops if 'a' in touched(L)]
    assert len(e_loops) == 1, "expected one e-recurrence loop"
    assert len(a_loops) >= 1 and not any(L is e_loops[0] for L in a_loops), \
        "a-group loop overlapped the e-recurrence loop"
    assert 'a' not in touched(e_loops[0]) and 'b' not in touched(e_loops[0]), "e-loop touched a/b"
    for L in a_loops:
        assert 'e' not in touched(L), "a-loop touched the e recurrence"

    out = dict(a=a.copy(), b=b.copy(), c=c.copy(), e=e.copy())
    sdfg(**out, N=n)

    # Independent analytic reference for the recurrence.
    e_exp = e.copy()
    for i in range(1, n):
        e_exp[i] = e_exp[i - 1] * e_exp[i - 1]
    assert np.allclose(out['e'], e_exp), f"recurrence wrong after split: {out['e']} vs {e_exp}"
    assert np.allclose(out['a'], a), "a must be unchanged (a+=bc; a-=bc nets to zero)"
    for k in ('a', 'b', 'c', 'e'):
        assert np.allclose(out[k], ref[k]), f"split not value-preserving on {k}"


def test_loop_fission_tsvc_s221_splits_per_iter_a_chain():
    """TSVC s221: ``a[i] = a[i] + c[i]*d[i]; b[i] = b[i-1] + a[i] + d[i]``.
    Stmt 2 reads stmt 1's ``a[i]`` at the same iteration index -- per-iter,
    safe to sequence as separate loops. The recurrence on ``b`` stays inside
    its own loop so downstream LoopToScan / BreakAntiDependence can lift it.
    Numerically identical to the un-fissioned form."""
    n = 12
    rng = np.random.default_rng(221)
    args = dict(a=rng.standard_normal(n), b=rng.standard_normal(n), c=rng.standard_normal(n), d=rng.standard_normal(n))

    base = tsvc_s221.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = tsvc_s221.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    loops = [cfg for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, LoopRegion)]
    assert len(loops) == 2, f"expected 2 loops, got {len(loops)}"

    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


def test_loop_fission_recurrence_plus_independent_splits():
    """A loop-carried recurrence and a data-independent statement: the legal
    distribution keeps the recurrence intact in its own loop and splits off
    the independent one (2 loops), numerically identical."""
    n = 24
    args = dict(a=np.random.rand(n), b=np.random.rand(n), x=np.zeros(n), y=np.random.rand(n))
    base = tsvc_recurrence_plus_independent.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = tsvc_recurrence_plus_independent.to_sdfg(simplify=True)
    assert LoopFission().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert _loop_count(sdfg) == 2, f"expected 2 loops, got {_loop_count(sdfg)}"
    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


def test_loop_fission_conditional_body_kept():
    """A conditional body is not a single state: conservative no-op, valid."""
    n = 7
    a = np.random.rand(n)
    sdfg = loop_conditional.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    A, B = np.zeros(n), np.zeros(n)
    sdfg(a=a.copy(), A=A, B=B, c=np.array([1], np.int32), N=n)
    assert np.allclose(A, a + 1.0) and np.allclose(B, a * 2.0)
    A0, B0 = np.full(n, 9.0), np.full(n, 9.0)
    sdfg(a=a.copy(), A=A0, B=B0, c=np.array([0], np.int32), N=n)
    assert np.allclose(A0, 9.0) and np.allclose(B0, 9.0)


# -----------------------------------------------------------------------------
# Additional Python-frontend tests covering gaps in the existing coverage.
# -----------------------------------------------------------------------------


@dace.program
def loop_same_array_two_writes(A: dace.float64[N], x: dace.float64[N], y: dace.float64[N]):
    """Two writes to the same array at the same per-iter index. The second
    overwrite cancels the first; both writes touch ``A`` so fission must keep
    them in one loop (their combined effect is what's value-preserving)."""
    for i in range(N):
        A[i] = x[i]
        A[i] = y[i]


def test_loop_fission_same_array_two_writes_kept_together():
    """Two writes to ``A[i]`` in the same body: fission must KEEP them in one
    loop because splitting would reorder the per-iter overwrite semantics.
    Numerics: ``A`` must equal ``y`` (the second write wins per-iter)."""
    n = 16
    rng = np.random.default_rng(0)
    args = dict(A=np.zeros(n), x=rng.standard_normal(n), y=rng.standard_normal(n))
    base = loop_same_array_two_writes.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = loop_same_array_two_writes.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    # The two writes share the same container; fission's write-merge keeps them
    # in a single loop (no sibling split).
    assert _loop_count(sdfg) == 1, f"expected 1 loop, got {_loop_count(sdfg)}"
    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    assert np.allclose(out['A'], args['y']), "second write must win per-iter"
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


@dace.program
def loop_per_iter_rmw(A: dace.float64[N], x: dace.float64[N]):
    """``A[i] = x[i]; A[i] = A[i] + 1.0`` -- write A[i] then read it back at the
    same iter index. The chain is RAW at the SAME index, which the per-iter
    relaxation MIGHT split, but the second statement still requires the
    first's write to have happened. With sequential post-fission ordering the
    sibling loops execute in turn, so the second loop sees A from the first."""
    for i in range(N):
        A[i] = x[i]
        A[i] = A[i] + 1.0


def test_loop_fission_per_iter_rmw_kept_together():
    """``A[i] = x[i]; A[i] = A[i] + 1.0`` -- both writes target ``A`` so the
    write-merge keeps them in one loop. Either way numerics must match
    (``A == x + 1.0``)."""
    n = 12
    rng = np.random.default_rng(42)
    args = dict(A=np.zeros(n), x=rng.standard_normal(n))
    base = loop_per_iter_rmw.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = loop_per_iter_rmw.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    assert np.allclose(out['A'], args['x'] + 1.0), "A must equal x + 1.0"
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


@dace.program
def loop_three_independent_writes(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], x: dace.float64[N],
                                  y: dace.float64[N], z: dace.float64[N]):
    """Three completely independent per-iter writes to three different arrays.
    Each touches its own container with no cross-array reads -- the maximally-
    fissioned shape is 3 sibling loops."""
    for i in range(N):
        A[i] = x[i] * 2.0
        B[i] = y[i] + 3.0
        C[i] = z[i] - 1.0


def test_loop_fission_three_independent_writes():
    """3 distinct arrays written independently from 3 distinct inputs -- max
    fission produces 3 sibling loops."""
    n = 20
    rng = np.random.default_rng(7)
    args = dict(A=np.zeros(n),
                B=np.zeros(n),
                C=np.zeros(n),
                x=rng.standard_normal(n),
                y=rng.standard_normal(n),
                z=rng.standard_normal(n))
    _run(loop_three_independent_writes, args, dict(N=n), 3)


@dace.program
def loop_mixed_partition(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], a: dace.float64[N],
                         b: dace.float64[N]):
    """One sibling is data-dependent (``A → B`` per-iter chain), the other is
    independent (``C`` from ``b``). Fission partitions into the A+B pair and
    the C-only sibling: 2 sibling loops."""
    for i in range(N):
        A[i] = a[i] + 1.0
        B[i] = A[i] * 2.0
        C[i] = b[i] - 0.5


def test_loop_fission_mixed_dep_and_independent_partitions():
    """Per-iter A→B chain AND independent C all split under the per-iter-shared
    relaxation -- max fission produces 3 sibling loops, executed in original
    order so A's loop completes before B reads it (the per-iter-bridge rewrite
    severs the A→B connection, mirroring
    ``test_loop_fission_dependent_kept_together_modified_splits_per_iter``)."""
    n = 15
    rng = np.random.default_rng(99)
    args = dict(A=np.zeros(n), B=np.zeros(n), C=np.zeros(n), a=rng.standard_normal(n), b=rng.standard_normal(n))
    _run(loop_mixed_partition, args, dict(N=n), 3)


@dace.program
def loop_reverse_step(A: dace.float64[N], B: dace.float64[N], a: dace.float64[N]):
    """Backward iteration ``range(N-1, -1, -1)`` with two independent per-iter
    writes. Loop direction shouldn't block fission -- both writes still touch
    distinct containers with the same iteration variable."""
    for i in range(N - 1, -1, -1):
        A[i] = a[i] * 3.0
        B[i] = a[i] - 7.0


def test_loop_fission_reverse_iteration_splits():
    """A reverse loop with 2 independent statements fissions just like a
    forward loop. Numerically identical to the un-fissioned form."""
    n = 18
    rng = np.random.default_rng(123)
    args = dict(A=np.zeros(n), B=np.zeros(n), a=rng.standard_normal(n))
    base = loop_reverse_step.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = loop_reverse_step.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    # Backward iteration is normalized by NormalizeNegativeStride to forward;
    # downstream we still expect 2 sibling loops (or 1 if the normalization
    # didn't run -- either way values match).
    assert _loop_count(sdfg) in (1, 2), f"expected 1 or 2 loops, got {_loop_count(sdfg)}"
    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


@dace.program
def loop_two_indep_with_one_recurrence(out_a: dace.float64[N], out_b: dace.float64[N], x: dace.float64[N],
                                       y: dace.float64[N]):
    """One pointwise sibling and one carried recurrence on a different array.
    Fission must split them: pointwise becomes parallelizable, recurrence
    stays sequential. (Mirrors TSVC s221 shape with a different access pattern.)"""
    for i in range(1, N):
        out_a[i] = x[i] * y[i] + 1.0  # pointwise
        out_b[i] = out_b[i - 1] + y[i]  # carried recurrence on out_b


def test_loop_fission_pointwise_and_carried_recurrence_split():
    """Pointwise out_a-loop fissions away from the carried out_b recurrence
    loop. Final loop count is 2."""
    n = 24
    rng = np.random.default_rng(221)
    args = dict(out_a=np.zeros(n), out_b=rng.standard_normal(n), x=rng.standard_normal(n), y=rng.standard_normal(n))
    _run(loop_two_indep_with_one_recurrence, args, dict(N=n), 2)


@dace.program
def loop_eight_independent(s0: dace.float64[N], s1: dace.float64[N], s2: dace.float64[N], s3: dace.float64[N],
                           s4: dace.float64[N], s5: dace.float64[N], s6: dace.float64[N], s7: dace.float64[N]):
    """Eight independent fills with constant values -- stresses the maximal-
    fission path on a wider body than ``loop_five_set_five_cpy``."""
    for i in range(N):
        s0[i] = 0.0
        s1[i] = 1.0
        s2[i] = 2.0
        s3[i] = 3.0
        s4[i] = 4.0
        s5[i] = 5.0
        s6[i] = 6.0
        s7[i] = 7.0


def test_loop_fission_eight_independent_writes_max_fission():
    """8 independent writes -> 8 sibling loops, each writing one array. Verify
    the post-fission shape AND that each surviving loop touches exactly one
    container (no leftover grouping)."""
    n = 9
    args = {f's{k}': np.full(n, -1.0) for k in range(8)}
    base = loop_eight_independent.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = loop_eight_independent.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    assert _loop_count(sdfg) == 8, f"expected 8 sibling loops, got {_loop_count(sdfg)}"

    # Each sibling loop's body must write exactly one of the 8 arrays.
    loops = [c for c in sdfg.nodes() if isinstance(c, LoopRegion)]
    sibling_writes = []
    for loop in loops:
        wset = set()
        for st in loop.all_states():
            for nd in st.nodes():
                if isinstance(nd, nodes.AccessNode) and st.in_degree(nd) > 0:
                    wset.add(nd.data)
        sibling_writes.append(wset)
    assert all(len(w) == 1
               for w in sibling_writes), (f"each sibling loop must write exactly one array; got {sibling_writes}")
    written_arrays = set().union(*sibling_writes)
    assert written_arrays == {f's{k}' for k in range(8)}, "all 8 arrays must be partitioned"

    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k_idx in range(8):
        assert np.allclose(out[f's{k_idx}'], float(k_idx)), f"s{k_idx} must equal {float(k_idx)}"
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


@dace.program
def loop_cross_shared_array_kept_together(A: dace.float64[N], x: dace.float64[N], y: dace.float64[N]):
    """Both statements read AND write the same non-transient array ``A`` --
    sibling 1 reads/writes A[i], sibling 2 also reads/writes A[i] -- but at
    DIFFERENT subsets (statement 1 writes A[i] from x[i], statement 2 reads
    A[i-1] for a recurrence-like access). The cross-iter ``A[i-1]`` access
    breaks per-iter independence -- fission must keep them together."""
    for i in range(1, N):
        A[i] = x[i] + 2.0
        A[i] = A[i - 1] * y[i]  # depends on previous-iter's write to A


def test_loop_fission_cross_iter_same_array_kept_together():
    """Statement 2 reads ``A[i-1]`` -- cross-iter dependency on ``A``. Fission
    must NOT split (would race) -- kept as a single loop."""
    n = 10
    rng = np.random.default_rng(1)
    A = rng.standard_normal(n)
    args = dict(A=A.copy(), x=rng.standard_normal(n), y=rng.standard_normal(n))
    base = loop_cross_shared_array_kept_together.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in args.items()}
    copy.deepcopy(base)(**ref, N=n)

    sdfg = loop_cross_shared_array_kept_together.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()
    assert _loop_count(sdfg) == 1, (f"cross-iter A[i-1] dep must keep loop intact; got {_loop_count(sdfg)} loops")
    out = {k: v.copy() for k, v in args.items()}
    sdfg(**out, N=n)
    for k in args:
        assert np.allclose(out[k], ref[k]), f"mismatch on {k}"


def test_loop_fission_refuses_constant_slot_intra_iter_carry():
    """Regression for TSVC s257 ``a[i] = aa[j,i] - a[i-1]; aa[j,i] = a[i] + bb[j,i]``
    fissioned over the inner ``j`` loop.

    The two statements share ``a[i]`` -- stmt1 writes, stmt2 reads, both at
    the constant slot ``i`` (the OUTER loop variable, not ``j``). Within
    ONE ``j``-iteration the read sees the just-written value. Fissioning
    into two separate ``j``-loops would let stmt1 overwrite ``a[i]`` ``N``
    times and stmt2 then read the LAST value for every iteration --
    breaking the per-iteration interleave.

    ``_is_per_iter_subset`` previously returned ``True`` for a subset that
    never referenced the loop variable (treating any constant slot as
    "safely sequenceable"). Fix: a per-iter subset must actually USE the
    loop variable; constant slots stay grouped with their producers.
    """
    from dace.transformation.passes.loop_fission import LoopFission

    LEN_2D = dace.symbol("LEN_2D")

    @dace.program
    def s257(a: dace.float64[LEN_2D], aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
        for i in range(8, LEN_2D):
            for j in range(LEN_2D):
                a[i] = aa[j, i] - a[i - 1]
                aa[j, i] = a[i] + bb[j, i]

    sdfg = s257.to_sdfg(simplify=True)
    LoopFission().apply_pass(sdfg, {})
    sdfg.validate()

    n = 12
    rng = np.random.default_rng(257)
    a = rng.random(n)
    aa = rng.random((n, n))
    bb = rng.random((n, n))
    ra = a.copy()
    raa = aa.copy()
    for i in range(8, n):
        for j in range(n):
            ra[i] = raa[j, i] - ra[i - 1]
            raa[j, i] = ra[i] + bb[j, i]

    sa, saa = a.copy(), aa.copy()
    sdfg(a=sa, aa=saa, bb=bb.copy(), LEN_2D=n)
    assert np.allclose(sa, ra), 'a diverges -- LoopFission split the intra-iter constant-slot carry'
    assert np.allclose(saa, raa), 'aa diverges -- LoopFission split the intra-iter constant-slot carry'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
