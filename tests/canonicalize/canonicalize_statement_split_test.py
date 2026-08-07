# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-frontend (numpy-style) value-preservation tests for the statement /
loop-nesting scenarios the redesign targets.

The hard invariant for EVERY case is that canonicalization preserves the value:
the canonicalized SDFG must compute exactly what the un-canonicalized (direct
frontend) SDFG does. Parallelization (maps) is best-effort on top; where the
current pipeline already achieves it we also assert it, otherwise we only require
correctness (and note the aspiration for SplitStatements / PerfectLoopNesting).

Scenarios (grouped):
* Independent / dependent-but-splittable statements (A=B+C; V=2*A, etc.).
* Anti-dependence by value -- a later statement reads the ORIGINAL (pre-write)
  value of an array the loop also writes (s1244 / s1213 shapes; a recurrence
  whose neighbour read sees the un-updated element).
* Genuinely loop-carried statements that must stay sequential (prefix sum).
* Perfect vs. imperfect 2-D nests.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.break_anti_dependence import BreakAntiDependence
from dace.transformation.passes.canonicalize.split_statements import SplitStatements

N = dace.symbol('N')
M = dace.symbol('M')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable)


def _canon_vs_raw(prog, inputs, **symbols):
    """Compile the un-canonicalized SDFG (ground truth) and the canonicalized SDFG
    on identical inputs; assert they agree. Returns the canonicalized SDFG so
    callers can additionally probe structure (maps / loops)."""
    raw = prog.to_sdfg(simplify=True)
    ref = {k: v.copy() for k, v in inputs.items()}
    raw.compile()(**ref, **symbols)

    cand = prog.to_sdfg(simplify=True)
    canonicalize(cand, validate=True, peel_limit=4, break_anti_dependence=True)
    got = {k: v.copy() for k, v in inputs.items()}
    cand.compile()(**got, **symbols)

    for k in inputs:
        assert np.allclose(ref[k], got[k], equal_nan=True), f"{prog.name}: '{k}' diverged after canonicalize"
    return cand


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).random(shape)


# ===========================================================================
# Independent / dependent-but-splittable statements
# ===========================================================================
@dace.program
def _dependent_same_index(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], V: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] + C[i]
        V[i] = 2.0 * A[i]


def test_dependent_same_index_splittable():
    """V depends on A at the SAME index (RAW per-iteration) -> two independent
    parallel statements after splitting; always value-preserving."""
    n = 48
    ins = {'A': np.zeros(n), 'B': _rand(n, seed=1), 'C': _rand(n, seed=2), 'V': np.zeros(n)}
    cand = _canon_vs_raw(_dependent_same_index, ins, N=n)
    assert _nloops(cand) == 0 and _nmaps(cand) >= 1, "elementwise dependent statements should parallelize"


@dace.program
def _two_independent(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N], E: dace.float64[N],
                     F: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] + C[i]
        D[i] = E[i] * F[i]


def test_two_independent_statements():
    n = 40
    ins = {k: (_rand(n, seed=ord(k)) if k not in 'AD' else np.zeros(n)) for k in 'ABCDEF'}
    cand = _canon_vs_raw(_two_independent, ins, N=n)
    assert _nloops(cand) == 0 and _nmaps(cand) >= 1


@dace.program
def _chain_three(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], V: dace.float64[N], W: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] + C[i]
        V[i] = 2.0 * A[i]
        W[i] = V[i] + 1.0


def test_chain_three_statements():
    n = 33
    ins = {'A': np.zeros(n), 'B': _rand(n, seed=3), 'C': _rand(n, seed=4), 'V': np.zeros(n), 'W': np.zeros(n)}
    _canon_vs_raw(_chain_three, ins, N=n)


@dace.program
def _pass_through_write_read(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in range(N):
        A[i] = B[i] * 2.0
        C[i] = A[i] + A[i]  # direct read of the just-written A[i] (same index)


def test_pass_through_write_then_direct_read():
    n = 36
    ins = {'A': np.zeros(n), 'B': _rand(n, seed=5), 'C': np.zeros(n)}
    _canon_vs_raw(_pass_through_write_read, ins, N=n)


# ===========================================================================
# Anti-dependence by value: a later statement reads the ORIGINAL value of an
# array the loop also writes.
# ===========================================================================
@dace.program
def _forward_read_antidep(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N]):
    for i in range(N - 1):
        A[i] = B[i] + 1.0
        D[i] = A[i] + A[i + 1]  # A[i+1] is the ORIGINAL (not-yet-written) value


def test_forward_read_antidependence_s1244():
    """s1244 shape: D reads A[i+1] which the fused loop has NOT written yet.
    Splitting D into its own loop needs a snapshot of the original A; the hard
    requirement here is that the value is preserved however it is lowered."""
    n = 50
    ins = {'A': _rand(n, seed=6), 'B': _rand(n, seed=7), 'D': np.zeros(n)}
    _canon_vs_raw(_forward_read_antidep, ins, N=n)


@dace.program
def _cross_antidep(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in range(1, N - 1):
        A[i] = B[i - 1] + C[i]
        B[i] = A[i + 1] * D[i]  # reads original A[i+1]; overwrites B (WAR vs the A read)


def test_cross_antidependence_s1213():
    n = 44
    ins = {'A': _rand(n, seed=8), 'B': _rand(n, seed=9), 'C': _rand(n, seed=10), 'D': _rand(n, seed=11)}
    _canon_vs_raw(_cross_antidep, ins, N=n)


K = dace.symbol('K')


@dace.program
def _sym_forward_antidep(A: dace.float64[N], B: dace.float64[N], D: dace.float64[N]):
    for i in range(N - K):
        A[i] = B[i] + 1.0
        D[i] = A[i] + A[i + K]  # A[i+K] is the ORIGINAL value; K a positive symbol


def test_symbolic_forward_offset_antidependence():
    """Same shape as s1244 but the read-ahead offset is the runtime symbol ``K``
    (``a[i + K]``). Under the nonnegative-symbol assumption this is a forward-read
    anti-dependence: SplitStatements snapshots ``A`` (guarding ``K >= 0``) and the
    value must be preserved for the concrete positive ``K``."""
    n, k = 50, 3
    ins = {'A': _rand(n, seed=21), 'B': _rand(n, seed=22), 'D': np.zeros(n)}
    cand = _canon_vs_raw(_sym_forward_antidep, ins, N=n, K=k)
    assert _nmaps(cand) >= 1, "the symbolic-offset anti-dependence should still parallelize a cone"


@dace.program
def _recurrence_then_read_original(A: dace.float64[N], Bout: dace.float64[N]):
    for i in range(1, N - 1):
        A[i] = A[i - 1] * 0.5  # sequential recurrence on A
        Bout[i] = A[i + 1]  # reads the ORIGINAL A[i+1] (pre-loop value)


def test_recurrence_plus_read_of_original_value():
    """User's 'second statement only relies on the pre-loop value': the B
    statement reads A[i+1] before the recurrence overwrites it. B is separable
    (snapshot of original A) while the A recurrence stays sequential."""
    n = 40
    ins = {'A': _rand(n, seed=12), 'Bout': np.zeros(n)}
    _canon_vs_raw(_recurrence_then_read_original, ins, N=n)


# ===========================================================================
# Genuinely loop-carried: must stay a correct sequential loop (or a scan).
# ===========================================================================
@dace.program
def _prefix_sum(A: dace.float64[N], B: dace.float64[N]):
    for i in range(1, N):
        A[i] = A[i - 1] + B[i]


def test_prefix_sum_stays_correct():
    """A[i] = A[i-1] + B[i] is a genuine carried recurrence. Correctness is the
    invariant: a naive parallel map over the carried axis would give WRONG values,
    so the value-preservation check (in ``_canon_vs_raw``) is what proves it was
    not mis-parallelized. The pipeline lowers it to a Scan -- the desired
    semantic-op recognition -- which is value-exact."""
    n = 32
    ins = {'A': _rand(n, seed=13), 'B': _rand(n, seed=14)}
    _canon_vs_raw(_prefix_sum, ins, N=n)


# ===========================================================================
# Perfect vs. imperfect 2-D nests.
# ===========================================================================
@dace.program
def _perfect_2d(A: dace.float64[N, M], B: dace.float64[N, M], C: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            A[i, j] = B[i, j] + C[i, j]


def test_perfect_2d_nest_parallelizes():
    n, m = 12, 10
    ins = {'A': np.zeros((n, m)), 'B': _rand(n, m, seed=15), 'C': _rand(n, m, seed=16)}
    cand = _canon_vs_raw(_perfect_2d, ins, N=n, M=m)
    assert _nloops(cand) == 0 and _nmaps(cand) >= 1


@dace.program
def _imperfect_2d_siblings(A: dace.float64[N, M], B: dace.float64[N, M], C: dace.float64[N, M], D: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            A[i, j] = B[i, j] + C[i, j]
        for j in range(M):
            D[i, j] = A[i, j] * 2.0


def test_imperfect_2d_sibling_inner_loops():
    """Two sibling inner loops under one outer loop -- an imperfect nest that
    needs fission to perfect-nest. Correctness is the invariant regardless."""
    n, m = 10, 8
    ins = {'A': np.zeros((n, m)), 'B': _rand(n, m, seed=17), 'C': _rand(n, m, seed=18), 'D': np.zeros((n, m))}
    _canon_vs_raw(_imperfect_2d_siblings, ins, N=n, M=m)


@dace.program
def _inner_carried_2d(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i in range(N):
        for j in range(1, M):
            A[i, j] = A[i, j - 1] + B[i, j]


def test_inner_carried_outer_parallel_2d():
    """Inner axis carries a recurrence, outer axis is data-parallel: correctness
    always; ideally the outer i becomes a map with j a sequential/scan inner."""
    n, m = 12, 9
    ins = {'A': _rand(n, m, seed=19), 'B': _rand(n, m, seed=20)}
    _canon_vs_raw(_inner_carried_2d, ins, N=n, M=m)


LEN_2D = dace.symbol('LEN_2D')


@dace.program
def _forward_read_big_array(a: dace.float64[N], flat: dace.float64[LEN_2D * LEN_2D]):
    for i in range(N):
        flat[4 + i] = flat[8 + i] + a[i]


def test_forward_read_anti_dependence_snapshots_and_stays_bit_exact():
    """s422 shape: a forward-read anti-dependence (``flat[4 + i]`` written, ``flat[8 + i]``
    read-ahead). Anti-dependence is allowed by default -- the array is snapshotted before
    the loop and the read-ahead is redirected to the snapshot -- and the result stays
    value-preserving regardless of how the read window relates to the container size."""
    n, l2 = 500, 64  # flat has l2*l2 = 4096 >= n + 8, so flat[8 + i] stays in bounds
    ins = {'a': _rand(n, seed=3), 'flat': _rand(l2 * l2, seed=4)}
    cand = _canon_vs_raw(_forward_read_big_array, ins, N=n, LEN_2D=l2)
    assert any('snap' in nm for nm in cand.arrays), "forward-read anti-dependence should snapshot the array"


# ===========================================================================
# The per-edge forward-read break -- one fixture per shape. These live with
# ``BreakAntiDependence(forward_reads=True)``, which owns the rewrite; ``SplitStatements``
# distributes and breaks nothing. Hand-built single-compute-state loops -- the frontend
# leaves slice states, which yield no single compute state.
# ===========================================================================
def _mixed_loop(name):
    """A ``for i in range(N - 1)`` loop with one (empty) body state."""
    sdfg = dace.SDFG(name)
    loop = LoopRegion('loop', 'i < N - 1', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    return sdfg, loop, body


def _pre_state(sdfg, loop):
    """The pre-loop snapshot state ``SplitStatements`` inserts before ``loop`` (or None)."""
    for e in loop.parent_graph.in_edges(loop):
        if isinstance(e.src, dace.SDFGState):
            return e.src
    return None


def _split_snaps(sdfg):
    return [nm for nm in sdfg.arrays if '_split_snap' in nm]


def test_forward_reads_mixed_forward_read_redirected_to_snapshot():
    """s1244 ``d[i] = a[i] + a[i+1]``: only the read-ahead ``a[i+1]`` moves to the
    snapshot; the same-index ``a[i]`` keeps its live-array (RAW) value.

    Regression pin: the write node ``A`` carries both the write and the forward read, so a
    guard that skipped nodes with ``in_degree > 0`` would miss the anti-dependence entirely.
    """
    sdfg, loop, body = _mixed_loop('split_mixed_s1244')
    for nm in ('A', 'B', 'D'):
        sdfg.add_array(nm, [N], dace.float64)
    tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
    wA = body.add_access('A')
    body.add_edge(body.add_read('B'), None, tw, 'b', dace.Memlet('B[i]'))
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    tr = body.add_tasklet('r', {'a0', 'a1'}, {'d'}, 'd = a0 + a1')
    body.add_edge(wA, None, tr, 'a0', dace.Memlet('A[i]'))
    body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + 1]'))
    body.add_edge(tr, 'd', body.add_write('D'), None, dace.Memlet('D[i]'))

    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    snap, = _split_snaps(sdfg)
    assert sdfg.arrays[snap].transient
    pre = _pre_state(sdfg, loop)
    assert pre is not None and any(n.data == snap for n in pre.data_nodes())
    assert next(e for e in body.in_edges(tr) if e.dst_conn == 'a1').src.data == snap
    assert next(e for e in body.in_edges(tr) if e.dst_conn == 'a0').src.data == 'A'


def test_forward_reads_copy_forward_read_preserves_destination_subset():
    """``Bout[i] = a[i+1]`` is an access-node copy: redirecting its source to the snapshot
    must keep the destination subset ``Bout[i]``, or the sink is written at ``None``."""
    sdfg, loop, body = _mixed_loop('split_mixed_copy')
    for nm in ('A', 'Bout'):
        sdfg.add_array(nm, [N], dace.float64)
    rAb = body.add_read('A')
    th = body.add_tasklet('h', {'x'}, {'y'}, 'y = x * 0.5')
    wA = body.add_access('A')
    body.add_edge(rAb, None, th, 'x', dace.Memlet('A[i - 1]'))
    body.add_edge(th, 'y', wA, None, dace.Memlet('A[i]'))
    wBout = body.add_write('Bout')
    body.add_edge(wA, None, wBout, None, dace.Memlet(data='A', subset='i + 1', other_subset='i'))

    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    snap, = _split_snaps(sdfg)
    in_e = next(e for e in body.in_edges(wBout) if e.data is not None)
    assert in_e.src.data == snap
    dst = in_e.data.get_dst_subset(in_e, body)
    assert dst is not None and str(dst) in ('i', 'i:i + 1')
    # The read-behind recurrence a[i-1] is RAW and stays on the live array.
    assert next(e for e in body.out_edges(rAb) if e.data is not None).data.data == 'A'


def test_forward_reads_symbolic_forward_offset_snapshots_with_guard():
    """``d[i] = a[i] + a[i + K]`` with symbolic ``K``: snapshot AND plant the runtime
    positivity guard that makes the rename sound.

    The guard must be STRICT (``K >= 1``), not the ``>= 0`` the whole-array pure-WAR
    rename in ``BreakAntiDependence`` uses. Here a sibling statement writes ``A[i]``
    earlier in the SAME iteration, so ``K == 0`` makes ``A[i + K]`` alias that
    just-written value -- reading the stale pre-loop snapshot instead would be a silent
    miscompile, and the guard is what turns it into a loud runtime fault."""
    sdfg, loop, body = _mixed_loop('split_mixed_symK')
    for nm in ('A', 'B', 'D'):
        sdfg.add_array(nm, [N], dace.float64)
    tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
    wA = body.add_access('A')
    body.add_edge(body.add_read('B'), None, tw, 'b', dace.Memlet('B[i]'))
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    tr = body.add_tasklet('r', {'a0', 'a1'}, {'d'}, 'd = a0 + a1')
    body.add_edge(wA, None, tr, 'a0', dace.Memlet('A[i]'))
    body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + K]'))
    body.add_edge(tr, 'd', body.add_write('D'), None, dace.Memlet('D[i]'))

    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) == 1
    snap, = _split_snaps(sdfg)
    assert next(e for e in body.in_edges(tr) if e.dst_conn == 'a1').src.data == snap
    pre = _pre_state(sdfg, loop)
    guards = [n for n in pre.nodes() if isinstance(n, nodes.Tasklet) and 'guard' in n.label]
    assert len(guards) == 1
    code = guards[0].code.as_string
    # ``K - 1 >= 0`` is exactly the strict ``K > 0`` for integer offsets; a bare ``K >= 0``
    # here would admit the aliasing K == 0 case.
    assert 'K - 1' in code and '>= 0' in code, code
    assert 'std::abort' in code, code
    # The guard carries no connectors, so only ``side_effects`` keeps dead-code elimination
    # from pruning it -- pruning would silently restore the unsound assume-positive path.
    assert guards[0].side_effects
    assert not guards[0].in_connectors and not guards[0].out_connectors


def test_forward_reads_symbolic_behind_offset_is_recurrence_noop():
    """``a[i - K]`` is a read-behind RAW recurrence, not an anti-dependence -> no snapshot."""
    sdfg, _loop, body = _mixed_loop('split_mixed_symK_behind')
    sdfg.add_array('A', [N], dace.float64)
    th = body.add_tasklet('h', {'x'}, {'y'}, 'y = x * 0.5')
    body.add_edge(body.add_read('A'), None, th, 'x', dace.Memlet('A[i - K]'))
    body.add_edge(th, 'y', body.add_access('A'), None, dace.Memlet('A[i]'))
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) is None
    assert not _split_snaps(sdfg)


def test_forward_reads_same_index_only_is_noop():
    """``c[i] = a[i]`` (offset 0) reads the value a sibling just wrote this iteration;
    redirecting it to the stale snapshot would be a silent miscompile."""
    sdfg, _loop, body = _mixed_loop('split_mixed_same')
    for nm in ('A', 'B', 'C'):
        sdfg.add_array(nm, [N], dace.float64)
    tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
    wA = body.add_access('A')
    body.add_edge(body.add_read('B'), None, tw, 'b', dace.Memlet('B[i]'))
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    body.add_edge(wA, None, body.add_write('C'), None, dace.Memlet(data='A', subset='i', other_subset='i'))
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) is None
    assert not _split_snaps(sdfg)


def test_forward_reads_reverse_stride_forward_read_is_not_snapshotted():
    """Reverse stride flips what ``a[i + 1]`` means: with ``i`` decreasing it is the value
    the PREVIOUS iteration wrote (RAW), not a read-ahead. Snapshotting it silently computes
    the wrong ``D`` -- so the split must refuse, and the values must match the un-split run."""

    def build(name):
        sdfg = dace.SDFG(name)
        loop = LoopRegion('loop', 'i > 0', 'i', 'i = N - 2', 'i = i - 1')
        sdfg.add_node(loop, is_start_block=True)
        body = loop.add_state('body', is_start_block=True)
        for nm in ('A', 'B', 'D'):
            sdfg.add_array(nm, [N], dace.float64)
        tw = body.add_tasklet('w', {'b'}, {'a'}, 'a = b + 1.0')
        wA = body.add_access('A')
        body.add_edge(body.add_read('B'), None, tw, 'b', dace.Memlet('B[i]'))
        body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
        tr = body.add_tasklet('r', {'a0', 'a1'}, {'d'}, 'd = a0 + a1')
        body.add_edge(wA, None, tr, 'a0', dace.Memlet('A[i]'))
        body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + 1]'))
        body.add_edge(tr, 'd', body.add_write('D'), None, dace.Memlet('D[i]'))
        return sdfg

    n = 16
    a0, b0 = _rand(n, seed=31), _rand(n, seed=32)
    ref = {'A': a0.copy(), 'B': b0.copy(), 'D': np.zeros(n)}
    build('rev_stride_ref').compile()(**ref, N=n)

    cand_sdfg = build('rev_stride_cand')
    assert SplitStatements().apply_pass(cand_sdfg, {}) is None
    got = {'A': a0.copy(), 'B': b0.copy(), 'D': np.zeros(n)}
    cand_sdfg.compile()(**got, N=n)
    for k in ref:
        assert np.allclose(ref[k], got[k]), f"reverse-stride split diverged on '{k}'"


def test_forward_reads_transient_write_not_snapshotted():
    """Only non-transient (global) arrays are snapshot candidates."""
    sdfg, _loop, body = _mixed_loop('split_mixed_transient')
    sdfg.add_array('A', [N], dace.float64, transient=True)
    sdfg.add_array('D', [N], dace.float64)
    tw = body.add_tasklet('w', {}, {'a'}, 'a = 1.0')
    wA = body.add_access('A')
    body.add_edge(tw, 'a', wA, None, dace.Memlet('A[i]'))
    tr = body.add_tasklet('r', {'a1'}, {'d'}, 'd = a1')
    body.add_edge(wA, None, tr, 'a1', dace.Memlet('A[i + 1]'))
    body.add_edge(tr, 'd', body.add_write('D'), None, dace.Memlet('D[i]'))
    assert BreakAntiDependence(forward_reads=True).apply_pass(sdfg, {}) is None
    assert not _split_snaps(sdfg)


# ===========================================================================
# Black-box (opaque) map bodies: statement splitting clones a body once per
# output, so a node whose effect is not just its out-memlets must block the split.
# ===========================================================================
def _two_output_map(name, second_tasklet):
    """``for i: A[i] = B[i] + 1; C[i] = <second_tasklet>`` as one straight-line map."""
    sdfg = dace.SDFG(name)
    for nm in ('A', 'B', 'C'):
        sdfg.add_array(nm, [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    entry, xit = state.add_map('m', {'i': '0:N'})
    rB = state.add_read('B')
    t1 = state.add_tasklet('t1', {'b'}, {'a'}, 'a = b + 1.0')
    state.add_memlet_path(rB, entry, t1, dst_conn='b', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(t1, xit, state.add_write('A'), src_conn='a', memlet=dace.Memlet('A[i]'))
    t2 = second_tasklet(state)
    state.add_memlet_path(rB, entry, t2, dst_conn='b', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(t2, xit, state.add_write('C'), src_conn='c', memlet=dace.Memlet('C[i]'))
    return sdfg


def test_split_maps_splits_a_transparent_two_output_map():
    """Baseline for the two guards below: an all-Python, effect-free body does split."""
    sdfg = _two_output_map('split_map_plain', lambda st: st.add_tasklet('t2', {'b'}, {'c'}, 'c = b * 2.0'))
    assert SplitStatements(split_maps=True).apply_pass(sdfg, {}) == 1


def test_split_maps_refuses_a_cpp_tasklet_body():
    """A non-Python tasklet is a black box -- the Python-AST side-effect scan has nothing to
    read, so ``has_side_effects`` answering False proves nothing and the body must not be cloned."""
    sdfg = _two_output_map('split_map_cpp',
                           lambda st: st.add_tasklet('t2', {'b'}, {'c'}, 'c = b * 2.0;', dace.Language.CPP))
    assert SplitStatements(split_maps=True).apply_pass(sdfg, {}) is None


def test_split_maps_refuses_a_side_effecting_body():
    """A declared side effect would run once per clone."""
    sdfg = _two_output_map('split_map_sideeffect',
                           lambda st: st.add_tasklet('t2', {'b'}, {'c'}, 'c = b * 2.0', side_effects=True))
    assert SplitStatements(split_maps=True).apply_pass(sdfg, {}) is None


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
