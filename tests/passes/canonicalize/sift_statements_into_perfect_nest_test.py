# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`SiftStatementsIntoPerfectNest`: sink outer-level statements into the
inner loop under boundary guards to make an imperfect nest perfect (GPU-only).

Each test builds a small SDFG (Python frontend or hand-built LoopRegions), applies the pass
with ``target='gpu'``, and asserts both the resulting CFG structure and bit-exact numerics
against a deep copy of the pre-sift SDFG. Refusal tests assert the pass is a no-op and that
the (unchanged) SDFG still matches its sequential oracle.
"""
import copy
import os

# dace lazily ``from mpi4py import MPI`` during ``to_sdfg``; steer Open MPI off UCX before
# that import so MPI_Init does not stall. ``setdefault`` defers to external configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize.sift_statements_into_perfect_nest import SiftStatementsIntoPerfectNest

N = dace.symbol('N')
M = dace.symbol('M')


def _loops(sdfg):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def _conds(sdfg):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, ConditionalBlock)]


def _inner_loop(sdfg, var='j'):
    return next(l for l in _loops(sdfg) if l.loop_variable == var)


def _outer_loop(sdfg, var='i'):
    return next(l for l in _loops(sdfg) if l.loop_variable == var)


def _cond_string(cb: ConditionalBlock) -> str:
    return cb.branches[0][0].as_string


def _region_writes(region, name: str) -> bool:
    """True iff ``name`` is written by any state inside ``region`` (a ConditionalBlock branch)."""
    for st in region.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == name and st.in_degree(n) > 0:
                return True
    return False


# ---------------------------------------------------------------------------------------------
# Frontend kernels (concrete inner bound => inner loop provably non-empty, so S1 passes).
# ---------------------------------------------------------------------------------------------
@dace.program
def _pre_only(a: dace.float64[N, 5], b: dace.float64[N, 5], s: dace.float64[N]):
    for i in range(N):
        s[i] = a[i, 0]
        for j in range(5):
            b[i, j] = a[i, j] + s[i]


@dace.program
def _post_only(a: dace.float64[N, 5], b: dace.float64[N, 5], c: dace.float64[N]):
    # Post block reads the inner loop's last-column output; element-wise (no reduction) so the
    # result is bit-exact and deterministic regardless of how the compiler vectorizes the body.
    for i in range(N):
        for j in range(5):
            b[i, j] = a[i, j] + 1.0
        c[i] = b[i, 4] * 2.0


@dace.program
def _pre_and_post(a: dace.float64[N, 5], c: dace.float64[N]):
    for i in range(N):
        acc = 0.0
        for j in range(5):
            acc += a[i, j]
        c[i] = acc


@dace.program
def _maybe_empty(a: dace.float64[N, M], c: dace.float64[N]):
    # M is a free symbol: range(M) is NOT provably non-empty, so the pass must refuse (S1).
    for i in range(N):
        acc = 0.0
        for j in range(M):
            acc += a[i, j]
        c[i] = acc


@dace.program
def _outer_carry(a: dace.float64[N, 5], y: dace.float64[N], z: dace.float64[N, 5]):
    # y written at i (post) and read at i-1 (pre): an outer-axis loop-carried dependence (S7).
    for i in range(1, N):
        pv = y[i - 1]
        for j in range(5):
            z[i, j] = a[i, j] + pv
        y[i] = z[i, 4]


def _build_interstate_sdfg() -> dace.SDFG:
    """Hand-built imperfect nest whose pre state is fed by an interstate-edge assignment.

    ``for i: { pre_head --(t = 2*i)--> [s[i] = a[i,0] + t] ; for j in 5: b[i,j] = a[i,j] + s[i] }``.
    The ``t = 2*i`` assignment lives on the internal pre edge; the sift must carry it into the
    ``j == 0`` guard together with the state that reads it.
    """
    sdfg = dace.SDFG('istate_kern')
    sdfg.add_array('a', [N, 5], dace.float64)
    sdfg.add_array('b', [N, 5], dace.float64)
    sdfg.add_array('s', [N], dace.float64, transient=True)
    sdfg.add_symbol('t', dace.float64)

    outer = LoopRegion('i_loop', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(outer, is_start_block=True)

    pre_head = outer.add_state('pre_head', is_start_block=True)
    pre_state = outer.add_state('pre_state')
    ar = pre_state.add_read('a')
    sw = pre_state.add_write('s')
    ptk = pre_state.add_tasklet('pre', {'inp'}, {'out'}, 'out = inp + t')
    pre_state.add_edge(ar, None, ptk, 'inp', dace.Memlet('a[i, 0]'))
    pre_state.add_edge(ptk, 'out', sw, None, dace.Memlet('s[i]'))

    inner = LoopRegion('j_loop', 'j < 5', 'j', 'j = 0', 'j = j + 1')
    body = inner.add_state('body', is_start_block=True)
    ba = body.add_read('a')
    bs = body.add_read('s')
    bw = body.add_write('b')
    btk = body.add_tasklet('body', {'ina', 'ins'}, {'out'}, 'out = ina + ins')
    body.add_edge(ba, None, btk, 'ina', dace.Memlet('a[i, j]'))
    body.add_edge(bs, None, btk, 'ins', dace.Memlet('s[i]'))
    body.add_edge(btk, 'out', bw, None, dace.Memlet('b[i, j]'))

    outer.add_node(inner)
    outer.add_edge(pre_head, pre_state, dace.InterstateEdge(assignments={'t': '2*i'}))
    outer.add_edge(pre_state, inner, dace.InterstateEdge())
    return sdfg


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
def test_pre_only_sift():
    """``for i: { s[i]=a[i,0]; for j: b=a+s }`` -> the pre state moves into a ``j == 0`` guard
    that is the first inner block; outer body left with one child; bit-exact."""
    n = 7
    a = np.random.rand(n, 5)
    base = _pre_only.to_sdfg(simplify=True)
    ref_b, ref_s = np.zeros((n, 5)), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), b=ref_b, s=ref_s, N=n)

    sdfg = _pre_only.to_sdfg(simplify=True)
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) == 1
    sdfg.validate()

    outer, inner = _outer_loop(sdfg), _inner_loop(sdfg)
    assert list(outer.nodes()) == [inner]  # exactly one child
    assert isinstance(inner.start_block, ConditionalBlock)  # pre-guard is the first inner block
    assert _cond_string(inner.start_block) == '(j == 0)'
    assert _region_writes(inner.start_block.branches[0][1], 's')  # the s[i]=... statement sifted in
    assert len(_conds(sdfg)) == 1

    out_b, out_s = np.zeros((n, 5)), np.zeros(n)
    sdfg(a=a.copy(), b=out_b, s=out_s, N=n)
    assert np.array_equal(out_b, ref_b)
    assert np.array_equal(out_s, ref_s)


def test_post_only_sift():
    """``for i: { for j: b=a+1; c[i]=b[i,4]*2 }`` -> the post state moves into a ``j == 4`` guard
    that is the last inner block; bit-exact vs the pre-sift SDFG."""
    n = 7
    a = np.random.rand(n, 5)
    base = _post_only.to_sdfg(simplify=True)
    ref_b, ref_c = np.zeros((n, 5)), np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), b=ref_b, c=ref_c, N=n)

    sdfg = _post_only.to_sdfg(simplify=True)
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) == 1
    sdfg.validate()

    inner = _inner_loop(sdfg)
    sinks = inner.sink_nodes()
    assert len(sinks) == 1 and isinstance(sinks[0], ConditionalBlock)  # post-guard is the last block
    assert _cond_string(sinks[0]) == '(j == 4)'
    assert _region_writes(sinks[0].branches[0][1], 'c')
    assert not isinstance(inner.start_block, ConditionalBlock)  # no pre-guard
    assert len(_conds(sdfg)) == 1

    out_b, out_c = np.zeros((n, 5)), np.zeros(n)
    sdfg(a=a.copy(), b=out_b, c=out_c, N=n)
    assert np.array_equal(out_c, ref_c)
    assert np.array_equal(out_b, ref_b)


def test_pre_and_post_sift():
    """``for i: { acc=0; for j: acc+=a; c[i]=acc }`` -> two guards: ``j == 0`` first (reset) and
    ``j == 4`` last (store); bit-exact reduction."""
    n = 8
    a = np.random.rand(n, 5)

    sdfg = _pre_and_post.to_sdfg(simplify=True)
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) == 1
    sdfg.validate()

    inner = _inner_loop(sdfg)
    assert isinstance(inner.start_block, ConditionalBlock) and _cond_string(inner.start_block) == '(j == 0)'
    sinks = inner.sink_nodes()
    assert len(sinks) == 1 and isinstance(sinks[0], ConditionalBlock) and _cond_string(sinks[0]) == '(j == 4)'
    assert len(_conds(sdfg)) == 2

    out = np.zeros(n)
    sdfg(a=a.copy(), c=out, N=n)
    # Oracle: a per-row sequential reduction (numpy ground truth). The sifted inner loop cannot
    # vectorize (the j==0 / j==4 guards break the reduction), so its scalar accumulation is
    # bit-exact with this order. A deep copy of the pre-sift SDFG is NOT a valid oracle here: the
    # C compiler auto-vectorizes its clean reduction loop, reassociating the sum by ~1 ULP.
    exp = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(5):
            acc += a[i, j]
        exp[i] = acc
    assert np.array_equal(out, exp)


def test_interstate_assignment_sifts_with_statement():
    """The interstate assignment ``t = 2*i`` feeding the pre state must sift down WITH the state
    into the ``j == 0`` guard; nothing is left assigning ``t`` in the outer body; bit-exact."""
    n = 6
    a = np.random.rand(n, 5)
    base = _build_interstate_sdfg()
    base.validate()
    ref = np.zeros((n, 5))
    copy.deepcopy(base)(a=a.copy(), b=ref, N=n)

    sdfg = _build_interstate_sdfg()
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) == 1
    sdfg.validate()

    # No stray interstate assignment left in the outer body.
    outer = _outer_loop(sdfg)
    assert all(not e.data.assignments for e in outer.edges())

    # Both the assignment and the writing state live inside the j == 0 guard.
    guard = _inner_loop(sdfg).start_block
    assert isinstance(guard, ConditionalBlock) and _cond_string(guard) == '(j == 0)'
    region = guard.branches[0][1]
    assert any('t' in e.data.assignments for e in region.edges())
    assert _region_writes(region, 's')

    out = np.zeros((n, 5))
    sdfg(a=a.copy(), b=out, N=n)
    assert np.array_equal(out, ref)
    # dace evaluates s[i] = a[i,0] + t (t = 2*i) then b[i,j] = a[i,j] + s[i]; match that association.
    assert np.array_equal(out, a + (a[:, 0:1] + 2.0 * np.arange(n).reshape(-1, 1)))


def test_refuses_possibly_empty_inner_loop():
    """A free-symbol inner bound ``range(M)`` is not provably >= 1: sifting the post store would
    drop it when M == 0, so the pass refuses (S1). Verified correct at M == 0."""
    sdfg = _maybe_empty.to_sdfg(simplify=True)
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) is None
    sdfg.validate()

    c = np.full(4, 9.0)
    sdfg(a=np.zeros((4, 0)), c=c, N=4, M=0)
    assert np.array_equal(c, np.zeros(4))  # post store ran even though the inner loop did not


def test_cpu_target_noop():
    """The pass is GPU-only: with ``target='cpu'`` it returns None and changes nothing."""
    n = 6
    a = np.random.rand(n, 5)
    base = _pre_and_post.to_sdfg(simplify=True)
    ref = np.zeros(n)
    copy.deepcopy(base)(a=a.copy(), c=ref, N=n)

    sdfg = _pre_and_post.to_sdfg(simplify=True)
    conds_before = len(_conds(sdfg))
    assert SiftStatementsIntoPerfectNest(target='cpu').apply_pass(sdfg, {}) is None
    sdfg.validate()
    assert len(_conds(sdfg)) == conds_before  # structurally unchanged

    out = np.zeros(n)
    sdfg(a=a.copy(), c=out, N=n)
    assert np.array_equal(out, ref)


def test_refuses_outer_axis_carry():
    """An outer-i loop-carried dependence (pre reads row i-1 that post wrote at i-1) is refused
    (S7): interchange after perfect nesting would break it. No-op, bit-exact vs sequential oracle."""
    n = 6
    a = np.random.rand(n, 5)
    y0 = np.random.rand(n)
    z0 = np.zeros((n, 5))

    sdfg = _outer_carry.to_sdfg(simplify=True)
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) is None
    sdfg.validate()

    out_y, out_z = y0.copy(), z0.copy()
    sdfg(a=a.copy(), y=out_y, z=out_z, N=n)

    # Sequential oracle honouring the i -> i-1 carry.
    exp_y, exp_z = y0.copy(), z0.copy()
    for i in range(1, n):
        pv = exp_y[i - 1]
        for j in range(5):
            exp_z[i, j] = a[i, j] + pv
        exp_y[i] = exp_z[i, 4]
    assert np.array_equal(out_y, exp_y)
    assert np.array_equal(out_z, exp_z)


def test_new_start_block_set_correctly():
    """After a pre-sift the inner loop's start block must BE the pre-guard ConditionalBlock, so
    dominator-based analysis stays correct: simplify() runs without KeyError and stays bit-exact."""
    n = 6
    a = np.random.rand(n, 5)

    sdfg = _pre_only.to_sdfg(simplify=True)
    assert SiftStatementsIntoPerfectNest(target='gpu').apply_pass(sdfg, {}) == 1

    inner = _inner_loop(sdfg)
    start = inner.start_block
    assert isinstance(start, ConditionalBlock) and _cond_string(start) == '(j == 0)'

    sdfg.simplify()  # dominator / dead-state analysis: must not raise on a mis-set start block
    sdfg.validate()

    out_b, out_s = np.zeros((n, 5)), np.zeros(n)
    sdfg(a=a.copy(), b=out_b, s=out_s, N=n)
    assert np.array_equal(out_b, a + a[:, 0:1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
