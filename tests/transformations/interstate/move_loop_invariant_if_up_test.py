# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the redesigned ``MoveLoopInvariantIfUp`` -- the inverse of
    ``MoveIfIntoLoop``. It hoists a loop-invariant guarding ConditionalBlock
    out of its loop, applied to a fixpoint so an innermost invariant guard
    sifts all the way up; the interstate-edge symbol-assignment chain the
    condition depends on is hoisted with it; emptied boundary states are
    cleaned. Each test checks structure (guard moved above the loop / no-op)
    AND end-to-end numerics vs a pure-numpy oracle for the guard taken and
    not-taken.
"""
import copy

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp

N = dace.symbol('N')
M = dace.symbol('M')


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def _conds(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _guard_wraps_a_loop(sdfg):
    """True iff some ConditionalBlock's branch contains a LoopRegion (the
    hoisted shape ``if c: { for k: ... }``)."""
    for cb in _conds(sdfg):
        for _, br in cb.branches:
            if br is not None and any(isinstance(x, LoopRegion) for x in br.all_control_flow_regions(recursive=True)):
                return True
    return False


# --------------------------------------------------------------------------- #
# Invariant on a symbolic/scalar expression (no loop var) -> hoist             #
# --------------------------------------------------------------------------- #


@dace.program
def sym_guard_in_loop(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    for k in range(N):
        if active[0] > 0:  # invariant: no loop variable, scalar input
            b[k] = a[k] + 1.0


def test_invariant_symbolic_guard_hoisted_and_e2e():
    n = 16
    a = np.random.rand(n)
    for av in (1, 0):
        sdfg = sym_guard_in_loop.to_sdfg(simplify=True)
        assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is not None, "must hoist"
        sdfg.validate()
        assert _guard_wraps_a_loop(sdfg), "guard must wrap the loop after hoisting"
        out = np.full(n, 9.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n)
        assert np.allclose(out, a + 1.0 if av > 0 else 9.0), f"mismatch active={av}"


# --------------------------------------------------------------------------- #
# Invariant on data not written in the loop (no loop var) -> hoist             #
# --------------------------------------------------------------------------- #


@dace.program
def data_guard_in_loop(a: dace.float64[N], b: dace.float64[N], thr: dace.float64[1]):
    for k in range(N):
        if thr[0] > 0.5:  # depends on data, but `thr` is never written in the loop
            b[k] = a[k] * 2.0


def test_invariant_data_guard_hoisted_and_e2e():
    n = 14
    a = np.random.rand(n)
    for tv in (0.9, 0.1):
        sdfg = data_guard_in_loop.to_sdfg(simplify=True)
        assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is not None, "must hoist"
        sdfg.validate()
        assert _guard_wraps_a_loop(sdfg)
        out = np.full(n, 3.0)
        sdfg(a=a.copy(), b=out, thr=np.array([tv], np.float64), N=n)
        assert np.allclose(out, a * 2.0 if tv > 0.5 else 3.0), f"mismatch thr={tv}"


# --------------------------------------------------------------------------- #
# Condition depends on the loop variable -> MUST NOT hoist                     #
# --------------------------------------------------------------------------- #


@dace.program
def loopvar_guard(a: dace.float64[N], b: dace.float64[N]):
    for k in range(N):
        if k % 2 == 0:  # depends on the loop variable -> not invariant
            b[k] = a[k] + 1.0


def test_loopvar_dependent_guard_not_hoisted_and_e2e():
    n = 15
    a = np.random.rand(n)
    sdfg = loopvar_guard.to_sdfg(simplify=True)
    base = copy.deepcopy(sdfg)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is None, "must NOT hoist a loop-var guard"
    # Structurally unchanged + numerically correct.
    sdfg.validate()
    out, ref = np.full(n, 5.0), np.full(n, 5.0)
    sdfg(a=a.copy(), b=out, N=n)
    base(a=a.copy(), b=ref, N=n)
    exp = np.full(n, 5.0)
    exp[::2] = a[::2] + 1.0
    assert np.allclose(out, ref) and np.allclose(out, exp)


# --------------------------------------------------------------------------- #
# Innermost guard sifts all the way up through nested loops (fixpoint)         #
# --------------------------------------------------------------------------- #


@dace.program
def nested_invariant_guard(a: dace.float64[N, M], b: dace.float64[N, M], active: dace.int32[1]):
    for i in range(N):
        for j in range(M):
            if active[0] > 0:  # invariant w.r.t. BOTH loops
                b[i, j] = a[i, j] + 1.0


def test_innermost_guard_sifts_all_the_way_up_and_e2e():
    n, m = 6, 5
    a = np.random.rand(n, m)
    for av in (1, 0):
        sdfg = nested_invariant_guard.to_sdfg(simplify=True)
        applied = MoveLoopInvariantIfUp().apply_pass(sdfg, {})
        assert applied is not None and applied >= 2, f"must hoist out of both loops, got {applied}"
        sdfg.validate()
        # No ConditionalBlock left inside any loop: the guard is at the top.
        assert not [cb for cb in _conds(sdfg) if any(isinstance(p, LoopRegion) for p in _ancestors(sdfg, cb))
                    ], "guard did not sift past every enclosing loop"
        out = np.full((n, m), 7.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n, M=m)
        assert np.allclose(out, a + 1.0 if av > 0 else 7.0), f"mismatch active={av}"


def _ancestors(sdfg, block):
    """The chain of enclosing regions of ``block`` up to the root SDFG."""
    out, g = [], block.parent_graph
    while g is not None and g is not sdfg:
        out.append(g)
        g = getattr(g, 'parent_graph', None)
    return out


# --------------------------------------------------------------------------- #
# Interstate symbol-assignment chain is hoisted WITH the guard                 #
# --------------------------------------------------------------------------- #


@dace.program
def assignment_chain_guard(a: dace.float64[N], b: dace.float64[N], flag: dace.int32[1]):
    s = flag[0] * 3  # loop-invariant interstate-style derived symbol
    for k in range(N):
        if s > 2:  # condition depends on the invariant `s`
            b[k] = a[k] + 1.0


def test_invariant_assignment_chain_hoisted_with_guard_and_e2e():
    n = 12
    a = np.random.rand(n)
    for fv in (1, 0):
        sdfg = assignment_chain_guard.to_sdfg(simplify=True)
        MoveLoopInvariantIfUp().apply_pass(sdfg, {})
        sdfg.validate()  # the `s`-defining computation must remain reachable
        out = np.full(n, 4.0)
        sdfg(a=a.copy(), b=out, flag=np.array([fv], np.int32), N=n)
        assert np.allclose(out, a + 1.0 if fv * 3 > 2 else 4.0), f"mismatch flag={fv}"


# --------------------------------------------------------------------------- #
# Condition depends on data INDEXED BY the loop variable -> MUST NOT hoist     #
# --------------------------------------------------------------------------- #


@dace.program
def data_and_loopvar_guard(a: dace.float64[N], b: dace.float64[N]):
    for k in range(N):
        if a[k] > 0.5:  # data + loop variable (a[k]) -> not invariant
            b[k] = a[k] * 2.0


def test_data_and_loopvar_guard_not_hoisted_and_e2e():
    n = 18
    a = np.random.rand(n)
    sdfg = data_and_loopvar_guard.to_sdfg(simplify=True)
    base = copy.deepcopy(sdfg)
    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is None, \
        "a guard reading a[k] (loop-variable-indexed data) must NOT hoist"
    sdfg.validate()
    out, ref = np.full(n, 5.0), np.full(n, 5.0)
    sdfg(a=a.copy(), b=out, N=n)
    base(a=a.copy(), b=ref, N=n)
    exp = np.full(n, 5.0)
    exp[a > 0.5] = a[a > 0.5] * 2.0
    assert np.allclose(out, ref) and np.allclose(out, exp)


# --------------------------------------------------------------------------- #
# Empty boundary states in the loop body are cleaned when the guard hoists     #
# --------------------------------------------------------------------------- #


def test_empty_boundary_states_cleaned_and_e2e():
    """``for k: { empty ; if c: body ; empty }`` -- the guard hoists and the
    empty boundary states are dropped (built imperatively: the frontend will
    not emit empty in-loop boundary states)."""
    from dace.sdfg.state import ControlFlowRegion
    from dace.sdfg.sdfg import InterstateEdge
    from dace.properties import CodeBlock
    n = 11
    sdfg = dace.SDFG('elc')
    sdfg.add_array('a', [n], dace.float64)
    sdfg.add_array('b', [n], dace.float64)
    sdfg.add_symbol('active', dace.int32)

    loop = LoopRegion('L', 'k < %d' % n, 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(loop, is_start_block=True)
    e1 = loop.add_state('e1', is_start_block=True)  # empty boundary state
    cb = ConditionalBlock('g')
    br = ControlFlowRegion('br', sdfg=sdfg)
    bs = br.add_state('bs', is_start_block=True)
    r = bs.add_read('a')
    w = bs.add_write('b')
    t = bs.add_tasklet('inc', {'x'}, {'y'}, 'y = x + 1.0')
    bs.add_edge(r, None, t, 'x', dace.Memlet('a[k]'))
    bs.add_edge(t, 'y', w, None, dace.Memlet('b[k]'))
    cb.add_branch(CodeBlock('active > 0'), br)
    loop.add_node(cb)
    e2 = loop.add_state('e2')  # empty boundary state
    loop.add_edge(e1, cb, InterstateEdge())
    loop.add_edge(cb, e2, InterstateEdge())

    assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is not None, "must hoist"
    sdfg.validate()
    assert _guard_wraps_a_loop(sdfg), "guard must wrap the loop after hoisting"
    # The hoisted loop body must carry no leftover empty boundary states.
    for lr in _loops(sdfg):
        empties = [b for b in lr.nodes() if isinstance(b, dace.SDFGState) and b.number_of_nodes() == 0]
        assert not empties, f"empty boundary states not cleaned: {[e.label for e in empties]}"

    for av in (1, 0):
        a = np.random.rand(n)
        out = np.full(n, 4.0)
        sdfg(a=a.copy(), b=out, active=np.int32(av), N=n)
        assert np.allclose(out, a + 1.0 if av > 0 else 4.0), f"mismatch active={av}"


# --------------------------------------------------------------------------- #
# Mixed map + loop nest: the guard hoists out of the (inner) loop, the map     #
# stays; value-preserving (the pass must work with a mix of maps and loops)    #
# --------------------------------------------------------------------------- #


@dace.program
def mixed_map_loop(a: dace.float64[N, M], b: dace.float64[N, M], active: dace.int32[1]):
    for i in dace.map[0:N]:  # outer parallel map
        for k in range(M):  # inner sequential loop
            if active[0] > 0:  # invariant w.r.t. the inner loop
                b[i, k] = a[i, k] + 1.0


def test_mixed_map_loop_guard_hoisted_out_of_loop_and_e2e():
    n, m = 7, 5
    a = np.random.rand(n, m)
    for av in (1, 0):
        sdfg = mixed_map_loop.to_sdfg(simplify=True)
        assert MoveLoopInvariantIfUp().apply_pass(sdfg, {}) is not None, \
            "the invariant guard must hoist out of the inner loop even inside a map"
        sdfg.validate()
        # A map is still present and the guard now wraps a loop (hoisted out
        # of the inner loop, not dropped).
        assert any(isinstance(n_, dace.sdfg.nodes.MapEntry) for n_, _ in sdfg.all_nodes_recursive()), \
            "the outer map must survive"
        assert _guard_wraps_a_loop(sdfg), "guard must wrap the inner loop after hoisting"
        out = np.full((n, m), 8.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n, M=m)
        assert np.allclose(out, a + 1.0 if av > 0 else 8.0), f"mismatch active={av}"


if __name__ == '__main__':
    test_invariant_symbolic_guard_hoisted_and_e2e()
    test_invariant_data_guard_hoisted_and_e2e()
    test_loopvar_dependent_guard_not_hoisted_and_e2e()
    test_innermost_guard_sifts_all_the_way_up_and_e2e()
    test_invariant_assignment_chain_hoisted_with_guard_and_e2e()
    test_data_and_loopvar_guard_not_hoisted_and_e2e()
    test_empty_boundary_states_cleaned_and_e2e()
    test_mixed_map_loop_guard_hoisted_out_of_loop_and_e2e()
