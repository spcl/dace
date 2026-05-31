# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.sdfg import nodes
from dace.properties import CodeBlock
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes import SymbolPropagation, ScalarToSymbolPromotion


def _count_loops(sdfg: dace.SDFG):
    loops = 0
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, LoopRegion):
            loops += 1
    return loops


def test_loop_carried_symbol():
    """
    Tests SymbolPropagation respects loop carried dependencies.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_symbol("LB", dace.int32)
    sdfg.add_symbol("UB", dace.int32)
    sdfg.add_symbol("idx", dace.int32)
    sdfg.add_symbol("cnt", dace.int32)
    sdfg.add_symbol("a", dace.int32)
    sdfg.add_symbol("b", dace.int32)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < UB", "i", "i = LB", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"cnt": "0"}))

    s = loop.add_state(is_start_block=True)
    s1 = loop.add_state()
    s2 = loop.add_state()
    e = loop.add_state()
    loop.add_edge(s, s1, dace.InterstateEdge(assignments={"a": "i", "c": "cnt + 1"}))
    loop.add_edge(s1, s2, dace.InterstateEdge(assignments={"b": "a+1"}))
    loop.add_edge(s2, e, dace.InterstateEdge(assignments={"idx": "b - 1 - LB", "cnt": "c"}))
    task = e.add_tasklet("init", {}, {"out"}, "out = 0")
    access = e.add_access("A")
    e.add_edge(task, "out", access, None, dace.Memlet("A[idx]"))

    e = sdfg.add_state()
    sdfg.add_edge(loop, e, dace.InterstateEdge(assignments={"cnt": "cnt+1"}))
    sdfg.validate()

    # Count loops before transformation
    assert _count_loops(sdfg) == 1

    # Apply SymbolPropagation, and LoopToMap
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.validate()

    # Should not have transformed the loop if the loop-carried dependency is respected as the
    assert _count_loops(sdfg) == 1

    # Validate correctness
    A = dace.ndarray([64], dtype=dace.float32)
    A[:] = np.random.rand(64).astype(dace.float32.type)
    sdfg(A=A, LB=0, UB=64)

    assert np.allclose(A[:], 0)


def test_nested_loop_carried_symbol():
    """
    Tests SymbolPropagation respects loop carried dependencies in nested loops.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.float32)
    sdfg.add_symbol("cnt", dace.int32)

    init = sdfg.add_state("init", is_start_block=True)
    loop = LoopRegion("loop", "i < 64", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"cnt": "0"}))

    loop2 = LoopRegion("loop2", "j < 64", "j", "j = 0", "j = j + 1")
    loop.add_node(loop2)
    s1 = loop2.add_state(is_start_block=True)
    s2 = loop2.add_state()
    loop2.add_edge(s1, s2, dace.InterstateEdge(assignments={"cnt": "cnt+1"}))

    e = sdfg.add_state()
    task = e.add_tasklet("init", {}, {"out"}, "out = cnt")
    access = e.add_access("A")
    e.add_edge(task, "out", access, None, dace.Memlet("A[0]"))
    sdfg.add_edge(loop, e, dace.InterstateEdge())
    sdfg.validate()

    # Apply SymbolPropagation, and LoopToMap
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()

    # Validate correctness
    A = dace.ndarray([64], dtype=dace.float32)
    A[:] = np.random.rand(64).astype(dace.float32.type)
    sdfg(A=A)
    assert np.allclose(A[0], 64 * 64)


def test_nested_symbol():
    """
    Tests that SymbolPropagation does not overwrite nested symbols.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("v", dace.int32)

    s = sdfg.add_state(is_start_block=True)
    cond = ConditionalBlock("cond", sdfg)
    edge1 = sdfg.add_edge(s, cond, dace.InterstateEdge(assignments={"v": "0"}))
    b1 = ControlFlowRegion("b1", sdfg)
    b1s = b1.add_state()
    b1e = b1.add_state()
    edge2 = b1.add_edge(b1s, b1e, dace.InterstateEdge(assignments={"v": "5"}))
    cond.add_branch(CodeBlock("v == 0"), b1)

    b2 = ControlFlowRegion("b2", sdfg)
    b2s = b2.add_state()
    b2e = b2.add_state()
    edge3 = b2.add_edge(b2s, b2e, dace.InterstateEdge(assignments={"v": "8"}))
    cond.add_branch(CodeBlock("v == 3"), b2)

    e = sdfg.add_state()
    edge4 = sdfg.add_edge(cond, e, dace.InterstateEdge(assignments={"v": "v+1"}))
    sdfg.validate()

    # Apply SymbolPropagation
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()

    # No assignment should have been changed
    assert edge1.data.assignments["v"] == "0"
    assert edge2.data.assignments["v"] == "5"
    assert edge3.data.assignments["v"] == "8"
    assert edge4.data.assignments["v"] == "v+1"


def test_multiple_sources():
    """
    Tests that SymbolPropagation handles multiple sources correctly.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("v", dace.int32)
    sdfg.add_symbol("a", dace.int32)

    s1 = sdfg.add_state(is_start_block=True)
    s2 = sdfg.add_state()
    s3 = sdfg.add_state()
    c = sdfg.add_state()
    e = sdfg.add_state()

    edge1 = sdfg.add_edge(s1, c, dace.InterstateEdge(assignments={"v": "0"}))
    edge2 = sdfg.add_edge(s2, c, dace.InterstateEdge(assignments={"a": "5"}))
    edge3 = sdfg.add_edge(s3, c, dace.InterstateEdge(assignments={"v": "8"}))
    edge4 = sdfg.add_edge(c, e, dace.InterstateEdge(assignments={"v": "v+a"}))

    # Apply SymbolPropagation
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()

    # No assignment should have been changed
    assert edge1.data.assignments["v"] == "0"
    assert edge2.data.assignments["a"] == "5"
    assert edge3.data.assignments["v"] == "8"
    assert edge4.data.assignments["v"] == "v+a"


def test_multiple_edge_assignments():
    """
    Tests that SymbolPropagation handles multiple edge assignments using views correctly.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [64], dace.int32)
    sdfg.add_view("A_view", [1], dace.int32)
    sdfg.add_symbol("v1", dace.int32)
    sdfg.add_symbol("v2", dace.int32)

    s1 = sdfg.add_state()
    access_A = s1.add_access("A")
    access_view = s1.add_access("A_view")
    access_view.add_in_connector("views")
    s1.add_edge(access_A, None, access_view, "views", dace.Memlet("A[0]"))

    s2 = sdfg.add_state()
    s3 = sdfg.add_state()
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"v1": "A_view"}))
    sdfg.add_edge(s2, s3, dace.InterstateEdge(assignments={"v2": "A[v1]"}))

    # Apply SymbolPropagation
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.compile()


def test_deeply_nested_sdfg():
    """
    Tests that SymbolPropagation handles deeply nested SDFGs correctly.
    """
    sdfg1 = dace.SDFG("nested1")
    sdfg1.add_symbol("v", dace.int32)
    sdfg1.add_symbol("a", dace.int32)

    s11 = sdfg1.add_state(is_start_block=True)
    s12 = sdfg1.add_state()
    edge1 = sdfg1.add_edge(s11, s12, dace.InterstateEdge(assignments={"v": "a"}))

    sdfg2 = dace.SDFG("nested2")
    s12.add_node(nodes.NestedSDFG("n2", sdfg2, {}, {}, symbol_mapping={"v": "v"}))
    sdfg2.add_symbol("v", dace.int32)
    s21 = sdfg2.add_state(is_start_block=True)

    sdfg3 = dace.SDFG("nested3")
    s21.add_node(nodes.NestedSDFG("n3", sdfg3, {}, {}, symbol_mapping={"v": "v"}))
    sdfg3.add_symbol("v", dace.int32)
    s31 = sdfg3.add_state(is_start_block=True)

    sdfg4 = dace.SDFG("nested4")
    s31.add_node(nodes.NestedSDFG("n4", sdfg4, {}, {}, symbol_mapping={"v": "v"}))
    sdfg4.add_symbol("c", dace.int32)
    sdfg4.add_symbol("v", dace.int32)
    s41 = sdfg4.add_state(is_start_block=True)
    s42 = sdfg4.add_state()
    edge4 = sdfg4.add_edge(s41, s42, dace.InterstateEdge(assignments={"c": "v+1"}))
    sdfg1.validate()

    # Apply SymbolPropagation
    SymbolPropagation().apply_pass(sdfg1, {})
    sdfg1.validate()

    # The outer iedge ``v = a`` was the only binding of ``v``; with propagation reaching
    # the NSDFG ``symbol_mapping`` (``{"v": "v"}`` -> ``{"v": "a"}``) the binding is
    # dead and gets swept, taking the now-unused ``v`` declaration with it so the
    # nested chain remains self-consistent. Same fate for the inner ``c = v+1``: its
    # destination state has no readers of ``c``, so the binding + the ``c`` declaration
    # both drop.
    assert "v" not in edge1.data.assignments, (
        f"propagation should have substituted v->a everywhere and dropped the dead binding; "
        f"got {dict(edge1.data.assignments)}")
    assert "v" not in sdfg1.symbols, "declaration of v should be removed with its binding"
    assert "c" not in edge4.data.assignments, (
        f"unused c=v+1 binding should be swept; got {dict(edge4.data.assignments)}")
    assert "c" not in sdfg4.symbols, "declaration of c should be removed with its binding"


def test_scalars():
    """
    Tests that SymbolPropagation handles scalars correctly.
    """
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("num", dace.int32)
    sdfg.add_array("A", [64], dace.int32)
    sdfg.add_scalar("B", dace.int32)

    s1 = sdfg.add_state(is_start_block=True)
    s2 = sdfg.add_state()
    edge1 = sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={"num": "B"}))

    task1 = s2.add_tasklet("init", {}, {"out"}, "out = -1")
    access1 = s2.add_access("B")
    s2.add_edge(task1, "out", access1, None, dace.Memlet("B[0]"))

    task2 = s2.add_tasklet("init", {}, {"out"}, "out = num")
    access2 = s2.add_access("A")
    s2.add_edge(task2, "out", access2, None, dace.Memlet("A[0]"))

    # Apply SymbolPropagation
    sdfg.validate()
    SymbolPropagation().apply_pass(sdfg, {})
    sdfg.validate()

    # Validate correctness
    A = dace.ndarray([10], dtype=dace.int32)
    A[:] = np.random.randint(0, 100, size=10).astype(dace.int32.type)
    sdfg(A=A, B=5)
    assert A[0] == 5


def test_cloudsc_kidia_kfdia_promote_then_propagate():
    """CloudSC subset: scalar arguments ``kidia`` / ``kfdia`` used as the
    inclusive horizontal loop bound ``range(kidia, kfdia + 1)`` across several
    level nests (the ``DO JK=1,KLEV; DO JL=KIDIA,KFDIA`` shape). ``simplify``
    promotes ``kfdia + 1`` to per-nest symbols ``kfdia_plus_1_N = kfdia + 1``.

    SymbolPropagation alone does NOT fold them: ``kfdia`` is a non-transient
    scalar ARGUMENT, so values referencing it are (correctly) skipped by the
    scalar filter -- the pass is a no-op (``apply_pass`` returns ``None``).
    Promoting the scalar arguments to symbols first with
    ``ScalarToSymbolPromotion(transients_only=False)`` makes ``kfdia`` a symbol,
    after which SymbolPropagation folds ``kfdia_plus_1 -> (kfdia + 1)``. The
    scalar-skip filter itself is unchanged (genuine scalars are still skipped --
    see ``test_scalars``). Value-preserving throughout."""
    klev, klon = dace.symbol('klev'), dace.symbol('klon')

    @dace.program
    def cloudsc_kidia_kfdia(pt: dace.float64[klev, klon], ptend: dace.float64[klev, klon], kidia: dace.int32,
                            kfdia: dace.int32):
        for jk in range(klev):
            for jl in range(kidia, kfdia + 1):
                ptend[jk, jl] = pt[jk, jl] * 2.0
        for jk in range(klev):
            for jl in range(kidia, kfdia + 1):
                ptend[jk, jl] = ptend[jk, jl] + 3.0

    def _kfdia_plus1_syms(g):
        return {k for e in g.all_interstate_edges() for k in e.data.assignments if k.startswith('kfdia_plus_1')}

    nlev, nlon = 5, 8
    rng = np.random.default_rng(0)
    pt = rng.standard_normal((nlev, nlon))

    # Reference (un-promoted) output.
    ref = cloudsc_kidia_kfdia.to_sdfg(simplify=True)
    ref_out = np.zeros((nlev, nlon))
    ref(pt=pt.copy(), ptend=ref_out, kidia=0, kfdia=nlon - 1, klev=nlev, klon=nlon)

    # (1) Without promotion: kfdia is a scalar argument -> symprop is a no-op.
    sdfg = cloudsc_kidia_kfdia.to_sdfg(simplify=True)
    assert _kfdia_plus1_syms(sdfg), 'simplify should promote kfdia + 1 to kfdia_plus_1 symbols'
    assert isinstance(sdfg.arrays.get('kfdia'), dace.data.Scalar)
    assert SymbolPropagation().apply_pass(sdfg, {}) is None, \
        'symprop must skip values referencing the scalar argument kfdia (no-op)'

    # (2) Promote the scalar arguments to symbols first, then symprop folds them.
    sdfg2 = cloudsc_kidia_kfdia.to_sdfg(simplify=True)
    s2s = ScalarToSymbolPromotion()
    s2s.transients_only = False
    promoted = s2s.apply_pass(sdfg2, {})
    assert promoted and {'kidia', 'kfdia'} <= promoted, f'expected kidia/kfdia promoted, got {promoted}'
    assert 'kfdia' in sdfg2.symbols and 'kfdia' not in sdfg2.arrays

    ret = SymbolPropagation().apply_pass(sdfg2, {})
    assert ret is not None and any(s.startswith('kfdia_plus_1') for s in ret), \
        f'after promotion symprop must fold kfdia_plus_1 -> (kfdia + 1); propagated={ret}'
    sdfg2.validate()

    # Value-preserving (kidia/kfdia are now symbols).
    out2 = np.zeros((nlev, nlon))
    sdfg2(pt=pt.copy(), ptend=out2, kidia=0, kfdia=nlon - 1, klev=nlev, klon=nlon)
    assert np.allclose(out2, ref_out)
    assert np.allclose(out2, pt * 2.0 + 3.0)


_SP_N = dace.symbol("_SP_N")


@dace.program
def _carried_index_kernel(a: dace.float64[_SP_N], b: dace.float64[_SP_N], c: dace.float64[_SP_N],
                          d: dace.float64[_SP_N]):
    j = -1
    for i in range(_SP_N // 2):
        k = j + 1
        a[i] = b[k] - d[i]
        j = k + 1
        b[k] = a[i] + c[k]


def test_carried_index_symbol_not_propagated_stale():
    """Reproducer (TSVC s128): a loop-carried index ``k = j + 1`` must not be
    propagated into a downstream block as ``j + 1`` once the loop has reassigned
    ``j = k + 1``. There the live ``j`` is two ahead, so the stale expression is an
    off-by-two on ``b[k]`` / ``c[k]``. SymbolPropagation must keep ``k`` live; this
    checks the propagated SDFG still matches the un-propagated reference."""
    import copy
    n = 64
    rng = np.random.default_rng(0)
    base = {name: rng.random(n) for name in "abcd"}

    ref = _carried_index_kernel.to_sdfg(simplify=True)
    cand = copy.deepcopy(ref)
    SymbolPropagation().apply_pass(cand, {})
    cand.validate()

    ra = {name: arr.copy() for name, arr in base.items()}
    ref(**ra, _SP_N=n)
    ca = {name: arr.copy() for name, arr in base.items()}
    cand(**ca, _SP_N=n)
    for name in "abcd":
        assert np.allclose(ra[name], ca[name]), f"{name}: SymbolPropagation changed the result"


def test_dead_iedge_assignment_eliminated_after_substitution():
    """A bound-symbol shorthand iedge assignment (``k_plus_1 = klev + 1``) survived
    symbol_propagation: its uses got substituted to ``klev + 1`` but the defining
    assignment was left in place. The fix sweeps such dead assignments to a fixed
    point at the end of the pass; nothing references ``k_plus_1`` after the
    substitution, so the iedge ends with an empty ``assignments`` dict.
    """
    sdfg = dace.SDFG('dead_iedge_repro')
    sdfg.add_array('out', [16], dace.float64)
    sdfg.add_symbol('klev', dace.int32)
    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'k_plus_1': '(klev + 1)'}))

    t = s2.add_tasklet('t', {}, {'_o'}, '_o = 1.0')
    w = s2.add_write('out')
    s2.add_edge(t, '_o', w, None, dace.Memlet(data='out', subset='k_plus_1'))
    sdfg.validate()

    res = SymbolPropagation().apply_pass(sdfg, {})
    assert res == {'k_plus_1'}, f'expected k_plus_1 to be reported propagated; got {res}'

    surviving = [(lhs, rhs) for e in sdfg.all_interstate_edges() for lhs, rhs in e.data.assignments.items()]
    assert surviving == [], f'dead k_plus_1 assignment must be eliminated; got {surviving}'

    # The substitution must reach the memlet: the write to s2's ``out`` now indexes
    # ``klev + 1`` directly, not via the shorthand symbol.
    seen = []
    for st in sdfg.states():
        for e in st.edges():
            if e.data is not None and e.data.data == 'out':
                seen.append(str(e.data.subset))
    assert 'klev + 1' in seen, f'expected memlet subset to be substituted to klev+1; got {seen}'


def test_dead_iedge_chain_unravels_to_fixed_point():
    """Chained shorthands (``a = klev + 1; b = a; c = b``) must all be eliminated
    once their uses are substituted -- the cleanup sweep iterates to a fixed point."""
    sdfg = dace.SDFG('chain_repro')
    sdfg.add_array('out', [16], dace.float64)
    sdfg.add_symbol('klev', dace.int32)
    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    s3 = sdfg.add_state('s3')
    s4 = sdfg.add_state('s4')
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'a': '(klev + 1)'}))
    sdfg.add_edge(s2, s3, dace.InterstateEdge(assignments={'b': 'a'}))
    sdfg.add_edge(s3, s4, dace.InterstateEdge(assignments={'c': 'b'}))

    t = s4.add_tasklet('t', {}, {'_o'}, '_o = 2.0')
    w = s4.add_write('out')
    s4.add_edge(t, '_o', w, None, dace.Memlet(data='out', subset='c'))
    sdfg.validate()

    SymbolPropagation().apply_pass(sdfg, {})

    surviving = [(lhs, rhs) for e in sdfg.all_interstate_edges() for lhs, rhs in e.data.assignments.items()]
    assert surviving == [], f'every link of the dead chain must be eliminated; got {surviving}'


def test_dead_iedge_with_array_shape_substituted_into_descriptor():
    """A symbol referenced *only* by an array descriptor's shape (cloudsc's
    ``[0:kfdia_plus_1, 0:klon]`` pattern) used to keep the defining iedge alive
    because the IR-level ``replace_dict`` does not reach into descriptors. The
    fix substitutes the symbol into descriptors as a final step before
    elimination, so the array shape becomes ``kfdia + 1`` directly and the
    iedge drops."""
    sdfg = dace.SDFG('array_shape_repro')
    sdfg.add_symbol('klev', dace.int32)
    sdfg.add_symbol('k_plus_1', dace.int32)
    sdfg.add_array('out', ['k_plus_1'], dace.float64)
    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'k_plus_1': '(klev + 1)'}))

    t = s2.add_tasklet('t', {}, {'_o'}, '_o = 3.0')
    w = s2.add_write('out')
    s2.add_edge(t, '_o', w, None, dace.Memlet(data='out', subset='0'))
    sdfg.validate()

    SymbolPropagation().apply_pass(sdfg, {})

    surviving = [(lhs, rhs) for e in sdfg.all_interstate_edges() for lhs, rhs in e.data.assignments.items()]
    assert surviving == [], (f'k_plus_1 should have been substituted into the array shape and the '
                             f'binding dropped; got {surviving}')
    shape_str = ', '.join(str(s) for s in sdfg.arrays['out'].shape)
    assert 'klev' in shape_str and 'k_plus_1' not in shape_str, (
        f'array shape must read klev + 1 directly; got {shape_str}')
    assert 'k_plus_1' not in sdfg.symbols, 'declaration of k_plus_1 should be removed with its binding'


if __name__ == "__main__":
    test_loop_carried_symbol()
    test_nested_loop_carried_symbol()
    test_nested_symbol()
    test_multiple_sources()
    test_multiple_edge_assignments()
    test_deeply_nested_sdfg()
    test_scalars()
    test_cloudsc_kidia_kfdia_promote_then_propagate()
    test_carried_index_symbol_not_propagated_stale()
    test_dead_iedge_assignment_eliminated_after_substitution()
    test_dead_iedge_chain_unravels_to_fixed_point()
    test_dead_iedge_preserved_when_lhs_still_used()
