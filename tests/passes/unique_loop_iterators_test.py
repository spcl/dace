# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``UniqueLoopIterators`` pass."""

import dace
import numpy as np
import pytest
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.passes.analysis import loop_analysis


@dace.program
def foo(A: dace.float64[10, 10], idx: dace.int32[10, 10], B: dace.float64[5, 10]):
    for i in range(5):
        for j, k in dace.map[0:10, 0:10]:
            A[j, k] = 1.1 * A[j, k] + 1.2 * B[i, idx[j, k]]


def test_nested_sdfg_symbol_mapping():
    """
    The map inside the loop body with nested SDFG.
    The loop variable ``i`` must appear in the nested SDFG's symbol_mapping.
    After UniqueLoopIterators, the symbol_mapping should reference the new
    ``_loop_it_<N>`` name, not the original ``i``.
    """

    sdfg = foo.to_sdfg(simplify=False)

    # Before: confirm ``i`` is the loop variable and appears in a nested SDFG mapping
    loops_before = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops_before) == 1
    assert loops_before[0].loop_variable == 'i'

    found_i_in_mapping = False
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if 'i' in node.symbol_mapping:
                    found_i_in_mapping = True
    assert found_i_in_mapping, "Expected 'i' in nested SDFG symbol_mapping before pass"

    # Apply pass
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    # After: the nested SDFG symbol_mapping should have _loop_it_0, not i
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                assert 'i' not in node.symbol_mapping, \
                    f"Original loop var 'i' should not be in symbol_mapping, got {node.symbol_mapping}"
                assert '_loop_it_0' in node.symbol_mapping, \
                    f"renamed loop var '_loop_it_0' should be in symbol_mapping, got {node.symbol_mapping}"

    A = np.random.rand(10, 10)
    idx = np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    B = np.random.rand(5, 10)

    A_ref = A.copy()
    for i in range(5):
        for j in range(10):
            for k in range(10):
                A_ref[j, k] = 1.1 * A_ref[j, k] + 1.2 * B[i, idx[j, k]]

    csdfg = sdfg.compile()
    csdfg(A=A, idx=idx, B=B)
    assert np.allclose(A, A_ref), f"Max error: {np.max(np.abs(A - A_ref))}"


@dace.program
def loop_var_used_after(A: dace.float64[10], B: dace.float64[10]):
    for i in range(10):
        A[i] = 2.0 * B[i]


def test_loop_var_reconstruction():
    """
    With ``assign_loop_iterator_post_value=True``, the pass must stage
    a postfix-assignment state that sets the original loop variable to
    its iterator-after-loop value (``init + diff - (diff mod step)``)
    so downstream reads see the same value gfortran / ifort / flang
    leave the iterator at after a counted DO exit.
    """

    sdfg = loop_var_used_after.to_sdfg(simplify=False)

    pass_ = UniqueLoopIterators()
    pass_.assign_loop_iterator_post_value = True
    pass_.apply_pass(sdfg, None)
    sdfg.validate()

    # Check that a reconstruction state was added
    reconstruction_states = [s for s in sdfg.all_states() if hasattr(s, 'label') and 'loop_iter_post_value' in s.label]
    assert len(reconstruction_states) == 1, f"Expected 1 reconstruction state, found {len(reconstruction_states)}"

    # Check that assignment is correct
    loops = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops) == 1
    loop = loops[0]

    out_edges = loop.parent_graph.out_edges(loop)
    assert len(out_edges) == 1

    assignments = out_edges[0].data.assignments
    assert 'i' in assignments, f"Expected assignment to 'i', got {assignments}"

    loop_end = loop_analysis.get_loop_end(loop)
    init = loop_analysis.get_init_assignment(loop)
    stride = loop_analysis.get_loop_stride(loop)
    import sympy as sp
    expected_post = init + (loop_end - init + stride) - sp.Mod(loop_end - init + stride, stride)
    assert str(assignments['i']) == f"({expected_post})", \
        f"Expected post-value {expected_post}, got {assignments['i']}"

    A = np.zeros(10)
    B = np.random.rand(10)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B)
    assert np.allclose(A, 2.0 * B)


@dace.program
def nested_loops(A: dace.float64[8, 8]):
    for i in range(8):
        for j in range(8):
            A[i, j] = A[i, j] + 1.0


def test_nested_loops():
    """
    Two nested LoopRegions with variables i and j.
    Both should be renamed to distinct unique names (_loop_it_0, _loop_it_1),
    and both should get reconstruction states.
    """

    sdfg = nested_loops.to_sdfg(simplify=False)

    loops_before = [cfg for cfg in sdfg.all_control_flow_regions() if isinstance(cfg, LoopRegion)]
    assert len(loops_before) == 2
    loop_vars_before = {lr.loop_variable for lr in loops_before}
    assert loop_vars_before == {'i', 'j'}

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    A = np.random.rand(8, 8)
    A_ref = A.copy() + 1.0
    csdfg = sdfg.compile()
    csdfg(A=A)
    assert np.allclose(A, A_ref), f"Max error: {np.max(np.abs(A - A_ref))}"


def test_loop_var_in_tasklet_body():
    """The pass must rename the loop iter inside tasklet expressions, not
    just on memlet subsets."""

    sdfg = dace.SDFG('iter_in_tasklet')
    sdfg.add_array('out', [10], dace.int64)
    body = LoopRegion('body', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(body, is_start_block=True)
    inner = body.add_state('inner', is_start_block=True)
    t = inner.add_tasklet('write_iter', set(), {'_o'}, '_o = i * 2')
    a = inner.add_access('out')
    inner.add_edge(t, '_o', a, None, dace.Memlet('out[i]'))

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    # The tasklet's code must now reference _loop_it_0, not i.
    found = False
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                code = node.code.as_string if hasattr(node.code, 'as_string') else str(node.code)
                assert 'i ' not in code and code.strip() != '_o = i * 2', \
                    f"Tasklet still references original 'i': {code}"
                if '_loop_it_0' in code:
                    found = True
    assert found, "Expected tasklet code to reference _loop_it_0"

    out = np.zeros(10, dtype=np.int64)
    csdfg = sdfg.compile()
    csdfg(out=out)
    assert (out == 2 * np.arange(10)).all()


def test_loop_var_on_interstate_edge():
    """The loop iter is read on an interstate edge inside the loop body.
    The pass must rewrite the edge's assignment / condition to use the
    renamed iterator."""

    sdfg = dace.SDFG('iter_on_edge')
    sdfg.add_array('out', [10], dace.int64)
    sdfg.add_symbol('k', dace.int64)

    body = LoopRegion('body', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(body, is_start_block=True)
    s1 = body.add_state('s1', is_start_block=True)
    s2 = body.add_state('s2')
    # Interstate edge inside the loop body that reads ``i``.
    body.add_edge(s1, s2, dace.InterstateEdge(assignments={'k': 'i + 100'}))

    t = s2.add_tasklet('write', set(), {'_o'}, '_o = k')
    a = s2.add_access('out')
    s2.add_edge(t, '_o', a, None, dace.Memlet('out[i]'))

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    found = False
    for e in body.edges():
        for tgt, rhs in e.data.assignments.items():
            assert 'i ' not in str(rhs), f"Edge assignment still references 'i': {rhs}"
            if '_loop_it_0' in str(rhs):
                found = True
    assert found, "Expected interstate-edge assignment to reference _loop_it_0"

    out = np.zeros(10, dtype=np.int64)
    csdfg = sdfg.compile()
    csdfg(out=out)
    assert (out == np.arange(10) + 100).all()


def test_loop_bound_with_indirect_array():
    """Loop bound is ``row_ptr(i+1) - 1`` -- sympy's default ``str``
    would render the array access as a function call.  The pass must use
    ``arrayexprs=`` so the reconstruction assignment carries the
    Python subscript form."""

    sdfg = dace.SDFG('indirect_bound')
    sdfg.add_array('row_ptr', [11], dace.int64)
    sdfg.add_array('values', [20], dace.float64)
    sdfg.add_array('acc', [10], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_symbol('j', dace.int64)

    outer = LoopRegion('outer', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(outer, is_start_block=True)
    body = outer.add_state('outer_body', is_start_block=True)
    inner = LoopRegion('inner',
                       condition_expr='j < row_ptr[i + 1]',
                       loop_var='j',
                       initialize_expr='j = row_ptr[i]',
                       update_expr='j = j + 1')
    outer.add_node(inner)
    outer.add_edge(body, inner, dace.InterstateEdge())

    inner_body = inner.add_state('inner_body', is_start_block=True)
    rd = inner_body.add_access('values')
    wr = inner_body.add_access('acc')
    t = inner_body.add_tasklet('add', {'_v', '_a'}, {'_o'}, '_o = _v + _a')
    inner_body.add_edge(rd, None, t, '_v', dace.Memlet('values[j]'))
    inner_body.add_edge(wr, None, t, '_a', dace.Memlet('acc[i]'))
    wr2 = inner_body.add_access('acc')
    inner_body.add_edge(t, '_o', wr2, None, dace.Memlet('acc[i]'))

    pass_ = UniqueLoopIterators()
    pass_.assign_loop_iterator_post_value = True
    pass_.apply_pass(sdfg, None)
    sdfg.validate()

    # The inner-loop reconstruction state must use Python subscript form.
    recon = [s for s in sdfg.all_states() if hasattr(s, 'label') and 'loop_iter_post_value' in s.label]
    rhs_strings = []
    for s in recon:
        for e in s.parent_graph.in_edges(s):
            for rhs in e.data.assignments.values():
                rhs_strings.append(str(rhs))

    assert any('row_ptr[' in s for s in rhs_strings), \
        f"Reconstruction RHS must use Python subscript form: {rhs_strings}"
    assert all('row_ptr(' not in s for s in rhs_strings), \
        f"Reconstruction RHS must not use function-call form: {rhs_strings}"


def test_while_loop_no_induction_var():
    """A LoopRegion that wasn't synthesised from a counted ``for`` (no
    ``loop_variable``) has nothing to rename.  The pass must skip it
    cleanly even with the postfix-assignment option on (there's no
    induction variable to leave a post-value for)."""

    sdfg = dace.SDFG('whileloop')
    sdfg.add_symbol('flag', dace.int64)

    body = LoopRegion('body', condition_expr='flag != 0', loop_var=None)
    sdfg.add_node(body, is_start_block=True)
    inner = body.add_state('inner', is_start_block=True)
    t = inner.add_tasklet('flip', set(), {'_o'}, '_o = 0')
    sdfg.add_array('out', [1], dace.int64)
    a = inner.add_access('out')
    inner.add_edge(t, '_o', a, None, dace.Memlet('out[0]'))

    pass_ = UniqueLoopIterators()
    pass_.assign_loop_iterator_post_value = True
    pass_.apply_pass(sdfg, None)

    recon = [s for s in sdfg.all_states() if hasattr(s, 'label') and 'loop_iter_post_value' in s.label]
    assert recon == [], f"Unexpected reconstruction states: {[s.label for s in recon]}"


_BIG_N = 4


@dace.program
def _big_nested_map_for_for_map(A: dace.float64[_BIG_N, _BIG_N], B: dace.float64[_BIG_N, _BIG_N]):
    """11 inline ``map -> for -> for -> map`` nests (22 maps, 22 loops).

    Iterator names ``i`` /``j`` /``k`` /``l`` are reused in every nest on
    purpose, so the pass must independently rename ~44 loop variables.
    """
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, j] * (k + 1)
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, j] - k
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[l, j] * 2 + k
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, k] + j
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[j, k] * 3
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, j] * A[k, l]
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[l, i] += A[i, j] + k * 2
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, j] / (k + 1)
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[l, k] - j
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, j] * k + l
    for i in dace.map[0:_BIG_N]:
        for j in range(_BIG_N):
            for k in range(_BIG_N):
                for l in dace.map[0:_BIG_N]:
                    B[i, l] += A[i, j] + A[i, k]


def test_large_nested_map_for_for_map_program():
    """>=20 maps with heavily aliased iterators across deep nests."""

    def n_maps(g):
        return sum(1 for n, _ in g.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))

    def loopvars(g):
        return [n.loop_variable for n, _ in g.all_nodes_recursive() if isinstance(n, LoopRegion)]

    base = _big_nested_map_for_for_map.to_sdfg(simplify=False)
    lv_before = loopvars(base)
    assert n_maps(base) >= 20
    assert len(lv_before) >= 20
    assert len(set(lv_before)) < len(lv_before), "expected aliased iterator names before the pass"

    rng = np.random.default_rng(0)
    A = rng.random((_BIG_N, _BIG_N))
    out_ref = np.zeros((_BIG_N, _BIG_N), np.float64)
    base(A=A, B=out_ref)

    sdfg = _big_nested_map_for_for_map.to_sdfg(simplify=False)
    UniqueLoopIterators().apply_pass(sdfg, {})
    sdfg.validate()

    lv_after = loopvars(sdfg)
    assert n_maps(sdfg) >= 20
    assert len(lv_after) == len(set(lv_after)), "loop iterators must be unique after the pass"

    out_pass = np.zeros((_BIG_N, _BIG_N), np.float64)
    sdfg(A=A, B=out_pass)
    assert np.allclose(out_ref, out_pass), "UniqueLoopIterators changed program semantics"


def test_no_postamble_drops_dead_symbol_declaration():
    """With ``assign_loop_iterator_post_value=False`` the renamed iterator
    has no surviving reader, so the stale ``sdfg.symbols[old_name]``
    declaration left by the frontend must be removed -- otherwise it leaks
    as a phantom free symbol on the enclosing NestedSDFG boundary
    (validation: ``Missing symbols on nested SDFG: ['i']``)."""

    @dace.program
    def sibling_for_loops(x: dace.float64[10, 10]):
        for j in dace.map[0:10]:
            for i in range(10):
                x[i, j] = 1.0
            for i in range(10):
                x[i, j] += 2.0

    sdfg = sibling_for_loops.to_sdfg(simplify=False)

    # Pre-condition: the body NestedSDFG declares ``i`` in its symbol table
    # and the parent does not provide it via ``symbol_mapping``; the
    # frontend keeps the declaration around because the original LoopRegions
    # use ``i`` as their ``loop_variable``.
    body_nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert any('i' in n.sdfg.symbols for n in body_nsdfgs)

    p = UniqueLoopIterators()
    p.assign_loop_iterator_post_value = False
    p.apply_pass(sdfg, {})

    # Every LoopRegion now has a unique ``_loop_it_<N>`` variable.
    loopvars = [n.loop_variable for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]
    assert loopvars == sorted(set(loopvars))
    assert all(v.startswith('_loop_it_') for v in loopvars)

    # The previously-leaking ``i`` declaration must be gone from every
    # SDFG-level symbol table that lost its last user.
    for n in body_nsdfgs:
        assert 'i' not in n.sdfg.symbols, f"stale 'i' still declared in {n.sdfg.name}.symbols"

    # And the resulting SDFG validates cleanly (no missing-symbol error on
    # the NestedSDFG boundary).
    sdfg.validate()


def test_no_postamble_clears_loop_var_for_inner_accumulator():
    """Regression: an inner reduction-style accumulator (``s += a[i, j-1] +
    a[i, j] + a[i, j+1]``) writes the per-iteration scalar ``s`` inside a
    ``for j: ...`` loop. With ``assign_loop_iterator_post_value = False``
    the rename must drop the dead ``j`` from ``sdfg.symbols`` of the
    enclosing NestedSDFG -- otherwise ``j`` lingers as a declared symbol
    that no longer corresponds to any loop variable, and the validator
    reports ``Missing symbols on nested SDFG: ['j']`` on the enclosing
    Map's body NestedSDFG.

    The previous gate ``old_name not in sdfg.free_symbols`` was circular:
    ``SDFG.free_symbols`` calls ``used_symbols(all_symbols=True)`` which
    unconditionally folds ``sdfg.symbols.keys()`` back into the "free"
    set (``ControlFlowRegion._used_symbols_internal``, the ``if
    all_symbols: free_syms |= set(self.symbols.keys())`` branch). The
    declared symbol therefore always appeared "free" by virtue of being
    declared and was never removed. The fix uses
    ``used_symbols(all_symbols=False)`` which reflects only actual
    code-generation usage.
    """

    @dace.program
    def redux(a: dace.float64[8, 9], b: dace.float64[8]):
        for i in dace.map[0:8]:
            s = a[i, 0] + a[i, 8]
            for j in range(1, 8):
                s += a[i, j - 1] + a[i, j] + a[i, j + 1]
            b[i] = s

    sdfg = redux.to_sdfg(simplify=True)
    # Pre-condition: at least one body NestedSDFG declares ``j`` -- the
    # frontend leaves it there for the original loop_var.
    nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert any('j' in n.sdfg.symbols for n in nsdfgs)

    p = UniqueLoopIterators()
    p.assign_loop_iterator_post_value = False
    p.apply_pass(sdfg, {})

    # Post-condition: no body NestedSDFG still declares ``j`` -- the
    # cleanup removed the stale declaration because nothing in the
    # body (memlets, tasklet code, interstate-edge assignments) actually
    # uses ``j`` after the rename to ``_loop_it_<N>``.
    nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    for n in nsdfgs:
        assert 'j' not in n.sdfg.symbols, \
            f"NSDFG {n.sdfg.name} still declares 'j'; symbols={sorted(n.sdfg.symbols)}"
    # And the SDFG validates as a whole (no Missing-symbols error).
    sdfg.validate()

    # Numerical equivalence against a pure-numpy oracle: the inner
    # accumulator semantics are preserved by canonicalize.
    a = np.random.rand(8, 9)
    b = np.zeros(8)
    sdfg.compile()(a=a.copy(), b=b)
    exp = np.zeros(8)
    for i in range(8):
        s = a[i, 0] + a[i, 8]
        for j in range(1, 8):
            s += a[i, j - 1] + a[i, j] + a[i, j + 1]
        exp[i] = s
    assert np.allclose(b, exp)


def test_postamble_preserves_symbol_declaration():
    """The dead-symbol cleanup must trigger only when the post-value
    epilogue is disabled. With the default ``assign_loop_iterator_post_value
    = True`` the epilogue's ``<old_name> = <post_value>`` assignment IS a
    surviving reader of the original symbol; removing the declaration would
    invalidate it."""

    @dace.program
    def sibling_for_loops(x: dace.float64[10, 10]):
        for j in dace.map[0:10]:
            for i in range(10):
                x[i, j] = 1.0
            for i in range(10):
                x[i, j] += 2.0

    sdfg = sibling_for_loops.to_sdfg(simplify=False)
    body_nsdfgs_before = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert any('i' in n.sdfg.symbols for n in body_nsdfgs_before)

    UniqueLoopIterators().apply_pass(sdfg, {})  # default: post-value ON

    # ``i`` declaration stays because the postamble assignments read/write it.
    body_nsdfgs_after = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    assert any('i' in n.sdfg.symbols for n in body_nsdfgs_after)
    sdfg.validate()


def test_idempotent_skips_already_unique_iterators():
    """Running the pass twice must be a no-op the second time and must NOT
    re-rename already-unique ``_loop_it_<N>`` iterators.

    Re-renaming an already-unique iterator that a deeply-nested SDFG
    imports used to drop the import from the grandchild's
    ``symbol_mapping`` (the re-key did not survive a second rename),
    producing "Missing symbols on nested SDFG: ['_loop_it_<N>']". The
    pass now skips iterators already in ``_loop_it_*`` form. This mirrors
    the canonicalize pipeline, which runs UniqueLoopIterators twice (the
    ``clean`` and ``ssa`` stages)."""

    # A map body (nested SDFG) with two nested loops, then a deeper nest --
    # the shape where a grandchild NSDFG imports the outer iterators.
    sdfg = nested_loops.to_sdfg(simplify=False)

    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    names_after_first = sorted(lr.loop_variable for lr in sdfg.all_control_flow_regions() if isinstance(lr, LoopRegion))
    assert all(n.startswith('_loop_it_') for n in names_after_first), names_after_first

    # Second application: the iterators are already unique; nothing changes
    # and the SDFG stays valid (no "Missing symbols" crash).
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    names_after_second = sorted(lr.loop_variable for lr in sdfg.all_control_flow_regions()
                                if isinstance(lr, LoopRegion))
    assert names_after_second == names_after_first, \
        f'second pass re-renamed already-unique iterators: {names_after_first} -> {names_after_second}'

    A = np.random.rand(8, 8)
    exp = A + 1.0
    csdfg = sdfg.compile()
    csdfg(A=A)
    assert np.allclose(A, exp)


N = dace.symbol('N')


@dace.program
def sibling_kbound_loops(out: dace.float64[N, N], arr: dace.float64[N, N], x: dace.int32):
    """Map body with two sibling ``k``-loops that share the name ``k`` and
    the same per-``i`` bounds, one of them inside an ``if`` guard. Both
    accumulate into ``out`` over the same sparse ``[beg:end)`` range."""
    for i in dace.map[0:N]:
        beg = i // 2 + 1
        end = beg + 2
        for k in range(beg, end):
            out[i, k] += arr[i, k]
        if x > 0:
            for k in range(beg, end):
                out[i, k] += 2.0 * arr[i, k]


def _sibling_kbound_oracle(arr, x, n):
    out = np.zeros((n, n))
    for i in range(n):
        beg, end = i // 2 + 1, i // 2 + 3
        for k in range(beg, min(end, n)):
            out[i, k] += arr[i, k]
        if x > 0:
            for k in range(beg, min(end, n)):
                out[i, k] += 2.0 * arr[i, k]
    return out


def test_value_preserving_sibling_kbound_loops():
    """Renaming two sibling loops that share the iterator name ``k`` must
    not cross-contaminate their memlets.

    The frontend's ``accesses`` cache hands out the *same* Range object for
    the identical ``arr[i, k]`` read in both loop bodies, and ``Memlet.simple``
    stores a Subset by reference, so the two read edges used to share one
    subset object. Renaming the first loop's ``k`` to ``_loop_it_0`` then
    rewrote that shared subset in place, leaving the *second* (guarded) loop
    reading ``arr[i, _loop_it_0]`` while writing ``out[i, _loop_it_1]`` -- a
    silent value corruption that only manifested when the guard was taken
    (``x = 1``). With each edge owning its subset the rename is local and the
    accumulation is preserved. Regression for the frontend slice-subset
    aliasing surfaced through UniqueLoopIterators."""
    n = 8
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((n, n))
    for x in (1, 0):
        sdfg = sibling_kbound_loops.to_sdfg(simplify=True)
        UniqueLoopIterators().apply_pass(sdfg, {})
        sdfg.validate()
        out = np.zeros((n, n))
        sdfg(out=out, arr=arr.copy(), x=np.int32(x), N=n)
        assert np.allclose(out, _sibling_kbound_oracle(arr, x, n)), f'value corrupted for x={x}'


@dace.program
def fission_then_duplicate(A: dace.float64[N, N], B: dace.float64[N, N]):
    """Outer ``i`` over an inner ``j`` body with a fully-parallel ``A`` write
    and a ``j``-carried ``B`` recurrence. LoopFission splits the inner ``j``
    loop into two siblings -- the shape that surfaces the duplicate iterator."""
    for i in range(N):
        for j in range(1, N):
            A[j, i] = A[j, i] * 2.0
            B[i, j] = B[i, j - 1] + B[i, j]


def test_disambiguates_fission_cloned_iterators():
    """LoopFission clones a loop into siblings that keep the same
    ``_loop_it_<N>`` name; re-running UniqueLoopIterators must give each its own
    name (so e.g. LoopToMap is not blocked by a sibling appearing to read the
    shared iterator). The pass skips *unique* ``_loop_it_*`` names (idempotent)
    but re-disambiguates duplicates.

    Reproduces ``map i { NestedSDFG { loop j fis0 (_loop_it_1): A[j,i] *= 2;
    loop j fis1 (_loop_it_1): B[i,j] = B[i,j-1] + B[i,j] } }``."""
    from dace.transformation.passes.loop_fission import LoopFission

    sdfg = fission_then_duplicate.to_sdfg(simplify=True)
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    LoopFission().apply_pass(sdfg, {})
    after_fission = [
        r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)
    ]
    assert len(after_fission) != len(set(after_fission)), \
        f'expected a duplicate iterator after fission, got {after_fission}'

    # Re-running the pass must disambiguate the fission-cloned duplicate.
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()
    after = [r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert len(after) == len(set(after)), f'iterators not disambiguated after re-run: {after}'

    n = 8
    rng = np.random.default_rng(0)
    A0, B0 = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    refA, refB = A0.copy(), B0.copy()
    ref = fission_then_duplicate.to_sdfg(simplify=True)
    ref(A=refA, B=refB, N=n)
    gotA, gotB = A0.copy(), B0.copy()
    sdfg(A=gotA, B=gotB, N=n)
    assert np.allclose(gotA, refA) and np.allclose(gotB, refB)


@dace.program
def triply_nested(A: dace.float64[6, 6, 6]):
    for i in range(6):
        for j in range(6):
            for k in range(6):
                A[i, j, k] = A[i, j, k] + (i + j + k)


def test_triply_nested_loops_unique_and_value_preserving():
    """Three nested counted loops (``i``, ``j``, ``k``) get three distinct
    ``_loop_it_<N>`` names, the rename cascades through every nesting depth
    (memlets, tasklet bodies, nested-SDFG symbol mappings), and the program
    stays value-preserving."""
    sdfg = triply_nested.to_sdfg(simplify=False)
    UniqueLoopIterators().apply_pass(sdfg, None)
    sdfg.validate()

    loopvars = [r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert len(loopvars) == 3
    assert len(set(loopvars)) == 3 and all(v.startswith('_loop_it_') for v in loopvars), loopvars

    A = np.random.rand(6, 6, 6)
    exp = A + (np.arange(6)[:, None, None] + np.arange(6)[None, :, None] + np.arange(6)[None, None, :])
    sdfg.compile()(A=A)
    assert np.allclose(A, exp)


@dace.program
def descending_loop(A: dace.float64[10], B: dace.float64[10]):
    for i in range(9, -1, -1):
        A[i] = 2.0 * B[i] + i


def test_negative_step_loop_value_preserving():
    """A counted DO with a negative step (``range(9, -1, -1)``) is renamed
    like any other; the post-value epilogue uses the descending exit value and
    the program stays value-preserving."""
    sdfg = descending_loop.to_sdfg(simplify=False)
    UniqueLoopIterators().apply_pass(sdfg, None)  # default: post-value ON
    sdfg.validate()

    loopvars = [r.loop_variable for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert loopvars and all(v.startswith('_loop_it_') for v in loopvars), loopvars

    A = np.zeros(10)
    B = np.random.rand(10)
    sdfg.compile()(A=A, B=B)
    assert np.allclose(A, 2.0 * B + np.arange(10))


def test_seeds_counter_past_existing_loop_it_names():
    """The per-call counter is seeded past any iterator already in
    ``_loop_it_<N>`` form, so a freshly-renamed loop never collides with an
    existing unique name. Here a loop already named ``_loop_it_5`` coexists
    with a loop named ``i``; ``i`` must become ``_loop_it_6`` (not reuse 0 or
    collide with 5)."""
    sdfg = dace.SDFG('seed_past_existing')
    sdfg.add_array('out', [10], dace.int64)

    # An already-unique iterator (left untouched by the idempotency skip).
    existing = LoopRegion('existing',
                          condition_expr='_loop_it_5 < 5',
                          loop_var='_loop_it_5',
                          initialize_expr='_loop_it_5 = 0',
                          update_expr='_loop_it_5 = _loop_it_5 + 1')
    sdfg.add_node(existing, is_start_block=True)
    es = existing.add_state('es', is_start_block=True)
    et = es.add_tasklet('w', set(), {'_o'}, '_o = _loop_it_5')
    ea = es.add_access('out')
    es.add_edge(et, '_o', ea, None, dace.Memlet('out[_loop_it_5]'))

    # A fresh loop to be renamed.
    fresh = LoopRegion('fresh', condition_expr='i < 10', loop_var='i', initialize_expr='i = 5', update_expr='i = i + 1')
    sdfg.add_node(fresh)
    sdfg.add_edge(existing, fresh, dace.InterstateEdge())
    fs = fresh.add_state('fs', is_start_block=True)
    ft = fs.add_tasklet('w', set(), {'_o'}, '_o = i')
    fa = fs.add_access('out')
    fs.add_edge(ft, '_o', fa, None, dace.Memlet('out[i]'))

    p = UniqueLoopIterators()
    p.assign_loop_iterator_post_value = False
    p.apply_pass(sdfg, None)
    sdfg.validate()

    loopvars = {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)}
    assert '_loop_it_5' in loopvars, f'existing unique iterator must be left alone: {loopvars}'
    assert '_loop_it_6' in loopvars, f'fresh iterator must seed past the existing max (-> 6): {loopvars}'
    assert len(loopvars) == 2, loopvars


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
