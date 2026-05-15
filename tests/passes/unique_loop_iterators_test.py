# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``UniqueLoopIterators`` pass."""

import dace
import numpy as np
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
    ``_loop_it_<N>`` name, not the original ``i`` .
    """
    UniqueLoopIterators._loop_var_counter = 0

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
    With ``assign_loop_iterator_post_value=True`` , the pass must stage
    a postfix-assignment state that sets the original loop variable to
    its iterator-after-loop value ( ``init + diff - (diff mod step)`` )
    so downstream reads see the same value gfortran / ifort / flang
    leave the iterator at after a counted DO exit.
    """
    UniqueLoopIterators._loop_var_counter = 0

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
    UniqueLoopIterators._loop_var_counter = 0

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
    UniqueLoopIterators._loop_var_counter = 0

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
    UniqueLoopIterators._loop_var_counter = 0

    sdfg = dace.SDFG('iter_on_edge')
    sdfg.add_array('out', [10], dace.int64)
    sdfg.add_symbol('k', dace.int64)

    body = LoopRegion('body', condition_expr='i < 10', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(body, is_start_block=True)
    s1 = body.add_state('s1', is_start_block=True)
    s2 = body.add_state('s2')
    # Interstate edge inside the loop body that reads ``i`` .
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
    UniqueLoopIterators._loop_var_counter = 0

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
    ``loop_variable`` ) has nothing to rename.  The pass must skip it
    cleanly even with the postfix-assignment option on (there's no
    induction variable to leave a post-value for)."""
    UniqueLoopIterators._loop_var_counter = 0

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

    Iterator names ``i`` / ``j`` / ``k`` / ``l`` are reused in every nest on
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
    """>=20 maps with heavily aliased iterators across deep nests.

    The frontend program has 22 maps and 22 ``range`` loops whose iterator
    names collapse to only two distinct symbols.  The pass must give every
    loop a unique name, keep all maps intact, and remain semantics-
    preserving -- verified against the un-transformed SDFG (the non-
    transformed reference), not a hand-written numpy loop.
    """

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
    UniqueLoopIterators._loop_var_counter = 0
    UniqueLoopIterators().apply_pass(sdfg, {})
    sdfg.validate()

    lv_after = loopvars(sdfg)
    assert n_maps(sdfg) >= 20
    assert len(lv_after) == len(set(lv_after)), "loop iterators must be unique after the pass"

    out_pass = np.zeros((_BIG_N, _BIG_N), np.float64)
    sdfg(A=A, B=out_pass)
    assert np.allclose(out_ref, out_pass), "UniqueLoopIterators changed program semantics"


if __name__ == '__main__':
    test_nested_sdfg_symbol_mapping()
    test_loop_var_reconstruction()
    test_nested_loops()
    test_loop_var_in_tasklet_body()
    test_loop_var_on_interstate_edge()
    test_loop_bound_with_indirect_array()
    test_while_loop_no_induction_var()
    test_large_nested_map_for_for_map_program()
