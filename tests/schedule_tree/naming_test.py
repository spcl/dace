# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree
from dace.transformation.passes.constant_propagation import ConstantPropagation

import pytest
from typing import List


def _irreducible_loop_to_loop():
    sdfg = dace.SDFG('irreducible')
    # Add a simple chain of two for loops with goto from second to first's body
    s1 = sdfg.add_state_after(sdfg.add_state_after(sdfg.add_state()))
    s2 = sdfg.add_state()
    e = sdfg.add_state()

    # Add a loop
    l1 = sdfg.add_state()
    l2 = sdfg.add_state_after(l1)
    sdfg.add_loop(s1, l1, s2, 'i', '0', 'i < 10', 'i + 1', loop_end_state=l2)

    l3 = sdfg.add_state()
    l4 = sdfg.add_state_after(l3)
    sdfg.add_loop(s2, l3, e, 'i', '0', 'i < 10', 'i + 1', loop_end_state=l4)

    # Irreducible part
    sdfg.add_edge(l3, l1, dace.InterstateEdge('i < 5'))

    # Avoiding undefined behavior
    sdfg.edges_between(l3, l4)[0].data.condition.as_string = 'i >= 5'

    return sdfg


def _nested_irreducible_loops():
    sdfg = _irreducible_loop_to_loop()
    nsdfg = _irreducible_loop_to_loop()

    l1 = sdfg.node(5)
    l1.add_nested_sdfg(nsdfg, None, {}, {})
    return sdfg


def test_clash_states():
    """
    Same test as test_irreducible_in_loops, but all states in the nested SDFG share names with the top SDFG
    """
    sdfg = _nested_irreducible_loops()

    stree = as_schedule_tree(sdfg)
    unique_names = set()
    for node in stree.preorder_traversal():
        if isinstance(node, tn.StateLabel):
            if node.state.name in unique_names:
                raise NameError('Name clash')
            unique_names.add(node.state.name)


@pytest.mark.parametrize('constprop', (False, True))
def test_clash_symbol_mapping(constprop):
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [200], dace.float64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('k', dace.int64)

    state = sdfg.add_state()
    state2 = sdfg.add_state()
    sdfg.add_edge(state, state2, dace.InterstateEdge(assignments={'k': 'M + 1'}))

    nsdfg = dace.SDFG('nester')
    nsdfg.add_symbol('M', dace.int64)
    nsdfg.add_symbol('N', dace.int64)
    nsdfg.add_symbol('k', dace.int64)
    nsdfg.add_array('out', [100], dace.float64)
    nsdfg.add_transient('tmp', [100], dace.float64)
    nstate = nsdfg.add_state()
    nstate2 = nsdfg.add_state()
    nsdfg.add_edge(nstate, nstate2, dace.InterstateEdge(assignments={'k': 'M + 1'}))

    # Copy
    # The code should end up as `tmp[N:N+2] <- out[M+1:M+3]`
    # In the outer SDFG: `tmp[N:N+2] <- A[M+101:M+103]`
    r = nstate.add_access('out')
    w = nstate.add_access('tmp')
    nstate.add_edge(r, None, w, None, dace.Memlet(data='out', subset='k:k+2', other_subset='M:M+2'))

    # Tasklet
    # The code should end up as `tmp[M] -> Tasklet -> out[N + 1]`
    # In the outer SDFG: `tmp[M] -> Tasklet -> A[N + 101]`
    r = nstate2.add_access('tmp')
    w = nstate2.add_access('out')
    t = nstate2.add_tasklet('dosomething', {'a'}, {'b'}, 'b = a + 1')
    nstate2.add_edge(r, None, t, 'a', dace.Memlet('tmp[N]'))
    nstate2.add_edge(t, 'b', w, None, dace.Memlet('out[k]'))

    # Connect nested SDFG to parent SDFG with an offset memlet
    nsdfg_node = state2.add_nested_sdfg(nsdfg, None, {}, {'out'}, {'N': 'M', 'M': 'N', 'k': 'k'})
    w = state2.add_write('A')
    state2.add_edge(nsdfg_node, 'out', w, None, dace.Memlet('A[100:200]'))

    # Get rid of k
    if constprop:
        ConstantPropagation().apply_pass(sdfg, {})

    stree = as_schedule_tree(sdfg)
    assert len(stree.children) in (2, 4)  # Either with assignments or without

    # With assignments
    if len(stree.children) == 4:
        assert constprop is False
        assert isinstance(stree.children[0], tn.AssignNode)
        assert isinstance(stree.children[1], tn.CopyNode)
        assert isinstance(stree.children[2], tn.AssignNode)
        assert isinstance(stree.children[3], tn.TaskletNode)
        assert stree.children[1].memlet.data == 'A'
        assert str(stree.children[1].memlet.src_subset) == 'k + 100:k + 102'
        assert str(stree.children[1].memlet.dst_subset) == 'N:N + 2'
        assert stree.children[3].in_memlets['a'].data == 'tmp'
        assert str(stree.children[3].in_memlets['a'].src_subset) == 'M'
        assert stree.children[3].out_memlets['b'].data == 'A'
        assert str(stree.children[3].out_memlets['b'].dst_subset) == 'k + 100'
    else:
        assert constprop is True
        assert isinstance(stree.children[0], tn.CopyNode)
        assert isinstance(stree.children[1], tn.TaskletNode)
        assert stree.children[0].memlet.data == 'A'
        assert str(stree.children[0].memlet.src_subset) == 'M + 101:M + 103'
        assert str(stree.children[0].memlet.dst_subset) == 'N:N + 2'
        assert stree.children[1].in_memlets['a'].data == 'tmp'
        assert str(stree.children[1].in_memlets['a'].src_subset) == 'M'
        assert stree.children[1].out_memlets['b'].data == 'A'
        assert str(stree.children[1].out_memlets['b'].dst_subset) == 'N + 101'


def test_edgecase_symbol_mapping():
    sdfg = dace.SDFG('tester')
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_symbol('N', dace.int64)

    state = sdfg.add_state()
    state2 = sdfg.add_state_after(state)

    nsdfg = dace.SDFG('nester')
    nsdfg.add_symbol('M', dace.int64)
    nsdfg.add_symbol('N', dace.int64)
    nsdfg.add_symbol('k', dace.int64)
    nstate = nsdfg.add_state()
    nstate2 = nsdfg.add_state()
    nsdfg.add_edge(nstate, nstate2, dace.InterstateEdge(assignments={'k': 'M + 1'}))

    state2.add_nested_sdfg(nsdfg, None, {}, {}, {'N': 'M', 'M': 'N', 'k': 'M + 1'})

    stree = as_schedule_tree(sdfg)
    assert len(stree.children) == 1
    assert isinstance(stree.children[0], tn.AssignNode)
    # TODO: "assign M + 1 = (N + 1)", target should stay "k"
    assert str(stree.children[0].name) == 'k'


def test_clash_iteration_symbols():
    sdfg = _nested_irreducible_loops()

    stree = as_schedule_tree(sdfg)

    def _traverse(node: tn.ScheduleTreeScope, scopes: List[str]):
        for child in node.children:
            if isinstance(child, tn.ForScope):
                itervar = child.header.itervar
                if itervar in scopes:
                    raise NameError('Nested scope redefines iteration variable')
                _traverse(child, scopes + [itervar])
            elif isinstance(child, tn.ScheduleTreeScope):
                _traverse(child, scopes)

    _traverse(stree, [])


if __name__ == '__main__':
    test_clash_states()
    test_clash_symbol_mapping(False)
    test_clash_symbol_mapping(True)
    test_edgecase_symbol_mapping()
    test_clash_iteration_symbols()
