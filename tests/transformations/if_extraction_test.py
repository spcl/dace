# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import os
from copy import deepcopy
from typing import Dict, Collection

import numpy as np

import dace
from dace import SDFG, InterstateEdge, Memlet, SDFGState
from dace.transformation.interstate import IfExtraction


def _add_map_with_connectors(st: SDFGState, name: str, ndrange: Dict[str, str],
                             en_conn_bases: Collection[str] = None, ex_conn_bases: Collection[str] = None):
    en, ex = st.add_map(name, ndrange)
    if en_conn_bases:
        for c in en_conn_bases:
            en.add_in_connector(f"IN_{c}")
            en.add_out_connector(f"OUT_{c}")
    if ex_conn_bases:
        for c in ex_conn_bases:
            ex.add_in_connector(f"IN_{c}")
            ex.add_out_connector(f"OUT_{c}")
    return en, ex


def make_branched_sdfg_that_does_not_depend_on_loop_var():
    """
    Construct a simple SDFG that does not depend on symbols defined or updated in the outer state, e.g., loop variables.
    """
    # First prepare the map-body.
    subg = SDFG('body')
    subg.add_array('tmp', (1,), dace.float32)
    subg.add_symbol('outval', dace.float32)
    ifh = subg.add_state('if_head')
    if1 = subg.add_state('if_b1')
    if2 = subg.add_state('if_b2')
    ift = subg.add_state('if_tail')
    subg.add_edge(ifh, if1, InterstateEdge(condition='(flag)', assignments={'outval': 1}))
    subg.add_edge(ifh, if2, InterstateEdge(condition='(not flag)', assignments={'outval': 2}))
    subg.add_edge(if1, ift, InterstateEdge())
    subg.add_edge(if2, ift, InterstateEdge())
    t0 = ift.add_tasklet('copy', inputs={}, outputs={'__out'}, code='__out = outval')
    tmp = ift.add_access('tmp')
    ift.add_edge(t0, '__out', tmp, None, Memlet(expr='tmp[0]'))
    subg.fill_scope_connectors()

    # Then prepare the parent graph.
    g = SDFG('prog')
    g.add_array('A', (10,), dace.float32)
    g.add_symbol('flag', dace.bool)
    st0 = g.add_state('outer', is_start_block=True)
    en, ex = _add_map_with_connectors(st0, 'map', {'i': '0:10'}, [], ['A'])
    body = st0.add_nested_sdfg(subg, None, {}, {'tmp'}, symbol_mapping={'flag': 'flag'})
    A = st0.add_access('A')
    st0.add_nedge(en, body, Memlet())
    st0.add_edge(body, 'tmp', ex, 'IN_A', Memlet(expr='A[i]'))
    st0.add_edge(ex, 'OUT_A', A, None, Memlet(expr='A[0:10]'))
    g.fill_scope_connectors()

    return g


def make_branched_sdfg_that_has_intermediate_branchlike_structure():
    """
    Construct an SDFG that has this structure:
              initial_state
                /      \\
        state_1         state_2
            |               |
        state_3         state_5
               \\       /
                state_5
              /         \
        state_6         state_7
              \\        /
             terminal_state
    """
    # First prepare the map-body.
    subg = SDFG('body')
    subg.add_array('tmp', (1,), dace.float32)
    subg.add_symbol('outval', dace.float32)
    ifh = subg.add_state('if_head')
    if1 = subg.add_state('state_1')
    if3 = subg.add_state('state_2')
    if2 = subg.add_state('state_3')
    if4 = subg.add_state('state_4')
    if5 = subg.add_state('state_5')
    if6 = subg.add_state('state_6')
    if7 = subg.add_state('state_7')
    ift = subg.add_state('if_tail')
    subg.add_edge(ifh, if1, InterstateEdge(condition='(flag)', assignments={'outval': 1}))
    subg.add_edge(ifh, if2, InterstateEdge(condition='(not flag)', assignments={'outval': 2}))
    subg.add_edge(if1, if3, InterstateEdge())
    subg.add_edge(if3, if5, InterstateEdge())
    subg.add_edge(if2, if4, InterstateEdge())
    subg.add_edge(if4, if5, InterstateEdge())
    subg.add_edge(if5, if6, InterstateEdge())
    subg.add_edge(if5, if7, InterstateEdge())
    subg.add_edge(if6, ift, InterstateEdge())
    subg.add_edge(if7, ift, InterstateEdge())
    t0 = ift.add_tasklet('copy', inputs={}, outputs={'__out'}, code='__out = outval')
    tmp = ift.add_access('tmp')
    ift.add_edge(t0, '__out', tmp, None, Memlet(expr='tmp[0]'))
    subg.fill_scope_connectors()

    # Then prepare the parent graph.
    g = SDFG('prog')
    g.add_array('A', (10,), dace.float32)
    g.add_symbol('flag', dace.bool)
    st0 = g.add_state('outer', is_start_block=True)
    en, ex = _add_map_with_connectors(st0, 'map', {'i': '0:10'}, [], ['A'])
    body = st0.add_nested_sdfg(subg, None, {}, {'tmp'}, symbol_mapping={'flag': 'flag'})
    A = st0.add_access('A')
    st0.add_nedge(en, body, Memlet())
    st0.add_edge(body, 'tmp', ex, 'IN_A', Memlet(expr='A[i]'))
    st0.add_edge(ex, 'OUT_A', A, None, Memlet(expr='A[0:10]'))
    g.fill_scope_connectors()

    return g


def make_branched_sdfg_that_depends_on_loop_var():
    """
    Construct a simple SDFG that depends on symbols defined or updated in the outer state, e.g., loop variables.
    """
    # First prepare the map-body.
    subg = SDFG('body')
    subg.add_array('tmp', (1,), dace.float32)
    subg.add_symbol('outval', dace.float32)
    ifh = subg.add_state('if_head')
    if1 = subg.add_state('if_b1')
    if2 = subg.add_state('if_b2')
    ift = subg.add_state('if_tail')
    subg.add_edge(ifh, if1, InterstateEdge(condition='(i == 0)', assignments={'outval': 1}))
    subg.add_edge(ifh, if2, InterstateEdge(condition='(not (i == 0))', assignments={'outval': 2}))
    subg.add_edge(if1, ift, InterstateEdge())
    subg.add_edge(if2, ift, InterstateEdge())
    t0 = ift.add_tasklet('copy', inputs={}, outputs={'__out'}, code='__out = outval')
    tmp = ift.add_access('tmp')
    ift.add_edge(t0, '__out', tmp, None, Memlet(expr='tmp[0]'))
    subg.fill_scope_connectors()

    # Then prepare the parent graph.
    g = SDFG('prog')
    g.add_array('A', (10,), dace.float32)
    st0 = g.add_state('outer', is_start_block=True)
    en, ex = _add_map_with_connectors(st0, 'map', {'i': '0:10'}, [], ['A'])
    body = st0.add_nested_sdfg(subg, None, {}, {'tmp'})
    A = st0.add_access('A')
    st0.add_nedge(en, body, Memlet())
    st0.add_edge(body, 'tmp', ex, 'IN_A', Memlet(expr='A[i]'))
    st0.add_edge(ex, 'OUT_A', A, None, Memlet(expr='A[0:10]'))
    g.fill_scope_connectors()

    return g


def test_simple_application():
    origA = np.zeros((10,), np.float32)

    g = make_branched_sdfg_that_does_not_depend_on_loop_var()
    g.save(os.path.join('_dacegraphs', 'simple-0.sdfg'))
    g.validate()
    g.compile()

    # Get the expected values.
    wantA_1 = deepcopy(origA)
    wantA_2 = deepcopy(origA)
    g(A=wantA_1, flag=True)
    g(A=wantA_2, flag=False)

    # Before, the outer graph had only one nested SDFG.
    assert len(g.nodes()) == 1

    assert g.apply_transformations_repeated([IfExtraction]) == 1
    g.save(os.path.join('_dacegraphs', 'simple-1.sdfg'))
    g.validate()
    g.compile()

    # Get the values from transformed program.
    gotA_1 = deepcopy(origA)
    gotA_2 = deepcopy(origA)
    g(A=gotA_1, flag=True)
    g(A=gotA_2, flag=False)

    # But now, the outer graph have four: two copies of the original nested SDFGs and two for branch management.
    assert len(g.nodes()) == 4
    assert g.start_state.is_empty()

    # Verify numerically.
    assert all(np.equal(wantA_1, gotA_1))
    assert all(np.equal(wantA_2, gotA_2))


def test_extracts_even_with_intermediate_branchlike_structure():
    origA = np.zeros((10,), np.float32)

    g = make_branched_sdfg_that_has_intermediate_branchlike_structure()
    g.save(os.path.join('_dacegraphs', 'intermediate_branch-0.sdfg'))
    g.validate()
    g.compile()

    # Get the expected values.
    wantA_1 = deepcopy(origA)
    wantA_2 = deepcopy(origA)
    g(A=wantA_1, flag=True)
    g(A=wantA_2, flag=False)

    # Before, the outer graph had only one nested SDFG.
    assert len(g.nodes()) == 1

    assert g.apply_transformations_repeated([IfExtraction]) == 1
    g.save(os.path.join('_dacegraphs', 'intermediate_branch-1.sdfg'))

    # Get the values from transformed program.
    gotA_1 = deepcopy(origA)
    gotA_2 = deepcopy(origA)
    g(A=gotA_1, flag=True)
    g(A=gotA_2, flag=False)

    # But now, the outer graph have four: two copies of the original nested SDFGs and two for branch management.
    assert len(g.nodes()) == 4
    assert g.start_state.is_empty()

    # Verify numerically.
    assert all(np.equal(wantA_1, gotA_1))
    assert all(np.equal(wantA_2, gotA_2))


def test_no_extraction_due_to_dependency_on_loop_var():
    g = make_branched_sdfg_that_depends_on_loop_var()
    g.save(os.path.join('_dacegraphs', 'dependent-0.sdfg'))

    assert g.apply_transformations_repeated([IfExtraction]) == 0


if __name__ == '__main__':
    test_simple_application()
    test_no_extraction_due_to_dependency_on_loop_var()
