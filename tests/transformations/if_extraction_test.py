# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import os
from copy import deepcopy

import numpy as np

import dace
from dace import SDFG, InterstateEdge, Memlet
from dace.transformation.interstate import IfExtraction

N = dace.symbol('N', dtype=dace.int32)


def make_simple_branched_sdfg():
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
    ift.add_memlet_path(t0, tmp, src_conn='__out', memlet=Memlet(expr='tmp[0]'))
    subg.fill_scope_connectors()

    g = SDFG('prog')
    g.add_array('A', (10,), dace.float32)
    g.add_symbol('flag', dace.bool)
    st0 = g.add_state('outer', is_start_block=True)
    en, ex = st0.add_map('map', {'i': '0:10'})
    body = st0.add_nested_sdfg(subg, None, {}, {'tmp'}, symbol_mapping={'flag': 'flag'})
    A = st0.add_access('A')
    st0.add_memlet_path(en, body, memlet=Memlet())
    st0.add_memlet_path(body, ex, src_conn='tmp', dst_conn='IN_A', memlet=Memlet(expr='A[i]'))
    st0.add_memlet_path(ex, A, src_conn='OUT_A', memlet=Memlet(expr='A[0:10]'))
    g.fill_scope_connectors()

    return g


@dace.program
def dependant_application(flag: dace.bool, arr: dace.float32[N]):
    for i in dace.map[0:N]:
        if i == 0:
            outval = 1
        else:
            outval = 2
        arr[i] = outval


def test_simple_application():
    origA = np.zeros((10,), np.float32)

    g = make_simple_branched_sdfg()
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


def test_fails_due_to_dependency():
    sdfg = dependant_application.to_sdfg(simplify=True)

    assert sdfg.apply_transformations_repeated([IfExtraction]) == 0


if __name__ == '__main__':
    test_simple_application()
    test_fails_due_to_dependency()
