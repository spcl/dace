import os
from copy import deepcopy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.const_assignment_fusion import ConstAssignmentMapFusion
from dace.transformation.interstate import StateFusionExtended

M = dace.symbol('M')
N = dace.symbol('N')


@dace.program
def assign_top_row(A: dace.float32[M, N]):
    for i in dace.map[0:N]:
        A[0, i] = 1


@dace.program
def assign_bottom_row(A: dace.float32[M, N]):
    for i in dace.map[0:N]:
        A[M - 1, i] = 1


@dace.program
def assign_left_col(A: dace.float32[M, N]):
    for i in dace.map[0:M]:
        A[i, 0] = 1


@dace.program
def assign_right_col(A: dace.float32[M, N]):
    for i in dace.map[0:M]:
        A[i, N - 1] = 1


def assign_bounary_sdfg():
    st0 = assign_top_row.to_sdfg(simplify=True, validate=True)
    st0.start_block.label = 'st0'

    st1 = assign_bottom_row.to_sdfg(simplify=True, validate=True)
    st1.start_block.label = 'st1'
    st0.add_edge(st0.start_state, st1.start_state, dace.InterstateEdge())

    st2 = assign_left_col.to_sdfg(simplify=True, validate=True)
    st2.start_block.label = 'st2'
    st0.add_edge(st1.start_state, st2.start_state, dace.InterstateEdge())

    st3 = assign_right_col.to_sdfg(simplify=True, validate=True)
    st3.start_block.label = 'st3'
    st0.add_edge(st2.start_state, st3.start_state, dace.InterstateEdge())

    return st0


def find_access_node_by_name(g, name):
    """ Finds the first data node by the given name"""
    return next((n, s) for n, s in g.all_nodes_recursive()
                if isinstance(n, nodes.AccessNode) and name == n.data)


def find_map_entry_by_name(g, name):
    """ Finds the first map entry node by the given name """
    return next((n, s) for n, s in g.all_nodes_recursive()
                if isinstance(n, nodes.MapEntry) and n.label.startswith(name))


def find_map_exit_by_name(g, name):
    """ Finds the first map entry node by the given name """
    return next((n, s) for n, s in g.all_nodes_recursive()
                if isinstance(n, nodes.MapExit) and n.label.startswith(name))


def test_within_state_fusion():
    A = np.random.uniform(size=(4, 5)).astype(np.float32)

    # Construct SDFG with the maps on separate states.
    g = assign_bounary_sdfg()
    g.save(os.path.join('_dacegraphs', 'simple-0.sdfg'))
    g.validate()
    actual_A = deepcopy(A)
    g(A=actual_A, M=4, N=5)

    # Fuse the two states so that the const-assignment-fusion is applicable.
    g.apply_transformations_repeated(StateFusionExtended, validate_all=True)
    g.save(os.path.join('_dacegraphs', 'simple-1.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'simple-2.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'simple-3.sdfg'))
    g.validate()

    g.apply_transformations(ConstAssignmentMapFusion)
    g.save(os.path.join('_dacegraphs', 'simple-4.sdfg'))
    g.validate()
    our_A = deepcopy(A)
    g(A=our_A, M=4, N=5)

    print(our_A)
    assert np.allclose(our_A, actual_A)


if __name__ == '__main__':
    test_within_state_fusion()
