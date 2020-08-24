import dace
from dace.transformation.subgraph import MultiExpansion
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.subgraph import ReduceExpansion
import dace.sdfg.utils as utils
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import dace.subsets as subsets
import numpy as np

import itertools

from dace.sdfg.graph import SubgraphView
from typing import Union, List

import sys


def fusion(sdfg: dace.SDFG,
           graph: dace.SDFGState,
           subgraph: Union[SubgraphView, List[SubgraphView]] = None,
           **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    map_fusion = SubgraphFusion()
    for (property, val) in kwargs.items():
        setattr(map_fusion, property, val)

    for sg in subgraph:
        map_entries = helpers.get_highest_scope_maps(sdfg, graph, sg)
        # remove map_entries and their corresponding exits from the subgraph
        # already before applying transformation
        if isinstance(sg, SubgraphView):
            for map_entry in map_entries:
                sg.nodes().remove(map_entry)
                if graph.exit_node(map_entry) in sg.nodes():
                    sg.nodes().remove(graph.exit_node(map_entry))
        print(f"Subgraph Fusion on map entries {map_entries}")
        map_fusion.fuse(sdfg, graph, map_entries)
        if isinstance(sg, SubgraphView):
            sg.nodes().append(map_fusion._global_map_entry)


N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)

A = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
B = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
C = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
out1 = np.ndarray((N.get(), M.get(), O.get()), np.float64)


@dace.program
def test_program(A: dace.float64[N, M, O], B: dace.float64[N, M, O],
         C: dace.float64[N, M, O]):
    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i, j, 0]
            in2 << B[i, j, 0]
            out >> C[i, j, 0]

            out = in1 + in2

    for i, j in dace.map[0:N, 0:M]:
        for z in range(1, O):
            with dace.tasklet:
                in1 << A[i, j, z]
                in2 << B[i, j, z]
                in3 << C[i, j, 0]
                out >> C[i, j, z]

                out = 2 * in1 + 2 * in2 + in3


@dace.program
def helper_sdfg(AA: dace.float64[O], BB: dace.float64[O], CC: dace.float64[O]):
    for z in range(1, 0):
        with dace.tasklet:
            in1 << AA[z]
            in2 << BB[z]
            in3 << CC[0]
            out >> CC[z]

            out = 2 * in1 + 2 * in2 + in3


def test_qualitatively(sdfg, graph):
    fusion(sdfg, graph)
    sdfg.validate()
    print("PASS")


def fix_sdfg(sdfg, graph):
    # fix sdfg as for now the SDFG gets parsed wrongly
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nested_original = node
            for edge in itertools.chain(graph.in_edges(node),
                                        graph.out_edges(node)):
                for e in graph.memlet_tree(edge):
                    if 'z' in e.data.subset.free_symbols:
                        new_subset = str(e.data.subset)
                        new_subset = new_subset.replace('z', '0:O')
                        e.data.subset = subsets.Range.from_string(new_subset)

    # next up replace sdfg
    inner_sdfg = helper_sdfg.to_sdfg()
    nnode = graph.add_nested_sdfg(inner_sdfg, sdfg, {'AA', 'BB', 'CC'}, {'CC'})
    # redirect edges
    connectors = []
    for e in graph.in_edges(nested_original):
        connectors.append(e.dst_conn)
    connectors.sort()

    for e in graph.in_edges(nested_original):
        if e.dst_conn == connectors[0]:
            graph.add_edge(e.src, e.src_conn, e.dst, 'AA', e.data)
            graph.remove_edge(e)
        if e.dst_conn == connectors[1]:
            graph.add_edge(e.src, e.src_conn, e.dst, 'BB', e.data)
            graph.remove_edge(e)
        if e.dst_conn == connectors[2]:
            graph.add_edge(e.src, e.src_conn, e.dst, 'CC', e.data)
            graph.remove_edge(e)
    e = graph.out_edges(nested_original)[0]
    graph.add_edge(e.src, 'CC', e.dst, e.dst_conn, e.data)
    graph.remove_edge(e)

    utils.change_edge_dest(graph, nested_original, nnode)
    utils.change_edge_src(graph, nested_original, nnode)
    graph.remove_node(nested_original)
    sdfg.validate()


def test_quantitatively(sdfg, graph):
    A = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
    B = np.random.rand(N.get(), M.get(), O.get()).astype(np.float64)
    C1 = np.zeros([N.get(), M.get(), O.get()], dtype=np.float64)
    C2 = np.zeros([N.get(), M.get(), O.get()], dtype=np.float64)

    sdfg.validate()
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, N=N, M=M, O=O)

    fusion(sdfg, graph)
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, N=N, M=M, O=O)

    assert np.allclose(C1, C2)
    print('PASS')

def test_invariant_dim():
    sdfg = test_program.to_sdfg()
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    fix_sdfg(sdfg, graph)
    test_quantitatively(sdfg, graph)

if __name__ == '__main__':
    test_invariant_dim()
