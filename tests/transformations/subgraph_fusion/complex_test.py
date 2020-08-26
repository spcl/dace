import dace
from dace.transformation.subgraph import MultiExpansion
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.subgraph import ReduceExpansion
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView
import dace.libraries.standard as stdlib
from typing import Union, List

import sys


def expand_reduce(sdfg: dace.SDFG,
                  graph: dace.SDFGState,
                  subgraph: Union[SubgraphView, List[SubgraphView]] = None,
                  **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    for sg in subgraph:
        reduce_nodes = []
        for node in sg.nodes():
            if isinstance(node, stdlib.Reduce):
                if not ReduceExpansion.can_be_applied(
                        graph=graph,
                        candidate={
                            ReduceExpansion._reduce: graph.node_id(node)
                        },
                        expr_index=0,
                        sdfg=sdfg):
                    print(f"WARNING: Cannot expand reduce node {node}: \
                            Can_be_applied() failed.")
                    continue
                reduce_nodes.append(node)

        trafo_reduce = ReduceExpansion(0, 0, {}, 0)
        for (property, val) in kwargs.items():
            setattr(trafo_reduce, property, val)

        for reduce_node in reduce_nodes:
            trafo_reduce.expand(sdfg, graph, reduce_node)
            if isinstance(sg, SubgraphView):
                sg.nodes().remove(reduce_node)
                sg.nodes().append(trafo_reduce._new_reduce)
                sg.nodes().append(trafo_reduce._outer_entry)


def expand_maps(sdfg: dace.SDFG,
                graph: dace.SDFGState,
                subgraph: Union[SubgraphView, List[SubgraphView]] = None,
                **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    trafo_expansion = MultiExpansion()
    for (property, val) in kwargs.items():
        setattr(trafo_expansion, property, val)

    for sg in subgraph:
        map_entries = helpers.get_highest_scope_maps(sdfg, graph, sg)
        trafo_expansion.expand(sdfg, graph, map_entries)


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

A = np.random.rand(N.get()).astype(np.float64)
B = np.random.rand(M.get()).astype(np.float64)
C = np.random.rand(O.get()).astype(np.float64)
out1 = np.ndarray((N.get(), M.get()), np.float64)
out2 = np.ndarray((1), np.float64)
out3 = np.ndarray((N.get(), M.get(), O.get()), np.float64)


@dace.program
def test_program(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O], \
         out1: dace.float64[N,M], out2: dace.float64[1], out3: dace.float64[N,M,O]):

    tmp1 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp2 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp3 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp4 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp5 = np.ndarray([N, M, O], dtype=dace.float64)

    t1 = np.ndarray([N, M], dtype=dace.float64)
    t2 = np.ndarray([N, M], dtype=dace.float64)
    t3 = np.ndarray([N, M], dtype=dace.float64)

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        tp = np.ndarray([1], dtype=dace.float64)
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tp

            out = in1 + in2 + in3

        with dace.tasklet:
            in1 << tp
            out >> tmp1[i, j, k]

            out = in1 + 42

    dace.reduce(lambda a, b: a + b, tmp1, t1, axis=2, identity=0)

    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            out >> t2[i, j]
            out = in1 + in2 + 42

    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << t2[i, j]
            in2 << A[i]
            out >> out1[i, j]

            out = in1 * in1 * in2 + in2

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << t1[i, j]
            in2 << t2[i, j]
            in3 << C[k]
            out >> tmp3[i, j, k]

            out = in1 + in2 + in3

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << tmp3[i, j, k]
            in2 << tmp1[i, j, k]
            out >> out3[i, j, k]

            out = in1 + in2

    @dace.tasklet
    def fun():
        in1 << tmp3[0, 0, 0]
        out >> out2

        out = in1 * 42


def test_qualitatively(sdfg, graph):
    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)
    fusion(sdfg, graph)
    sdfg.validate()
    print("PASS")


def test_quantitatively(sdfg, graph):

    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(M.get()).astype(np.float64)
    C = np.random.rand(O.get()).astype(np.float64)
    out1_base = np.ndarray((N.get(), M.get()), np.float64)
    out2_base = np.ndarray((1), np.float64)
    out3_base = np.ndarray((N.get(), M.get(), O.get()), np.float64)
    out1 = np.ndarray((N.get(), M.get()), np.float64)
    out2 = np.ndarray((1), np.float64)
    out3 = np.ndarray((N.get(), M.get(), O.get()), np.float64)
    csdfg = sdfg.compile()
    csdfg(A=A,
          B=B,
          C=C,
          out1=out1_base,
          out2=out2_base,
          out3=out3_base,
          N=N,
          M=M,
          O=O)

    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)
    sgf = SubgraphFusion()
    matcher = sgf.match(sdfg, SubgraphView(graph, [node for node in graph.nodes()]))
    assert matcher == True
    fusion(sdfg, graph)
    sdfg.validate()
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, out1=out1, out2=out2, out3=out3, N=N, M=M, O=O)

    assert np.allclose(out1, out1_base)
    assert np.allclose(out2, out2_base)
    assert np.allclose(out3, out3_base)
    print('PASS')

def test_complex():
    sdfg = test_program.to_sdfg()
    sdfg.apply_strict_transformations()
    test_quantitatively(sdfg, sdfg.nodes()[0])

if __name__ == "__main__":
    test_complex()
