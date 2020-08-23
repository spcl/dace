import dace
import numpy as np
import sys

from dace.transformation.subgraph import ReduceExpansion, SubgraphFusion, MultiExpansion
import dace.transformation.subgraph.helpers as helpers

import dace.dtypes as dtypes
from dace.sdfg.graph import SubgraphView
import dace.libraries.standard as stdlib
import dace.sdfg.nodes as nodes
from typing import Union, List


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


dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity=0)

    tmp_out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


sdfg = softmax.to_sdfg()
sdfg.apply_strict_transformations()
H.set(10)
B.set(10)
SN.set(20)
SM.set(20)
A = np.ndarray((H.get(), B.get(), SN.get(), SM.get()), dtype=np.float32)


def get_partition(sdfg, graph):
    subgraph1 = SubgraphView(graph, [])
    subgraph2 = SubgraphView(graph, [])

    cnt1 = 0
    for node in dace.sdfg.utils.dfs_topological_sort(graph):
        if isinstance(node, stdlib.nodes.reduce.Reduce):
            if cnt1 < 2:
                subgraph1._subgraph_nodes.append(node)
                cnt1 += 1
            else:
                subgraph2._subgraph_nodes.append(node)

        if isinstance(node, nodes.MapEntry):
            if cnt1 < 2:
                subgraph1._subgraph_nodes.append(node)
                cnt1 += 1
            else:
                subgraph2._subgraph_nodes.append(node)

    return [subgraph1, subgraph2]


def test_pipeline():

    X_in = np.random.rand(H.get(), B.get(), SN.get(),
                          SM.get()).astype(np.float32)

    csdfg = sdfg.compile()
    res1 = csdfg(X_in=X_in, H=H, B=B, SN=SN, SM=SM)

    subgraph = get_partition(sdfg, sdfg.nodes()[0])
    expand_reduce(sdfg, sdfg.nodes()[0], subgraph)
    expand_maps(sdfg, sdfg.nodes()[0], subgraph)
    fusion(sdfg, sdfg.nodes()[0], subgraph)

    csdfg = sdfg.compile()
    res2 = csdfg(X_in=X_in, H=H, B=B, SN=SN, SM=SM)

    assert np.allclose(res1, res2)
    print("PASS")
    return


if __name__ == "__main__":
    test_pipeline()
