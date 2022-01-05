# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.subgraph import MultiExpansion
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.subgraph import ReduceExpansion
import dace.transformation.subgraph.helpers as helpers

from dace.sdfg.graph import SubgraphView

import dace.libraries.standard as stdlib
from typing import Union, List


def expand_reduce(sdfg: dace.SDFG,
                  graph: dace.SDFGState,
                  subgraph: Union[SubgraphView, List[SubgraphView]] = None,
                  **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, list):
        subgraph = [subgraph]

    for sg in subgraph:
        reduce_nodes = []
        for node in sg.nodes():
            if isinstance(node, stdlib.Reduce):
                rexp = ReduceExpansion(sdfg, sdfg.sdfg_id, sdfg.node_id(graph),
                                       {ReduceExpansion.reduce: graph.node_id(node)}, 0)
                if not rexp.can_be_applied(graph, 0, sdfg):
                    print(f"WARNING: Cannot expand reduce node {node}:" "can_be_applied() failed.")
                    continue
                reduce_nodes.append(node)

        trafo_reduce = ReduceExpansion(sdfg, sdfg.sdfg_id, sdfg.node_id(graph), {}, 0)
        for (property, val) in kwargs.items():
            setattr(trafo_reduce, property, val)

        for reduce_node in reduce_nodes:
            trafo_reduce.expand(sdfg, graph, reduce_node)
            if isinstance(sg, SubgraphView):
                sg.nodes().remove(reduce_node)
                sg.nodes().append(trafo_reduce._reduce)
                sg.nodes().append(trafo_reduce._outer_entry)


def expand_maps(sdfg: dace.SDFG,
                graph: dace.SDFGState,
                subgraph: Union[SubgraphView, List[SubgraphView]] = None,
                **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, list):
        subgraph = [subgraph]

    trafo_expansion = MultiExpansion(subgraph[0])
    for (property, val) in kwargs.items():
        setattr(trafo_expansion, property, val)

    for sg in subgraph:
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, sg)
        trafo_expansion.expand(sdfg, graph, map_entries)


def fusion(sdfg: dace.SDFG, graph: dace.SDFGState, subgraph: Union[SubgraphView, List[SubgraphView]] = None, **kwargs):

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, list):
        subgraph = [subgraph]

    map_fusion = SubgraphFusion(subgraph[0])
    for (property, val) in kwargs.items():
        setattr(map_fusion, property, val)

    for sg in subgraph:
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, sg)
        # remove map_entries and their corresponding exits from the subgraph
        # already before applying transformation
        if isinstance(sg, SubgraphView):
            for map_entry in map_entries:
                sg.nodes().remove(map_entry)
                if graph.exit_node(map_entry) in sg.nodes():
                    sg.nodes().remove(graph.exit_node(map_entry))
        map_fusion.fuse(sdfg, graph, map_entries)
        if isinstance(sg, SubgraphView):
            sg.nodes().append(map_fusion._global_map_entry)
