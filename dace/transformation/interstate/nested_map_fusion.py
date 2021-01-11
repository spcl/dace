# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy as dcpy
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
import networkx as nx
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace import memlet
from dace.sdfg import graph as dace_graph
from dace.libraries.standard.nodes.barrier import Barrier
from dace.sdfg.nodes import Map
from dace.transformation.helpers import nest_state_subgraph


@registry.autoregister_params()
class NestedMapFusion(transformation.Transformation):

    first_state = transformation.PatternNode(dace_sdfg.SDFGState)
    second_state = transformation.PatternNode(dace_sdfg.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestedMapFusion.first_state,
                NestedMapFusion.second_state
            )
        ]

    @staticmethod
    def can_be_applied(sdfg: dace_sdfg.SDFG, candidate, expr_index, _sdfg, strict=False):

        first_state: dace_state.SDFGState = sdfg.nodes()[candidate[NestedMapFusion.first_state]]
        second_state: dace_state.SDFGState = sdfg.nodes()[candidate[NestedMapFusion.second_state]]

        sources1 = first_state.source_nodes()
        sinks1 = first_state.sink_nodes()

        sources2 = second_state.source_nodes()
        sinks2 = second_state.sink_nodes()

        # we expect top level map scope enclosing everything else in both states

        if len(sources1) != 1 or len(sinks1) != 1:
            return False

        if len(sources2) != 1 or len(sinks2) != 1:
            return False

        map_entry1 = sources1[0]
        if not isinstance(map_entry1, nodes.MapEntry):
            return False

        map_exit1 = sinks1[0]
        if not isinstance(map_exit1, nodes.MapExit):
            return False

        map_entry2 = sources2[0]
        if not isinstance(map_entry2, nodes.MapEntry):
            return False

        map_exit2 = sinks2[0]
        if not isinstance(map_exit2, nodes.MapExit):
            return False

        # now we check that both maps have same ranges

        map1: Map = map_entry1.map
        map2: Map = map_entry2.map

        # This could be improved by reusing find_permutation() from map_fusion.py.
        if map1.range != map2.range:
            return False

        return True

    @staticmethod
    def match_to_str(sdfg, candidate):
        first_state = sdfg.nodes()[candidate[NestedMapFusion.first_state]]
        second_state = sdfg.nodes()[candidate[NestedMapFusion.second_state]]

        return " -> ".join(state.label for state in [first_state, second_state])

    def apply(self, sdfg: dace_sdfg.SDFG):
        first_state: dace_state.SDFGState = sdfg.nodes()[self.subgraph[NestedMapFusion.first_state]]
        second_state: dace_state.SDFGState = sdfg.nodes()[self.subgraph[NestedMapFusion.second_state]]

        sources1 = first_state.source_nodes()
        sinks1 = first_state.sink_nodes()

        sources2 = second_state.source_nodes()
        sinks2 = second_state.sink_nodes()

        map_entry1: nodes.MapEntry = sources1[0]
        map_exit1: nodes.MapExit = sinks1[0]

        map_entry2: nodes.MapEntry = sources2[0]
        map_exit2: nodes.MapExit = sinks2[0]

        nodes1 = first_state.all_nodes_between(map_entry1, map_exit1)
        nodes2 = second_state.all_nodes_between(map_entry2, map_exit2)

        nest_state_subgraph(sdfg, first_state, dace_graph.SubgraphView(first_state, nodes1))
        nest_state_subgraph(sdfg, second_state, dace_graph.SubgraphView(second_state, nodes2))

        # state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        # candidate = self.subgraph
        #
        # map_exit: nodes.MapExit = state.nodes()[candidate[NestedMapFusion.map_exit]]
        # map_entry: nodes.MapEntry = state.nodes()[candidate[NestedMapFusion.map_entry]]
        #
        # barrier: Barrier = Barrier(name='barrier')
        #
        # exit_edges = state.in_edges(map_exit)
        # entry_edges = state.out_edges(map_entry)
        #
        # map1: Map = map_exit.map
        # map_exit2: nodes.MapExit = state.exit_node(map_entry)
        #
        # map1_exit_nodes = [e.src for e in exit_edges]
        # map2_entry_nodes = [e.dst for e in entry_edges]
        #
        # map_exit2.map = map1
        #
        # for e in exit_edges:
        #     state.remove_edge(e)
        #
        # for e in entry_edges:
        #     state.remove_edge(e)
        #
        # state.remove_node(map_exit)
        # state.remove_node(map_entry)
        #
        # for n in map1_exit_nodes:
        #     state.add_nedge(n, barrier, memlet.Memlet())
        #
        # for n in map2_entry_nodes:
        #     state.add_nedge(barrier, n, memlet.Memlet())
