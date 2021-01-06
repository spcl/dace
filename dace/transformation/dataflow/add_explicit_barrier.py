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


@registry.autoregister_params(singlestate=True)
class AddExplicitBarrier(transformation.Transformation):
    """
    Takes "AccessNode--WCR->AccessNode->MapExit" and adds "AccessNode--WCR->AccessNode->BarrierLibraryNode->MapExit"
    """

    access_node1 = transformation.PatternNode(nodes.AccessNode)
    access_node2 = transformation.PatternNode(nodes.AccessNode)
    map_exit = transformation.PatternNode(nodes.MapExit)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                AddExplicitBarrier.access_node1,
                AddExplicitBarrier.access_node2,
                AddExplicitBarrier.map_exit,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        access_node1 = state.nodes()[candidate[AddExplicitBarrier.access_node1]]
        access_node2 = state.nodes()[candidate[AddExplicitBarrier.access_node2]]
        map_exit = state.nodes()[candidate[AddExplicitBarrier.map_exit]]

        # edge between access nodes has WCR
        edges12 = state.edges_between(access_node1, access_node2)

        if len(edges12) != 1:
            return False
        edge12: dace_graph.MultiConnectorEdge = edges12[0]
        edge12_data: memlet.Memlet = edge12.data

        if edge12_data.wcr is None:
            return False

        # edge between access node 2 and MapExit is empty memlet
        edges2e = state.edges_between(access_node2, map_exit)

        if len(edges2e) != 1:
            return False
        edge2e: dace_graph.MultiConnectorEdge = edges2e[0]
        edge2e_data: memlet.Memlet = edge2e.data

        if not edge2e_data.is_empty():
            return False

        return True

    @staticmethod
    def match_to_str(state, candidate):
        access_node1: nodes.AccessNode = state.nodes()[candidate[AddExplicitBarrier.access_node1]]
        access_node2: nodes.AccessNode = state.nodes()[candidate[AddExplicitBarrier.access_node2]]
        map_exit: nodes.MapExit = state.nodes()[candidate[AddExplicitBarrier.map_exit]]

        return f"{access_node1.label} -> {access_node2.label} -> {map_exit.map.label} : {str(map_exit.map.params)}"

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph

        access_node1: nodes.AccessNode = state.nodes()[candidate[AddExplicitBarrier.access_node1]]
        access_node2: nodes.AccessNode = state.nodes()[candidate[AddExplicitBarrier.access_node2]]
        map_exit: nodes.MapExit = state.nodes()[candidate[AddExplicitBarrier.map_exit]]

        barrier: Barrier = Barrier(name='barrier')

        edges = state.edges_between(access_node2, map_exit)
        for e in edges:
            state.remove_edge(e)

        state.add_nedge(access_node2, barrier, memlet.Memlet())
        state.add_nedge(barrier, map_exit, memlet.Memlet())
