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
import itertools
from dace import data


@registry.autoregister_params()
class RemoveUnusedStates(transformation.Transformation):
    """
    If some state only does writes to transients that are not used anymore, then
    entire state content is removed. The removal of state is a separate question, because
    it can be involved in complex state transitions with conditions.
    """

    unused_state = transformation.PatternNode(dace_sdfg.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                RemoveUnusedStates.unused_state,
            )
        ]

    @staticmethod
    def can_be_applied(sdfg: dace_sdfg.SDFG, candidate, expr_index, _sdfg, strict=False):

        unused_state: dace_state.SDFGState = sdfg.nodes()[candidate[RemoveUnusedStates.unused_state]]

        # we skip states that are already empty
        if unused_state.is_empty():
            return False

        # we consider only states with the following content AccessNode -> Something (Tasklet,NestedSDFG) -> AccessNode
        input_access_nodes: List[nodes.AccessNode] = unused_state.source_nodes()
        output_access_nodes: List[nodes.AccessNode] = unused_state.sink_nodes()

        for n in itertools.chain(input_access_nodes, output_access_nodes):
            if not isinstance(n, nodes.AccessNode):
                return False

        intermediate_node = None

        for n in unused_state.nodes():
            if n in input_access_nodes:
                continue
            if n in output_access_nodes:
                continue
            if intermediate_node is None:
                intermediate_node = n
            else:
                # there are more than one other nodes, transformation doesn't work for such pattern
                return False

        # pattern violation
        if intermediate_node is None:
            return False

        # we can remove state if it writes into transients that are not used by following states

        data_descs: List[str] = [n.data for n in output_access_nodes]

        # check they all transient
        for dd in data_descs:
            if not sdfg.arrays[dd].transient:
                # access node is not a transient
                return False

        # check they all are not used anymore
        for state_edge in sdfg.bfs_edges(unused_state):
            state_edge: dace_graph.Edge
            unchecked_state: dace_state.SDFGState = state_edge.dst
            for n in unchecked_state.nodes():
                if not isinstance(n, nodes.AccessNode):
                    continue
                n: nodes.AccessNode
                if n.data not in data_descs:
                    continue
                if n.has_reads(unchecked_state):
                    # we found read from node in later state, can't apply transformation
                    return False

        return True

    def apply(self, sdfg: dace_sdfg.SDFG):
        candidate = self.subgraph
        unused_state: dace_state.SDFGState = sdfg.nodes()[candidate[RemoveUnusedStates.unused_state]]

        all_nodes = unused_state.nodes()
        for n in all_nodes:
            unused_state.remove_node(n)
