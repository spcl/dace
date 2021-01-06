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
from dace.transformation.helpers import nest_state_subgraph
from dace.sdfg.graph import SubgraphView


@registry.autoregister_params(singlestate=True)
class NestMaps(transformation.Transformation):

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestMaps.map_entry,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        map_entry = state.nodes()[candidate[NestMaps.map_entry]]
        map_exit = state.exit_node(map_entry)

        # transformation can be applied to maps that are not already nested

        in_edges = state.in_edges(map_entry)
        out_edges = state.out_edges(map_exit)

        e: dace_graph.MultiConnectorEdge

        # check if input access nodes have incoming edges
        for e in in_edges:
            if state.in_edges(e.src):
                return True

        # check if output access nodes have outgoing edges
        for e in out_edges:
            if state.out_edges(e.dst):
                return True

        return False # already nested

    @staticmethod
    def match_to_str(state: dace_state.SDFGState, candidate):
        map_entry: nodes.MapEntry = state.nodes()[candidate[NestMaps.map_entry]]

        return f"{map_entry.map.label} : {str(map_entry.map.params)}"

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        map_entry: nodes.MapEntry = state.nodes()[candidate[NestMaps.map_entry]]

        nest_state_subgraph(sdfg=sdfg, state=state, subgraph=state.scope_subgraph(map_entry))

