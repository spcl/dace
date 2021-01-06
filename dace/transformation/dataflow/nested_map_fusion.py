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


@registry.autoregister_params(singlestate=True)
class NestedMapFusion(transformation.Transformation):

    map_exit = transformation.PatternNode(nodes.MapExit)
    map_entry = transformation.PatternNode(nodes.MapEntry)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestedMapFusion.map_exit,
                NestedMapFusion.map_entry,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        map_exit = state.nodes()[candidate[NestedMapFusion.map_exit]]
        map_entry = state.nodes()[candidate[NestedMapFusion.map_entry]]

        map1: Map = map_exit.map
        map2: Map = map_entry.map

        # This could be improved by reusing find_permutation() from map_fusion.py.
        if map1.range != map2.range:
            return False

        # This could be replaced by renaming of matching parameters.
        # The problem is that such replacement can be not unique.
        if map1.params != map2.params:
            return False

        return True

    @staticmethod
    def match_to_str(state, candidate):
        map_exit: nodes.MapExit = state.nodes()[candidate[NestedMapFusion.map_exit]]
        map_entry: nodes.MapEntry = state.nodes()[candidate[NestedMapFusion.map_entry]]

        return f"{map_exit.map.label} : {str(map_exit.map.params)} -> {map_entry.map.label} : {str(map_entry.map.params)}"

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph

        map_exit: nodes.MapExit = state.nodes()[candidate[NestedMapFusion.map_exit]]
        map_entry: nodes.MapEntry = state.nodes()[candidate[NestedMapFusion.map_entry]]

        barrier: Barrier = Barrier(name='barrier')

        exit_edges = state.in_edges(map_exit)
        entry_edges = state.out_edges(map_entry)

        map1: Map = map_exit.map
        map_exit2: nodes.MapExit = state.exit_node(map_entry)

        map1_exit_nodes = [e.src for e in exit_edges]
        map2_entry_nodes = [e.dst for e in entry_edges]

        map_exit2.map = map1

        for e in exit_edges:
            state.remove_edge(e)

        for e in entry_edges:
            state.remove_edge(e)

        state.remove_node(map_exit)
        state.remove_node(map_entry)

        for n in map1_exit_nodes:
            state.add_nedge(n, barrier, memlet.Memlet())

        for n in map2_entry_nodes:
            state.add_nedge(barrier, n, memlet.Memlet())
