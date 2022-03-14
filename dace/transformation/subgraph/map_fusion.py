# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement subgraph fusion
"""
import dace

from dace.sdfg.graph import SubgraphView
from dace.transformation import transformation
from dace.properties import make_properties, Property
from dace.transformation.subgraph import helpers

from collections import OrderedDict

from dace.transformation.subgraph import OnTheFlyMapFusion

@make_properties
class MapFusion(transformation.SubgraphTransformation):

    debug = Property(desc="Show debug info", dtype=bool, default=False)

    @staticmethod
    def can_be_applied(sdfg: dace.SDFG, state: dace.SDFGState) -> bool:
        return True

    def apply(self, sdfg, state):
        graph = state._graph

        converged = False
        while not converged:
            state_subgraph = SubgraphView(graph, state.nodes())
            map_entries = helpers.get_outermost_scope_maps(sdfg, graph, state_subgraph)

            children_dict = OrderedDict()
            parent_dict = OrderedDict()
            for map_entry in map_entries:
                map_exit = graph.exit_node(map_entry)
                for e in graph.out_edges(map_exit):
                    if isinstance(e.dst, dace.nodes.AccessNode):
                        for oe in graph.out_edges(e.dst):
                            if oe.dst in map_entries:
                                other_entry = oe.dst

                                if map_entry not in children_dict:
                                    children_dict[map_entry] = []
                                children_dict[map_entry].append(other_entry)

                                if other_entry not in parent_dict:
                                    parent_dict[other_entry] = []

                                parent_dict[other_entry].append(map_entry)

            applied = False
            leaves = [me for me in map_entries if me not in children_dict]
            while len(leaves) > 0 and not applied:
                child = leaves.pop(0)
                if child not in parent_dict:
                    continue

                parents = parent_dict[child]
                while len(parents) > 0 and not applied:
                    parent = parents.pop(0)
                    fusion = OnTheFlyMapFusion(state, sdfg_id=sdfg.sdfg_id, state_id=sdfg.node_id(state))
                    if OnTheFlyMapFusion.can_be_applied(state, parent, child):
                        fusion.apply(sdfg, parent, child)
                        applied = True
                        break
            
            converged = not applied
