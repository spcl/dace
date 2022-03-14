# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement subgraph fusion
"""
import dace

from dace.transformation import transformation
from dace.properties import make_properties, Property
from dace.transformation.subgraph import helpers

from collections import OrderedDict

from dace.transformation.subgraph import OnTheFlyMapFusion

@make_properties
class MapFusion(transformation.SubgraphTransformation):

    def can_be_applied(self, *args, **kwargs) -> bool:
        return True

    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG):
        state_subgraph = dace.sdfg.state.StateSubgraphView(state, map(state.node, self.subgraph))
        map_entries = helpers.get_outermost_scope_maps(sdfg, state, state_subgraph)
        children_dict, parent_dict = self.topology(state, map_entries)

        fuse_counter = 0
        queue = [me for me in map_entries if me not in children_dict]
        while len(queue) > 0:
            child = queue.pop(0)
            if child not in parent_dict:
                continue

            parents = parent_dict[child]
            while len(parents) > 0:
                parent = parents.pop(0)
                fusion = OnTheFlyMapFusion(state, sdfg_id=sdfg.sdfg_id, state_id=sdfg.node_id(state))
                if fusion.can_be_applied(state, sdfg, parent, child):
                    fusion.apply(state, sdfg, parent, child)
                    fuse_counter += 1 
                    break
                else:
                    queue.append(parent)

            queue.extend(parents)

            subgraph = []
            for node_id in self.subgraph:
                try:
                    state.node(node_id)
                    subgraph.append(node_id)
                except:
                    pass

            self.subgraph = subgraph
            state_subgraph = dace.sdfg.state.StateSubgraphView(state, map(state.node, self.subgraph))
            map_entries = helpers.get_outermost_scope_maps(sdfg, state, state_subgraph)
            children_dict, parent_dict = self.topology(state, map_entries)

        return fuse_counter

    def topology(self, state, map_entries):
        children_dict = OrderedDict()
        parent_dict = OrderedDict()
        for map_entry in map_entries:
            map_exit = state.exit_node(map_entry)
            for e in state.out_edges(map_exit):
                if isinstance(e.dst, dace.nodes.AccessNode):
                    for oe in state.out_edges(e.dst):
                        if oe.dst in map_entries:
                            other_entry = oe.dst

                            if map_entry not in children_dict:
                                children_dict[map_entry] = []
                            children_dict[map_entry].append(other_entry)

                            if other_entry not in parent_dict:
                                parent_dict[other_entry] = []

                            parent_dict[other_entry].append(map_entry)

        return children_dict, parent_dict
