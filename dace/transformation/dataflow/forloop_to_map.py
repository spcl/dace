# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the ForLoopToMap transformation. """

import dace

from dace.sdfg import SDFG, SDFGState, utils as sdutil
from dace.transformation import transformation


class ForLoopToMap(transformation.SingleStateTransformation):
    """ Implements the ForLoopToMap transformation.
    
        The transformation simply converts the ForLoop, ForLoopEntry, and ForLoopExit nodes into Map, MapEntry, and
        MapExit nodes.
    """

    forloop_entry = transformation.PatternNode(dace.nodes.ForLoopEntry)

    @staticmethod
    def annotates_memlets():
        return True
    
    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.forloop_entry)]
    
    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> dace.nodes.MapEntry:
        """ Applies the transformation and returns the MapEntry of the new Map. """

        # Retrieve ForLoopEntry and ForLoopExit nodes.
        forloop_entry = self.forloop_entry
        forloop_exit = graph.exit_node(forloop_entry)
        forloop = forloop_entry.map

        # Create new map
        map_node = dace.nodes.Map(forloop.label, forloop.params, forloop.range)
        map_entry = dace.nodes.MapEntry(map_node)
        map_entry.in_connectors = forloop_entry.in_connectors
        map_entry.out_connectors = forloop_entry.out_connectors
        map_exit = dace.nodes.MapExit(map_node)
        map_exit.in_connectors = forloop_exit.in_connectors
        map_exit.out_connectors = forloop_exit.out_connectors

        # Add map to graph
        graph.add_node(map_entry)
        graph.add_node(map_exit)

        # Redirect edges
        sdutil.change_edge_src(graph, forloop_entry, map_entry)
        sdutil.change_edge_dest(graph, forloop_entry, map_entry)
        sdutil.change_edge_src(graph, forloop_exit, map_exit)
        sdutil.change_edge_dest(graph, forloop_exit, map_exit)

        # Remove old nodes
        graph.remove_nodes_from((forloop_entry, forloop_exit))

        return map_entry
