# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the map-expansion transformation. """

from dace.sdfg.utils import consolidate_edges
from typing import Dict, List
import copy
import dace
from dace import dtypes, subsets, symbolic
from dace.properties import EnumProperty, make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import OrderedMultiDiConnectorGraph
from dace.transformation import transformation as pm
from dace.sdfg.propagation import propagate_memlets_scope


@make_properties
class MapSymbolDefinition(pm.SingleStateTransformation):
    """
    Takes the pattern MapEntry -> NestedSDFG{State -symbol definitions-> State -> ...} -> MapExit
    and moves the symbol definitions in a multidimensional one-iteration nested map
    """

    map_entry = pm.PatternNode(nodes.MapEntry)
    nested_sdfg = pm.PatternNode(nodes.NestedSDFG)
    map_exit = pm.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False):
        nested_sdfg = self.nested_sdfg

        start_state = nested_sdfg.sdfg.start_state

        # the first state needs to be empty and used only for symbol definition
        if len(start_state.nodes()) != 0:
            return False

        if len(nested_sdfg.sdfg.out_edges(start_state)) != 1:
            return False
        
        return True

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        current_map = self.map_entry
        nested_sdfg = self.nested_sdfg
        current_exit = self.map_exit
        
        # get the symbol definitions
        e = nested_sdfg.sdfg.out_edges(nested_sdfg.sdfg.start_state)[0]
        params = list(e.data.assignments.keys())
        ranges = subsets.Range([(e.data.assignments[k], e.data.assignments[k], 1) for k in params])
        ranges.replace(nested_sdfg.symbol_mapping)
        new_map = nodes.Map(current_map.label + '_' + str(params), params, ranges, schedule=current_map.schedule)

        new_entry = nodes.MapEntry(new_map)
        new_exit = nodes.MapExit(new_map)
        
        graph.add_node(new_entry)
        graph.add_node(new_exit)
        
        used_connections = set()
        for nstate in nested_sdfg.sdfg.nodes():
            for node in nstate.nodes():
                if isinstance(node, nodes.AccessNode):
                    used_connections.add(node.data)
        
        for edge in list(graph.out_edges(current_map)):
            # NOTE: do not connect inputs to the nested sdfg if used only in the interstate edge! -> connect to map only
            if edge.dst_conn in used_connections:
                new_in_connector = 'IN' + edge.src_conn[3:]
                new_entry.add_out_connector(edge.src_conn)
                graph.add_edge(new_entry, edge.src_conn, nested_sdfg, edge.dst_conn, memlet=copy.deepcopy(edge.data))
            else:
                new_in_connector = edge.data.data
                nested_sdfg.remove_in_connector(edge.dst_conn)
            new_entry.add_in_connector(new_in_connector)
            graph.add_edge(current_map, edge.src_conn, new_entry, new_in_connector, memlet=copy.deepcopy(edge.data))
            graph.remove_edge(edge)

        for edge in list(graph.in_edges(current_exit)):
            new_out_connector = 'OUT' + edge.dst_conn[2:]
            new_exit.add_in_connector(edge.dst_conn)
            new_exit.add_out_connector(new_out_connector)
            graph.add_edge(nested_sdfg, edge.src_conn, new_exit, edge.dst_conn, memlet=copy.deepcopy(edge.data))
            graph.add_edge(new_exit, new_out_connector, edge.dst, edge.dst_conn, memlet=copy.deepcopy(edge.data))
            graph.remove_edge(edge)

        e.data.assignments = {}
        from dace.transformation.interstate import StateFusion
        xform = StateFusion()
        xform.first_state = nested_sdfg.sdfg.start_state
        xform.second_state = e.dst
        xform.apply(nested_sdfg.sdfg, None)
