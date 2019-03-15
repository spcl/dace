""" Contains classes that implement the map-expansion transformation. """

import copy
from typing import Dict
import dace
from dace import types, subsets, symbolic
from dace.graph import nodes, nxutil
from dace.graph.graph import OrderedMultiDiConnectorGraph
from dace.transformation import pattern_matching as pm


class MapExpansion(pm.Transformation):
    """ Implements the map-expansion pattern.

        Map-expansion takes an N-dimensional map and expands it to N 
        unidimensional maps.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MapExpansion._map_entry)]

    @staticmethod
    def can_be_applied(graph: dace.graph.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[dace.graph.nodes.Node, int],
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):
        # A candidate subgraph matches the map-expansion pattern when it includes
        # a N-dimensional map, with N greater than one.
        map_entry = graph.nodes()[candidate[MapExpansion._map_entry]]
        return map_entry.map.get_param_num() > 1

    @staticmethod
    def match_to_str(graph: dace.graph.graph.OrderedMultiDiConnectorGraph,
                     candidate: Dict[dace.graph.nodes.Node, int]):
        map_entry = graph.nodes()[candidate[MapExpansion._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg: dace.SDFG):
        # Extract the map and its entry and exit nodes.
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[MapExpansion._map_entry]]
        map_exit = graph.exit_nodes(map_entry)[0]
        current_map = map_entry.map

        # Create new maps
        maps = [
            nodes.Map(
                current_map.label + '_' + str(param), [param],
                subsets.Range([param_range]),
                schedule=types.ScheduleType.Sequential) for param, param_range
            in zip(current_map.params, current_map.range)
        ]
        maps[0]._schedule = types.ScheduleType.Default

        # Create new map entries
        entries = [nodes.MapEntry(new_map) for new_map in maps]
        entries[0].in_connectors = map_entry.in_connectors
        entries[0].out_connectors = map_entry.out_connectors
        num_entry_out_edges = len(graph.out_edges(map_entry))
        for i in range(1, len(entries)):
            entries[i].in_connectors = set(
                'IN_' + str(i + 1) for i in range(num_entry_out_edges))
            entries[i].out_connectors = set(
                'OUT_' + str(i + 1) for i in range(num_entry_out_edges))

        # Create new map exits
        exits = [nodes.MapExit(new_map) for new_map in maps]
        exits.reverse()
        exits[-1].in_connectors = map_exit.in_connectors
        exits[-1].out_connectors = map_exit.out_connectors
        num_entry_out_edges = len(graph.out_edges(map_exit))
        for i in range(0, len(exits) - 1):
            exits[i].in_connectors = set(
                'IN_' + str(i + 1) for i in range(num_entry_out_edges))
            exits[i].out_connectors = set(
                'OUT_' + str(i + 1) for i in range(num_entry_out_edges))

        # Add new nodes to state
        graph.add_nodes_from(entries)
        graph.add_nodes_from(exits)

        # Redirect edges to new nodes
        dace.graph.nxutil.change_edge_dest(graph, map_entry, entries[0])
        dace.graph.nxutil.change_edge_src(graph, map_exit, exits[-1])

        for i, e in enumerate(graph.out_edges(map_entry)):
            graph.remove_edge(e)
            graph.add_edge(entries[0], e.src_conn, entries[1],
                           'IN_' + str(i + 1), copy.deepcopy(e.data))
            graph.add_edge(entries[-1], 'OUT_' + str(i + 1), e.dst, e.dst_conn,
                           copy.deepcopy(e.data))
            for j in range(1, len(entries) - 1):
                graph.add_edge(entries[j], 'OUT_' + str(i + 1), entries[j + 1],
                               'IN_' + str(i + 1), copy.deepcopy(e.data))
        for i, e in enumerate(graph.in_edges(map_exit)):
            graph.remove_edge(e)
            graph.add_edge(e.src, e.src_conn, exits[0], 'IN_' + str(i + 1),
                           copy.deepcopy(e.data))
            graph.add_edge(exits[-2], 'OUT_' + str(i + 1), exits[-1],
                           e.dst_conn, copy.deepcopy(e.data))
            for j in range(0, len(exits) - 2):
                graph.add_edge(exits[j], 'OUT_' + str(i + 1), exits[j + 1],
                               'IN_' + str(i + 1), copy.deepcopy(e.data))

        # Remove old nodes
        graph.remove_node(map_entry)
        graph.remove_node(map_exit)


pm.Transformation.register_pattern(MapExpansion)
