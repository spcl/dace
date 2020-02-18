""" Contains classes that implement the map-expansion transformation. """

from typing import Dict
import dace
from dace import dtypes, registry, subsets, symbolic
from dace.graph import nodes, nxutil
from dace.graph.graph import OrderedMultiDiConnectorGraph
from dace.transformation import pattern_matching as pm


@registry.autoregister_params(singlestate=True)
class MapExpansion(pm.Transformation):
    """ Implements the map-expansion pattern.

        Map-expansion takes an N-dimensional map and expands it to N 
        unidimensional maps.

        New edges abide by the following rules:
          1. If there are no edges coming from the outside, use empty memlets
          2. Edges with IN_* connectors replicate along the maps
          3. Edges for dynamic map ranges replicate until reaching range(s)
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
        # A candidate subgraph matches the map-expansion pattern when it
        # includes an N-dimensional map, with N greater than one.
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
        new_maps = [
            nodes.Map(
                current_map.label + '_' + str(param), [param],
                subsets.Range([param_range]),
                schedule=dtypes.ScheduleType.Sequential) for param, param_range
            in zip(current_map.params[1:], current_map.range[1:])
        ]
        current_map.params = [current_map.params[0]]
        current_map.range = subsets.Range([current_map.range[0]])

        # Create new map entries and exits
        entries = [nodes.MapEntry(new_map) for new_map in new_maps]
        exits = [nodes.MapExit(new_map) for new_map in new_maps]

        # Create edges, abiding by the following rules:
        # 1. If there are no edges coming from the outside, use empty memlets
        # 2. Edges with IN_* connectors replicate along the maps
        # 3. Edges for dynamic map ranges replicate until reaching range(s)
        for edge in graph.out_edges(map_entry):
            graph.remove_edge(edge)
            graph.add_memlet_path(
                map_entry,
                *entries,
                edge.dst,
                src_conn=edge.src_conn,
                memlet=edge.data,
                dst_conn=edge.dst_conn)

        # Modify dynamic map ranges
        dynamic_edges = dace.sdfg.dynamic_map_inputs(graph, map_entry)
        for edge in dynamic_edges:
            # Remove old edge and connector
            graph.remove_edge(edge)
            edge.dst._in_connectors.remove(edge.dst_conn)

            # Propagate to each range it belongs to
            path = []
            for mapnode in [map_entry] + entries:
                path.append(mapnode)
                if any(
                        edge.dst_conn in map(str, symbolic.symlist(r))
                        for r in mapnode.map.range):
                    graph.add_memlet_path(
                        edge.src,
                        *path,
                        memlet=edge.data,
                        src_conn=edge.src_conn,
                        dst_conn=edge.dst_conn)

        # Create new map exits
        for edge in graph.in_edges(map_exit):
            graph.remove_edge(edge)
            graph.add_memlet_path(
                edge.src,
                *exits[::-1],
                map_exit,
                memlet=edge.data,
                src_conn=edge.src_conn,
                dst_conn=edge.dst_conn)
