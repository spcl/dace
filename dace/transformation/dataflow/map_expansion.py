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
class MapExpansion(pm.SingleStateTransformation):
    """ Implements the map-expansion pattern.

        Map-expansion takes an N-dimensional map and expands it to N 
        unidimensional maps.

        New edges abide by the following rules:
          1. If there are no edges coming from the outside, use empty memlets
          2. Edges with IN_* connectors replicate along the maps
          3. Edges for dynamic map ranges replicate until reaching range(s)
    """

    map_entry = pm.PatternNode(nodes.MapEntry)

    inner_schedule = EnumProperty(desc="Schedule for inner maps",
                                  dtype=dtypes.ScheduleType,
                                  default=dtypes.ScheduleType.Sequential,
                                  allow_none=True)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False):
        # A candidate subgraph matches the map-expansion pattern when it
        # includes an N-dimensional map, with N greater than one.
        return self.map_entry.map.get_param_num() > 1

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        # Extract the map and its entry and exit nodes.
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        current_map = map_entry.map

        # Create new maps
        inner_schedule = self.inner_schedule or current_map.schedule
        new_maps = [
            nodes.Map(current_map.label + '_' + str(param), [param],
                      subsets.Range([param_range]),
                      schedule=inner_schedule)
            for param, param_range in zip(current_map.params[1:], current_map.range[1:])
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
        for edge in list(graph.out_edges(map_entry)):
            if edge.src_conn is not None and edge.src_conn not in entries[-1].out_connectors:
                entries[-1].add_out_connector(edge.src_conn)

            graph.add_edge(entries[-1], edge.src_conn, edge.dst, edge.dst_conn, memlet=copy.deepcopy(edge.data))
            graph.remove_edge(edge)

        if graph.in_degree(map_entry) == 0:
            graph.add_memlet_path(map_entry, *entries, memlet=dace.Memlet())
        else:
            for edge in graph.in_edges(map_entry):
                if not edge.dst_conn.startswith("IN_"):
                    continue
                
                in_conn = edge.dst_conn
                out_conn = "OUT_" + in_conn[3:]
                if in_conn not in entries[-1].in_connectors:
                    graph.add_memlet_path(map_entry,
                                          *entries,
                                          memlet=copy.deepcopy(edge.data),
                                          src_conn=out_conn,
                                          dst_conn=in_conn)

        # Modify dynamic map ranges
        dynamic_edges = dace.sdfg.dynamic_map_inputs(graph, map_entry)
        for edge in dynamic_edges:
            # Remove old edge and connector
            graph.remove_edge(edge)
            edge.dst.remove_in_connector(edge.dst_conn)

            # Propagate to each range it belongs to
            path = []
            for mapnode in [map_entry] + entries:
                path.append(mapnode)
                if any(edge.dst_conn in map(str, symbolic.symlist(r)) for r in mapnode.map.range):
                    graph.add_memlet_path(edge.src,
                                          *path,
                                          memlet=edge.data,
                                          src_conn=edge.src_conn,
                                          dst_conn=edge.dst_conn)

        # Create new map exits
        for edge in graph.in_edges(map_exit):
            graph.remove_edge(edge)
            graph.add_memlet_path(edge.src,
                                  *exits[::-1],
                                  map_exit,
                                  memlet=edge.data,
                                  src_conn=edge.src_conn,
                                  dst_conn=edge.dst_conn)

        from dace.sdfg.scope import ScopeTree
        scope = None
        queue: List[ScopeTree] = graph.scope_leaves()
        while len(queue) > 0:
            tnode = queue.pop()
            if tnode.entry == entries[-1]:
                scope = tnode
                break
            elif tnode.parent is not None:
                queue.append(tnode.parent)
        else:
            raise ValueError('Cannot find scope in state')

        propagate_memlets_scope(sdfg, state=graph, scopes=scope)
        consolidate_edges(sdfg, scope)

        return [map_entry] + entries
