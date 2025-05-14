# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the map-expansion transformation. """

from dace.sdfg.utils import consolidate_edges
from typing import Dict, List
import copy
import dace
from dace import dtypes, subsets, symbolic
from dace.properties import EnumProperty, make_properties, Property
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import OrderedMultiDiConnectorGraph
from dace.transformation import transformation as pm
from dace.sdfg.propagation import propagate_memlets_scope


@make_properties
class MapExpansion(pm.SingleStateTransformation):
    """ Implements the map-expansion pattern.

        Map-expansion takes an N-dimensional map and expands it.
        It will generate the k nested unidimensional map and a (N-k)-dimensional inner most map.
        If k is not specified all maps are expanded.

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
    expansion_limit = Property(desc="How many unidimensional maps will be created, known as k. "
                               "If None, the default no limit is in place.",
                               dtype=int,
                               allow_none=True,
                               default=None)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive: bool = False):
        # A candidate subgraph matches the map-expansion pattern when it
        # includes an N-dimensional map, with N greater than one.
        return self.map_entry.map.get_param_num() > 1

    def generate_new_maps(self,
                          current_map: nodes.Map):
        if self.expansion_limit is None:
            full_expand = True
        elif isinstance(self.expansion_limit, int):
            full_expand = False
            if self.expansion_limit <= 0:   # These are invalid, so we make a full expansion
                full_expand = True
            elif (self.map_entry.map.get_param_num() - self.expansion_limit) <= 1:
                full_expand = True
        else:
            raise TypeError(f"Does not know how to handle type {type(self.expansion_limit).__name__}")

        inner_schedule = self.inner_schedule or current_map.schedule
        if full_expand:
            new_maps = [
                nodes.Map(
                    current_map.label + '_' + str(param), [param],
                    subsets.Range([param_range]),
                    schedule=inner_schedule if dim != 0 else current_map.schedule)
                for dim, param, param_range in zip(range(len(current_map.params)), current_map.params, current_map.range)
            ]
            for i, new_map in enumerate(new_maps):
                new_map.range.tile_sizes[0] = current_map.range.tile_sizes[i]

        else:
            k = self.expansion_limit
            new_maps: list[nodes.Map] = []

            # Unidimensional maps
            for dim in range(0, k):
                dim_param = current_map.params[dim]
                dim_range = current_map.range.ranges[dim]
                dim_tile  = current_map.range.tile_sizes[dim]
                new_maps.append(
                    nodes.Map(
                        current_map.label + '_' + str(dim_param),
                        [dim_param],
                        subsets.Range([dim_range]),
                        schedule=inner_schedule if dim != 0 else current_map.schedule ))
                new_maps[-1].range.tile_sizes[0] = dim_tile

            # Multidimensional maps
            mdim_params = current_map.params[k:]
            mdim_ranges = current_map.range.ranges[k:]
            mdim_tiles  = current_map.range.tile_sizes[k:]
            new_maps.append(
                    nodes.Map(
                        current_map.label,  # The original name
                        mdim_params,
                        mdim_ranges,
                        schedule=inner_schedule ))
            new_maps[-1].range.tile_sizes = mdim_tiles
        return new_maps

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        # Extract the map and its entry and exit nodes.
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)
        current_map = map_entry.map

        # Generate the new maps that we should use.
        new_maps = self.generate_new_maps(current_map)

        if not new_maps:        # No changes should be made -> noops
            return

        # Reuse the map that is already existing for the first one.
        current_map.params = new_maps[0].params
        current_map.range  = new_maps[0].range
        new_maps.pop(0)

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
