# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

from typing import Dict
import dace
from copy import deepcopy as dcpy
from dace import dtypes, subsets, symbolic
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, Property, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.dataflow import MapInterchange
from dace.transformation.dataflow.strip_mining import calc_set_image, calc_set_union
import sympy


@make_properties
class GPUGridStridedTiling(transformation.SingleStateTransformation):
    """
    Implements the grid-strided map tiling transformation on two nested maps.

    E.g.
    i = ib:ie:is -> j = jb:je:js
    After transformation:
    i0 = 0:GridDim -> j0 = 0:BlockDim -> i1 = ib+i0*is:ie:GridDim*is -> j1 = jb+j0*js:je:BlockDim*js
    where GridDim = min(MaxGridDim, (ie-ib)//is)
    """

    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties

    new_dim_prefix = Property(dtype=str, default="tile", desc="Prefix for new dimension name")
    max_grid_dim = SymbolicProperty(default=65535, desc="Maximum grid dimension")
    block_dim = Property(default=128, desc="Block dimension")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):

        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry

        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for e in graph.out_edges(outer_map_entry):
            if e.dst != inner_map_entry:
                return False
        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for e in graph.in_edges(inner_map_entry):
            if e.src != outer_map_entry:
                return False

        # Check the edges between the exits of the two maps.
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for e in graph.out_edges(inner_map_exit):
            if e.dst != outer_map_exit:
                return False
        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for e in graph.in_edges(outer_map_exit):
            if e.src != inner_map_exit:
                return False

        # Currently only support nested maps with a single dimension in each.
        if len(outer_map_entry.map.params) != 1 or len(inner_map_entry.map.params) != 1:
            return False

        return True

    def _find_new_dim(self, sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry, prefix: str, target_dim: str):
        """ Finds a variable that is not already defined in scope. """
        candidate = '%s_%s' % (prefix, target_dim)
        index = 1
        defined_vars = set(str(s) for s in (state.symbols_defined_at(entry).keys() | sdfg.symbols.keys()))
        while candidate in defined_vars:
            candidate = '%s%d_%s' % (prefix, index, target_dim)
            index += 1
        return candidate

    def apply(self, graph: SDFGState, sdfg: SDFG):
        i_entry = self.inner_map_entry
        o_entry = self.outer_map_entry
        i_exit = graph.exit_node(i_entry)
        o_exit = graph.exit_node(o_entry)

        new_dim_prefix = self.new_dim_prefix
        max_grid_dim = self.max_grid_dim
        block_dim = self.block_dim

        max_grid_dim = symbolic.pystr_to_symbolic(max_grid_dim)
        block_dim = symbolic.pystr_to_symbolic(block_dim)

        # Get the map params
        o_from, o_to, o_step = o_entry.map.range[0]
        i_from, i_to, i_step = i_entry.map.range[0]

        tile_o_dim_new = self._find_new_dim(sdfg, graph, o_entry, new_dim_prefix, o_entry.map.params[0])
        tile_i_dim_new = self._find_new_dim(sdfg, graph, i_entry, new_dim_prefix, i_entry.map.params[0])

        grid_dim = sympy.Min(max_grid_dim, (o_to + 1 - o_from) // o_step)

        # TODO: how to deal with approximated values?
        # begin, end, step of all four maps
        tile_o_range_new = (0, grid_dim - 1, 1)
        tile_i_range_new = (0, block_dim - 1, 1)
        o_range_new = (o_from + symbolic.pystr_to_symbolic(tile_o_dim_new) * o_step, o_to, grid_dim * o_step)
        i_range_new = (i_from + symbolic.pystr_to_symbolic(tile_i_dim_new) * i_step, i_to, block_dim * i_step)

        # Create the new maps
        tile_o_map = nodes.Map(o_entry.map.label, [tile_o_dim_new],
                               subsets.Range([tile_o_range_new]),
                               schedule=dtypes.ScheduleType.GPU_Device)
        tile_i_map = nodes.Map(i_entry.map.label, [tile_i_dim_new],
                               subsets.Range([tile_i_range_new]),
                               schedule=dtypes.ScheduleType.GPU_ThreadBlock)

        # Create the new map entries and exits
        tile_o_entry = nodes.MapEntry(tile_o_map)
        tile_i_entry = nodes.MapEntry(tile_i_map)
        tile_o_exit = nodes.MapExit(tile_o_map)
        tile_i_exit = nodes.MapExit(tile_i_map)

        # Set block size
        tile_i_entry.map.gpu_block_size = [self.block_dim, 1, 1]

        # Update Range and ScheduleType of the maps
        o_entry.map.range = subsets.Range([o_range_new])
        o_entry.map.schedule = dtypes.ScheduleType.Sequential
        i_entry.map.range = subsets.Range([i_range_new])
        i_entry.map.schedule = dtypes.ScheduleType.Sequential

        # Redirect edges
        tile_o_entry.in_connectors = dcpy(o_entry.in_connectors)
        tile_i_entry.in_connectors = dcpy(i_entry.in_connectors)
        tile_o_exit.out_connectors = dcpy(o_exit.out_connectors)
        tile_i_exit.out_connectors = dcpy(i_exit.out_connectors)
        sdutil.change_edge_src(graph, o_exit, tile_o_exit)
        sdutil.change_edge_src(graph, i_exit, tile_i_exit)
        sdutil.change_edge_dest(graph, o_entry, tile_o_entry)
        sdutil.change_edge_dest(graph, i_entry, tile_i_entry)

        # Connect previous map nodes and corresponding tile map nodes
        # Code borrowed from StripMining transformation
        for map_entry, new_map_entry, map_exit, new_map_exit in [
            (o_entry, tile_o_entry, o_exit, tile_o_exit),
            (i_entry, tile_i_entry, i_exit, tile_i_exit),
        ]:
            # Create new entry edges
            new_in_edges = dict()
            entry_in_conn = {}
            entry_out_conn = {}
            for _src, src_conn, _dst, _, memlet in graph.out_edges(map_entry):
                if (src_conn is not None and src_conn[:4] == 'OUT_'
                        and not isinstance(sdfg.arrays[memlet.data], dace.data.Scalar)):
                    new_subset = calc_set_image(
                        map_entry.map.params,
                        map_entry.map.range,
                        memlet.subset,
                    )
                    conn = src_conn[4:]
                    key = (memlet.data, 'IN_' + conn, 'OUT_' + conn)
                    if key in new_in_edges.keys():
                        old_subset = new_in_edges[key].subset
                        new_in_edges[key].subset = calc_set_union(old_subset, new_subset)
                    else:
                        entry_in_conn['IN_' + conn] = None
                        entry_out_conn['OUT_' + conn] = None
                        new_memlet = dcpy(memlet)
                        new_memlet.subset = new_subset
                        if memlet.dynamic:
                            new_memlet.num_accesses = memlet.num_accesses
                        else:
                            new_memlet.num_accesses = new_memlet.num_elements().simplify()
                        new_in_edges[key] = new_memlet
                else:
                    if src_conn is not None and src_conn[:4] == 'OUT_':
                        conn = src_conn[4:]
                        in_conn = 'IN_' + conn
                        out_conn = 'OUT_' + conn
                    else:
                        in_conn = src_conn
                        out_conn = src_conn
                    if in_conn:
                        entry_in_conn[in_conn] = None
                    if out_conn:
                        entry_out_conn[out_conn] = None
                    new_in_edges[(memlet.data, in_conn, out_conn)] = dcpy(memlet)
            new_map_entry.out_connectors = entry_out_conn
            map_entry.in_connectors = entry_in_conn
            for (_, in_conn, out_conn), memlet in new_in_edges.items():
                graph.add_edge(new_map_entry, out_conn, map_entry, in_conn, memlet)

            # Create new exit edges
            new_out_edges = dict()
            exit_in_conn = {}
            exit_out_conn = {}
            for _src, _, _dst, dst_conn, memlet in graph.in_edges(map_exit):
                if (dst_conn is not None and dst_conn[:3] == 'IN_'
                        and not isinstance(sdfg.arrays[memlet.data], dace.data.Scalar)):
                    new_subset = calc_set_image(
                        map_entry.map.params,
                        map_entry.map.range,
                        memlet.subset,
                    )
                    conn = dst_conn[3:]
                    key = (memlet.data, 'IN_' + conn, 'OUT_' + conn)
                    if key in new_out_edges.keys():
                        old_subset = new_out_edges[key].subset
                        new_out_edges[key].subset = calc_set_union(old_subset, new_subset)
                    else:
                        exit_in_conn['IN_' + conn] = None
                        exit_out_conn['OUT_' + conn] = None
                        new_memlet = dcpy(memlet)
                        new_memlet.subset = new_subset
                        if memlet.dynamic:
                            new_memlet.num_accesses = memlet.num_accesses
                        else:
                            new_memlet.num_accesses = new_memlet.num_elements().simplify()
                        new_out_edges[key] = new_memlet
                else:
                    if dst_conn is not None and dst_conn[:3] == 'IN_':
                        conn = dst_conn[3:]
                        in_conn = 'IN_' + conn
                        out_conn = 'OUT_' + conn
                    else:
                        in_conn = dst_conn
                        out_conn = dst_conn
                    if in_conn:
                        exit_in_conn[in_conn] = None
                    if out_conn:
                        exit_out_conn[out_conn] = None
                    new_out_edges[(memlet.data, in_conn, out_conn)] = dcpy(memlet)
            new_map_exit.in_connectors = exit_in_conn
            map_exit.out_connectors = exit_out_conn
            for (_, in_conn, out_conn), memlet in new_out_edges.items():
                graph.add_edge(map_exit, out_conn, new_map_exit, in_conn, memlet)

        # if inner map contains dynamic range, need to move the dynamic connectors
        # from tile_inner map entry to inner map entry to facilitate MapInterchange.
        # Because we brute-forcely did sdutil.change_edge_dest(graph, i_entry, tile_i_entry)
        # TODO: what about map exit connectors?
        data_dict: Dict[str, dace.Memlet] = {}  # map data array to memlet
        for e in graph.edges_between(o_entry, tile_i_entry):
            if e.dst_conn is not None and e.dst_conn[:3] != 'IN_' and e.src_conn[:4] == 'OUT_':
                # trim edges
                graph.remove_edge(e)
                # add edges between tile_i_entry and i_entry
                graph.add_edge(tile_i_entry, e.src_conn, i_entry, e.dst_conn, dcpy(e.data))

                # add edges between o_entry and tile_i_entry
                if e.data.data not in data_dict.keys():
                    # new edge data, add to data_dict
                    data_dict[e.data.data] = dcpy(e.data)
                    in_conn = 'IN_' + e.src_conn[4:]
                    assert e.src_conn[4:] == e.data.data
                    graph.add_edge(o_entry, e.src_conn, tile_i_entry, in_conn, data_dict[e.data.data])
                else:
                    # already added edge data, just add edge volume
                    # TODO: how to add subset?
                    data_dict[e.data.data].volume += e.data.volume

                # trim connectors
                tile_i_entry.remove_in_connector(e.dst_conn)

                # TODO: fix missing added connectors

        # sdfg.view()

        # Interchange middle two maps
        MapInterchange.apply_to(sdfg, outer_map_entry=o_entry, inner_map_entry=tile_i_entry)

    @staticmethod
    def annotates_memlets():
        return True
