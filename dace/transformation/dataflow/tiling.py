""" This module contains classes and functions that implement the orthogonal
    tiling transformation. """

import copy
import dace
from copy import deepcopy as dcpy
from dace import types, subsets, symbolic
from dace.properties import make_properties, Property, ParamsProperty, ShapeProperty
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from math import ceil
import sympy
import networkx as nx


def calc_set_image_index(map_idx, map_set, array_idx):
    image = []
    for a_idx in array_idx.indices:
        new_range = [a_idx, a_idx, a_idx]
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            new_range[0] = new_range[0].subs(
                symbol, dace.symbolic.overapproximate(m_range[0]))
            new_range[1] = new_range[1].subs(
                symbol, dace.symbolic.overapproximate(m_range[1]))
            new_range[2] = new_range[2].subs(
                symbol, dace.symbolic.overapproximate(m_range[2]))
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image_range(map_idx, map_set, array_range, strided):
    image = []
    n = len(array_range) - len(strided)
    if n > 0:
        strided.append([strided[-1]] * n)
    for a_range, stride in zip(array_range, strided):
        new_range = list(a_range)
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            new_range[0] = new_range[0].subs(
                symbol, dace.symbolic.overapproximate(m_range[0]))
            new_range[1] = new_range[1].subs(
                symbol, dace.symbolic.overapproximate(m_range[1]))
            if stride:
                new_range[2] = symbolic.pystr_to_symbolic('%s / %s' % (str(
                    new_range[2]), symbolic.symstr(m_range[1])))
            else:
                new_range[2] = new_range[2].subs(
                    symbol, dace.symbolic.overapproximate(m_range[2]))
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image(map_idx, map_set, array_set, strided):
    if isinstance(array_set, subsets.Range):
        return calc_set_image_range(map_idx, map_set, array_set, strided)
    if isinstance(array_set, subsets.Indices):
        return calc_set_image_index(map_idx, map_set, array_set)


def calc_set_union(aname, array, set_a, set_b):
    if isinstance(set_a, subsets.Indices) or isinstance(
            set_b, subsets.Indices):
        # raise NotImplementedError('Set union with indices is not implemented.')
        return subsets.Range.from_array(array)
    if not (isinstance(set_a, subsets.Range)
            and isinstance(set_b, subsets.Range)):
        raise TypeError('Can only compute the union of ranges.')
    if len(set_a) != len(set_b):
        raise ValueError('Range dimensions do not match')
    union = []
    for range_a, range_b in zip(set_a, set_b):
        union.append([
            sympy.Min(range_a[0], range_b[0]),
            sympy.Max(range_a[1], range_b[1]),
            sympy.Min(range_a[2], range_b[2]),
        ])
    return subsets.Range(union)


@make_properties
class OrthogonalTiling(pattern_matching.Transformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    # Properties
    prefix = Property(
        dtype=str, default="tile", desc="Prefix for new iterators")
    tile_sizes = ShapeProperty(
        dtype=tuple, default=(128, 128, 128), desc="Tile size per dimension")
    divides_evenly = Property(
        dtype=bool,
        default=False,
        desc="Tile size divides dimension length evenly")

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(OrthogonalTiling._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[OrthogonalTiling._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        # Tile map.
        target_dim, new_dim, new_map = self.__stripmine(
            sdfg, graph, self.subgraph)
        return new_map

    def __stripmine(self, sdfg, graph, candidate):
        # Retrieve map entry and exit nodes.
        map_entry = graph.nodes()[candidate[OrthogonalTiling._map_entry]]
        map_exit = graph.exit_nodes(map_entry)[0]

        # Map subgraph
        map_subgraph = graph.scope_subgraph(map_entry)

        # Retrieve transformation properties.
        prefix = self.prefix
        tile_sizes = self.tile_sizes
        divides_evenly = self.divides_evenly

        new_param = []
        new_range = []

        for dim_idx in range(len(map_entry.map.params)):

            if dim_idx >= len(tile_sizes):
                tile_size = tile_sizes[-1]
            else:
                tile_size = tile_sizes[dim_idx]

            # Retrieve parameter and range of dimension to be strip-mined.
            target_dim = map_entry.map.params[dim_idx]
            td_from, td_to, td_step = map_entry.map.range[dim_idx]

            new_dim = prefix + '_' + target_dim

            # Basic values
            if divides_evenly:
                tile_num = '(%s + 1 - %s) / %s' % (symbolic.symstr(td_to),
                                                   symbolic.symstr(td_from),
                                                   str(tile_size))
            else:
                tile_num = 'int_ceil((%s + 1 - %s), %s)' % (symbolic.symstr(
                    td_to), symbolic.symstr(td_from), str(tile_size))

            # Outer map values (over all tiles)
            nd_from = 0
            nd_to = symbolic.pystr_to_symbolic(str(tile_num) + ' - 1')
            nd_step = 1

            # Inner map values (over one tile)
            td_from_new = dace.symbolic.pystr_to_symbolic(td_from)
            td_to_new_exact = symbolic.pystr_to_symbolic(
                'min(%s + 1 - %s * %s, %s + %s) - 1' %
                (symbolic.symstr(td_to), str(new_dim), str(tile_size),
                 td_from_new, str(tile_size)))
            td_to_new_approx = symbolic.pystr_to_symbolic(
                '%s + %s - 1' % (td_from_new, str(tile_size)))

            # Outer map (over all tiles)
            new_dim_range = (nd_from, nd_to, nd_step)
            new_param.append(new_dim)
            new_range.append(new_dim_range)

            # Inner map (over one tile)
            if divides_evenly:
                td_to_new = td_to_new_approx
            else:
                td_to_new = dace.symbolic.SymExpr(td_to_new_exact,
                                                  td_to_new_approx)
            map_entry.map.range[dim_idx] = (td_from_new, td_to_new, td_step)

            # Fix subgraph memlets
            target_dim = dace.symbolic.pystr_to_symbolic(target_dim)
            offset = dace.symbolic.pystr_to_symbolic(
                '%s * %s' % (new_dim, str(tile_size)))
            for _, _, _, _, memlet in map_subgraph.edges():
                old_subset = memlet.subset
                if isinstance(old_subset, dace.subsets.Indices):
                    new_indices = []
                    for idx in old_subset:
                        new_idx = idx.subs(target_dim, target_dim + offset)
                        new_indices.append(new_idx)
                    memlet.subset = dace.subsets.Indices(new_indices)
                elif isinstance(old_subset, dace.subsets.Range):
                    new_ranges = []
                    for i, old_range in enumerate(old_subset):
                        if len(old_range) == 3:
                            b, e, s, = old_range
                            t = old_subset.tile_sizes[i]
                        else:
                            raise ValueError(
                                'Range %s is invalid.' % old_range)
                        new_b = b.subs(target_dim, target_dim + offset)
                        new_e = e.subs(target_dim, target_dim + offset)
                        new_s = s.subs(target_dim, target_dim + offset)
                        new_t = t.subs(target_dim, target_dim + offset)
                        new_ranges.append((new_b, new_e, new_s, new_t))
                    memlet.subset = dace.subsets.Range(new_ranges)
                else:
                    raise NotImplementedError

        new_map = nodes.Map(prefix + '_' + map_entry.map.label, new_param,
                            subsets.Range(new_range))
        new_map_entry = nodes.MapEntry(new_map)
        new_exit = nodes.MapExit(new_map)

        # Make internal map's schedule to "not parallel"
        map_entry.map._schedule = types.ScheduleType.Default

        # Redirect/create edges.
        new_in_edges = {}
        for _src, conn, _dest, _, memlet in graph.out_edges(map_entry):
            if not isinstance(sdfg.arrays[memlet.data], dace.data.Scalar):
                new_subset = copy.deepcopy(memlet.subset)
                # new_subset = calc_set_image(map_entry.map.params,
                #                             map_entry.map.range, memlet.subset,
                #                             cont_or_strided)
                if memlet.data in new_in_edges:
                    src, src_conn, dest, dest_conn, new_memlet, num = \
                        new_in_edges[memlet.data]
                    new_memlet.subset = calc_set_union(
                        new_memlet.data, sdfg.arrays[nnew_memlet.data],
                        new_memlet.subset, new_subset)
                    new_memlet.num_accesses = new_memlet.num_elements()
                    new_in_edges.update({
                        memlet.data: (src, src_conn, dest, dest_conn,
                                      new_memlet, min(num, int(conn[4:])))
                    })
                else:
                    new_memlet = dcpy(memlet)
                    new_memlet.subset = new_subset
                    new_memlet.num_accesses = new_memlet.num_elements()
                    new_in_edges.update({
                        memlet.data: (new_map_entry, None, map_entry, None,
                                      new_memlet, int(conn[4:]))
                    })
        nxutil.change_edge_dest(graph, map_entry, new_map_entry)

        new_out_edges = {}
        for _src, conn, _dest, _, memlet in graph.in_edges(map_exit):
            if not isinstance(sdfg.arrays[memlet.data], dace.data.Scalar):
                new_subset = memlet.subset
                # new_subset = calc_set_image(map_entry.map.params,
                #                             map_entry.map.range,
                #                             memlet.subset, cont_or_strided)
                if memlet.data in new_out_edges:
                    src, src_conn, dest, dest_conn, new_memlet, num = \
                        new_out_edges[memlet.data]
                    new_memlet.subset = calc_set_union(
                        new_memlet.data, sdfg.arrays[nnew_memlet.data],
                        new_memlet.subset, new_subset)
                    new_memlet.num_accesses = new_memlet.num_elements()
                    new_out_edges.update({
                        memlet.data: (src, src_conn, dest, dest_conn,
                                      new_memlet, min(num, conn[4:]))
                    })
                else:
                    new_memlet = dcpy(memlet)
                    new_memlet.subset = new_subset
                    new_memlet.num_accesses = new_memlet.num_elements()
                    new_out_edges.update({
                        memlet.data: (map_exit, None, new_exit, None,
                                      new_memlet, conn[4:])
                    })
        nxutil.change_edge_src(graph, map_exit, new_exit)

        # Connector related work follows
        # 1. Dictionary 'old_connector_number': 'new_connector_numer'
        # 2. New node in/out connectors
        # 3. New edges

        in_conn_nums = []
        for _, e in new_in_edges.items():
            _, _, _, _, _, num = e
            in_conn_nums.append(num)
        in_conn = {}
        for i, num in enumerate(in_conn_nums):
            in_conn.update({num: i + 1})

        entry_in_connectors = set()
        entry_out_connectors = set()
        for i in range(len(in_conn_nums)):
            entry_in_connectors.add('IN_' + str(i + 1))
            entry_out_connectors.add('OUT_' + str(i + 1))
        new_map_entry.in_connectors = entry_in_connectors
        new_map_entry.out_connectors = entry_out_connectors

        for _, e in new_in_edges.items():
            src, _, dst, _, memlet, num = e
            graph.add_edge(src, 'OUT_' + str(in_conn[num]), dst,
                           'IN_' + str(in_conn[num]), memlet)

        out_conn_nums = []
        for _, e in new_out_edges.items():
            _, _, dst, _, _, num = e
            if dst is not new_exit:
                continue
            out_conn_nums.append(num)
        out_conn = {}
        for i, num in enumerate(out_conn_nums):
            out_conn.update({num: i + 1})

        exit_in_connectors = set()
        exit_out_connectors = set()
        for i in range(len(out_conn_nums)):
            exit_in_connectors.add('IN_' + str(i + 1))
            exit_out_connectors.add('OUT_' + str(i + 1))
        new_exit.in_connectors = exit_in_connectors
        new_exit.out_connectors = exit_out_connectors

        for _, e in new_out_edges.items():
            src, _, dst, _, memlet, num = e
            graph.add_edge(src, 'OUT_' + str(out_conn[num]), dst,
                           'IN_' + str(out_conn[num]), memlet)

        # Return strip-mined dimension.
        return target_dim, new_dim, new_map

    @staticmethod
    def __modify_edges(sdfg, graph, candidate, target_dim, new_dim):
        map_entry = graph.nodes()[candidate[OrthogonalTiling._map_entry]]

        processed = []
        for src, _dest, memlet, _scope in nxutil.traverse_sdfg_scope(
                graph, map_entry, True):
            if memlet in processed:
                continue
            processed.append(memlet)

            # Corner cases
            if isinstance(sdfg.arrays[memlet.data], dace.data.Stream):
                continue
            if memlet.wcr is not None:
                memlet.num_accesses = 1
                continue

            for i, dim in enumerate(memlet.subset):
                if isinstance(dim, tuple):
                    dim = tuple(
                        symbolic.pystr_to_symbolic(d).subs(
                            symbolic.pystr_to_symbolic(target_dim),
                            symbolic.pystr_to_symbolic(
                                '%s + %s' % (str(new_dim), str(target_dim))))
                        for d in dim)
                else:
                    dim = symbolic.pystr_to_symbolic(dim).subs(
                        symbolic.pystr_to_symbolic(target_dim),
                        symbolic.pystr_to_symbolic(
                            '%s + %s' % (str(new_dim), str(target_dim))))

                memlet.subset[i] = dim
        return


pattern_matching.Transformation.register_pattern(OrthogonalTiling)
