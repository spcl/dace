""" This module contains classes and functions that implement the strip-mining
    transformation."""

import dace
from copy import deepcopy as dcpy
from dace import types, subsets, symbolic
from dace.properties import make_properties, Property
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from math import ceil
import sympy
import networkx as nx


def calc_set_image_index(map_idx, map_set, array_idx):
    image = []
    for a_idx in array_idx.indices:
        new_range = [a_idx, a_idx, 1]
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            new_range[0] = new_range[0].subs(
                symbol, dace.symbolic.overapproximate(m_range[0]))
            new_range[1] = new_range[1].subs(
                symbol, dace.symbolic.overapproximate(m_range[1]))
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image_range(map_idx, map_set, array_range):
    image = []
    for a_range in array_range:
        new_range = a_range
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            new_range = [
                new_range[i].subs(symbol,
                                  dace.symbolic.overapproximate(m_range[i]))
                for i in range(0, 3)
            ]
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image(map_idx, map_set, array_set):
    if isinstance(array_set, subsets.Range):
        return calc_set_image_range(map_idx, map_set, array_set)
    if isinstance(array_set, subsets.Indices):
        return calc_set_image_index(map_idx, map_set, array_set)


def calc_set_union(set_a, set_b):
    if isinstance(set_a, subsets.Indices) or isinstance(
            set_b, subsets.Indices):
        raise NotImplementedError('Set union with indices is not implemented.')
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
class StripMining(pattern_matching.Transformation):
    """ Implements the strip-mining transformation.

        Strip-mining takes as input a map dimension and splits it into
        two dimensions. The new dimension iterates over the range of
        the original one with a parameterizable step, called the tile
        size. The original dimension is changed to iterates over the
        range of the tile size, with the same step as before.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    # Properties
    dim_idx = Property(
        dtype=int, default=-1, desc="Index of dimension to be strip-mined")
    new_dim_prefix = Property(
        dtype=str, default="tile", desc="Prefix for new dimension name")
    tile_size = Property(
        dtype=str, default="64", desc="Tile size of strip-mined dimension")
    divides_evenly = Property(
        dtype=bool,
        default=False,
        desc="Tile size divides dimension range evenly?")
    strided = Property(
        dtype=bool,
        default=False,
        desc="Continuous (false) or strided (true) elements in tile")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(StripMining._map_entry)
            # kStripMining._tasklet, StripMining._map_exit)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[StripMining._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        # Strip-mine selected dimension.
        target_dim, new_dim, new_map = self.__stripmine(
            sdfg, graph, self.subgraph)
        # Modify edges to match strip-mined dimension.
        # StripMining.__modify_edges(sdfg, graph, self.subgraph, target_dim, new_dim)
        return new_map

    # def __init__(self, tag=True):
    def __init__(self, *args, **kwargs):
        self._entry = nodes.EntryNode()
        self._tasklet = nodes.Tasklet('_')
        self._exit = nodes.ExitNode()
        super().__init__(*args, **kwargs)
        # self.tag = tag

    @property
    def entry(self):
        return self._entry

    @property
    def exit(self):
        return self._exit

    @property
    def tasklet(self):
        return self._tasklet

    def print_match_pattern(self, candidate):
        gentry = candidate[self.entry]
        return str(gentry.map.params[-1])

    def modifies_graph(self):
        return True

    def __stripmine(self, sdfg, graph, candidate):

        # Retrieve map entry and exit nodes.
        map_entry = graph.nodes()[candidate[StripMining._map_entry]]
        map_exits = graph.exit_nodes(map_entry)

        # Retrieve transformation properties.
        dim_idx = self.dim_idx
        new_dim_prefix = self.new_dim_prefix
        tile_size = self.tile_size
        divides_evenly = self.divides_evenly
        strided = self.strided

        # Retrieve parameter and range of dimension to be strip-mined.
        target_dim = map_entry.map.params[dim_idx]
        td_from, td_to, td_step = map_entry.map.range[dim_idx]

        # Create new map. Replace by cloning???
        new_dim = new_dim_prefix + '_' + target_dim
        nd_from = 0
        nd_to = symbolic.pystr_to_symbolic(
            'int_ceil(%s + 1 - %s, %s) - 1' %
            (symbolic.symstr(td_to), symbolic.symstr(td_from), tile_size))
        nd_step = 1
        new_dim_range = (nd_from, nd_to, nd_step)
        new_map = nodes.Map(new_dim + '_' + map_entry.map.label, [new_dim],
                            subsets.Range([new_dim_range]))
        new_map_entry = nodes.MapEntry(new_map)

        # Change the range of the selected dimension to iterate over a single
        # tile
        if strided:
            td_from_new = symbolic.pystr_to_symbolic(new_dim)
            td_to_new_approx = td_to
            td_step = symbolic.pystr_to_symbolic(tile_size)
        else:
            td_from_new = symbolic.pystr_to_symbolic(
                '%s + %s * %s' % (symbolic.symstr(td_from), str(new_dim),
                                  tile_size))
            td_to_new_exact = symbolic.pystr_to_symbolic(
                'min(%s + 1, %s + %s * %s + %s) - 1' %
                (symbolic.symstr(td_to), symbolic.symstr(td_from), tile_size,
                 str(new_dim), tile_size))
            td_to_new_approx = symbolic.pystr_to_symbolic(
                '%s + %s * %s + %s - 1' % (symbolic.symstr(td_from), tile_size,
                                           str(new_dim), tile_size))
        if divides_evenly or strided:
            td_to_new = td_to_new_approx
        else:
            td_to_new = dace.symbolic.SymExpr(td_to_new_exact,
                                              td_to_new_approx)
        map_entry.map.range[dim_idx] = (td_from_new, td_to_new, td_step)

        # Make internal map's schedule to "not parallel"
        map_entry.map._schedule = types.ScheduleType.Default

        # Redirect/create edges.
        new_in_edges = {}
        for _src, conn, _dest, _, memlet in graph.out_edges(map_entry):
            if not isinstance(sdfg.arrays[memlet.data], dace.data.Scalar):
                new_subset = calc_set_image(
                    map_entry.map.params,
                    map_entry.map.range,
                    memlet.subset,
                )
                if memlet.data in new_in_edges:
                    src, src_conn, dest, dest_conn, new_memlet, num = \
                        new_in_edges[memlet.data]
                    new_memlet.subset = calc_set_union(new_memlet.subset,
                                                       new_subset)
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
        new_exits = []
        for map_exit in map_exits:
            if isinstance(map_exit, nodes.MapExit):
                new_exit = nodes.MapExit(new_map)
                new_exits.append(new_exit)
            for _src, conn, _dest, _, memlet in graph.in_edges(map_exit):
                if not isinstance(sdfg.arrays[memlet.data], dace.data.Scalar):
                    new_subset = calc_set_image(
                        map_entry.map.params,
                        map_entry.map.range,
                        memlet.subset,
                    )
                    if memlet.data in new_out_edges:
                        src, src_conn, dest, dest_conn, new_memlet, num = \
                            new_out_edges[memlet.data]
                        new_memlet.subset = calc_set_union(
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

        for new_exit in new_exits:

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
        map_entry = graph.nodes()[candidate[StripMining._map_entry]]

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


pattern_matching.Transformation.register_pattern(StripMining)
