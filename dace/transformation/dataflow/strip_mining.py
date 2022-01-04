# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the strip-mining
    transformation."""

import dace
from copy import deepcopy as dcpy
from dace import dtypes, registry, subsets, symbolic
from dace.sdfg import SDFG, SDFGState
from dace.properties import EnumProperty, make_properties, Property, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.symbolic import issymbolic, overapproximate, SymExpr
from dace.transformation import transformation, helpers as xfh
import sympy


def calc_set_image_index(map_idx, map_set, array_idx):
    image = []
    for a_idx in array_idx.indices:
        new_range = [a_idx, a_idx, SymExpr(1, 1)]
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            for i in range(2):
                if isinstance(m_range[i], SymExpr):
                    exact = m_range[i].expr
                    approx = m_range[i].approx
                else:
                    exact = m_range[i]
                    approx = overapproximate(m_range[i])
                if isinstance(new_range[i], SymExpr):
                    new_range[i] = SymExpr(new_range[i].expr.subs([(symbol, exact)]),
                                           new_range[i].approx.subs([(symbol, approx)]))
                elif issymbolic(new_range[i]):
                    new_range[i] = SymExpr(new_range[i].subs([(symbol, exact)]), new_range[i].subs([(symbol, approx)]))
                else:
                    new_range[i] = SymExpr(new_range[i], new_range[i])
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image_range(map_idx, map_set, array_range):
    image = []
    for a_range in array_range:
        new_range = list(a_range)
        for m_idx, m_range in zip(map_idx, map_set):
            symbol = symbolic.pystr_to_symbolic(m_idx)
            for i in range(3):
                if isinstance(m_range[i], SymExpr):
                    exact = m_range[i].expr
                    approx = m_range[i].approx
                else:
                    exact = m_range[i]
                    approx = overapproximate(m_range[i])
                if isinstance(new_range[i], SymExpr):
                    new_range[i] = SymExpr(new_range[i].expr.subs([(symbol, exact)]),
                                           new_range[i].approx.subs([(symbol, approx)]))
                elif issymbolic(new_range[i]):
                    new_range[i] = SymExpr(new_range[i].subs([(symbol, exact)]), new_range[i].subs([(symbol, approx)]))
                else:
                    new_range[i] = SymExpr(new_range[i], new_range[i])
            if isinstance(new_range[0], SymExpr):
                start = new_range[0].approx
            else:
                start = new_range[0]
            if isinstance(new_range[1], SymExpr):
                stop = new_range[1].approx
            else:
                stop = new_range[1]
            if isinstance(new_range[2], SymExpr):
                step = new_range[2].approx
            else:
                step = new_range[2]
            descending = (start > stop) == True
            posstep = (step > 0) == True
            if descending and posstep:
                new_range[0], new_range[1] = new_range[1], new_range[0]
        image.append(new_range)
    return subsets.Range(image)


def calc_set_image(map_idx, map_set, array_set):
    if isinstance(array_set, subsets.Range):
        return calc_set_image_range(map_idx, map_set, array_set)
    if isinstance(array_set, subsets.Indices):
        return calc_set_image_index(map_idx, map_set, array_set)


def calc_set_union(set_a, set_b):
    if isinstance(set_a, subsets.Indices) or isinstance(set_b, subsets.Indices):
        raise NotImplementedError('Set union with indices is not implemented.')
    if not (isinstance(set_a, subsets.Range) and isinstance(set_b, subsets.Range)):
        raise TypeError('Can only compute the union of ranges.')
    if len(set_a) != len(set_b):
        raise ValueError('Range dimensions do not match')
    union = []
    for range_a, range_b in zip(set_a, set_b):
        r_union = []
        for i in range(3):
            if isinstance(range_a[i], SymExpr):
                a_exact = range_a[i].expr
                a_approx = range_a[i].approx
            else:
                a_exact = range_a[i]
                a_approx = range_a[i]
            if isinstance(range_b[i], SymExpr):
                b_exact = range_b[i].expr
                b_approx = range_b[i].approx
            else:
                b_exact = range_b[i]
                b_approx = range_b[i]
            if i in {0, 2}:
                r_union.append(SymExpr(sympy.Min(a_exact, b_exact), sympy.Min(a_approx, b_approx)))
            else:
                r_union.append(SymExpr(sympy.Max(a_exact, b_exact), sympy.Max(a_approx, b_approx)))
        union.append(r_union)
        # union.append([
        #     sympy.Min(range_a[0], range_b[0]),
        #     sympy.Max(range_a[1], range_b[1]),
        #     sympy.Min(range_a[2], range_b[2]),
        # ])
    return subsets.Range(union)


@make_properties
class StripMining(transformation.SingleStateTransformation):
    """ Implements the strip-mining transformation.

        Strip-mining takes as input a map dimension and splits it into
        two dimensions. The new dimension iterates over the range of
        the original one with a parameterizable step, called the tile
        size. The original dimension is changed to iterates over the
        range of the tile size, with the same step as before.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    dim_idx = Property(dtype=int, default=-1, desc="Index of dimension to be strip-mined")
    new_dim_prefix = Property(dtype=str, default="tile", desc="Prefix for new dimension name")
    tile_size = SymbolicProperty(default=64,
                                 desc="Tile size of strip-mined dimension, "
                                 "or number of tiles if tiling_type=number_of_tiles")
    tile_stride = SymbolicProperty(default=0,
                                   desc="Stride between two tiles of the "
                                   "strip-mined dimension. If zero, it is set "
                                   "equal to the tile size.")
    tile_offset = SymbolicProperty(default=0, desc="Tile stride offset (negative)")
    divides_evenly = Property(dtype=bool, default=False, desc="Tile size divides dimension range evenly?")
    strided = Property(dtype=bool, default=False, desc="Continuous (false) or strided (true) elements in tile")

    tiling_type = EnumProperty(dtype=dtypes.TilingType,
                               default=dtypes.TilingType.Normal,
                               allow_none=True,
                               desc="normal: the outerloop increments with tile_size, "
                               "ceilrange: uses ceiling(N/tile_size) in outer range, "
                               "number_of_tiles: tiles the map into the number of provided tiles, "
                               "provide the number of tiles over tile_size")

    skew = Property(dtype=bool, default=False, desc="If True, offsets inner tile back such that it starts with zero")

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def match_to_str(self, graph: SDFGState) -> str:
        return self.map_entry.map.label + ': ' + str(self.map_entry.map.params)

    def apply(self, graph: SDFGState, sdfg: SDFG) -> nodes.Map:
        # Strip-mine selected dimension.
        _, _, new_map = self._stripmine(sdfg, graph, self.map_entry)
        return new_map

    def _find_new_dim(self, sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry, prefix: str, target_dim: str):
        """ Finds a variable that is not already defined in scope. """
        stree = state.scope_tree()
        if len(prefix) == 0:
            return target_dim
        candidate = '%s_%s' % (prefix, target_dim)
        index = 1
        defined_vars = set(str(s) for s in (state.symbols_defined_at(entry).keys() | sdfg.symbols.keys()))
        while candidate in defined_vars:
            candidate = '%s%d_%s' % (prefix, index, target_dim)
            index += 1
        return candidate

    def _create_strided_range(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
        map_exit = state.exit_node(map_entry)
        dim_idx = self.dim_idx
        new_dim_prefix = self.new_dim_prefix
        tile_size = self.tile_size
        divides_evenly = self.divides_evenly
        tile_stride = self.tile_stride
        if tile_stride == 0:
            tile_stride = tile_size
        if tile_stride != tile_size:
            raise NotImplementedError

        # Retrieve parameter and range of dimension to be strip-mined.
        target_dim = map_entry.map.params[dim_idx]
        td_from, td_to, td_step = map_entry.map.range[dim_idx]
        new_dim = self._find_new_dim(sdfg, state, map_entry, new_dim_prefix, target_dim)
        new_dim_range = (td_from, td_to, tile_size)
        new_map = nodes.Map(map_entry.map.label, [new_dim], subsets.Range([new_dim_range]))

        dimsym = dace.symbolic.pystr_to_symbolic(new_dim)
        td_from_new = dimsym
        if divides_evenly:
            td_to_new = dimsym + tile_size - 1
        else:
            if isinstance(td_to, dace.symbolic.SymExpr):
                td_to = td_to.expr
            td_to_new = dace.symbolic.SymExpr(sympy.Min(dimsym + tile_size - 1, td_to), dimsym + tile_size - 1)
        td_step_new = td_step

        return new_dim, new_map, (td_from_new, td_to_new, td_step_new)

    def _create_ceil_range(self, sdfg: SDFG, graph: SDFGState, map_entry: nodes.MapEntry):
        map_exit = graph.exit_node(map_entry)

        # Retrieve transformation properties.
        dim_idx = self.dim_idx
        new_dim_prefix = self.new_dim_prefix
        tile_size = self.tile_size
        divides_evenly = self.divides_evenly
        strided = self.strided
        offset = self.tile_offset

        tile_stride = self.tile_stride
        if tile_stride == 0:
            tile_stride = tile_size

        # Retrieve parameter and range of dimension to be strip-mined.
        target_dim = map_entry.map.params[dim_idx]
        td_from, td_to, td_step = map_entry.map.range[dim_idx]
        # Create new map. Replace by cloning map object?
        new_dim = self._find_new_dim(sdfg, graph, map_entry, new_dim_prefix, target_dim)
        nd_from = 0
        if tile_stride == 1:
            nd_to = td_to - td_from
        else:
            nd_to = symbolic.pystr_to_symbolic(
                'int_ceil(%s + 1 - %s, %s) - 1' %
                (symbolic.symstr(td_to), symbolic.symstr(td_from), symbolic.symstr(tile_stride)))
        nd_step = 1
        new_dim_range = (nd_from, nd_to, nd_step)
        new_map = nodes.Map(new_dim + '_' + map_entry.map.label, [new_dim], subsets.Range([new_dim_range]))

        # Change the range of the selected dimension to iterate over a single
        # tile
        if strided:
            td_from_new = symbolic.pystr_to_symbolic(new_dim)
            td_to_new_approx = td_to
            td_step = tile_size

        elif offset == 0:
            td_from_new = symbolic.pystr_to_symbolic(
                '%s + %s * %s' % (symbolic.symstr(td_from), symbolic.symstr(new_dim), symbolic.symstr(tile_stride)))
            td_to_new_exact = symbolic.pystr_to_symbolic(
                'min(%s + 1, %s + %s * %s + %s) - 1' %
                (symbolic.symstr(td_to), symbolic.symstr(td_from), symbolic.symstr(tile_stride),
                 symbolic.symstr(new_dim), symbolic.symstr(tile_size)))
            td_to_new_approx = symbolic.pystr_to_symbolic('%s + %s * %s + %s - 1' %
                                                          (symbolic.symstr(td_from), symbolic.symstr(tile_stride),
                                                           symbolic.symstr(new_dim), symbolic.symstr(tile_size)))

        else:
            # include offset
            td_from_new_exact = symbolic.pystr_to_symbolic(
                'max(%s,%s + %s * %s - %s)' %
                (symbolic.symstr(td_from), symbolic.symstr(td_from), symbolic.symstrtr(tile_stride),
                 symbolic.symstr(new_dim), symbolic.symstr(offset)))
            td_from_new_approx = symbolic.pystr_to_symbolic('%s + %s * %s - %s ' %
                                                            (symbolic.symstr(td_from), symbolic.symstr(tile_stride),
                                                             symbolic.symstr(new_dim), symbolic.symstr(offset)))
            td_from_new = dace.symbolic.SymExpr(td_from_new_exact, td_from_new_approx)

            td_to_new_exact = symbolic.pystr_to_symbolic(
                'min(%s + 1, %s + %s * %s + %s - %s) -1' %
                (symbolic.symstr(td_to), symbolic.symstr(td_from), symbolic.symstr(tile_stride),
                 symbolic.symstr(new_dim), symbolic.symstr(tile_size), symbolic.symstr(offset)))
            td_to_new_approx = symbolic.pystr_to_symbolic(
                '%s + %s * %s + %s - %s - 1' %
                (symbolic.symstr(td_from), symbolic.symstr(tile_stride), symbolic.symstr(new_dim),
                 symbolic.symstr(tile_size), symbolic.symstr(offset)))

        if divides_evenly or strided:
            td_to_new = td_to_new_approx
        else:
            td_to_new = dace.symbolic.SymExpr(td_to_new_exact, td_to_new_approx)
        return new_dim, new_map, (td_from_new, td_to_new, td_step)

    def _create_from_tile_numbers(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
        map_exit = state.exit_node(map_entry)

        # Retrieve transformation properties.
        dim_idx = self.dim_idx
        new_dim_prefix = self.new_dim_prefix
        divides_evenly = self.divides_evenly
        number_of_tiles = self.tile_size
        tile_stride = self.tile_stride

        number_of_tiles = dace.symbolic.pystr_to_symbolic(number_of_tiles)

        # Retrieve parameter and range of dimension to be strip-mined.
        target_dim = map_entry.map.params[dim_idx]
        td_from, td_to, td_step = map_entry.map.range[dim_idx]
        size = map_entry.map.range.size_exact()[dim_idx]

        if tile_stride != 0:
            raise NotImplementedError

        new_dim = self._find_new_dim(sdfg, state, map_entry, new_dim_prefix, target_dim)
        new_dim_range = (td_from, number_of_tiles - 1, 1)
        new_map = nodes.Map(map_entry.map.label, [new_dim], subsets.Range([new_dim_range]))

        dimsym = dace.symbolic.pystr_to_symbolic(new_dim)
        td_from_new = (dimsym * size) // number_of_tiles
        if divides_evenly:
            td_to_new = ((dimsym + 1) * size) // number_of_tiles - 1
        else:
            if isinstance(td_to, dace.symbolic.SymExpr):
                td_to = td_to.expr
            td_to_new = dace.symbolic.SymExpr(
                sympy.Min(((dimsym + 1) * size) // number_of_tiles, td_to + 1) - 1,
                ((dimsym + 1) * size) // number_of_tiles - 1)
        td_step_new = td_step
        return new_dim, new_map, (td_from_new, td_to_new, td_step_new)

    def _stripmine(self, sdfg: SDFG, graph: SDFGState, map_entry: nodes.MapEntry):
        # Retrieve map entry and exit nodes.
        map_exit = graph.exit_node(map_entry)

        # Retrieve transformation properties.
        dim_idx = self.dim_idx
        target_dim = map_entry.map.params[dim_idx]

        if self.tiling_type == dtypes.TilingType.CeilRange:
            new_dim, new_map, td_rng = self._create_ceil_range(sdfg, graph, map_entry)
        elif self.tiling_type == dtypes.TilingType.NumberOfTiles:
            new_dim, new_map, td_rng = self._create_from_tile_numbers(sdfg, graph, map_entry)
        else:
            new_dim, new_map, td_rng = self._create_strided_range(sdfg, graph, map_entry)

        new_map_entry = nodes.MapEntry(new_map)
        new_map_exit = nodes.MapExit(new_map)

        td_to_new_approx = td_rng[1]
        if isinstance(td_to_new_approx, dace.symbolic.SymExpr):
            td_to_new_approx = td_to_new_approx.approx

        # Special case: If range is 1 and no prefix was specified, skip range
        if td_rng[0] == td_to_new_approx and target_dim == new_dim:
            map_entry.map.range = subsets.Range([r for i, r in enumerate(map_entry.map.range) if i != dim_idx])
            map_entry.map.params = [p for i, p in enumerate(map_entry.map.params) if i != dim_idx]
            if len(map_entry.map.params) == 0:
                raise ValueError('Strip-mining all dimensions of the map with ' 'empty tiles is disallowed')
        else:
            map_entry.map.range[dim_idx] = td_rng

        # Make internal map's schedule to "not parallel"
        new_map.schedule = map_entry.map.schedule
        map_entry.map.schedule = dtypes.ScheduleType.Sequential

        # Redirect edges
        new_map_entry.in_connectors = dcpy(map_entry.in_connectors)
        sdutil.change_edge_dest(graph, map_entry, new_map_entry)
        new_map_exit.out_connectors = dcpy(map_exit.out_connectors)
        sdutil.change_edge_src(graph, map_exit, new_map_exit)

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

        # Skew if necessary
        if self.skew:
            xfh.offset_map(sdfg, graph, map_entry, dim_idx, td_rng[0])

        # Return strip-mined dimension.
        return target_dim, new_dim, new_map
