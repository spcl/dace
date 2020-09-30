""" Block-cyclic distribution of data and maps. """

import copy
import functools
import warnings
from abc import ABC
import random
from sympy import ceiling, floor, Mod, Rational

from dace import dtypes, registry, symbolic, subsets, sdfg as sd
from dace.properties import (LambdaProperty, Property, ShapeProperty, DictProperty,
                             TypeProperty, ListProperty, make_properties)
from dace.sdfg import nodes, utils
from dace.transformation import pattern_matching
from dace.symbolic import pystr_to_symbolic as strsym


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@registry.autoregister_params(singlestate=True)
@make_properties
class BlockCyclicData(pattern_matching.Transformation):
    """ Block-Cycic distributed data.
    """

    _access_node = nodes.AccessNode('')

    dataname = Property(
        dtype=str,
        desc="Name of data to distributed",
        allow_none=False)

    gridname = Property(
        dtype=str,
        desc="Name of process grid to distribute data to",
        allow_none=False)

    block = ShapeProperty(default=[], allow_none=False)
    
    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        super().__init__(sdfg_id, state_id, subgraph, expr_index)

    @staticmethod
    def expressions():
        return [utils.node_path_graph(BlockCyclicData._access_node)]

    @staticmethod
    def match_to_str(graph, candidate):
        node = graph.nodes()[candidate[BlockCyclicData._access_node]]
        return '%s' % node

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        node = graph.nodes()[candidate[BlockCyclicData._access_node]]
        data = sdfg.arrays[node.data]
        if len(data.dist_shape) > 0:
            return False
        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        node = graph.nodes()[self.subgraph[BlockCyclicData._access_node]]

        if self.dataname:
            for n in graph.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == self.dataname:
                    node = n
                    break

        data = sdfg.arrays[node.data]
        data.process_grid = self.gridname

        # Add block sizes to symbols
        for bsize in self.block:
            if symbolic.issymbolic(bsize) and bsize not in sdfg.symbols:
                sdfg.add_symbol(str(bsize), int)

        pgrid = sdfg.process_grids[self.gridname]
        dims = len(data.shape)
        pdims = len(pgrid.grid)

        # TODO: Shouldn't the dimensions agree after all?
        # (Based on new theory and not the old common spaces)
        new_dist_shape = [None] * pdims
        new_shape = [None] * 2 * dims

        for i in range(dims):
            new_dist_shape[i] = pgrid.grid[i]
            new_shape[i] = ceiling(data.shape[i] / (pgrid.grid[i] * self.block[i]))
            new_shape[i + dims] = self.block[i]

        # Change data properties
        data.storage = dtypes.StorageType.Distributed
        data.dist_shape = new_dist_shape
        data.shape = new_shape
        data.total_size = _prod(new_shape)

        # TODO: What happens if subset of edge is true range instead of index?
        edges = set()
        visited_edges = set()
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    for e1 in state.all_edges(node):
                        for e2 in state.memlet_tree(e1):
                            if e2 in visited_edges:
                                continue
                            if (not isinstance(e2.src, nodes.CodeNode) and
                                    not isinstance(e2.dst, nodes.CodeNode)):
                                visited_edges.add(e2)
                                continue
                            if e2.data.data != self.dataname:
                                visited_edges.add(e2)
                                continue
                            edges.add(e2)
                            visited_edges.add(e2)

        for e in edges:
            dist_ranges = [None] * pdims
            local_ranges = [None] * 2 * dims
            for i in range(dims):
                if isinstance(e.data.subset, subsets.Range):
                    oldval = e.data.subset[i][0]
                else:
                    oldval = e.data.subset[i]
                pval = Mod(floor(oldval / self.block[i]), pgrid.grid[i])
                lval = floor(oldval / (pgrid.grid[i] * self.block[i]))
                oval = Mod(oldval, self.block[i])
                dist_ranges[i] = (pval, pval, 1)
                local_ranges[i] = (lval, lval, 1)
                local_ranges[i + dims] = (oval, oval, 1)
            e.data.subset = subsets.Range(local_ranges)
            e.data.dist_subset = subsets.Range(dist_ranges)


@registry.autoregister_params(singlestate=True)
@make_properties
class BlockCyclicMap(pattern_matching.Transformation):
    """ Block-cyclic distribution of a Map.
    """

    _map_entry = nodes.MapEntry(nodes.Map('', [], []))

    gridname = Property(
        dtype=str,
        desc="Name of process grid to distribute Map to",
        allow_none=False)

    block = ShapeProperty(default=[], allow_none=False)
    
    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        super().__init__(sdfg_id, state_id, subgraph, expr_index)

    @staticmethod
    def expressions():
        return [utils.node_path_graph(BlockCyclicMap._map_entry)]

    @staticmethod
    def match_to_str(graph, candidate):
        node = graph.nodes()[candidate[BlockCyclicMap._map_entry]]
        return '%s' % node

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        mentry = graph.nodes()[self.subgraph[BlockCyclicMap._map_entry]]
        mexit = graph.exit_node(mentry)
        mname = mentry.map.label
        midx = mentry.map.params
        mspace = mentry.map.range

        # Add block sizes to symbols
        for bsize in self.block:
            if symbolic.issymbolic(bsize) and bsize not in sdfg.symbols:
                sdfg.add_symbol(str(bsize), int)

        pgrid = sdfg.process_grids[self.gridname]
        dims = len(mspace)
        pdims = len(pgrid.grid)

        pidx = ['p_' + idx for idx in midx]
        lidx = ['l_' + idx for idx in midx]
        oidx = [idx for idx in midx]
        pspace = [None] * dims
        lspace = [None] * dims
        ospace = [None] * dims

        for i in range(dims):
            pspace[i] = (0, pgrid.grid[i] - 1, 1)
            lspace[i] = (0, ceiling((mspace[i][1] - mspace[i][0] + 1) / (self.block[i] * pgrid.grid[i])) - 1, 1)
            ospace[i] = (mspace[i][0] + (strsym(lidx[i]) * pgrid.grid[i] + strsym(pidx[i])) * self.block[i],
                         mspace[i][0] + (strsym(lidx[i]) * pgrid.grid[i] + strsym(pidx[i]) + 1) * self.block[i] - 1, 1)
        pmap = nodes.Map('p_' + mname, pidx, subsets.Range(pspace),
                         schedule=dtypes.ScheduleType.MPI)
        pentry = nodes.MapEntry(pmap)
        pexit = nodes.MapExit(pmap)
        lmap = nodes.Map('l_' + mname, lidx, subsets.Range(lspace))
        lentry = nodes.MapEntry(lmap)
        lexit = nodes.MapExit(lmap)
        omap = nodes.Map('o_' + mname, oidx, subsets.Range(ospace))
        oentry = nodes.MapEntry(omap)
        oexit = nodes.MapExit(omap)

        mentry_in_edges = graph.in_edges(mentry)
        mentry_out_edges = graph.out_edges(mentry)
        mexit_in_edges = graph.in_edges(mexit)
        mexit_out_edges = graph.out_edges(mexit)

        graph.remove_nodes_from([mentry, mexit])

        for e1 in mentry_out_edges:
            for e2 in mentry_in_edges:
                if e2.data.data == e1.data.data:
                    mem = e1.data
                    graph.add_memlet_path(
                        e2.src, pentry, lentry, oentry, e1.dst, memlet=mem,
                        src_conn=e2.src_conn, dst_conn=e1.dst_conn)

        for e1 in mexit_in_edges:
            for e2 in mexit_out_edges:
                if e2.data.data == e1.data.data:
                    mem = e1.data
                    graph.add_memlet_path(
                        e1.src, oexit, lexit, pexit, e2.dst, memlet=mem,
                        src_conn=e1.src_conn, dst_conn=e2.dst_conn)
