""" Contains classes that implement transformations related to distributed
    data. """

import copy
import warnings
from abc import ABC
import random
import sympy

from dace import dtypes, registry, symbolic, subsets, sdfg as sd
from dace.properties import (LambdaProperty, Property, ShapeProperty, DictProperty,
                             TypeProperty, ListProperty, make_properties)
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching


@registry.autoregister_params(singlestate=True)
@make_properties
class DataDistribution(pattern_matching.Transformation):
    """ Implements the Data Distribution transformation, that
        distributes data among multiple ranks.
    """

    _access_node = nodes.AccessNode('')

    array = Property(
        dtype=str,
        desc="Array to distribute",
        allow_none=False)

    space = Property(
        dtype=str,
        desc="Space to use",
        allow_none=False)

    arrayspace_mapping = DictProperty(
        key_type=int,
        value_type=int,
        desc="Mapping from array to coordinate space dimensions",
        allow_none=False)

    constant_offsets = ListProperty(
        element_type=int,
        desc="Constant offsets for every space dimension",
        allow_none=False)
    
    dependent_offsets = ListProperty(
        element_type=int,
        desc="Dependent offsets for every space dimension",
        allow_none=False)
    
    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        super().__init__(sdfg_id, state_id, subgraph, expr_index)

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(DataDistribution._access_node)
        ]

    @staticmethod
    def match_to_str(graph, candidate):
        node = graph.nodes()[candidate[DataDistribution._access_node]]
        return '%s' % node

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        node = graph.nodes()[candidate[DataDistribution._access_node]]
        data = sdfg.arrays[node.data]
        if data.dist_shape:
            return False
        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        node = graph.nodes()[self.subgraph[DataDistribution._access_node]]

        if self.array:
            for n in graph.nodes():
                if isinstance(n, nodes.AccessNode) and n.data == self.array:
                    node = n
                    break

        data = sdfg.arrays[node.data]
        data.space = self.space
        data.arrayspace_mapping = self.arrayspace_mapping
        data.constant_offsets = self.constant_offsets
        data.dependent_offsets = self.dependent_offsets

        space = sdfg.spaces[self.space]
        dims = len(data.shape)
        pdims = len(space.process_grid)

        inv_mapping = {}
        for k, v in self.arrayspace_mapping.items():
            inv_mapping[v] = k

        new_shape = [1] * 2 * dims
        new_dist_shape = [1] * pdims
        for k, v in self.arrayspace_mapping.items():
            new_shape[k] = symbolic.pystr_to_symbolic(
                "int_ceil({}, ({}) * ({}))".format(data.shape[k],
                                                   space.block_sizes[v],
                                                   space.process_grid[v]))
            new_shape[k + dims] = space.block_sizes[v]
        for k, v in inv_mapping.items():
            new_dist_shape[k] = space.process_grid[k]
        data.shape = new_shape
        data.dist_shape = new_dist_shape

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
                            if e2.data.data != self.array:
                                visited_edges.add(e2)
                                continue
                            edges.add(e2)
                            visited_edges.add(e2)

        for e in edges:
            local_ranges = [(0, 0, 1)] * 2 * dims
            dist_ranges = [(0, 0, 1)] * pdims
            for k, v in self.arrayspace_mapping.items():
                # Nasty hack because sympy cannot do 0 // symbol
                # but does 0 / symbol
                op0 = "//"
                op1 = "//"
                if isinstance(e.data.subset, subsets.Range):
                    subset0 = e.data.subset[k][0]
                    subset1 = e.data.subset[k][1]
                    if e.data.subset[k][0] == 0:
                        op0 = "/"
                    if e.data.subset[k][1] == 0:
                        op1 = "/"
                else:
                    subset0 = e.data.subset[k]
                    subset1 = e.data.subset[k]
                    if e.data.subset[k] == 0:
                        op0 = "/"
                        op1 = "/"
                local_ranges[k] = (
                    symbolic.pystr_to_symbolic(
                        "({}) {} (({}) * ({}))".format(
                            subset0, op0,
                            space.block_sizes[v],
                            space.process_grid[v])),
                    symbolic.pystr_to_symbolic(
                        "({}) {} (({}) * ({}))".format(
                            subset1, op1,
                            space.block_sizes[v],
                            space.process_grid[v])), 1)
                local_ranges[k + dims] = (
                    symbolic.pystr_to_symbolic(
                        "({}) % ({})".format(subset0, space.block_sizes[v])),
                    symbolic.pystr_to_symbolic(
                        "({}) % ({})".format(subset1, space.block_sizes[v])), 1)
            for k, v in inv_mapping.items():
                # Nasty hack because sympy cannot do 0 // symbol
                # but does 0 / symbol
                op0 = "//"
                op1 = "//"
                if isinstance(e.data.subset, subsets.Range):
                    subset0 = e.data.subset[v][0]
                    subset1 = e.data.subset[v][1]
                    if e.data.subset[v][0] == 0:
                        op0 = "/"
                    if e.data.subset[v][1] == 0:
                        op1 = "/"
                else:
                    subset0 = e.data.subset[v]
                    subset1 = e.data.subset[v]
                    if e.data.subset[v] == 0:
                        op0 = "/"
                        op1 = "/"
                dist_ranges[k] = (
                    symbolic.pystr_to_symbolic(
                        "(({}) {} ({})) % ({})".format(
                            subset0, op0, 
                            space.block_sizes[k],
                            space.process_grid[k])),
                    symbolic.pystr_to_symbolic(
                        "(({}) {} ({})) % ({})".format(
                            subset1, op1,
                            space.block_sizes[k],
                            space.process_grid[k])), 1)
            e.data.subset = subsets.Range(local_ranges)
            e.data.dist_subset = subsets.Range(dist_ranges)


@registry.autoregister_params(singlestate=True)
@make_properties
class MapDistribution(pattern_matching.Transformation):
    """ Implements the Map Distribution transformation, that
        distributes the iteration space of a map among multiple ranks.
    """

    _map_entry = nodes.MapEntry(nodes.Map('', [], []))

    space = Property(
        dtype=str,
        desc="Space to use",
        allow_none=False)

    iterationspace_mapping = DictProperty(
        key_type=int,
        value_type=int,
        desc="Mapping from iteration to coordinate space dimensions",
        allow_none=False)

    constant_offsets = ListProperty(
        element_type=int,
        desc="Constant offsets for every space dimension",
        allow_none=False)
    
    dependent_offsets = ListProperty(
        element_type=int,
        desc="Dependent offsets for every space dimension",
        allow_none=False)
    
    def __init__(self, sdfg_id, state_id, subgraph, expr_index):
        super().__init__(sdfg_id, state_id, subgraph, expr_index)

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(MapDistribution._map_entry)
        ]

    @staticmethod
    def match_to_str(graph, candidate):
        node = graph.nodes()[candidate[MapDistribution._map_entry]]
        return '%s' % node

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # node = graph.nodes()[candidate[MapDistribution._map_entry]]
        # data = sdfg.arrays[node.data]
        # if data.dist_shape:
        #     return False
        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        mentry = graph.nodes()[self.subgraph[MapDistribution._map_entry]]
        mexit = graph.exit_nodes(mentry)[0]
        mname = mentry.map.label
        midx = mentry.map.params
        mspace = mentry.map.range

        space = sdfg.spaces[self.space]
        dims = len(mspace)
        pdims = len(space.process_grid)

        inv_mapping = {}
        for k, v in self.iterationspace_mapping.items():
            inv_mapping[v] = k

        # pidx = ['p_' + str(i) for i in range(pdims)]
        # for k, v in inv_mapping.items():
        #     pidx[k] = 'p_' + midx[v]
        # pspace = [(0, s-1, 1) for s in space.process_grid]
        pidx = ['p_' + idx for idx in midx]
        pspace = [None] * dims
        for k, v in self.iterationspace_mapping.items():
            pspace[k] = (0, space.process_grid[v] - 1, 1)
        pmap = nodes.Map('p_' + mname, pidx, subsets.Range(pspace))
        pentry = nodes.MapEntry(pmap)
        pexit = nodes.MapExit(pmap)

        lidx = ['l_' + idx for idx in midx]
        oidx = [idx for idx in midx]
        lspace = [None] * dims
        ospace = [None] * dims
        for k, v in self.iterationspace_mapping.items():
            lspace[k] = (
                0, "int_ceil({} - {} + 1, ({}) * ({}))".format(
                    mspace[k][1], mspace[k][0], space.block_sizes[v],
                    space.process_grid[v]), 1)
            ospace[k] = (
                "{} + ({} * ({}) + {}) * ({})".format(
                    mspace[k][0], lidx[k], space.process_grid[v],
                    pidx[k], space.block_sizes[v]),
                 "{} + ({} * ({}) + {} + 1) * ({}) - 1".format(
                    mspace[k][0], lidx[k], space.process_grid[v],
                    pidx[k], space.block_sizes[v]), 1)
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

        fidx = sympy.Wild("fidx")
        proc = sympy.Wild("proc")
        block = sympy.Wild("block")

        for e1 in mentry_out_edges:
            for e2 in mentry_in_edges:
                if e2.data.data == e1.data.data:
                    mem = e1.data
                    data = sdfg.arrays[e1.data.data]
                    if data.space == self.space:
                        new_subset = [None] * len(data.shape)
                        new_dist_subset = [(0, 0, 1)] * pdims
                        for k, v in data.arrayspace_mapping.items():
                            simplify = False
                            matches = mem.subset[k][0].match(
                                fidx // (proc * block))
                            if matches and v in inv_mapping.keys():
                                idx_expr = matches[fidx]
                                actual_v = inv_mapping[v]
                                if (symbolic.pystr_to_symbolic(oidx[actual_v])
                                        in idx_expr.free_symbols):
                                    simplify = True
                            if simplify:
                                new_subset[k] = (lidx[actual_v],
                                                 lidx[actual_v], 1)
                                new_subset[k + len(data.shape) // 2] = (
                                    "{} - ({})".format(oidx[actual_v],
                                                       ospace[actual_v][0]),
                                    "{} - ({})".format(oidx[actual_v],
                                                       ospace[actual_v][0]), 1)
                                new_dist_subset[v] = (pidx[actual_v],
                                                      pidx[actual_v], 1)
                            else:
                                new_subset[k] = mem.subset[k]
                                new_subset[k + len(data.shape) // 2] = mem.subset[k + len(data.shape) // 2]
                                new_dist_subset[v] = mem.dist_subset[v]
                        mem.subset = subsets.Range(new_subset)
                        mem.dist_subset = subsets.Range(new_dist_subset)
                    graph.add_memlet_path(
                        e2.src, pentry, lentry, oentry, e1.dst, memlet=mem,
                        src_conn=e2.src_conn, dst_conn=e1.dst_conn)

        for e1 in mexit_in_edges:
            for e2 in mexit_out_edges:
                if e2.data.data == e1.data.data:
                    mem = e1.data
                    data = sdfg.arrays[e1.data.data]
                    if data.space == self.space:
                        new_subset = [None] * len(data.shape)
                        new_dist_subset = [(0, 0, 1)] * pdims
                        for k, v in data.arrayspace_mapping.items():
                            simplify = False
                            matches = mem.subset[k][0].match(
                                fidx // (proc * block))
                            if matches and v in inv_mapping.keys():
                                idx_expr = matches[fidx]
                                actual_v = inv_mapping[v]
                                if (symbolic.pystr_to_symbolic(oidx[actual_v])
                                        in idx_expr.free_symbols):
                                    simplify = True
                            if simplify:
                                new_subset[k] = (lidx[actual_v],
                                                 lidx[actual_v], 1)
                                new_subset[k + len(data.shape) // 2] = (
                                    "{} - ({})".format(oidx[actual_v],
                                                       ospace[actual_v][0]),
                                    "{} - ({})".format(oidx[actual_v],
                                                       ospace[actual_v][0]), 1)
                                new_dist_subset[v] = (pidx[actual_v],
                                                      pidx[actual_v], 1)
                            else:
                                new_subset[k] = mem.subset[k]
                                new_subset[k + len(data.shape) // 2] = mem.subset[k + len(data.shape) // 2]
                                new_dist_subset[v] = mem.dist_subset[v]
                        mem.subset = subsets.Range(new_subset)
                        mem.dist_subset = subsets.Range(new_dist_subset)
                    graph.add_memlet_path(
                        e1.src, oexit, lexit, pexit, e2.dst, memlet=mem,
                        src_conn=e1.src_conn, dst_conn=e2.dst_conn)
