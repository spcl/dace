# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Moves a loop around a map into the map """

import copy
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
import dace.transformation.helpers as helpers
import networkx as nx
from dace.sdfg.scope import ScopeTree
from dace import Memlet, nodes, sdfg as sd, subsets as sbs, symbolic, symbol
from dace.sdfg import nodes, propagation, utils as sdutil
from dace.transformation import transformation
from sympy import diff
from typing import List, Set, Tuple

from dace.transformation.passes.analysis import loop_analysis


def fold(memlet_subset_ranges, itervar, lower, upper):
    return [(r[0].replace(symbol(itervar), lower), r[1].replace(symbol(itervar), upper), r[2])
            for r in memlet_subset_ranges]


def offset(memlet_subset_ranges, value):
    return (memlet_subset_ranges[0] + value, memlet_subset_ranges[1] + value, memlet_subset_ranges[2])


@transformation.explicit_cf_compatible
class MoveLoopIntoMap(transformation.MultiStateTransformation):
    """
    Moves a loop around a map into the map
    """

    loop = transformation.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        if step not in [-1, 1]:
            return False

        # Body must contain a single state
        if len(self.loop.nodes()) != 1 or not isinstance(self.loop.nodes()[0], SDFGState):
            return False
        body: SDFGState = self.loop.nodes()[0]

        # Body must have only a single connected component
        # NOTE: This is a strict check that can be potentially relaxed.
        # If only one connected component includes a Map and the others do not create RW dependencies, then we could
        # proceed with the transformation. However, that would be a case of an SDFG with redundant computation/copying,
        # which is unlikely after simplification transformations. Alternatively, we could try to apply the
        # transformation to each component separately, but this would require a lot more checks.
        if len(list(nx.weakly_connected_components(body._nx))) > 1:
            return False

        # Check if body contains exactly one map
        maps = [node for node in body.nodes() if isinstance(node, nodes.MapEntry)]
        if len(maps) != 1:
            return False

        map_entry = maps[0]
        map_exit = body.exit_node(map_entry)
        subgraph = body.scope_subgraph(map_entry)
        read_set, write_set = body.read_and_write_sets()

        # Check for iteration variable in map and data descriptors
        if str(itervar) in map_entry.free_symbols:
            return False
        for arr in (read_set | write_set):
            if str(itervar) in set(map(str, sdfg.arrays[arr].free_symbols)):
                return False

        # Check that everything else outside the Map is independent of the loop's itervar
        for e in body.edges():
            if e.src in subgraph.nodes() or e.dst in subgraph.nodes():
                continue
            if e.dst is map_entry and isinstance(e.src, nodes.AccessNode):
                continue
            if e.src is map_exit and isinstance(e.dst, nodes.AccessNode):
                continue
            if str(itervar) in e.data.free_symbols:
                return False
            if isinstance(e.dst, nodes.AccessNode) and e.dst.data in read_set:
                # NOTE: This is strict check that can be potentially relaxed.
                # If some data written indirectly by the Map (i.e., it is not an immediate output of the MapExit) is
                # also read, then abort. In practice, we could follow the edges and with subset compositions figure out
                # if there is a RW dependency on the loop variable. However, in such complicated cases, it is far more
                # likely that the simplification redundant array/copying transformations trigger first. If they don't,
                # this is a good hint that there is a RW dependency.
                if nx.has_path(body._nx, map_exit, e.dst):
                    return False
        for n in body.nodes():
            if n in subgraph.nodes():
                continue
            if str(itervar) in n.free_symbols:
                return False

        def test_subset_dependency(subset: sbs.Subset, mparams: Set[int]) -> Tuple[bool, List[int]]:
            dims = []
            for i, r in enumerate(subset):
                if not isinstance(r, (list, tuple)):
                    r = [r]
                fsymbols = set()
                for token in r:
                    if symbolic.issymbolic(token):
                        fsymbols = fsymbols.union({str(s) for s in token.free_symbols})
                if itervar in fsymbols:
                    if fsymbols.intersection(mparams):
                        return (False, [])
                    else:
                        # Strong checks
                        if not permissive:
                            # Only indices allowed
                            if len(r) > 1 and r[0] != r[1]:
                                return (False, [])
                            derivative = diff(r[0])
                            # Index function must be injective
                            if not (((derivative > 0) == True) or ((derivative < 0) == True)):
                                return (False, [])
                        dims.append(i)
            return (True, dims)

        # Check that Map memlets depend on itervar in a consistent manner
        # a. A container must either not depend at all on itervar, or depend on it always in the same dimensions.
        # b. Abort when a dimension depends on both the itervar and a Map parameter.
        mparams = set(map_entry.map.params)
        data_dependency = dict()
        for e in body.edges():
            if e.src in subgraph.nodes() and e.dst in subgraph.nodes():
                if itervar in e.data.free_symbols:
                    e.data.try_initialize(sdfg, subgraph, e)
                    for i, subset in enumerate((e.data.src_subset, e.data.dst_subset)):
                        if subset:
                            if i == 0:
                                access = body.memlet_path(e)[0].src
                            else:
                                access = body.memlet_path(e)[-1].dst
                            passed, dims = test_subset_dependency(subset, mparams)
                            if not passed:
                                return False
                            if dims:
                                if access.data in data_dependency:
                                    if data_dependency[access.data] != dims:
                                        return False
                                else:
                                    data_dependency[access.data] = dims

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        body: sd.SDFGState = self.loop.nodes()[0]
        itervar = self.loop.loop_variable

        for node in body.nodes():
            if isinstance(node, nodes.MapEntry):
                map_entry = node
            if isinstance(node, nodes.MapExit):
                map_exit = node

        # nest map's content in sdfg
        map_subgraph = body.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        nsdfg = helpers.nest_state_subgraph(sdfg, body, map_subgraph, full_data=True)
        nested_state: SDFGState = nsdfg.sdfg.nodes()[0]

        # replicate loop in nested sdfg
        inner_loop = LoopRegion(self.loop.label, self.loop.loop_condition, self.loop.loop_variable,
                                self.loop.init_statement, self.loop.update_statement, self.loop.inverted, nsdfg,
                                self.loop.update_before_condition)
        inner_loop.add_node(nested_state, is_start_block=True)
        nsdfg.sdfg.remove_node(nested_state)
        nsdfg.sdfg.add_node(inner_loop, is_start_block=True)

        graph.add_node(body, is_start_block=(graph.start_block is self.loop))
        for ie in graph.in_edges(self.loop):
            graph.add_edge(ie.src, body, ie.data)
        for oe in graph.out_edges(self.loop):
            graph.add_edge(body, oe.dst, oe.data)
        graph.remove_node(self.loop)

        if itervar in nsdfg.symbol_mapping:
            del nsdfg.symbol_mapping[itervar]
        if itervar in sdfg.symbols:
            del sdfg.symbols[itervar]

        # Add missing data/symbols
        for s in nsdfg.sdfg.free_symbols:
            if s in nsdfg.symbol_mapping:
                continue
            if s in sdfg.symbols:
                nsdfg.symbol_mapping[s] = s
            elif s in sdfg.arrays:
                desc = sdfg.arrays[s]
                access = body.add_access(s)
                conn = nsdfg.sdfg.add_datadesc(s, copy.deepcopy(desc))
                nsdfg.sdfg.arrays[s].transient = False
                nsdfg.add_in_connector(conn)
                body.add_memlet_path(access, map_entry, nsdfg, memlet=Memlet.from_array(s, desc), dst_conn=conn)
            else:
                raise NotImplementedError(f"Free symbol {s} is neither a symbol nor data.")
        to_delete = set()
        for s in nsdfg.symbol_mapping:
            if s not in nsdfg.sdfg.free_symbols:
                to_delete.add(s)
        for s in to_delete:
            del nsdfg.symbol_mapping[s]

        # propagate scope for correct volumes
        scope_tree = ScopeTree(map_entry, map_exit)
        scope_tree.parent = ScopeTree(None, None)
        # The first execution helps remove apperances of symbols
        # that are now defined only in the nested SDFG in memlets.
        propagation.propagate_memlets_scope(sdfg, body, scope_tree)

        for s in to_delete:
            if helpers.is_symbol_unused(sdfg, s):
                sdfg.remove_symbol(s)

        sdfg.reset_cfg_list()

        from dace.transformation.interstate import RefineNestedAccess
        transformation = RefineNestedAccess()
        transformation.setup_match(sdfg, body.parent_graph.cfg_id, body.block_id,
                                   {RefineNestedAccess.nsdfg: body.node_id(nsdfg)}, 0)
        transformation.apply(body, sdfg)

        # Second propagation for refined accesses.
        propagation.propagate_memlets_scope(sdfg, body, scope_tree)
