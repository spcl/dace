# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Moves a loop around a map into the map """

import copy
import dace.transformation.helpers as helpers
import networkx as nx
from dace.sdfg.scope import ScopeTree
from dace import data as dt, Memlet, nodes, sdfg as sd, subsets as sbs, symbolic, symbol
from dace.properties import CodeBlock
from dace.sdfg import nodes, propagation
from dace.transformation import transformation
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
from sympy import diff
from typing import List, Set, Tuple


def fold(memlet_subset_ranges, itervar, lower, upper):
    return [(r[0].replace(symbol(itervar), lower), r[1].replace(symbol(itervar), upper), r[2])
            for r in memlet_subset_ranges]


def offset(memlet_subset_ranges, value):
    return (memlet_subset_ranges[0] + value, memlet_subset_ranges[1] + value, memlet_subset_ranges[2])


class MoveLoopIntoMap(DetectLoop, transformation.MultiStateTransformation):
    """
    Moves a loop around a map into the map
    """

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin
        after: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range, and stride
        loop_info = find_for_loop(sdfg, guard, body)
        if not loop_info:
            return False
        itervar, (start, end, step), (_, body_end) = loop_info

        if step not in [-1, 1]:
            return False

        # Body must contain a single state
        if body != body_end:
            return False

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

    def apply(self, _, sdfg: sd.SDFG):
        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin

        # Obtain iteration variable, range, and stride
        itervar, (start, end, step), _ = find_for_loop(sdfg, guard, body)

        forward_loop = step > 0

        for node in body.nodes():
            if isinstance(node, nodes.MapEntry):
                map_entry = node
            if isinstance(node, nodes.MapExit):
                map_exit = node

        # nest map's content in sdfg
        map_subgraph = body.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        nsdfg = helpers.nest_state_subgraph(sdfg, body, map_subgraph, full_data=True)

        # replicate loop in nested sdfg
        new_before, new_guard, new_after, new_loop = nsdfg.sdfg.add_loop(
            before_state=None,
            loop_state=nsdfg.sdfg.nodes()[0],
            loop_end_state=None,
            after_state=None,
            loop_var=itervar,
            initialize_expr=f'{start}',
            condition_expr=f'{itervar} <= {end}' if forward_loop else f'{itervar} >= {end}',
            increment_expr=f'{itervar} + {step}' if forward_loop else f'{itervar} - {abs(step)}')

        for state in nsdfg.sdfg.nodes():
            new_loop.states.add(state)

        # remove outer loop
        before_guard_edge = nsdfg.sdfg.edges_between(new_before, new_guard)[0]
        for e in nsdfg.sdfg.out_edges(new_guard):
            if e.dst is new_after:
                guard_after_edge = e
            else:
                guard_body_edge = e

        for body_inedge in sdfg.in_edges(body):
            if body_inedge.src is guard:
                guard_body_edge.data.assignments.update(body_inedge.data.assignments)
            sdfg.remove_edge(body_inedge)
        for body_outedge in sdfg.out_edges(body):
            sdfg.remove_edge(body_outedge)
        for guard_inedge in sdfg.in_edges(guard):
            before_guard_edge.data.assignments.update(guard_inedge.data.assignments)
            guard_inedge.data.assignments = {}
            sdfg.add_edge(guard_inedge.src, body, guard_inedge.data)
            sdfg.remove_edge(guard_inedge)
        for guard_outedge in sdfg.out_edges(guard):
            if guard_outedge.dst is body:
                guard_body_edge.data.assignments.update(guard_outedge.data.assignments)
            else:
                guard_after_edge.data.assignments.update(guard_outedge.data.assignments)
            guard_outedge.data.condition = CodeBlock("1")
            sdfg.add_edge(body, guard_outedge.dst, guard_outedge.data)
            sdfg.remove_edge(guard_outedge)
        sdfg.remove_node(guard)
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

        from dace.transformation.interstate import RefineNestedAccess
        transformation = RefineNestedAccess()
        transformation.setup_match(sdfg, 0, sdfg.node_id(body), {RefineNestedAccess.nsdfg: body.node_id(nsdfg)}, 0)
        transformation.apply(body, sdfg)

        # Second propagation for refined accesses.
        propagation.propagate_memlets_scope(sdfg, body, scope_tree)
