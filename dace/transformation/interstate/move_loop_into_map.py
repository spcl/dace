# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Moves a loop around a map into the map """

import ast
import copy
import dace
import dace.transformation.helpers as helpers
import networkx as nx
from dace.codegen import control_flow as cf
from dace.sdfg.scope import ScopeTree
from dace import data as dt, Memlet, nodes, sdfg as sd, subsets as sbs, symbolic, symbol
from dace.properties import CodeBlock
from dace.sdfg import graph, nodes, propagation, utils as sdutil
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
            map_dims = 0
            found_conlict = False
            for i, r in enumerate(subset):
                if not isinstance(r, (list, tuple)):
                    r = [r]
                fsymbols = set()
                for token in r:
                    if symbolic.issymbolic(token):
                        fsymbols = fsymbols.union({str(s) for s in token.free_symbols})
                if any(p in fsymbols for p in mparams) and itervar not in fsymbols:
                    map_dims += 1
                if itervar in fsymbols:
                    if fsymbols.intersection(mparams):
                        return (False, [])
                    else:
                        # Strong checks
                        if not permissive:
                            # Only indices allowed
                            if len(r) > 1 and r[0] != r[1]:
                                # return (False, [])
                                found_conlict = True
                            derivative = diff(r[0])
                            # Index function must be injective
                            if not (((derivative > 0) == True) or ((derivative < 0) == True)):
                                # return (False, [])
                                found_conlict = True
                        dims.append(i)
            if map_dims == len(mparams):
                return (True, dims)
            elif found_conlict:
                return (False, [])
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
        new_before, new_guard, new_after = nsdfg.sdfg.add_loop(
            before_state=None,
            loop_state=nsdfg.sdfg.nodes()[0],
            loop_end_state=None,
            after_state=None,
            loop_var=itervar,
            initialize_expr=f'{start}',
            condition_expr=f'{itervar} <= {end}' if forward_loop else f'{itervar} >= {end}',
            increment_expr=f'{itervar} + {step}' if forward_loop else f'{itervar} - {abs(step)}')

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
            guard_inedge.data.assignments = {}  # Could there be here assignments needed outside?
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


class MoveMapIntoLoop(transformation.SingleStateTransformation):
    """
    Moves a map around a loop into the loop
    """

    map_entry = transformation.PatternNode(nodes.EntryNode)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    map_exit = transformation.PatternNode(nodes.ExitNode)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):

        # If the body a loop?
        nsdfg = self.nested_sdfg.sdfg
        components = helpers.find_sdfg_control_flow(nsdfg)
        # Body must contain a single control-flow component
        if len(components) != 1:
            return False
        cf_node: cf.ControlFlow
        _, cf_node = list(components.values())[0]
        # Component must be ForScope
        if not isinstance(cf_node, cf.ForScope):
            return False

        mparams = set(self.map_entry.map.params)

        # Obtain loop information
        guard: sd.SDFGState = cf_node.guard
        body: sd.SDFGState = cf_node.body.first_state

        # Obtain iteration variable, range, and stride
        loop_info = find_for_loop(nsdfg, guard, body)
        if not loop_info:
            return False
        itervar, (start, end, step), (_, body_end) = loop_info
        dependent_symbols = set(mparams)
        for k, v in self.nested_sdfg.symbol_mapping.items():
            try:
                fsymbols = v.free_symbols
            except AttributeError:
                fsymbols = set()
            if any(str(f) in dependent_symbols for f in fsymbols):
                dependent_symbols.add(k)
        for s in (start, end, step):
            if any(str(s) in dependent_symbols for s in s.free_symbols):
                return False

        # Collect read and writes from states
        read_set: Set[str] = set()
        write_set: Set[str] = set()
        for state in nsdfg.states():
            rset, wset = state.read_and_write_sets()
            read_set |= rset
            write_set |= wset

        # Check for map parameters in data descriptors
        for arr in (read_set | write_set):
            if any(p in set(map(str, nsdfg.arrays[arr].free_symbols)) for p in mparams):
                return False

        def test_subset_dependency(subset: sbs.Subset) -> Tuple[bool, List[int]]:
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

        # Check that NestedSDFG memlets depend on map params in a consistent manner
        # a. A container must either not depend at all on itervar, or depend on it always in the same dimensions.
        # b. Abort when a dimension depends on both the itervar and a Map parameter.
        data_dependency = dict()
        for state in nsdfg.states():
            for e in state.edges():
                if any(p in e.data.free_symbols for p in mparams):
                    e.data.try_initialize(nsdfg, state, e)
                    for i, subset in enumerate((e.data.src_subset, e.data.dst_subset)):
                        if subset:
                            if i == 0:
                                access = state.memlet_path(e)[0].src
                            else:
                                access = state.memlet_path(e)[-1].dst
                            passed, dims = test_subset_dependency(subset)
                            if not passed:
                                return False
                            if dims:
                                if access.data in data_dependency:
                                    if data_dependency[access.data] != dims:
                                        return False
                                else:
                                    data_dependency[access.data] = dims

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):

        nsdfg = self.nested_sdfg.sdfg
        components = helpers.find_sdfg_control_flow(nsdfg)
        cf_node: cf.ForScope
        _, cf_node = list(components.values())[0]
        mparams = set(self.map_entry.map.params)

        # Obtain loop information
        guard: sd.SDFGState = cf_node.guard
        body: sd.SDFGState = cf_node.body.first_state

        # Obtain iteration variable, range, and stride
        loop_info = find_for_loop(nsdfg, guard, body)
        if not loop_info:
            return False
        itervar, (start, end, step), (_, body_end) = loop_info

        forward_loop = step > 0

        # Pull all transients of the nested SDFG into the outer SDFG
        map_state = nsdfg.parent
        map_shape = self.map_entry.map.range.size()
        thread_locals = sdutil.get_thread_local_data(nsdfg)
        for name, desc in nsdfg.arrays.items():
            if name in thread_locals:
                continue
            if desc.transient:

                if isinstance(desc, dt.Scalar):
                    outer_desc = dt.Array(desc.dtype, desc.shape, desc.transient, desc.allow_conflicts, desc.storage,
                                    desc.location, desc.strides, desc.offset, False, desc.lifetime, 0, desc.debuginfo,
                                    desc.total_size, desc.start_offset)
                else:
                    outer_desc = copy.deepcopy(desc)
                for sz in reversed(map_shape):
                    outer_desc.strides = [outer_desc.total_size] + list(outer_desc.strides)
                    outer_desc.total_size = outer_desc.total_size * sz
                outer_desc.shape = map_shape + list(desc.shape)
                # Try to keep consistent offsets.
                offset = desc.offset[0]
                if any(o != offset for o in desc.offset):
                    offset = 0
                outer_desc.offset = [offset] * len(map_shape) + list(desc.offset)

                new_name = sdfg.add_datadesc(name, outer_desc, find_new_name=True)

                desc.transient = False
                # NOTE: Need to take into account the offset
                subset = sbs.Range([(f"{p} - {r[0]}", f"{p} - {r[0]}", 1) for p, r in zip(self.map_entry.map.params, self.map_entry.map.range)] + [(0, s - 1, 1) for s in desc.shape])
                read = map_state.add_access(new_name)
                self.nested_sdfg.add_in_connector(name)
                map_state.add_memlet_path(read, self.map_entry, self.nested_sdfg, dst_conn=name, memlet=Memlet(data=new_name, subset=copy.deepcopy(subset)))
                write = map_state.add_access(new_name)
                self.nested_sdfg.add_out_connector(name, force=True)
                map_state.add_memlet_path(self.nested_sdfg, self.map_exit, write, src_conn=name, memlet=Memlet(data=new_name, subset=copy.deepcopy(subset)))

        # nest map's content in sdfg
        map_subgraph = graph.scope_subgraph(self.map_entry, include_entry=True, include_exit=True)
        new_nsdfg = helpers.nest_state_subgraph(sdfg, graph, map_subgraph, full_data=True)

        # replicate loop in nested sdfg
        new_before, new_guard, new_after = new_nsdfg.sdfg.add_loop(
            before_state=None,
            loop_state=new_nsdfg.sdfg.nodes()[0],
            loop_end_state=None,
            after_state=None,
            loop_var=itervar,
            initialize_expr=f'{start}',
            condition_expr=f'{itervar} <= {end}' if forward_loop else f'{itervar} >= {end}',
            increment_expr=f'{itervar} + {step}' if forward_loop else f'{itervar} - {abs(step)}')
        new_nsdfg.sdfg.start_state = new_nsdfg.sdfg.node_id(new_before)

        # remove inner loop
        for e in nsdfg.in_edges(guard):
            if e.src not in (body, body_end):
                nsdfg.remove_node(e.src)
                # nsdfg.remove_edge(e)
        for e in nsdfg.out_edges(guard):
            if e.dst not in (body, body_end):
                nsdfg.remove_node(e.dst)
                # nsdfg.remove_edge(e)
        nsdfg.remove_node(guard)

        # Add itervar to nested-nested SDFG
        if itervar in nsdfg.symbols:
            nsdfg.parent_nsdfg_node.symbol_mapping[itervar] = dace.symbol(itervar, nsdfg.symbols[itervar])
        else:
            nsdfg.add_symbol(itervar, dace.int32)
            nsdfg.parent_nsdfg_node.symbol_mapping[itervar] = dace.symbol(itervar, dace.int32)

        from dace.transformation.interstate import RefineNestedAccess
        nsdfg.apply_transformations_repeated(RefineNestedAccess)
        sdfg.apply_transformations_repeated(RefineNestedAccess)
        propagation.propagate_states(new_nsdfg.sdfg)
        propagation.propagate_memlets_state(new_nsdfg.sdfg, nsdfg.parent)
        propagation.propagate_memlets_state(sdfg, graph)
        nsdfg.apply_transformations_repeated(RefineNestedAccess)
        sdfg.apply_transformations_repeated(RefineNestedAccess)


class MoveMapIntoIf(transformation.SingleStateTransformation):
    """
    Moves a Map around an IfScope into the IfScope.
    """

    map_entry = transformation.PatternNode(nodes.EntryNode)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    map_exit = transformation.PatternNode(nodes.ExitNode)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):

        # Is the body an IfScope?
        nsdfg = self.nested_sdfg.sdfg
        components = helpers.find_sdfg_control_flow(nsdfg)
        # Body must contain a single control-flow component
        if len(components) != 1:
            return False
        cf_node: cf.ControlFlow
        _, cf_node = list(components.values())[0]
        # Component must be IfScope
        if not isinstance(cf_node, cf.IfScope):
            return False

        mparams = set(self.map_entry.map.params)

        # Check basic structure of the IfScope.
        # The guard state must be empty
        if_guard: dace.SDFGState = cf_node.branch_state
        if len(if_guard.nodes()) != 0:
            return False
        # There must be a single sink state, the if-exit state.
        sink_states = nsdfg.sink_nodes()
        if len(sink_states) != 1:
            return False
        # The exit state must be empty.
        if_exit: dace.SDFGState = sink_states[0]
        if len(if_exit.nodes()) != 0:
            return False
        # We do not handle "orelse" yet.
        if cf_node.orelse is not None:
            return False
        # The condition must not depend on the Map parameters.
        condition = cf_node.condition
        symbols_to_check = set(mparams)
        for k, v in self.nested_sdfg.symbol_mapping.items():
            try:
                if symbolic.issymbolic(v):
                    fsymbols = v.free_symbols
                else:
                    fsymbols = symbolic.pystr_to_symbolic(v).free_symbols
            except AttributeError:
                fsymbols = set()
            if any(str(f) in symbols_to_check for f in fsymbols):
                symbols_to_check.add(k)
        if any(str(s) in symbols_to_check for s in condition.get_free_symbols()):
            return False

        # Check the read memlets of the condition
        cond_edge = next(e for e in nsdfg.out_edges(if_guard) if e.dst is not if_exit)
        cond_memlets = cond_edge.data.get_read_memlets(nsdfg.arrays)
        for m in cond_memlets:
            in_data = m.data
            in_desc = nsdfg.arrays[in_data]
            out_edge = next(e for e in nsdfg.parent.in_edges_by_connector(self.nested_sdfg, in_data))
            out_data = nsdfg.parent.memlet_path(out_edge)[0].src.data
            out_desc = sdfg.arrays[out_data]
            new_m = helpers.unsqueeze_memlet(m, out_edge.data, map=self.map_entry.map)
            if any(str(s) in symbols_to_check for s in new_m.free_symbols):
                return False

        # Collect read and writes from states
        read_set: Set[str] = set()
        write_set: Set[str] = set()
        for state in nsdfg.states():
            rset, wset = state.read_and_write_sets()
            read_set |= rset
            write_set |= wset

        # Check for map parameters in data descriptors
        for arr in (read_set | write_set):
            if any(p in set(map(str, nsdfg.arrays[arr].free_symbols)) for p in mparams):
                return False

        def test_subset_dependency(subset: sbs.Subset) -> Tuple[bool, List[int]]:
            dims = []
            for i, r in enumerate(subset):
                if not isinstance(r, (list, tuple)):
                    r = [r]
                fsymbols = set()
                for token in r:
                    if symbolic.issymbolic(token):
                        fsymbols = fsymbols.union({str(s) for s in token.free_symbols})
                # NOTE: IfScopes don't have an iteration variable. Does this mean that we can ignore everything below?
                # if itervar in fsymbols:
                #     if fsymbols.intersection(mparams):
                #         return (False, [])
                #     else:
                #         # Strong checks
                #         if not permissive:
                #             # Only indices allowed
                #             if len(r) > 1 and r[0] != r[1]:
                #                 return (False, [])
                #             derivative = diff(r[0])
                #             # Index function must be injective
                #             if not (((derivative > 0) == True) or ((derivative < 0) == True)):
                #                 return (False, [])
                #         dims.append(i)
            return (True, dims)

        # Check that NestedSDFG memlets depend on map params in a consistent manner
        # a. A container must either not depend at all on itervar, or depend on it always in the same dimensions.
        # b. Abort when a dimension depends on both the itervar and a Map parameter.
        data_dependency = dict()
        for state in nsdfg.states():
            for e in state.edges():
                if any(p in e.data.free_symbols for p in mparams):
                    e.data.try_initialize(nsdfg, state, e)
                    for i, subset in enumerate((e.data.src_subset, e.data.dst_subset)):
                        if subset:
                            if i == 0:
                                access = state.memlet_path(e)[0].src
                            else:
                                access = state.memlet_path(e)[-1].dst
                            passed, dims = test_subset_dependency(subset)
                            if not passed:
                                return False
                            if dims:
                                if access.data in data_dependency:
                                    if data_dependency[access.data] != dims:
                                        return False
                                else:
                                    data_dependency[access.data] = dims

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):

        nsdfg = self.nested_sdfg.sdfg
        components = helpers.find_sdfg_control_flow(nsdfg)
        cf_node: cf.ControlFlow
        _, cf_node = list(components.values())[0]
        mparams = set(self.map_entry.map.params)

        if_guard: dace.SDFGState = cf_node.branch_state
        ipostdom = sdutil.postdominators(nsdfg)
        if_exit: dace.SDFGState = ipostdom[if_guard]
        condition = cf_node.condition
        inv_condition = nsdfg.edges_between(if_guard, if_exit)[0].data.condition

        # Unsqueeze memlets in conditions
        fsymbols = set()
        cond_edge = next(e for e in nsdfg.out_edges(if_guard) if e.dst is not if_exit)
        cond_memlets = cond_edge.data.get_read_memlets(nsdfg.arrays)
        new_memlets = []
        for m in cond_memlets:
            in_data = m.data
            in_desc = nsdfg.arrays[in_data]
            out_edge = next(e for e in nsdfg.parent.in_edges_by_connector(self.nested_sdfg, in_data))
            out_data = nsdfg.parent.memlet_path(out_edge)[0].src.data
            out_desc = sdfg.arrays[out_data]
            new_m = helpers.unsqueeze_memlet(m, out_edge.data, map=self.map_entry.map)
            new_m = propagation.propagate_memlet(nsdfg.parent, new_m, self.map_entry, False)
            new_memlets.append(new_m)
            fsymbols.update(new_m.free_symbols)
        for node in ast.walk(cond_edge.data.condition.code[0]):
            if isinstance(node, ast.Subscript):
                        m = new_memlets.pop(0)
                        subscript: ast.Subscript = ast.parse(str(m)).body[0].value
                        assert isinstance(node.value, ast.Name) and node.value.id == m.data
                        node.slice = ast.copy_location(subscript.slice, node.slice)
        
        inv_cond_edge = nsdfg.edges_between(if_guard, if_exit)[0]
        inv_cond_memlets = inv_cond_edge.data.get_read_memlets(nsdfg.arrays)
        new_memlets = []
        for m in cond_memlets:
            in_data = m.data
            in_desc = nsdfg.arrays[in_data]
            out_edge = next(e for e in nsdfg.parent.in_edges_by_connector(self.nested_sdfg, in_data))
            out_data = nsdfg.parent.memlet_path(out_edge)[0].src.data
            out_desc = sdfg.arrays[out_data]
            new_m = helpers.unsqueeze_memlet(m, out_edge.data, map=self.map_entry.map)
            new_m = propagation.propagate_memlet(nsdfg.parent, new_m, self.map_entry)
            new_memlets.append(new_m)
            fsymbols.update(new_m.free_symbols)
        for node in ast.walk(inv_cond_edge.data.condition.code[0]):
            if isinstance(node, ast.Subscript):
                        m = new_memlets.pop(0)
                        subscript: ast.Subscript = ast.parse(str(m)).body[0].value
                        assert isinstance(node.value, ast.Name) and node.value.id == m.data
                        node.slice = ast.copy_location(subscript.slice, node.slice)
        
        condition = cond_edge.data.condition
        inv_condition = inv_cond_edge.data.condition

        # remove IfScope.
        nsdfg.remove_nodes_from([if_guard, if_exit])

        # Pull all transients of the nested SDFG into the outer SDFG
        map_state = nsdfg.parent
        map_shape = self.map_entry.map.range.size()
        thread_locals = sdutil.get_thread_local_data(nsdfg)
        for name, desc in nsdfg.arrays.items():
            if name in thread_locals:
                continue
            if desc.transient:

                if isinstance(desc, dt.Scalar):
                    outer_desc = dt.Array(desc.dtype, desc.shape, desc.transient, desc.allow_conflicts, desc.storage,
                                    desc.location, desc.strides, desc.offset, False, desc.lifetime, 0, desc.debuginfo,
                                    desc.total_size, desc.start_offset)
                else:
                    outer_desc = copy.deepcopy(desc)
                for sz in reversed(map_shape):
                    outer_desc.strides = [outer_desc.total_size] + list(outer_desc.strides)
                    outer_desc.total_size = outer_desc.total_size * sz
                outer_desc.shape = map_shape + list(desc.shape)
                # Try to keep consistent offsets.
                offset = desc.offset[0]
                if any(o != offset for o in desc.offset):
                    offset = 0
                outer_desc.offset = [offset] * len(map_shape) + list(desc.offset)

                new_name = sdfg.add_datadesc(name, outer_desc, find_new_name=True)

                desc.transient = False
                # NOTE: Need to take into account the offset
                subset = sbs.Range([(f"{p} - {r[0]}", f"{p} - {r[0]}", 1) for p, r in zip(self.map_entry.map.params, self.map_entry.map.range)] + [(0, s - 1, 1) for s in desc.shape])
                read = map_state.add_access(new_name)
                self.nested_sdfg.add_in_connector(name)
                map_state.add_memlet_path(read, self.map_entry, self.nested_sdfg, dst_conn=name, memlet=Memlet(data=new_name, subset=copy.deepcopy(subset)))
                write = map_state.add_access(new_name)
                self.nested_sdfg.add_out_connector(name, force=True)
                map_state.add_memlet_path(self.nested_sdfg, self.map_exit, write, src_conn=name, memlet=Memlet(data=new_name, subset=copy.deepcopy(subset)))

        # nest map's content in sdfg
        map_subgraph = graph.scope_subgraph(self.map_entry, include_entry=True, include_exit=True)
        new_nsdfg = helpers.nest_state_subgraph(sdfg, graph, map_subgraph, full_data=True)

        # replicate IfScope in nested sdfg

        body = new_nsdfg.sdfg.nodes()[0]
        if_guard = new_nsdfg.sdfg.add_state('if_guard')
        if_exit = new_nsdfg.sdfg.add_state('if_exit')
        new_nsdfg.symbol_mapping.update({s: s for s in fsymbols if s not in new_nsdfg.sdfg.symbols})
        new_nsdfg.sdfg.add_edge(if_guard, body, dace.InterstateEdge(condition=condition))
        new_nsdfg.sdfg.add_edge(if_guard, if_exit, dace.InterstateEdge(condition=inv_condition))
        new_nsdfg.sdfg.add_edge(body, if_exit, dace.InterstateEdge())

        from dace.transformation.interstate import RefineNestedAccess
        nsdfg.apply_transformations_repeated(RefineNestedAccess)
        sdfg.apply_transformations_repeated(RefineNestedAccess)
        propagation.propagate_states(new_nsdfg.sdfg)
        propagation.propagate_memlets_state(new_nsdfg.sdfg, nsdfg.parent)
        propagation.propagate_memlets_state(sdfg, graph)
        nsdfg.apply_transformations_repeated(RefineNestedAccess)
        sdfg.apply_transformations_repeated(RefineNestedAccess)
