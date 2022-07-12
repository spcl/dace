# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop to map transformation """

from collections import defaultdict
import copy
import itertools
import sympy as sp
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple

from dace import dtypes, memlet, nodes, registry, sdfg as sd, symbolic, subsets
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr, nodes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
import dace.transformation.helpers as helpers
from dace.transformation import transformation as xf


def _check_range(subset, a, itersym, b,c, step):
    found = False
    for rb, re, _ in subset.ndrange():
        m = rb.match(a * itersym + b)
        if m is None:
            m = rb.match(a * (itersym + c) + b)
            if m is None:
                continue
        if (m[a] >= 1) != True and (m[a] != 1/step):
            continue
        if re != rb:
            if isinstance(rb, symbolic.SymExpr):
                rb = rb.approx
            if isinstance(re, symbolic.SymExpr):
                re = re.approx

            # If False or indeterminate, the range may
            # overlap across iterations
            if ((re - rb) > m[a] * step) != False:
                continue

            m = re.match(a * itersym + b)
            if m is None:
                m = re.match(a * (itersym + c) + b)
                if m is None:
                    continue
            if (m[a] >= 1) != True and (m[a] != 1/step):
                continue
        found = True
        break
    return found


def _dependent_indices(itervar: str, subset: subsets.Subset) -> Set[int]:
    """ Finds the indices or ranges of a subset that depend on the iteration
        variable. Returns their index in the subset's indices/ranges list.
    """
    if isinstance(subset, subsets.Indices):
        return {
            i
            for i, idx in enumerate(subset)
            if symbolic.issymbolic(idx) and itervar in {str(s)
                                                        for s in idx.free_symbols}
        }
    else:
        return {
            i
            for i, rng in enumerate(subset) if any(
                symbolic.issymbolic(t) and itervar in {str(s)
                                                       for s in t.free_symbols} for t in rng)
        }


def _sanitize_by_index(indices: Set[int], subset: subsets.Subset) -> subsets.Range:
    """ Keeps the indices or ranges of subsets that are in `indices`. """
    return type(subset)([t for i, t in enumerate(subset) if i in indices])


@make_properties
class LoopToMap(DetectLoop, xf.MultiStateTransformation):
    """Convert a control flow loop into a dataflow map. Currently only supports
       the simple case where there is no overlap between inputs and outputs in
       the body of the loop, and where the loop body only consists of a single
       state.
    """

    itervar = Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc='The name of the iteration variable (optional).',
    )

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Is this even a loop
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        guard = self.loop_guard
        begin = self.loop_begin

        # Guard state should not contain any dataflow
        if len(guard.nodes()) != 0:
            return False

        # If loop cannot be detected, fail
        found = find_for_loop(graph, guard, begin, itervar=self.itervar)
        if not found:
            return False

        itervar, (start, end, step), (_, body_end) = found

        # We cannot handle symbols read from data containers unless they are
        # scalar
        for expr in (start, end, step):
            if symbolic.contains_sympy_functions(expr):
                return False

        # Find all loop-body states
        states = set()
        to_visit = [begin]
        while to_visit:
            state = to_visit.pop(0)
            for _, dst, _ in sdfg.out_edges(state):
                if dst not in states and dst is not guard:
                    to_visit.append(dst)
            states.add(state)

        assert (body_end in states)

        write_set = set()
        for state in states:
            _, wset = state.read_and_write_sets()
            write_set |= wset

        # Get access nodes from other states to isolate local loop variables
        other_access_nodes = set()
        for state in sdfg.nodes():
            if state in states:
                continue
            other_access_nodes |= set(n.data for n in state.data_nodes() if sdfg.arrays[n.data].transient)
        # Add non-transient nodes from loop state
        for state in states:
            other_access_nodes |= set(n.data for n in state.data_nodes() if not sdfg.arrays[n.data].transient)

        write_memlets = defaultdict(list)

        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        c = sp.Wild('c', exclude=[itersym])

        for state in states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                # Take all writes that are not conflicted into consideration
                if dn.data in write_set:
                    for e in state.in_edges(dn):
                        #if e.data.dynamic and e.data.wcr is None:
                        #    # If pointers are involved, give up
                        #    return False
                        # To be sure that the value is only written at unique
                        # indices per loop iteration, we want to match symbols
                        # of the form "a*i+b" where a >= 1, and i is the iteration
                        # variable. The iteration variable must be used.
                        if e.data.wcr is None:
                            dst_subset = e.data.get_dst_subset(e, state)
                            if not (dst_subset and _check_range(dst_subset, a, itersym, b,c, step)):
                                return False
                        # End of check
                        e.data.try_initialize(sdfg, state, e)
                        write_memlets[dn.data].append(e.data)

        # TODO: also check interstate edges with reads
        # anames = sdfg.arrays.keys()
        # for e in gr.SubgraphView(sdfg, states).edges():
        #     read_set |= e.data.free_symbols & anames

        # After looping over relevant writes, consider reads that may overlap
        for state in states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                data = dn.data
                if data in write_memlets:
                    # Import as necessary
                    from dace.sdfg.propagation import propagate_subset

                    for e in state.out_edges(dn):
                        # If the same container is both read and written, only match if
                        # it read and written at locations that will not create data races
                        #if (e.data.dynamic and e.data.src_subset.num_elements() != 1):
                        #    # If pointers are involved, give up
                        #    return False
                        src_subset = e.data.get_src_subset(e, state)
                        if not _check_range(src_subset, a, itersym, b,c, step):
                            return False

                        pread = propagate_subset([e.data], sdfg.arrays[data], [itervar],
                                                 subsets.Range([(start, end, step)]))
                        for candidate in write_memlets[data]:
                            # Simple case: read and write are in the same subset
                            read = src_subset
                            write = candidate.dst_subset
                            if read == write:
                                continue
                            ridx = _dependent_indices(itervar, read)
                            widx = _dependent_indices(itervar, write)
                            indices = set(ridx) | set(widx)
                            if not indices:
                                indices = set(range(len(read)))
                            read = _sanitize_by_index(indices, read)
                            write = _sanitize_by_index(indices, write)
                            if read == write:
                                continue
                            # Propagated read does not overlap with propagated write
                            pwrite = propagate_subset([candidate],
                                                      sdfg.arrays[data], [itervar],
                                                      subsets.Range([(start, end, step)]),
                                                      use_dst=True)
                            t_pread = _sanitize_by_index(indices, pread.src_subset)
                            pwrite = _sanitize_by_index(indices, pwrite.dst_subset)
                            if subsets.intersects(t_pread, pwrite) is False:
                                continue
                            return False

        # Check that the iteration variable is not used on other edges or states
        # before it is reassigned
        prior_states = True
        for state in cfg.stateorder_topological_sort(sdfg):
            # Skip all states up to guard
            if prior_states:
                if state is begin:
                    prior_states = False
                continue
            # We do not need to check the loop-body states
            if state in states:
                continue
            if itervar in state.free_symbols:
                return False
            # Don't continue in this direction, as the variable has
            # now been reassigned
            # TODO: Handle case of subset of out_edges
            if all(itervar in e.data.assignments for e in sdfg.out_edges(state)):
                break

        return True

    def apply(self, _, sdfg: sd.SDFG):
        # Obtain loop information
        guard: sd.SDFGState = self.loop_guard
        body: sd.SDFGState = self.loop_begin
        after: sd.SDFGState = self.exit_state

        # Obtain iteration variable, range, and stride
        itervar, (start, end, step), (_, body_end) = find_for_loop(sdfg, guard, body, itervar=self.itervar)

        # Find all loop-body states
        states = set()
        to_visit = [body]
        while to_visit:
            state = to_visit.pop(0)
            for _, dst, _ in sdfg.out_edges(state):
                if dst not in states and dst is not guard:
                    to_visit.append(dst)
            states.add(state)

        # Register WCR edges, if exist
        wcrs: Dict[str, str] = {}
        for state in states:
            for e in state.edges():
                if isinstance(e.dst, nodes.AccessNode):
                    if e.data.wcr is not None:
                        if e.data.data not in wcrs:
                            wcrs[e.data.data] = e.data.wcr
                        elif wcrs[e.data.data] != e.data.wcr:  # Ambiguous WCR, register as None
                            wcrs[e.data.data] = None

        # Nest loop-body states
        if len(states) > 1:

            # Find read/write sets
            read_set, write_set = set(), set()
            for state in states:
                rset, wset = state.read_and_write_sets()
                read_set |= rset
                write_set |= wset
                for e in state.edges():
                    # Add to write set also scalars between tasklets
                    if not isinstance(e.src, nodes.Tasklet):
                        continue
                    if not isinstance(e.dst, nodes.Tasklet):
                        continue
                    if e.data.data and e.data.data in sdfg.arrays:
                        write_set.add(e.data.data)
                # Add data from edges
                subgraph = gr.SubgraphView(sdfg, states)
                for edge in subgraph.edges():
                    for s in edge.data.free_symbols:
                        if s in sdfg.arrays:
                            read_set.add(s)

            # Find NestedSDFG's unique data
            rw_set = read_set | write_set
            unique_set = set()
            for name in rw_set:
                if not sdfg.arrays[name].transient:
                    continue
                found = False
                for state in sdfg.states():
                    if state in states:
                        continue
                    for node in state.nodes():
                        if (isinstance(node, nodes.AccessNode) and node.data == name):
                            found = True
                            break
                for isedge in sdfg.edges():
                    if name in isedge.data.free_symbols:
                        found = True
                        break        
                if not found:
                    unique_set.add(name)

            # Find NestedSDFG's connectors
            read_set = {n for n in read_set if n not in unique_set or not sdfg.arrays[n].transient}
            write_set = {n for n in write_set if n not in unique_set or not sdfg.arrays[n].transient}

            # Create NestedSDFG and add all loop-body states and edges
            # Also, find defined symbols in NestedSDFG
            fsymbols = set(sdfg.free_symbols)
            new_body = sdfg.add_state('single_state_body')
            nsdfg = SDFG("loop_body", constants=sdfg.constants_prop, parent=new_body)
            nsdfg.add_node(body, is_start_state=True)
            body.parent = nsdfg
            exit_state = nsdfg.add_state('exit')
            nsymbols = dict()
            for state in states:
                if state is body:
                    continue
                nsdfg.add_node(state)
                state.parent = nsdfg
            for state in states:
                #if state is body:
                #    continue
                for src, dst, data in sdfg.in_edges(state):
                    if src is guard and dst is body:
                        continue
                    nsymbols.update({s: sdfg.symbols[s] for s in data.assignments.keys() if s in sdfg.symbols})
                    nsymbols.update({s: sdfg.symbols[s] for s in data.free_symbols if s in sdfg.symbols})
                    nsdfg.add_edge(src, dst, data)
            
            # Move guard -> body edge to guard -> new_body
            for src, dst, data, in sdfg.edges_between(guard, body):
                sdfg.add_edge(src, new_body, data)
            # Move body_end -> guard edge to body_end -> exit_state
            for src, dst, data in sdfg.edges_between(body_end, guard):
                if itervar in data.assignments:
                    del data.assignments[itervar]
                nsdfg.add_edge(body_end, exit_state, data)
            sdfg.add_edge(new_body, guard, InterstateEdge())
            # # Move body_end -> guard edge to new_body -> guard
            # if body_end is body:
            #     for src, dst, data in sdfg.edges_between(body_end, guard):
            #         nsdfg.add_edge(body_end, exit_state, data)
            #     sdfg.add_edge(new_body, guard, InterstateEdge())
            # else:
            #     nsdfg.add_edge(body_end, exit_state, InterstateEdge())
            #     for src, dst, data in sdfg.edges_between(body_end, guard):
            #         sdfg.add_edge(new_body, dst, data)    

            # Delete loop-body states and edges from parent SDFG
            for state in states:
                for e in sdfg.all_edges(state):
                    sdfg.remove_edge(e)
                sdfg.remove_node(state)

            # Add NestedSDFG arrays
            for name in read_set | write_set:
                nsdfg.arrays[name] = copy.deepcopy(sdfg.arrays[name])
                nsdfg.arrays[name].transient = False
            for name in unique_set:
                nsdfg.arrays[name] = sdfg.arrays[name]
                del sdfg.arrays[name]

            # Add NestedSDFG node
            cnode = new_body.add_nested_sdfg(nsdfg, None, read_set, write_set)
            if sdfg.parent:
                for s, m in sdfg.parent_nsdfg_node.symbol_mapping.items():
                    if s not in cnode.symbol_mapping:
                        cnode.symbol_mapping[s] = m
                        nsdfg.add_symbol(s, sdfg.symbols[s])
            for name in read_set:
                r = new_body.add_read(name)
                new_body.add_edge(r, None, cnode, name, memlet.Memlet.from_array(name, sdfg.arrays[name]))
            for name in write_set:
                w = new_body.add_write(name)
                mm = memlet.Memlet.from_array(name, sdfg.arrays[name])
                if name in wcrs and wcrs[name] is not None:
                    mm.wcr = wcrs[name]
                new_body.add_edge(cnode, name, w, None, mm)

            # Fix SDFG symbols
            for sym in sdfg.free_symbols - fsymbols:
                del sdfg.symbols[sym]
            for sym, dtype in nsymbols.items():
                nsdfg.symbols[sym] = dtype
            if itervar not in nsdfg.symbols:
                nsdfg.add_symbol(itervar, sdfg.symbols[itervar])
            if itervar not in cnode.symbol_mapping:
                cnode.symbol_mapping[itervar] = itervar

            # Change body state reference
            multistate_nsdfg = nsdfg
            body = new_body




        if (step < 0) == True:
            # If step is negative, we have to flip start and end to produce a
            # correct map with a positive increment
            start, end, step = end, start, -step

        # If necessary, make a nested SDFG with assignments
        isedge = sdfg.edges_between(guard, body)[0]
        symbols_to_remove = set()
        data_symbols = set()
        if len(isedge.data.assignments) > 0:
            if len(states) == 1:
                nsdfg = helpers.nest_state_subgraph(sdfg, body, gr.SubgraphView(body, body.nodes()))
                for sym in isedge.data.free_symbols:
                    if sym in nsdfg.symbol_mapping or sym in nsdfg.in_connectors:
                        continue
                    if sym in sdfg.symbols:
                        nsdfg.symbol_mapping[sym] = symbolic.pystr_to_symbolic(sym)
                        nsdfg.sdfg.add_symbol(sym, sdfg.symbols[sym])
                    elif sym in sdfg.arrays:
                        if sym in nsdfg.sdfg.arrays:
                            raise NotImplementedError
                        rnode = body.add_read(sym)
                        nsdfg.add_in_connector(sym)
                        desc = copy.deepcopy(sdfg.arrays[sym])
                        desc.transient = False
                        nsdfg.sdfg.add_datadesc(sym, desc)
                        body.add_edge(rnode, None, nsdfg, sym, memlet.Memlet(sym))
            else:
                nsdfg = multistate_nsdfg.parent_nsdfg_node

            # # If a nested SDFG was already created, then we need to ensure that all symbols are mapped properly
            # if len(states) > 1:
            #     for sym, dtype in nsymbols.items():
            #         if sym in multistate_nsdfg.symbols or sym in multistate_nsdfg.free_symbols:
            #             if sym not in nsdfg.symbol_mapping:
            #                 nsdfg.symbol_mapping[sym] = sym
            #             if sym not in multistate_nsdfg.parent_nsdfg_node.symbol_mapping:
            #                 multistate_nsdfg.parent_nsdfg_node.symbol_mapping[sym] = sym

            nstate = nsdfg.sdfg.node(0)
            # init_state = nsdfg.sdfg.add_state_before(nstate)
            init_state = nsdfg.sdfg.add_state('init_state', is_start_state=True)
            nsdfg.sdfg.add_edge(init_state, nstate, InterstateEdge())
            nisedge = nsdfg.sdfg.edges_between(init_state, nstate)[0]
            nisedge.data.assignments = isedge.data.assignments
            symbols_to_remove = set(nisedge.data.assignments.keys())
            for k in nisedge.data.assignments.keys():
                if k in nsdfg.symbol_mapping:
                    del nsdfg.symbol_mapping[k]
            for s in nisedge.data.free_symbols:
                if s in sdfg.symbols:
                    if s not in nsdfg.sdfg.symbols:
                        nsdfg.symbol_mapping[s] = s
                        nsdfg.sdfg.add_symbol(s, sdfg.symbols[s])
                elif s in sdfg.arrays:
                    if s not in nsdfg.sdfg.arrays:
                        nsdfg.sdfg.arrays[s] = copy.deepcopy(sdfg.arrays[s])
                        nsdfg.sdfg.arrays[s].transient = False
                        nsdfg.add_in_connector(s)
                        data_symbols.add(s)

            isedge.data.assignments = {}

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        map = nodes.Map(body.label + "_map", [itervar], [(start, end, step)])
        entry = nodes.MapEntry(map)
        exit = nodes.MapExit(map)
        body.add_node(entry)
        body.add_node(exit)

        # Add reads for data used on the entry edge
        if data_symbols:
            for rd in data_symbols:
                access_node = body.add_read(rd)
                body.add_memlet_path(access_node, entry, nsdfg, dst_conn=rd,
                                     memlet=memlet.Memlet.from_array(rd, sdfg.arrays[rd]))

        # If the map uses symbols from data containers, instantiate reads
        containers_to_read = entry.free_symbols & sdfg.arrays.keys()
        for rd in containers_to_read:
            # We are guaranteed that this is always a scalar, because
            # can_be_applied makes sure there are no sympy functions in each of
            # the loop expresions
            access_node = body.add_read(rd)
            body.add_memlet_path(access_node, entry, dst_conn=rd, memlet=memlet.Memlet(rd))

        # Reroute all memlets through the entry and exit nodes
        for n in source_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.out_edges(n):
                    body.remove_edge(e)
                    body.add_edge_pair(entry, e.dst, n, e.data, internal_connector=e.dst_conn)
            else:
                body.add_nedge(entry, n, memlet.Memlet())
        for n in sink_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.in_edges(n):
                    body.remove_edge(e)
                    body.add_edge_pair(exit, e.src, n, e.data, internal_connector=e.src_conn)
            else:
                body.add_nedge(n, exit, memlet.Memlet())

        # Get rid of the loop exit condition edge
        after_edge = sdfg.edges_between(guard, after)[0]
        sdfg.remove_edge(after_edge)

        # Remove the assignment on the edge to the guard
        for e in sdfg.in_edges(guard):
            if itervar in e.data.assignments:
                del e.data.assignments[itervar]

        # Remove the condition on the entry edge
        condition_edge = sdfg.edges_between(guard, body)[0]
        condition_edge.data.condition = CodeBlock("1")

        # Get rid of backedge to guard
        sdfg.remove_edge(sdfg.edges_between(body, guard)[0])

        # Route body directly to after state, maintaining any other assignments
        # it might have had
        sdfg.add_edge(body, after, sd.InterstateEdge(assignments=after_edge.data.assignments))

        # If this had made the iteration variable a free symbol, we can remove
        # it from the SDFG symbols
        if itervar in sdfg.free_symbols:
            sdfg.remove_symbol(itervar)
        for sym in symbols_to_remove:
            if sym in sdfg.symbols and helpers.is_symbol_unused(sdfg, sym):
                sdfg.remove_symbol(sym)
