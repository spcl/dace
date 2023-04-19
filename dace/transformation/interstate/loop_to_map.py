# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop to map transformation """

from collections import defaultdict
import copy
import itertools
import sympy as sp
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple

from dace import data as dt, dtypes, memlet, nodes, registry, sdfg as sd, symbolic, subsets
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr, nodes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation.interstate.loop_detection import (DetectLoop, find_for_loop)
import dace.transformation.helpers as helpers
from dace.transformation import transformation as xf


def _check_range(subset, a, itersym, b, step):
    found = False
    for rb, re, _ in subset.ndrange():
        m = rb.match(a * itersym + b)
        if m is None:
            continue
        if (abs(m[a]) >= 1) != True:
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
                continue
            if (abs(m[a]) >= 1) != True:
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

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False):
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

        in_order_states = list(cfg.stateorder_topological_sort(sdfg))
        loop_begin_idx = in_order_states.index(begin)
        loop_end_idx = in_order_states.index(body_end)

        if loop_end_idx < loop_begin_idx:  # Malformed loop
            return False

        # Find all loop-body states
        states: List[SDFGState] = list(sdutil.dfs_conditional(sdfg, [begin], lambda _, c: c is not guard))

        assert (body_end in states)

        write_set: Set[str] = set()
        for state in states:
            _, wset = state.read_and_write_sets()
            write_set |= wset

        # Collect symbol reads and writes from inter-state assignments
        symbols_that_may_be_used: Set[str] = {itervar}
        used_before_assignment: Set[str] = set()
        for state in states:
            for e in sdfg.out_edges(state):
                # Collect read-before-assigned symbols (this works because the states are always in order,
                # see above call to `stateorder_topological_sort`)
                read_symbols = e.data.read_symbols()
                read_symbols -= symbols_that_may_be_used
                used_before_assignment |= read_symbols
                # If symbol was read before it is assigned, the loop cannot be parallel
                assigned_symbols = set()
                for k, v in e.data.assignments.items():
                    try:
                        fsyms = symbolic.pystr_to_symbolic(v).free_symbols
                    except AttributeError:
                        fsyms = set()
                    if not k in fsyms:
                        assigned_symbols.add(k)
                if assigned_symbols & used_before_assignment:
                    return False

                symbols_that_may_be_used |= e.data.assignments.keys()

        # Get access nodes from other states to isolate local loop variables
        other_access_nodes: Set[str] = set()
        for state in sdfg.nodes():
            if state in states:
                continue
            other_access_nodes |= set(n.data for n in state.data_nodes() if sdfg.arrays[n.data].transient)
        # Add non-transient nodes from loop state
        for state in states:
            other_access_nodes |= set(n.data for n in state.data_nodes() if not sdfg.arrays[n.data].transient)

        write_memlets: Dict[str, List[memlet.Memlet]] = defaultdict(list)

        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])

        for state in states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                # Take all writes that are not conflicted into consideration
                if dn.data in write_set:
                    for e in state.in_edges(dn):
                        if e.data.dynamic and e.data.wcr is None:
                            # If pointers are involved, give up
                            return False
                        # To be sure that the value is only written at unique
                        # indices per loop iteration, we want to match symbols
                        # of the form "a*i+b" where |a| >= 1, and i is the iteration
                        # variable. The iteration variable must be used.
                        if e.data.wcr is None:
                            dst_subset = e.data.get_dst_subset(e, state)
                            if not (dst_subset and _check_range(dst_subset, a, itersym, b, step)):
                                return False
                        # End of check

                        write_memlets[dn.data].append(e.data)

        # After looping over relevant writes, consider reads that may overlap
        for state in states:
            for dn in state.data_nodes():
                if dn.data not in other_access_nodes:
                    continue
                data = dn.data
                if data in write_memlets:
                    for e in state.out_edges(dn):
                        # If the same container is both read and written, only match if
                        # it read and written at locations that will not create data races
                        src_subset = e.data.get_src_subset(e, state)
                        if not self.test_read_memlet(sdfg, state, e, itersym, itervar, start, end, step, write_memlets,
                                                     e.data, src_subset):
                            return False

        # Consider reads in inter-state edges (could be in assignments or in condition)
        isread_set: Set[memlet.Memlet] = set()
        for s in states:
            for e in sdfg.all_edges(s):
                isread_set |= set(e.data.get_read_memlets(sdfg.arrays))
        for mmlt in isread_set:
            if mmlt.data in write_memlets:
                if not self.test_read_memlet(sdfg, None, None, itersym, itervar, start, end, step, write_memlets, mmlt,
                                             mmlt.subset):
                    return False

        # Check that the iteration variable and other symbols are not used on other edges or states
        # before they are reassigned
        for state in in_order_states[loop_begin_idx + 1:]:
            if state in states:
                continue
            # Don't continue in this direction, as all loop symbols have been reassigned
            if not symbols_that_may_be_used:
                break

            # Check state contents
            if symbols_that_may_be_used & state.free_symbols:
                return False

            # Check inter-state edges
            reassigned_symbols: Set[str] = None
            for e in sdfg.out_edges(state):
                if symbols_that_may_be_used & e.data.read_symbols():
                    return False

                # Check for symbols that are set by all outgoing edges
                # TODO: Handle case of subset of out_edges
                if reassigned_symbols is None:
                    reassigned_symbols = set(e.data.assignments.keys())
                else:
                    reassigned_symbols &= e.data.assignments.keys()

            # Remove reassigned symbols
            if reassigned_symbols is not None:
                symbols_that_may_be_used -= reassigned_symbols

        return True

    def test_read_memlet(self, sdfg: SDFG, state: SDFGState, edge: gr.MultiConnectorEdge[memlet.Memlet],
                         itersym: symbolic.SymbolicType, itervar: str, start: symbolic.SymbolicType,
                         end: symbolic.SymbolicType, step: symbolic.SymbolicType,
                         write_memlets: Dict[str, List[memlet.Memlet]], mmlt: memlet.Memlet, src_subset: subsets.Range):
        # Import as necessary
        from dace.sdfg.propagation import propagate_subset, align_memlet

        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])
        data = mmlt.data

        if (mmlt.dynamic and mmlt.src_subset.num_elements() != 1):
            # If pointers are involved, give up
            return False
        if not _check_range(src_subset, a, itersym, b, step):
            return False

        # Always use the source data container for the memlet test
        if state is not None and edge is not None:
            mmlt = align_memlet(state, edge, dst=False)
            data = mmlt.data

        pread = propagate_subset([mmlt], sdfg.arrays[data], [itervar], subsets.Range([(start, end, step)]))
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

        return True

    def _is_array_thread_local(self, name: str, itervar: str, sdfg: SDFG, states: List[SDFGState]) -> bool:
        """
        This helper method checks whether an array used exclusively in the body of a detected for-loop is thread-local,
        i.e., its whole range is may be used in every loop iteration, or is can be shared by multiple iterations.

        For simplicity, it is assumed that the for-loop can be safely transformed to a Map. The method applies only to
        bodies that become a NestedSDFG.

        :param name: The name of array.
        :param itervar: The for-loop iteration variable.
        :param sdfg: The SDFG containing the states that comprise the body of the for-loop.
        :param states: A list of states that comprise the body of the for-loop.
        :return: True if the array is thread-local, otherwise False.
        """

        desc = sdfg.arrays[name]
        if not isinstance(desc, dt.Array):
            # Scalars are always thread-local.
            return True
        if itervar in (str(s) for s in desc.free_symbols):
            # If the shape or strides of the array depend on the iteration variable, then the array is thread-local.
            return True
        for state in states:
            for node in state.data_nodes():
                if node.data != name:
                    continue
                for e in state.out_edges(node):
                    src_subset = e.data.get_src_subset(e, state)
                    # If the iteration variable is in the subsets symbols, then the array cannot be thread-local.
                    # Here we use the assumption that the for-loop can be turned to a valid Map, i.e., all other edges
                    # carrying the array depend on the iteration variable in a consistent manner.
                    if src_subset and itervar in src_subset.free_symbols:
                        return False
                for e in state.in_edges(node):
                    dst_subset = e.data.get_dst_subset(e, state)
                    # If the iteration variable is in the subsets symbols, then the array cannot be thread-local.
                    # Here we use the assumption that the for-loop can be turned to a valid Map, i.e., all other edges
                    # carrying the array depend on the iteration variable in a consistent manner.
                    if dst_subset and itervar in dst_subset.free_symbols:
                        return False
        return True

    def apply(self, _, sdfg: sd.SDFG):
        from dace.sdfg.propagation import align_memlet

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

        # Nest loop-body states
        if len(states) > 1:

            # Find read/write sets
            read_set, write_set = set(), set()
            for state in states:
                rset, wset = state.read_and_write_sets()
                read_set |= rset
                write_set |= wset
                # Add to write set also scalars between tasklets
                for src_node in state.nodes():
                    if not isinstance(src_node, nodes.Tasklet):
                        continue
                    for dst_node in state.nodes():
                        if src_node is dst_node:
                            continue
                        if not isinstance(dst_node, nodes.Tasklet):
                            continue
                        for e in state.edges_between(src_node, dst_node):
                            if e.data.data and e.data.data in sdfg.arrays:
                                write_set.add(e.data.data)
                # Add data from edges
                for src in states:
                    for dst in states:
                        for edge in sdfg.edges_between(src, dst):
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
                if not found and self._is_array_thread_local(name, itervar, sdfg, states):
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
                if state is body:
                    continue
                for src, dst, data in sdfg.in_edges(state):
                    nsymbols.update({s: sdfg.symbols[s] for s in data.assignments.keys() if s in sdfg.symbols})
                    nsdfg.add_edge(src, dst, data)
            nsdfg.add_edge(body_end, exit_state, InterstateEdge())

            # Move guard -> body edge to guard -> new_body
            for src, dst, data, in sdfg.edges_between(guard, body):
                sdfg.add_edge(src, new_body, data)
            # Move body_end -> guard edge to new_body -> guard
            for src, dst, data in sdfg.edges_between(body_end, guard):
                sdfg.add_edge(new_body, dst, data)

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
                new_body.add_edge(cnode, name, w, None, memlet.Memlet.from_array(name, sdfg.arrays[name]))

            # Fix SDFG symbols
            for sym in sdfg.free_symbols - fsymbols:
                if sym in sdfg.symbols:
                    del sdfg.symbols[sym]
            for sym, dtype in nsymbols.items():
                nsdfg.symbols[sym] = dtype

            # Change body state reference
            body = new_body

        if (step < 0) == True:
            # If step is negative, we have to flip start and end to produce a
            # correct map with a positive increment
            start, end, step = end, start, -step

        # If necessary, make a nested SDFG with assignments
        isedge = sdfg.edges_between(guard, body)[0]
        symbols_to_remove = set()
        if len(isedge.data.assignments) > 0:
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
            for name, desc in nsdfg.sdfg.arrays.items():
                if desc.transient and not self._is_array_thread_local(name, itervar, nsdfg.sdfg, nsdfg.sdfg.states()):
                    odesc = copy.deepcopy(desc)
                    sdfg.arrays[name] = odesc
                    desc.transient = False
                    wnode = body.add_access(name)
                    nsdfg.add_out_connector(name)
                    body.add_edge(nsdfg, name, wnode, None, memlet.Memlet.from_array(name, odesc))

            nstate = nsdfg.sdfg.node(0)
            init_state = nsdfg.sdfg.add_state_before(nstate)
            nisedge = nsdfg.sdfg.edges_between(init_state, nstate)[0]
            nisedge.data.assignments = isedge.data.assignments
            symbols_to_remove = set(nisedge.data.assignments.keys())
            for k in nisedge.data.assignments.keys():
                if k in nsdfg.symbol_mapping:
                    del nsdfg.symbol_mapping[k]
            isedge.data.assignments = {}

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        # Check intermediate notes
        intermediate_nodes = []
        for node in body.nodes():
            if isinstance(node, nodes.AccessNode) and body.in_degree(node) > 0 and node not in sink_nodes:
                # Scalars written without WCR must be thread-local
                if isinstance(node.desc(sdfg), dt.Scalar) and any(e.data.wcr is None for e in body.in_edges(node)):
                    continue
                # Arrays written with subsets that do not depend on the loop variable must be thread-local
                map_dependency = False
                for e in state.in_edges(node):
                    subset = e.data.get_dst_subset(e, state)
                    if any(str(s) == itervar for s in subset.free_symbols):
                        map_dependency = True
                        break
                if not map_dependency:
                    continue
                intermediate_nodes.append(node)

        map = nodes.Map(body.label + "_map", [itervar], [(start, end, step)])
        entry = nodes.MapEntry(map)
        exit = nodes.MapExit(map)
        body.add_node(entry)
        body.add_node(exit)

        # If the map uses symbols from data containers, instantiate reads
        containers_to_read = entry.free_symbols & sdfg.arrays.keys()
        for rd in containers_to_read:
            # We are guaranteed that this is always a scalar, because
            # can_be_applied makes sure there are no sympy functions in each of
            # the loop expresions
            access_node = body.add_read(rd)
            body.add_memlet_path(access_node, entry, dst_conn=rd, memlet=memlet.Memlet(rd))

        # Direct edges among source and sink access nodes must pass through a tasklet.
        # We first gather them and handle them later.
        direct_edges = set()
        for n1 in source_nodes:
            if not isinstance(n1, nodes.AccessNode):
                continue
            for n2 in sink_nodes:
                if not isinstance(n2, nodes.AccessNode):
                    continue
                for e in body.edges_between(n1, n2):
                    e.data.try_initialize(sdfg, body, e)
                    direct_edges.add(e)
                    body.remove_edge(e)

        # Reroute all memlets through the entry and exit nodes
        for n in source_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.out_edges(n):
                    # Fix memlet to contain outer data as subset
                    new_memlet = align_memlet(body, e, dst=False)

                    body.remove_edge(e)
                    body.add_edge_pair(entry, e.dst, n, new_memlet, internal_connector=e.dst_conn)
            else:
                body.add_nedge(entry, n, memlet.Memlet())
        for n in sink_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.in_edges(n):
                    # Fix memlet to contain outer data as subset
                    new_memlet = align_memlet(body, e, dst=True)

                    body.remove_edge(e)
                    body.add_edge_pair(exit, e.src, n, new_memlet, internal_connector=e.src_conn)
            else:
                body.add_nedge(n, exit, memlet.Memlet())
        intermediate_sinks = {}
        for n in intermediate_nodes:
            if isinstance(sdfg.arrays[n.data], dt.View):
                continue
            if n.data in intermediate_sinks:
                sink = intermediate_sinks[n.data]
            else:
                sink = body.add_access(n.data)
                intermediate_sinks[n.data] = sink
            helpers.make_map_internal_write_external(sdfg, body, exit, n, sink)

        # Here we handle the direct edges among source and sink access nodes.
        for e in direct_edges:
            src = e.src.data
            dst = e.dst.data
            if e.data.subset.num_elements() == 1:
                t = body.add_tasklet(f"{n1}_{n2}", {'__inp'}, {'__out'}, "__out =  __inp")
                src_conn, dst_conn = '__out', '__inp'
            else:
                desc = sdfg.arrays[src]
                tname, _ = sdfg.add_transient('tmp',
                                              e.data.src_subset.size(),
                                              desc.dtype,
                                              desc.storage,
                                              find_new_name=True)
                t = body.add_access(tname)
                src_conn, dst_conn = None, None
            body.add_memlet_path(n1,
                                 entry,
                                 t,
                                 memlet=memlet.Memlet(data=src, subset=e.data.src_subset),
                                 dst_conn=dst_conn)
            body.add_memlet_path(t,
                                 exit,
                                 n2,
                                 memlet=memlet.Memlet(data=dst,
                                                      subset=e.data.dst_subset,
                                                      wcr=e.data.wcr,
                                                      wcr_nonatomic=e.data.wcr_nonatomic),
                                 src_conn=src_conn)

        if not source_nodes and not sink_nodes:
            body.add_nedge(entry, exit, memlet.Memlet())

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
