# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop to map transformation """

from collections import defaultdict
import copy
import itertools
import sympy as sp
import networkx as nx
from typing import Dict, List, Optional, Tuple

from dace import dtypes, memlet, nodes, registry, sdfg as sd, symbolic, subsets
from dace.properties import Property, make_properties, CodeBlock
from dace.sdfg import graph as gr, nodes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg
from dace.frontend.python.astutils import ASTFindReplace
from dace.transformation.interstate.loop_detection import (DetectLoop,
                                                           find_for_loop)
import dace.transformation.helpers as helpers
from dace.transformation import transformation as xf


def _check_range(subset, a, itersym, b, step):
    found = False
    for rb, re, _ in subset.ndrange():
        m = rb.match(a * itersym + b)
        if m is None:
            continue
        if (m[a] >= 1) != True:
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
            if (m[a] >= 1) != True:
                continue
        found = True
        break
    return found


@registry.autoregister
class LoopToMap(DetectLoop):
    """Convert a control flow loop into a dataflow map. Currently only supports
       the simple case where there is no overlap between inputs and outputs in
       the body of the loop, and where the loop body only consists of a single
       state.
    """
    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # Is this even a loop
        if not DetectLoop.can_be_applied(graph, candidate, expr_index, sdfg,
                                         strict):
            return False

        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])

        # Guard state should contain any dataflow
        if len(guard.nodes()) != 0:
            return False

        # Only support loops with a single-state body
        begin_outedges = graph.out_edges(begin)
        if len(begin_outedges) != 1 or begin_outedges[0].dst != guard:
            return False

        # If loop cannot be detected, fail
        found = find_for_loop(graph, guard, begin)
        if not found:
            return False

        itervar, (start, end, step), _ = found

        # We cannot handle symbols read from data containers unless they are
        # scalar
        for expr in (start, end, step):
            if symbolic.contains_sympy_functions(expr):
                return False

        _, write_set = begin.read_and_write_sets()
        code_nodes = [n for n in begin.nodes() if isinstance(n, nodes.CodeNode)]

        # Get access nodes from other states to isolate local loop variables
        other_access_nodes = set()
        for state in sdfg.nodes():
            if state is begin:
                continue
            other_access_nodes |= set(n.data for n in state.data_nodes()
                                      if sdfg.arrays[n.data].transient)
        # Add non-transient nodes from loop state
        other_access_nodes |= set(n.data for n in begin.data_nodes()
                                  if not sdfg.arrays[n.data].transient)

        write_memlets = defaultdict(list)

        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])

        for cn in code_nodes:
            # Take all writes that are not conflicted into consideration
            for e in begin.out_edges(cn):
                data = e.data.data
                if data not in other_access_nodes:
                    continue
                subset = e.data.subset
                if data in write_set:
                    if e.data.dynamic and e.data.wcr is None:
                        # If pointers are involved, give up
                        return False
                    # To be sure that the value is only written at unique
                    # indices per loop iteration, we want to match symbols
                    # of the form "a*i+b" where a >= 1, and i is the iteration
                    # variable. The iteration variable must be used.
                    if e.data.wcr is None:
                        if not _check_range(e.data.subset, a, itersym, b, step):
                            return False
                    # End of check

                    write_memlets[data].append(e.data)

        # After looping over relevant writes, consider reads that may overlap
        for cn in code_nodes:
            for e in begin.in_edges(cn):
                data = e.data.data
                subset = e.data.subset
                from dace.sdfg.propagation import propagate_subset
                # If the same container is both read and written, only match if
                # it read and written at locations that will not create data races
                if data in write_memlets:
                    if e.data.dynamic and subset.num_elements() != 1:
                        # If pointers are involved, give up
                        return False
                    if not _check_range(e.data.subset, a, itersym, b, step):
                        return False

                    pread = propagate_subset([e.data], sdfg.arrays[data],
                                             [itervar],
                                             subsets.Range([(start, end, step)
                                                            ]))
                    for candidate in write_memlets[data]:
                        # Simple case: read and write are in the same subset
                        if e.data.subset == candidate.subset:
                            break
                        # Propagated read does not overlap with propagated write
                        pwrite = propagate_subset([candidate],
                                                  sdfg.arrays[data], [itervar],
                                                  subsets.Range([(start, end,
                                                                  step)]))
                        if subsets.intersects(pread.subset,
                                              pwrite.subset) is False:
                            break
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
            if itervar in state.free_symbols:
                return False
            # Don't continue in this direction, as the variable has
            # now been reassigned
            # TODO: Handle case of subset of out_edges
            if all(itervar in e.data.assignments
                   for e in sdfg.out_edges(state)):
                break

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        guard = graph.node(candidate[DetectLoop._loop_guard])
        begin = graph.node(candidate[DetectLoop._loop_begin])
        sexit = graph.node(candidate[DetectLoop._exit_state])

        return (' -> '.join(state.label
                            for state in [guard, begin, sexit]) + ' (for loop)')

    def apply(self, sdfg: sd.SDFG):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        body: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        itervar, (start, end, step), _ = find_for_loop(sdfg, guard, body)

        if (step < 0) == True:
            # If step is negative, we have to flip start and end to produce a
            # correct map with a positive increment
            start, end, step = end, start, -step

        # If necessary, make a nested SDFG with assignments
        isedge = sdfg.edges_between(guard, body)[0]
        symbols_to_remove = set()
        if len(isedge.data.assignments) > 0:
            nsdfg = helpers.nest_state_subgraph(
                sdfg, body, gr.SubgraphView(body, body.nodes()))
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
            body.add_memlet_path(access_node,
                                 entry,
                                 dst_conn=rd,
                                 memlet=memlet.Memlet(rd))

        # Reroute all memlets through the entry and exit nodes
        for n in source_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.out_edges(n):
                    body.remove_edge(e)
                    body.add_edge_pair(entry,
                                       e.dst,
                                       n,
                                       e.data,
                                       internal_connector=e.dst_conn)
            else:
                body.add_nedge(entry, n, memlet.Memlet())
        for n in sink_nodes:
            if isinstance(n, nodes.AccessNode):
                for e in body.in_edges(n):
                    body.remove_edge(e)
                    body.add_edge_pair(exit,
                                       e.src,
                                       n,
                                       e.data,
                                       internal_connector=e.src_conn)
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
        sdfg.add_edge(
            body, after,
            sd.InterstateEdge(assignments=after_edge.data.assignments))

        # If this had made the iteration variable a free symbol, we can remove
        # it from the SDFG symbols
        if itervar in sdfg.free_symbols:
            sdfg.remove_symbol(itervar)
        for sym in symbols_to_remove:
            if helpers.is_symbol_unused(sdfg, sym):
                sdfg.remove_symbol(sym)
