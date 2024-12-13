# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop to map transformation """

from collections import defaultdict
import copy
import sympy as sp
from typing import Dict, List, Set

from dace import data as dt, dtypes, memlet, nodes, sdfg as sd, symbolic, subsets, properties
from dace.codegen.tools.type_inference import infer_expr_type
from dace.sdfg import graph as gr, nodes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import BreakBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock
import dace.transformation.helpers as helpers
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis


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


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToMap(xf.MultiStateTransformation):
    """
    Convert a control flow loop into a dataflow map. Currently only supports the simple case where there is no overlap
    between inputs and outputs in the body of the loop, and where the loop body only consists of a single state.
    """

    loop = xf.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive = False):
        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or end is None or step is None or itervar is None:
            return False

        sset = {}
        sset.update(sdfg.symbols)
        sset.update(sdfg.arrays)
        t = dtypes.result_type_of(infer_expr_type(start, sset), infer_expr_type(step, sset), infer_expr_type(end, sset))
        # We may only convert something to map if the bounds are all integer-derived types. Otherwise most map schedules
        # except for sequential would be invalid.
        if not t in dtypes.INTEGER_TYPES:
            return False

        # Loops containing break, continue, or returns may not be turned into a map.
        for blk in self.loop.all_control_flow_blocks():
            if isinstance(blk, (BreakBlock, ContinueBlock, ReturnBlock)):
                return False

        # We cannot handle symbols read from data containers unless they are scalar.
        for expr in (start, end, step):
            if symbolic.contains_sympy_functions(expr):
                return False

        _, write_set = self.loop.read_and_write_sets()
        loop_states = set(self.loop.all_states())
        all_loop_blocks = set(self.loop.all_control_flow_blocks())

        # Collect symbol reads and writes from inter-state assignments
        in_order_loop_blocks = list(cfg_analysis.blockorder_topological_sort(self.loop, recursive=True,
                                                                             ignore_nonstate_blocks=False))
        symbols_that_may_be_used: Set[str] = {itervar}
        used_before_assignment: Set[str] = set()
        for block in in_order_loop_blocks:
            for e in block.parent_graph.out_edges(block):
                # Collect read-before-assigned symbols (this works because the states are always in order,
                # see above call to `blockorder_topological_sort`)
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
        for state in sdfg.states():
            if state in loop_states:
                continue
            other_access_nodes |= set(n.data for n in state.data_nodes() if sdfg.arrays[n.data].transient)
        # Add non-transient nodes from loop state
        for state in loop_states:
            other_access_nodes |= set(n.data for n in state.data_nodes() if not sdfg.arrays[n.data].transient)

        write_memlets: Dict[str, List[memlet.Memlet]] = defaultdict(list)

        itersym = symbolic.pystr_to_symbolic(itervar)
        a = sp.Wild('a', exclude=[itersym])
        b = sp.Wild('b', exclude=[itersym])

        for state in loop_states:
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
        for state in loop_states:
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
        for e in self.loop.all_interstate_edges():
            isread_set |= set(e.data.get_read_memlets(sdfg.arrays))
        for mmlt in isread_set:
            if mmlt.data in write_memlets:
                if not self.test_read_memlet(sdfg, None, None, itersym, itervar, start, end, step, write_memlets, mmlt,
                                             mmlt.subset):
                    return False

        # Check that the iteration variable and other symbols are not used on other edges or blocks before they are
        # reassigned.
        in_order_blocks = list(cfg_analysis.blockorder_topological_sort(sdfg, recursive=True,
                                                                        ignore_nonstate_blocks=False))
        # First check the outgoing edges of the loop itself.
        reassigned_symbols: Set[str] = None
        for oe in graph.out_edges(self.loop):
            if symbols_that_may_be_used & oe.data.read_symbols():
                return False
            # Check for symbols that are set by all outgoing edges
            # TODO: Handle case of subset of out_edges
            if reassigned_symbols is None:
                reassigned_symbols = set(oe.data.assignments.keys())
            else:
                reassigned_symbols &= oe.data.assignments.keys()
        # Remove reassigned symbols
        if reassigned_symbols is not None:
            symbols_that_may_be_used -= reassigned_symbols
        loop_idx = in_order_blocks.index(self.loop)
        for block in in_order_blocks[loop_idx + 1:]:
            if block in all_loop_blocks:
                continue
            # Don't continue in this direction, as all loop symbols have been reassigned
            if not symbols_that_may_be_used:
                break

            # Check state contents
            if symbols_that_may_be_used & block.free_symbols:
                return False

            # Check inter-state edges
            reassigned_symbols = None
            for e in block.parent_graph.out_edges(block):
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

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        from dace.sdfg.propagation import align_memlet

        # Obtain loop information
        itervar = self.loop.loop_variable
        start = loop_analysis.get_init_assignment(self.loop)
        end = loop_analysis.get_loop_end(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)

        nsdfg = None

        # Nest loop-body states
        states = set(self.loop.all_states())
        # Find read/write sets
        read_set, write_set = set(), set()
        for state in self.loop.all_states():
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
        for edge in self.loop.all_interstate_edges():
            for s in edge.data.free_symbols:
                if s in sdfg.arrays:
                    read_set.add(s)

        # Find NestedSDFG's / Loop's unique data
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

        # Create NestedSDFG and add the loop contents to it. Gaher symbols defined in the NestedSDFG.
        fsymbols = set(sdfg.free_symbols)
        body = graph.add_state_before(self.loop, 'single_state_body')
        nsdfg = SDFG('loop_body', constants=sdfg.constants_prop, parent=body)
        nsdfg.add_node(self.loop.start_block, is_start_block=True)
        nsymbols = dict()
        for block in self.loop.nodes():
            if block is self.loop.start_block:
                continue
            nsdfg.add_node(block)
        for e in self.loop.edges():
            nsymbols.update({s: sdfg.symbols[s] for s in e.data.assignments.keys() if s in sdfg.symbols})
            nsdfg.add_edge(e.src, e.dst, e.data)

        # Add NestedSDFG arrays
        for name in read_set | write_set:
            nsdfg.arrays[name] = copy.deepcopy(sdfg.arrays[name])
            nsdfg.arrays[name].transient = False
        for name in unique_set:
            nsdfg.arrays[name] = sdfg.arrays[name]
            del sdfg.arrays[name]

        # Add NestedSDFG node
        cnode = body.add_nested_sdfg(nsdfg, body, read_set, write_set)
        if sdfg.parent:
            for s, m in sdfg.parent_nsdfg_node.symbol_mapping.items():
                if s not in cnode.symbol_mapping:
                    cnode.symbol_mapping[s] = m
                    nsdfg.add_symbol(s, sdfg.symbols[s])
        for name in read_set:
            r = body.add_read(name)
            body.add_edge(r, None, cnode, name, memlet.Memlet.from_array(name, sdfg.arrays[name]))
        for name in write_set:
            w = body.add_write(name)
            body.add_edge(cnode, name, w, None, memlet.Memlet.from_array(name, sdfg.arrays[name]))

        # Fix SDFG symbols
        for sym in sdfg.free_symbols - fsymbols:
            if sym in sdfg.symbols:
                del sdfg.symbols[sym]
        for sym, dtype in nsymbols.items():
            nsdfg.symbols[sym] = dtype

        if (step < 0) == True:
            # If step is negative, we have to flip start and end to produce a correct map with a positive increment.
            start, end, step = end, start, -step

        source_nodes = body.source_nodes()
        sink_nodes = body.sink_nodes()

        # Check intermediate notes
        intermediate_nodes: List[nodes.AccessNode] = []
        for node in body.nodes():
            if isinstance(node, nodes.AccessNode) and body.in_degree(node) > 0 and node not in sink_nodes:
                # Scalars written without WCR must be thread-local
                if isinstance(node.desc(sdfg), dt.Scalar) and any(e.data.wcr is None for e in body.in_edges(node)):
                    continue
                # Arrays written with subsets that do not depend on the loop variable must be thread-local
                map_dependency = False
                for e in body.in_edges(node):
                    subset = e.data.get_dst_subset(e, body)
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
        direct_edges: Set[gr.MultiConnectorEdge[memlet.Memlet]] = set()
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
        intermediate_sinks: Dict[str, nodes.AccessNode] = {}
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
            src: str = e.src.data
            dst: str = e.dst.data
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

        # Redirect outgoing edges connected to the loop to connect to the body state instead.
        for e in graph.out_edges(self.loop):
            graph.add_edge(body, e.dst, e.data)
        # Delete the loop and connected edges.
        graph.remove_node(self.loop)

        # If this had made the iteration variable a free symbol, we can remove it from the SDFG symbols
        if itervar in sdfg.free_symbols:
            sdfg.remove_symbol(itervar)

        sdfg.reset_cfg_list()
        for n, p in sdfg.all_nodes_recursive():
            if isinstance(n, nodes.NestedSDFG):
                n.sdfg.parent = p
                n.sdfg.parent_nsdfg_node = n
                n.sdfg.parent_sdfg = p.sdfg
