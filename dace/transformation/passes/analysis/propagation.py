# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from collections import OrderedDict, defaultdict, deque
import copy
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx
import sympy

from dace import data as dt
from dace import properties, symbolic, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.sdfg.propagation import align_memlet, propagate_memlet, propagate_subset
from dace.sdfg.scope import ScopeTree
from dace.sdfg.sdfg import SDFG, InterstateEdge, memlets_in_ast
from dace.sdfg.state import (ConditionalBlock, ControlFlowBlock, ControlFlowRegion, GlobalDepDataRecordT, LoopRegion,
                             SDFGState)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.helpers import unsqueeze_memlet
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.analysis import ControlFlowBlockReachability


MemletPropResultT = Dict[ControlFlowBlock, Tuple[Dict[str, GlobalDepDataRecordT],
                                                 Dict[str, GlobalDepDataRecordT],
                                                 Dict[str, GlobalDepDataRecordT],
                                                 Dict[str, GlobalDepDataRecordT]]]

@transformation.explicit_cf_compatible
class StatePropagation(ppl.ControlFlowRegionPass):
    """
    Analyze a control flow region to determine the number of times each block inside of it is executed in the form of a
    symbolic expression, or a concrete number where possible.
    Each control flow block is marked with a symbolic expression for the number of executions, and a boolean flag to
    indicate whether the number of executions is dynamic or not. A combination of dynamic being set to true and the
    number of executions being 0 indicates that the number of executions is dynamically unbounded.
    Additionally, the pass annotates each block with a `ranges` property, which indicates for loop variables defined
    at that block what range of values the variable may take on.
    Note: This path directly annotates the graph.
    This pass supersedes ``dace.sdfg.propagation.propagate_states`` and is based on its algorithm, with significant
    simplifications thanks to the use of control flow regions.
    """

    CATEGORY: str = 'Analysis'

    def __init__(self):
        super().__init__()
        self.top_down = True
        self.apply_to_conditionals = True

    def depends_on(self):
        return {ControlFlowBlockReachability}

    def _propagate_in_cfg(self, cfg: ControlFlowRegion, reachable: Dict[ControlFlowBlock, Set[ControlFlowBlock]],
                          starting_executions: int, starting_dynamic_executions: bool):
        visited_blocks: Set[ControlFlowBlock] = set()
        traversal_q: deque[Tuple[ControlFlowBlock, int, bool, List[str]]] = deque()
        traversal_q.append((cfg.start_block, starting_executions, starting_dynamic_executions, []))
        while traversal_q:
            (block, proposed_executions, proposed_dynamic, itvar_stack) = traversal_q.pop()
            out_edges = cfg.out_edges(block)
            if block in visited_blocks:
                # This block has already been visited, meaning there are multiple paths towards this block.
                if proposed_executions == 0 and proposed_dynamic:
                    block.executions = 0
                    block.dynamic_executions = True
                else:
                    block.executions = sympy.Max(block.executions, proposed_executions).doit()
                    block.dynamic_executions = (block.dynamic_executions or proposed_dynamic)
            elif proposed_dynamic and proposed_executions == 0:
                # We're propagating a dynamic unbounded number of executions, which always gets propagated
                # unconditionally. Propagate to all children.
                visited_blocks.add(block)
                block.executions = proposed_executions
                block.dynamic_executions = proposed_dynamic
                # This gets pushed through to all children unconditionally.
                if len(out_edges) > 0:
                    for oedge in out_edges:
                        traversal_q.append((oedge.dst, proposed_executions, proposed_dynamic, itvar_stack))
            else:
                # If the state hasn't been visited yet and we're not propagating a dynamic unbounded number of
                # executions, we calculate the number of executions for the next state(s) and continue propagating.
                visited_blocks.add(block)
                block.executions = proposed_executions
                block.dynamic_executions = proposed_dynamic
                if len(out_edges) == 1:
                    # Continue with the only child state.
                    if not out_edges[0].data.is_unconditional():
                        # If the transition to the child state is based on a condition, this state could be an implicit
                        # exit state. The child state's number of executions is thus only given as an upper bound and
                        # marked as dynamic.
                        proposed_dynamic = True
                    traversal_q.append((out_edges[0].dst, proposed_executions, proposed_dynamic, itvar_stack))
                elif len(out_edges) > 1:
                    # Conditional split
                    for oedge in out_edges:
                        traversal_q.append((oedge.dst, block.executions, True, itvar_stack))

        # Check if the CFG contains any cycles. Any cycles left in the graph (after control flow raising) are
        # irreducible control flow and thus lead to a dynamically unbounded number of executions. Mark any block
        # inside and reachable from any block inside the cycle as dynamically unbounded, irrespectively of what it was
        # marked as before.
        cycles: Iterable[Iterable[ControlFlowBlock]] = cfg.find_cycles()
        for cycle in cycles:
            for blk in cycle:
                blk.executions = 0
                blk.dynamic_executions = True
                for reached in reachable[blk]:
                    reached.executions = 0
                    blk.dynamic_executions = True

    def apply(self, region, pipeline_results) -> None:
        if isinstance(region, ConditionalBlock):
            # In a conditional block, each branch is executed up to as many times as the conditional block itself is.
            # TODO(later): We may be able to derive ranges here based on the branch conditions too.
            for _, b in region.branches:
                b.executions = region.executions
                b.dynamic_executions = True
                b.ranges = region.ranges
        else:
            if isinstance(region, SDFG):
                # The root SDFG is executed exactly once, any other, nested SDFG is executed as many times as the parent
                # state is.
                if region is region.root_sdfg:
                    region.executions = 1
                    region.dynamic_executions = False
                elif region.parent:
                    region.executions = region.parent.executions
                    region.dynamic_executions = region.parent.dynamic_executions

            # Clear existing annotations.
            for blk in region.nodes():
                blk.executions = 0
                blk.dynamic_executions = True
                blk.ranges = region.ranges

            # Determine the number of executions for the start block within this region. In the case of loops, this
            # is dependent on the number of loop iterations - where they can be determined. Where they may not be
            # determined, the number of iterations is assumed to be dynamically unbounded. For any other control flow
            # region, the start block is executed as many times as the region itself is.
            starting_execs = region.executions
            starting_dynamic = region.dynamic_executions
            if isinstance(region, LoopRegion):
                # If inside a loop, add range information if possible.
                start = loop_analysis.get_init_assignment(region)
                stop = loop_analysis.get_loop_end(region)
                stride = loop_analysis.get_loop_stride(region)
                if start is not None and stop is not None and stride is not None and region.loop_variable:
                    # This inequality needs to be checked exactly like this due to constraints in sympy/symbolic
                    # expressions, do not simplify!
                    if (stride < 0) == True:
                        rng = (stop, start, -stride)
                    else:
                        rng = (start, stop, stride)
                    for blk in region.nodes():
                        blk.ranges[str(region.loop_variable)] = subsets.Range([rng])

                    # Get surrounding iteration variables for the case of nested loops.
                    itvar_stack = []
                    par = region.parent_graph
                    while par is not None and not isinstance(par, SDFG):
                        if isinstance(par, LoopRegion) and par.loop_variable:
                            itvar_stack.append(par.loop_variable)
                        par = par.parent_graph

                    # Calculate the number of loop executions.
                    # This resolves ranges based on the order of iteration variables from surrounding loops.
                    loop_executions = sympy.ceiling(((stop + 1) - start) / stride)
                    for outer_itvar_string in itvar_stack:
                        outer_range = region.ranges[outer_itvar_string]
                        outer_start = outer_range[0][0]
                        outer_stop = outer_range[0][1]
                        outer_stride = outer_range[0][2]
                        outer_itvar = symbolic.pystr_to_symbolic(outer_itvar_string)
                        exec_repl = loop_executions.subs({outer_itvar: (outer_itvar * outer_stride + outer_start)})
                        sum_rng = (outer_itvar, 0, sympy.ceiling((outer_stop - outer_start) / outer_stride))
                        loop_executions = sympy.Sum(exec_repl, sum_rng)
                    starting_execs = loop_executions.doit()
                    starting_dynamic = region.dynamic_executions
                else:
                    starting_execs = 0
                    starting_dynamic = True

            # Propagate the number of executions.
            self._propagate_in_cfg(region, pipeline_results[ControlFlowBlockReachability.__name__][region.cfg_id],
                                   starting_execs, starting_dynamic)


@properties.make_properties
@transformation.explicit_cf_compatible
class MemletPropagation(ppl.ControlFlowRegionPass):
    """
    TODO
    """

    CATEGORY: str = 'Analysis'

    def __init__(self) -> None:
        super().__init__()
        self.top_down = False
        self.apply_to_conditionals = True

    def modifies(self):
        return ppl.Modifies.Memlets

    def should_reapply(self, modified):
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets)

    def _propagate_node(self, state: SDFGState, node: Union[nodes.EntryNode, nodes.ExitNode]):
        if isinstance(node, nodes.EntryNode):
            internal_edges = [e for e in state.out_edges(node) if e.src_conn and e.src_conn.startswith('OUT_')]
            external_edges = [e for e in state.in_edges(node) if e.dst_conn and e.dst_conn.startswith('IN_')]
            geticonn = lambda e: e.src_conn[4:]
            geteconn = lambda e: e.dst_conn[3:]
            use_dst = False
        else:
            internal_edges = [e for e in state.in_edges(node) if e.dst_conn and e.dst_conn.startswith('IN_')]
            external_edges = [e for e in state.out_edges(node) if e.src_conn and e.src_conn.startswith('OUT_')]
            geticonn = lambda e: e.dst_conn[3:]
            geteconn = lambda e: e.src_conn[4:]
            use_dst = True

        for edge in external_edges:
            if edge.data.is_empty():
                new_memlet = Memlet()
            else:
                internal_edge = next(e for e in internal_edges if geticonn(e) == geteconn(edge))
                aligned_memlet = align_memlet(state, internal_edge, dst=use_dst)
                new_memlet = propagate_memlet(state, aligned_memlet, node, True, connector=geteconn(edge))
            edge.data = new_memlet

    def _propagate_scope(self,
                         state: SDFGState,
                         scopes: List[ScopeTree],
                         propagate_entry: bool = True,
                         propagate_exit: bool = True) -> None:
        scopes_to_process = scopes
        next_scopes = set()

        # Process scopes from the inputs upwards, propagating edges at the
        # entry and exit nodes
        while len(scopes_to_process) > 0:
            for scope in scopes_to_process:
                if scope.entry is None:
                    continue

                # Propagate out of entry
                if propagate_entry:
                    self._propagate_node(state, scope.entry)

                # Propagate out of exit
                if propagate_exit:
                    self._propagate_node(state, scope.exit)

                # Add parent to next frontier
                next_scopes.add(scope.parent)
            scopes_to_process = next_scopes
            next_scopes = set()

    def _propagate_nsdfg(self, parent_sdfg: SDFG, parent_state: SDFGState, nsdfg_node: nodes.NestedSDFG):
        outer_symbols = parent_state.symbols_defined_at(nsdfg_node)
        sdfg = nsdfg_node.sdfg

        possible_reads = copy.deepcopy(sdfg.possible_reads)
        possible_writes = copy.deepcopy(sdfg.possible_writes)
        certain_reads = copy.deepcopy(sdfg.certain_reads)
        certain_writes = copy.deepcopy(sdfg.certain_writes)

        # Make sure any potential NSDFG symbol mapping is correctly reversed when propagating out.
        for mapping in [possible_reads, possible_writes, certain_reads, certain_writes]:
            for border_memlet in mapping.values():
                border_memlet.replace(nsdfg_node.symbol_mapping)

                # Also make sure that there's no symbol in the border memlet's range that only exists inside the
                # nested SDFG. If that's the case, use the entire range.
                if border_memlet.src_subset is not None:
                    if any(str(s) not in outer_symbols.keys() for s in border_memlet.src_subset.free_symbols):
                        border_memlet.src_subset = subsets.Range.from_array(sdfg.arrays[border_memlet.data])
                if border_memlet.dst_subset is not None:
                    if any(str(s) not in outer_symbols.keys() for s in border_memlet.dst_subset.free_symbols):
                        border_memlet.dst_subset = subsets.Range.from_array(sdfg.arrays[border_memlet.data])

        # Propagate the inside 'border' memlets outside the SDFG by offsetting, and unsqueezing if necessary.
        for iedge in parent_state.in_edges(nsdfg_node):
            if iedge.dst_conn in possible_reads:
                try:
                    inner_memlet = possible_reads[iedge.dst_conn]
                    iedge.data = unsqueeze_memlet(inner_memlet, iedge.data, True)
                    if isinstance(iedge.data.subset, subsets.SubsetUnion):
                        iedge.data.subset = iedge.data.subset.to_bounding_box_subset()
                    # If no appropriate memlet found, use array dimension
                    for i, (rng, s) in enumerate(zip(iedge.data.subset, parent_sdfg.arrays[iedge.data.data].shape)):
                        if rng[1] + 1 == s:
                            iedge.data.subset[i] = (iedge.data.subset[i][0], s - 1, 1)
                        if symbolic.issymbolic(iedge.data.volume):
                            if any(str(s) not in outer_symbols for s in iedge.data.volume.free_symbols):
                                iedge.data.volume = 0
                                iedge.data.dynamic = True
                except (ValueError, NotImplementedError):
                    # In any case of memlets that cannot be unsqueezed (i.e., reshapes), use dynamic unbounded memlets.
                    iedge.data.volume = 0
                    iedge.data.dynamic = True
        for oedge in parent_state.out_edges(nsdfg_node):
            if oedge.src_conn in possible_writes:
                try:
                    inner_memlet = possible_writes[oedge.src_conn]
                    oedge.data = unsqueeze_memlet(inner_memlet, oedge.data, True)
                    if isinstance(oedge.data.subset, subsets.SubsetUnion):
                        oedge.data.subset = oedge.data.subset.to_bounding_box_subset()
                    # If no appropriate memlet found, use array dimension
                    for i, (rng, s) in enumerate(zip(oedge.data.subset, parent_sdfg.arrays[oedge.data.data].shape)):
                        if rng[1] + 1 == s:
                            oedge.data.subset[i] = (oedge.data.subset[i][0], s - 1, 1)
                        if symbolic.issymbolic(oedge.data.volume):
                            if any(str(s) not in outer_symbols for s in oedge.data.volume.free_symbols):
                                oedge.data.volume = 0
                                oedge.data.dynamic = True
                except (ValueError, NotImplementedError):
                    # In any case of memlets that cannot be unsqueezed (i.e., reshapes), use dynamic unbounded memlets.
                    oedge.data.volume = 0
                    oedge.data.dynamic = True

    def _propagate_state(self, state: SDFGState, result: MemletPropResultT) -> None:
        # Ensure memlets around nested SDFGs are propagated correctly.
        for nd in state.nodes():
            if isinstance(nd, nodes.NestedSDFG):
                self._propagate_nsdfg(state.sdfg, state, nd)

        # Propagate memlets through the scopes, bottom up, starting at the scope leaves.
        # TODO: Make sure this propagation happens without overapproximation, i.e., using SubsetUnions.
        self._propagate_scope(state, state.scope_leaves())

        # Gather all writes and reads inside this state now to determine the state-wide reads and writes.
        # Collect write memlets.
        writes: Dict[str, List[Tuple[MultiConnectorEdge[Memlet], nodes.AccessNode]]] = defaultdict(lambda: [])
        for anode in state.data_nodes():
            is_view = isinstance(state.sdfg.data(anode.data), dt.View)
            for iedge in state.in_edges(anode):
                if not iedge.data.is_empty() and not (is_view and iedge.dst_conn == 'views'):
                    root_edge = state.memlet_tree(iedge).root().edge
                    writes[anode.data].append((root_edge, anode))

        # Go over (overapproximated) reads and check if they are covered by writes.
        not_covered_reads: Dict[str, List[Tuple[MultiConnectorEdge[Memlet],
                                                nodes.AccessNode]]] = defaultdict(lambda: [])
        for anode in state.data_nodes():
            for oedge in state.out_edges(anode):
                if not oedge.data.is_empty() and not (isinstance(oedge.dst, nodes.AccessNode)
                                                      and oedge.dst_conn == 'views'):
                    if oedge.data.data != anode.data:
                        # Special case for memlets copying data out of the scope, which are by default aligned with the
                        # outside data container. In this case, the source container must either be a scalar, or the
                        # read subset is contained in the memlet's `other_subset` property.
                        # See `dace.sdfg.propagation.align_memlet` for more.
                        desc = state.sdfg.data(oedge.data.data)
                        if oedge.data.other_subset is not None:
                            read_subset = oedge.data.other_subset
                        elif oedge.data.dst_subset is not None:
                            read_subset = oedge.data.dst_subset
                        elif isinstance(desc, dt.Scalar) or (isinstance(desc, dt.Array) and desc.total_size == 1):
                            read_subset = subsets.Range([(0, 0, 1)] * len(desc.shape))
                        else:
                            raise RuntimeError('Invalid memlet range detected in MemletPropagation')
                    else:
                        read_subset = oedge.data.src_subset or oedge.data.subset
                    covered = False
                    for [write, to] in writes[anode.data]:
                        if write.data.subset.covers_precise(read_subset) and nx.has_path(state.nx, to, anode):
                            covered = True
                            break
                    if not covered:
                        not_covered_reads[anode.data].append((oedge, anode))

        # Filter out any writes that are overwritten later on
        for data in writes:
            if len(writes[data]) > 1:
                to_remove: Set[nodes.AccessNode] = set()
                grouped_by_anodes: Dict[nodes.AccessNode, Set[MultiConnectorEdge[Memlet]]] = {}
                for edge, nd in writes[data]:
                    if nd in grouped_by_anodes:
                        grouped_by_anodes[nd].add(edge)
                    else:
                        grouped_by_anodes[nd] = set([edge])
                for nd in grouped_by_anodes.keys():
                    if nd in to_remove:
                        continue
                    write_subset = None
                    for write in grouped_by_anodes[nd]:
                        if write_subset is None:
                            write_subset = subsets.SubsetUnion(write.data.subset)
                        else:
                            write_subset.union(write.data.subset)
                    if write_subset is None:
                        continue
                    for other_write_nd in grouped_by_anodes.keys():
                        if other_write_nd is nd or other_write_nd in to_remove:
                            continue
                        if nx.has_path(state.nx, nd, other_write_nd):
                            other_write_subset = None
                            for other_write in grouped_by_anodes[other_write_nd]:
                                if other_write_subset is None:
                                    other_write_subset = subsets.SubsetUnion(other_write.data.subset)
                                else:
                                    other_write_subset.union(other_write.data.subset)
                            if other_write_subset is not None and other_write_subset.covers_precise(write_subset):
                                to_remove.add(nd)
                filtered_writes = []
                for nd in grouped_by_anodes.keys():
                    if nd not in to_remove:
                        for write in grouped_by_anodes[nd]:
                            filtered_writes.append((write, nd))
                writes[data] = filtered_writes

        state.certain_writes = {}
        state.possible_writes = {}
        for data in writes:
            if len(writes[data]) > 0:
                subset = None
                volume = None
                is_dynamic = False
                for edge, _ in writes[data]:
                    memlet = edge.data
                    is_dynamic |= memlet.dynamic
                    if subset is None:
                        subset = subsets.SubsetUnion(memlet.dst_subset or memlet.subset)
                    else:
                        subset.union(memlet.dst_subset or memlet.subset)
                    if memlet.volume == 0:
                        volume = 0
                    else:
                        if volume is None:
                            volume = memlet.volume
                        elif volume != 0:
                            volume += memlet.volume
                new_memlet = Memlet(data=data, subset=subset)
                new_memlet.dynamic = is_dynamic
                new_memlet.volume = volume if volume is not None else 0
                state.certain_writes[data] = new_memlet
                state._certain_writes_moredata[data] = GlobalDepDataRecordT(memlet=new_memlet, accesses=writes[data])
                state.possible_writes[data] = new_memlet
                state._possible_writes_moredata[data] = GlobalDepDataRecordT(memlet=new_memlet, accesses=writes[data])

        state.certain_reads = {}
        state._certain_reads_moredata = {}
        state.possible_reads = {}
        state._possible_reads_moredata = {}
        for data in not_covered_reads:
            subset = None
            volume = None
            is_dynamic = False
            for edge, _ in not_covered_reads[data]:
                memlet = edge.data
                is_dynamic |= memlet.dynamic
                if subset is None:
                    subset = subsets.SubsetUnion(memlet.dst_subset or memlet.subset)
                else:
                    subset.union(memlet.dst_subset or memlet.subset)
                if memlet.volume == 0:
                    volume = 0
                else:
                    if volume is None:
                        volume = memlet.volume
                    elif volume != 0:
                        volume += memlet.volume
            new_memlet = Memlet(data=data, subset=subset)
            new_memlet.dynamic = is_dynamic
            new_memlet.volume = volume
            state.certain_reads[data] = new_memlet
            state._certain_reads_moredata[data] = GlobalDepDataRecordT(memlet=new_memlet,
                                                                       accesses=not_covered_reads[data])
            state.possible_reads[data] = new_memlet
            state._possible_reads_moredata[data] = GlobalDepDataRecordT(memlet=new_memlet,
                                                                        accesses=not_covered_reads[data])

        result[state] = (state._certain_reads_moredata, state._possible_reads_moredata,
                         state._certain_writes_moredata, state._possible_writes_moredata)

    def _propagate_conditional(self, conditional: ConditionalBlock, result: MemletPropResultT) -> None:
        # The union of all reads between all conditions and branches gives the set of _possible_ reads, while the
        # intersection gives the set of _guaranteed_ reads. The first condition can also be counted as a guaranteed
        # read. The same applies for writes, except that conditions do not contain writes.

        def add_memlet(memlet: Memlet,
                       mlt_dict: Dict[str, Memlet],
                       mlt_dict_moredata: Dict[str, GlobalDepDataRecordT],
                       use_intersection: bool = False,
                       accesses: Optional[List[Union[Tuple[MultiConnectorEdge[Memlet], nodes.AccessNode],
                                                     Edge[InterstateEdge]]]] = None):
            if memlet.data not in mlt_dict:
                propagated_memlet = Memlet(data=memlet.data, subset=(memlet.src_subset or memlet.subset))
                propagated_memlet.volume = memlet.volume
                propagated_memlet.dynamic = memlet.dynamic
                mlt_dict[memlet.data] = propagated_memlet
            else:
                propagated_memlet = mlt_dict[memlet.data]
                if use_intersection:
                    isect = propagated_memlet.subset.intersection(memlet.src_subset or memlet.subset)
                    if isect is not None:
                        propagated_memlet.subset = isect
                    else:
                        propagated_memlet.subset = subsets.SubsetUnion([])

                    propagated_memlet.volume = sympy.Min(memlet.volume, propagated_memlet.volume)
                    propagated_memlet.dynamic |= memlet.dynamic
                else:
                    mlt_subset = propagated_memlet.subset
                    if not isinstance(mlt_subset, subsets.SubsetUnion):
                        mlt_subset = subsets.SubsetUnion([mlt_subset])
                    mlt_subset.union(memlet.src_subset or memlet.subset)
                    propagated_memlet.subset = mlt_subset

                    if propagated_memlet.volume != 0:
                        if memlet.volume == 0:
                            propagated_memlet.volume = memlet.volume
                        else:
                            propagated_memlet.volume = sympy.Max(memlet.volume, propagated_memlet.volume)
                    propagated_memlet.dynamic |= memlet.dynamic
            if memlet.data not in mlt_dict_moredata:
                mlt_dict_moredata[memlet.data] = GlobalDepDataRecordT(memlet=mlt_dict[memlet.data], accesses=accesses)
            else:
                mlt_dict_moredata[memlet.data].memlet = mlt_dict[memlet.data]
                mlt_dict_moredata[memlet.data].accesses.extend(accesses)

        conditional.possible_reads = {}
        conditional.certain_reads = {}
        conditional.possible_writes = {}
        conditional.certain_writes = {}
        conditional._possible_reads_moredata = {}
        conditional._certain_reads_moredata = {}
        conditional._possible_writes_moredata = {}
        conditional._certain_writes_moredata = {}

        # Gather the union of possible reads and writes. At the same time, determine if there is an else branch present.
        has_else = False
        for cond, branch in conditional.branches:
            if cond is not None:
                read_memlets = memlets_in_ast(cond.code[0], conditional.sdfg.arrays)
                for read_memlet in read_memlets:
                    # TODO: passing the condition like this probably won't work, needs adjusting
                    add_memlet(read_memlet, conditional.possible_reads, conditional._possible_reads_moredata,
                               False, [cond])
            else:
                has_else = True
            for read_data in branch.possible_reads:
                read_memlet = branch.possible_reads[read_data]
                add_memlet(read_memlet, conditional.possible_reads, conditional._possible_reads_moredata,
                           False, branch._possible_reads_moredata[read_data].accesses)
            for write_data in branch.possible_writes:
                write_memlet = branch.possible_writes[write_data]
                add_memlet(write_memlet, conditional.possible_writes, conditional._possible_writes_moredata,
                           False, branch._possible_writes_moredata[write_data].accesses)

        # If there is no else branch or only one branch exists, there are no certain reads or writes.
        if len(conditional.branches) > 1 and has_else:
            # Gather the certain reads (= Intersection of certain reads for each branch)
            for container in conditional.possible_reads.keys():
                candidates = []
                skip = False
                for cond, branch in conditional.branches:
                    found = False
                    if cond is not None:
                        read_memlets = memlets_in_ast(cond.code[0], conditional.sdfg.arrays)
                        for read_memlet in read_memlets:
                            if read_memlet.data == container:
                                found = True
                                candidates.append((read_memlet, [cond]))
                    if container in branch.certain_reads:
                        found = True
                        candidates.append((branch.certain_reads[container],
                                           branch._certain_reads_moredata[container].accesses))
                    if not found:
                        skip = True
                        break
                if skip:
                    continue
                for cand_memlet, cand_accesses in candidates:
                    add_memlet(cand_memlet, conditional.certain_reads, conditional._certain_reads_moredata,
                               use_intersection=True, accesses=cand_accesses)
            # Gather the certain writes (= Intersection of certain writes for each branch)
            for container in conditional.possible_writes.keys():
                candidates = []
                skip = False
                for _, branch in conditional.branches:
                    if container in branch.certain_writes:
                        candidates.append((branch.certain_writes[container],
                                           branch._certain_writes_moredata[container].accesses))
                    else:
                        skip = True
                        break
                if skip:
                    continue
                for cand_memlet, cand_accesses in candidates:
                    add_memlet(cand_memlet, conditional.certain_writes, conditional._certain_writes_moredata,
                               use_intersection=True, accesses=cand_accesses)

        # Ensure the first condition's reads are part of the certain reads.
        first_cond = conditional.branches[0][0]
        if first_cond is not None:
            read_memlets = memlets_in_ast(first_cond.code[0], conditional.sdfg.arrays)
            for read_memlet in read_memlets:
                add_memlet(read_memlet, conditional.certain_reads, conditional._certain_reads_moredata,
                           False, [first_cond])

        result[conditional] = (conditional._certain_reads_moredata, conditional._possible_reads_moredata,
                               conditional._certain_writes_moredata, conditional._possible_writes_moredata)

    def _propagate_loop(self, loop: LoopRegion, result: MemletPropResultT) -> None:
        # First propagate the contents of the loop for one iteration.
        self._propagate_cfg(loop, result)

        # Propagate memlets from inside the loop through the loop ranges.
        # Collect loop information and form the loop variable range first.
        itvar = loop.loop_variable
        if not itvar:
            # For a while loop we don't have the ability to propagate much.
            # TODO: We may need to propagate dynamic unbounded, actually...
            return
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if itvar and start is not None and end is not None and stride is not None:
            loop_range = subsets.Range([(start, end, stride)])
            deps_ret = loop_analysis.get_loop_carry_dependencies(loop)
            loop_carry_dependencies: Dict[str, Dict[GlobalDepDataRecordT, GlobalDepDataRecordT]] = {}
            for loop_read in deps_ret.keys():
                if loop_read.memlet.data in loop_carry_dependencies:
                    loop_carry_dependencies[loop_read.memlet.data][loop_read] = deps_ret[loop_read]
                else:
                    loop_carry_dependencies[loop_read.memlet.data] = { loop_read: deps_ret[loop_read] }
            loop._carry_dependencies_moredata = loop_carry_dependencies
        else:
            loop_range = None
            loop_carry_dependencies = {}

        # Collect all symbols and variables (i.e., scalar data containers) defined at this point, particularly by
        # looking at defined loop variables in the parent chain up the control flow tree.
        variables_at_loop = OrderedDict(loop.sdfg.symbols)
        for k, v in loop.sdfg.arrays.items():
            if isinstance(v, dt.Scalar):
                variables_at_loop[k] = v
        pivot = loop
        while pivot is not None:
            if isinstance(pivot, LoopRegion):
                new_symbols = pivot.new_symbols(loop.sdfg.symbols)
                variables_at_loop.update(new_symbols)
            pivot = pivot.parent_graph
        defined_variables = [symbolic.pystr_to_symbolic(s) for s in variables_at_loop.keys()]
        # Propagate memlet subsets through the loop variable and its range.
        # TODO: Remove loop-carried dependencies from the writes (i.e., only the first read would be a true read)
        for repo, repo_moredata in [(loop.certain_reads, loop._certain_reads_moredata),
                                    (loop.possible_reads, loop._possible_reads_moredata)]:
            for dat in repo_moredata.keys():
                read_memlet = repo_moredata[dat].memlet
                desc = loop.sdfg.data(dat)
                if dat in loop_carry_dependencies and read_memlet in loop_carry_dependencies[dat]:
                    #dep_write = loop_carry_deps[read_memlet]
                    #diff = subsets.difference(read.subset, dep_write.subset)
                    #if isinstance(diff, subsets.SubsetUnion):
                    #    diff = diff.to_bounding_box_subset()
                    #tgt_expr = symbolic.pystr_to_symbolic(itvar) - loop_range.ranges[0][-1]
                    #for i in range(diff.dims()):
                    #    dim = diff.dim_to_string(i)
                    #    ...
                    ## Check if the remaining read subset is only in the direction opposing the loop iteration. If
                    #diff.__getitem__(0)
                    propagated_read_memlet = propagate_subset([read_memlet], desc, [itvar], loop_range,
                                                              defined_variables, use_dst=False)
                    repo[dat] = propagated_read_memlet
                    repo_moredata[dat].memlet = propagated_read_memlet
                else:
                    propagated_read_memlet = propagate_subset([read_memlet], desc, [itvar], loop_range,
                                                              defined_variables, use_dst=False)
                    repo[dat] = propagated_read_memlet
                    repo_moredata[dat].memlet = propagated_read_memlet
        for repo, repo_moredata in [(loop.certain_writes, loop._certain_writes_moredata),
                                    (loop.possible_writes, loop._possible_writes_moredata)]:
            for dat in repo_moredata.keys():
                write_memlet = repo_moredata[dat].memlet
                desc = loop.sdfg.data(dat)
                propagated_write_memlet = propagate_subset([write_memlet], desc, [itvar], loop_range, defined_variables,
                                                           use_dst=True)
                repo[dat] = propagated_write_memlet
                repo_moredata[dat].memlet = propagated_write_memlet

        result[loop] = (loop._certain_reads_moredata, loop._possible_reads_moredata,
                        loop._certain_writes_moredata, loop._possible_writes_moredata)

    def _propagate_cfg(self, cfg: ControlFlowRegion, result: MemletPropResultT) -> None:
        cfg.possible_reads = {}
        cfg.possible_writes = {}
        cfg.certain_reads = {}
        cfg.certain_writes = {}
        cfg._possible_reads_moredata = {}
        cfg._possible_writes_moredata = {}
        cfg._certain_reads_moredata = {}
        cfg._certain_writes_moredata = {}

        alldoms = cfg_analysis.all_dominators(cfg)
        allpostdom = cfg_analysis.all_post_dominators(cfg)

        # For each node in the CFG, check what reads are covered by exactly covering writes in dominating nodes. If such
        # a dominating write is found, the read is contained to read data originating from within the same CFG, and thus
        # is not counted as an input to the CFG.
        for nd in cfg.nodes():
            # For each node, also determine possible reads from interstate edges. For this, any read from any outgoing
            # interstate edge is counted as a possible read. The only time it is NOT counted as a read, is when there
            # is a certain write in the block itself tha covers the read.
            odeg = cfg.out_degree(nd)
            for oedge in cfg.out_edges(nd):
                for read_memlet in oedge.data.get_read_memlets(cfg.sdfg.arrays):
                    covered = False
                    if (read_memlet.data not in nd.certain_writes
                            or not nd.certain_writes[read_memlet.data].subset.covers_precise(read_memlet.subset)):
                        repos = [(nd.possible_reads, nd._possible_reads_moredata)]
                        if odeg == 1:
                            repos.append((nd.certain_reads, nd._certain_reads_moredata))
                        for repo, repo_moredata in repos:
                            if read_memlet.data in repo:
                                existing_memlet: Memlet = repo[read_memlet.data]
                                if isinstance(existing_memlet.subset, subsets.SubsetUnion):
                                    existing_memlet.subset.union(read_memlet.subset)
                                else:
                                    subset = subsets.SubsetUnion(read_memlet.subset)
                                    subset.union(existing_memlet.subset)
                                    existing_memlet.subset = subset
                            else:
                                repo[read_memlet.data] = read_memlet
                            if read_memlet.data in repo_moredata:
                                existing_entry = repo_moredata[read_memlet.data]
                                if isinstance(existing_entry.memlet.subset, subsets.SubsetUnion):
                                    existing_entry.memlet.subset.union(read_memlet.subset)
                                else:
                                    subset = subsets.SubsetUnion(read_memlet.subset)
                                    subset.union(existing_entry.memlet.subset)
                                    existing_entry.memlet.subset = subset
                                existing_entry.accesses.append(oedge)
                            else:
                                repo_moredata[read_memlet.data] = GlobalDepDataRecordT(memlet=read_memlet,
                                                                                       accesses=[oedge])
            for repo, repo_moredata, cfg_repo, cfg_repo_moredata in [
                (nd.possible_reads, nd._possible_reads_moredata, cfg.possible_reads, cfg._possible_reads_moredata),
                (nd.certain_reads, nd._certain_reads_moredata, cfg.certain_reads, cfg._certain_reads_moredata)
            ]:
                for read_data in repo:
                    read_memlet: Memlet = repo[read_data]
                    covered = False
                    for dom in alldoms[nd]:
                        if read_data in dom.certain_writes:
                            write_memlet: Memlet = dom.certain_writes[read_data]
                            if write_memlet.subset.covers_precise(read_memlet.subset):
                                covered = True
                                break
                    if not covered:
                        if read_data in cfg_repo:
                            existing_memlet: Memlet = cfg_repo[read_data]
                            if isinstance(existing_memlet.subset, subsets.SubsetUnion):
                                existing_memlet.subset.union(read_memlet.subset)
                            else:
                                subset = subsets.SubsetUnion(read_memlet.subset)
                                subset.union(existing_memlet.subset)
                                existing_memlet.subset = subset
                        else:
                            cfg_repo[read_data] = copy.deepcopy(read_memlet)
                        if read_data in cfg_repo_moredata:
                            existing_entry = cfg_repo_moredata[read_data]
                            if isinstance(existing_entry.memlet.subset, subsets.SubsetUnion):
                                existing_entry.memlet.subset.union(read_memlet.subset)
                            else:
                                subset = subsets.SubsetUnion(read_memlet.subset)
                                subset.union(existing_entry.memlet.subset)
                                existing_entry.memlet.subset = subset
                            existing_entry.accesses.extend(repo_moredata[read_data].accesses)
                        else:
                            cfg_repo_moredata[read_data] = copy.deepcopy(repo_moredata[read_data])

        # For each node in the CFG, check what writes are covered by exactly covering writes in a postdominating node.
        # If such a postdominating write is found, the write does not leave the CFG, and thus is not counted as a write
        # or output from the CFG.
        for nd in cfg.nodes():
            for cont in nd.possible_writes:
                covered = False
                for postdom in allpostdom[nd]:
                    if cont in postdom.certain_writes:
                        write_memlet: Memlet = postdom.certain_writes[cont]
                        if write_memlet.subset.covers_precise(nd.possible_writes[cont].subset):
                            covered = True
                            break
                if not covered:
                    if cont in cfg.possible_writes:
                        union = subsets.SubsetUnion(cfg.possible_writes[cont].subset)
                        union.union(nd.possible_writes[cont].subset)
                        cfg.possible_writes[cont] = Memlet(data=cont, subset=union)
                    else:
                        cfg.possible_writes[cont] = copy.deepcopy(nd.possible_writes[cont])
                    if cont in cfg._possible_writes_moredata:
                        existing_entry = cfg._possible_writes_moredata[cont]
                        if isinstance(existing_entry.memlet.subset, subsets.SubsetUnion):
                            existing_entry.memlet.subset.union(nd.possible_writes[cont].subset)
                        else:
                            subset = subsets.SubsetUnion(nd.possible_writes[cont].subset)
                            subset.union(existing_entry.memlet.subset)
                            existing_entry.memlet.subset = subset
                        existing_entry.accesses.extend(nd._possible_writes_moredata[cont].accesses)
                    else:
                        cfg._possible_writes_moredata[cont] = copy.deepcopy(nd._possible_writes_moredata[cont])
            for cont in nd.certain_writes:
                covered = False
                for postdom in allpostdom[nd]:
                    if cont in postdom.certain_writes:
                        write_memlet: Memlet = postdom.certain_writes[cont]
                        if write_memlet.subset.covers_precise(nd.certain_writes[cont].subset):
                            covered = True
                            break
                if not covered:
                    if cont in cfg.certain_writes:
                        union = subsets.SubsetUnion(cfg.certain_writes[cont].subset)
                        union.union(nd.certain_writes[cont].subset)
                        cfg.certain_writes[cont] = Memlet(data=cont, subset=union)
                    else:
                        cfg.certain_writes[cont] = copy.deepcopy(nd.certain_writes[cont])
                    if cont in cfg._certain_writes_moredata:
                        existing_entry = cfg._certain_writes_moredata[cont]
                        if isinstance(existing_entry.memlet.subset, subsets.SubsetUnion):
                            existing_entry.memlet.subset.union(nd.certain_writes[cont].subset)
                        else:
                            subset = subsets.SubsetUnion(nd.certain_writes[cont].subset)
                            subset.union(existing_entry.memlet.subset)
                            existing_entry.memlet.subset = subset
                        existing_entry.accesses.extend(nd._certain_writes_moredata[cont].accesses)
                    else:
                        cfg._certain_writes_moredata[cont] = copy.deepcopy(nd._certain_writes_moredata[cont])

        if isinstance(cfg, ControlFlowBlock):
            result[cfg] = (cfg._certain_reads_moredata, cfg._possible_reads_moredata,
                           cfg._certain_writes_moredata, cfg._possible_writes_moredata)

    def apply(self, region: ControlFlowRegion, _) -> Optional[MemletPropResultT]:
        result: MemletPropResultT = {}

        for nd in region.nodes():
            if isinstance(nd, SDFGState):
                self._propagate_state(nd, result)
        if isinstance(region, ConditionalBlock):
            self._propagate_conditional(region, result)
        elif isinstance(region, LoopRegion):
            self._propagate_loop(region, result)
        else:
            self._propagate_cfg(region, result)

        if result:
            return result
        else:
            return None
