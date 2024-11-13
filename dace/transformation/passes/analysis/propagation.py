# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from collections import OrderedDict, defaultdict, deque
import copy
from typing import Dict, Iterable, List, Set, Tuple, Union

import networkx as nx
import sympy

from dace import data as dt
from dace import properties, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.propagation import align_memlet, propagate_memlet, propagate_subset
from dace.sdfg.scope import ScopeTree
from dace.sdfg.sdfg import SDFG, memlets_in_ast
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.subsets import Range, Subset, SubsetUnion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.helpers import unsqueeze_memlet
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.analysis.analysis import ControlFlowBlockReachability


@transformation.experimental_cfg_block_compatible
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
    This pass supersedes `dace.sdfg.propagation.propagate_states` and is based on its algorithm, with significant
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
                        blk.ranges[str(region.loop_variable)] = Range([rng])

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
@transformation.experimental_cfg_block_compatible
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

    def _propagate_scope(self, state: SDFGState, scopes: List[ScopeTree], propagate_entry: bool = True,
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

        possible_reads = copy.deepcopy(sdfg._possible_reads)
        possible_writes = copy.deepcopy(sdfg._possible_writes)
        certain_reads = copy.deepcopy(sdfg._certain_reads)
        certain_writes = copy.deepcopy(sdfg._certain_writes)

        # Make sure any potential NSDFG symbol mapping is correctly reversed when propagating out.
        for mapping in [possible_reads, possible_writes, certain_reads, certain_writes]:
            for border_memlet in mapping.values():
                border_memlet.replace(nsdfg_node.symbol_mapping)

                # Also make sure that there's no symbol in the border memlet's range that only exists inside the
                # nested SDFG. If that's the case, use the entire range.
                if border_memlet.src_subset is not None:
                    if any(str(s) not in outer_symbols.keys() for s in border_memlet.src_subset.free_symbols):
                        border_memlet.src_subset = Range.from_array(sdfg.arrays[border_memlet.data])
                if border_memlet.dst_subset is not None:
                    if any(str(s) not in outer_symbols.keys() for s in border_memlet.dst_subset.free_symbols):
                        border_memlet.dst_subset = Range.from_array(sdfg.arrays[border_memlet.data])

        # Propagate the inside 'border' memlets outside the SDFG by offsetting, and unsqueezing if necessary.
        for iedge in parent_state.in_edges(nsdfg_node):
            if iedge.dst_conn in possible_reads:
                try:
                    inner_memlet = possible_reads[iedge.dst_conn]
                    iedge.data = unsqueeze_memlet(inner_memlet, iedge.data, True)
                    if isinstance(iedge.data.subset, SubsetUnion):
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
                    if isinstance(oedge.data.subset, SubsetUnion):
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

    def _propagate_state(self, state: SDFGState) -> None:
        # Ensure memlets around nested SDFGs are propagated correctly.
        for nd in state.nodes():
            if isinstance(nd, nodes.NestedSDFG):
                self._propagate_nsdfg(state.sdfg, state, nd)

        # Propagate memlets through the scopes, bottom up, starting at the scope leaves.
        # TODO: Make sure this propagation happens without overapproximation, i.e., using SubsetUnions.
        self._propagate_scope(state, state.scope_leaves())

        # Gather all writes and reads inside this state now to determine the state-wide reads and writes.
        # Collect write memlets.
        writes: Dict[str, List[Tuple[Memlet, nodes.AccessNode]]] = defaultdict(lambda: [])
        for anode in state.data_nodes():
            for iedge in state.in_edges(anode):
                if not iedge.data.is_empty():
                    root_edge = state.memlet_tree(iedge).root().edge
                    writes[anode.data].append([root_edge.data, anode])

        # Go over (overapproximated) reads and check if they are covered by writes.
        not_covered_reads: Dict[str, Set[Memlet]] = defaultdict(set)
        for anode in state.data_nodes():
            for oedge in state.out_edges(anode):
                if not oedge.data.is_empty():
                    if oedge.data.data != anode.data:
                        # Special case for memlets copying data out of the scope, which are by default aligned with the
                        # outside data container. In this case, the source container must either be a scalar, or the
                        # read subset is contained in the memlet's `other_subset` property.
                        # See `dace.sdfg.propagation.align_memlet` for more.
                        desc = state.sdfg.data(anode.data)
                        if oedge.data.other_subset is not None:
                            read_subset = oedge.data.other_subset
                        elif isinstance(desc, dt.Scalar) or (isinstance(desc, dt.Array) and desc.total_size == 1):
                            read_subset = Range([(0, 0, 1)] * len(desc.shape))
                        else:
                            raise RuntimeError('Invalid memlet range detected in MemletPropagation')
                    else:
                        read_subset = oedge.data.src_subset or oedge.data.subset
                    covered = False
                    for [write, to] in writes[anode.data]:
                        if write.subset.covers_precise(read_subset) and nx.has_path(state.nx, to, anode):
                            covered = True
                            break
                    if not covered:
                        not_covered_reads[anode.data].add(oedge.data)

        state._certain_writes = {}
        state._possible_writes = {}
        for data in writes:
            subset = None
            volume = None
            is_dynamic = False
            for memlet, _ in writes[data]:
                is_dynamic |= memlet.dynamic
                if subset is None:
                    subset = SubsetUnion(memlet.dst_subset or memlet.subset)
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
            state._certain_writes[data] = new_memlet
            state._possible_writes[data] = new_memlet

        state._certain_reads = {}
        state._possible_reads = {}
        for data in not_covered_reads:
            subset = None
            volume = None
            is_dynamic = False
            for memlet in not_covered_reads[data]:
                is_dynamic |= memlet.dynamic
                if subset is None:
                    subset = SubsetUnion(memlet.dst_subset or memlet.subset)
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
            state._certain_reads[data] = new_memlet
            state._possible_reads[data] = new_memlet

    def _propagate_conditional(self, conditional: ConditionalBlock) -> None:
        # The union of all reads between all conditions and branches gives the set of _possible_ reads, while the
        # intersection gives the set of _guaranteed_ reads. The first condition can also be counted as a guaranteed
        # read. The same applies for writes, except that conditions do not contain writes.

        def add_memlet(memlet: Memlet, mlt_dict: Dict[str, Memlet], use_intersection: bool = False):
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
                        propagated_memlet.subset = SubsetUnion([])

                    propagated_memlet.volume = sympy.Min(memlet.volume, propagated_memlet.volume)
                    propagated_memlet.dynamic |= memlet.dynamic
                else:
                    mlt_subset = propagated_memlet.subset
                    if not isinstance(mlt_subset, SubsetUnion):
                        mlt_subset = SubsetUnion([mlt_subset])
                    mlt_subset.union(memlet.src_subset or memlet.subset)
                    propagated_memlet.subset = mlt_subset

                    if propagated_memlet.volume != 0:
                        if memlet.volume == 0:
                            propagated_memlet.volume = memlet.volume
                        else:
                            propagated_memlet.volume = sympy.Max(memlet.volume, propagated_memlet.volume)
                    propagated_memlet.dynamic |= memlet.dynamic

        conditional._possible_reads = {}
        conditional._certain_reads = {}
        conditional._possible_writes = {}
        conditional._certain_writes = {}

        # Gather the union of possible reads and writes. At the same time, determine if there is an else branch present.
        has_else = False
        for cond, branch in conditional.branches:
            if cond is not None:
                read_memlets = memlets_in_ast(cond.code[0], conditional.sdfg.arrays)
                for read_memlet in read_memlets:
                    add_memlet(read_memlet, conditional._possible_reads)
            else:
                has_else = True
            for read_data in branch._possible_reads:
                read_memlet = branch._possible_reads[read_data]
                add_memlet(read_memlet, conditional._possible_reads)
            for write_data in branch._possible_writes:
                write_memlet = branch._possible_writes[write_data]
                add_memlet(write_memlet, conditional._possible_writes)

        # If there is no else branch or only one branch exists, there are no certain reads or writes.
        if len(conditional.branches) > 1 and has_else:
            # Gather the certain reads (= Intersection of certain reads for each branch)
            for container in conditional._possible_reads.keys():
                candidate_memlets = []
                skip = False
                for cond, branch in conditional.branches:
                    found = False
                    if cond is not None:
                        read_memlets = memlets_in_ast(cond.code[0], conditional.sdfg.arrays)
                        for read_memlet in read_memlets:
                            if read_memlet.data == container:
                                found = True
                                candidate_memlets.append(read_memlet)
                    if container in branch._certain_reads:
                        found = True
                        candidate_memlets.append(branch._certain_reads[container])
                    if not found:
                        skip = True
                        break
                if skip:
                    continue
                for cand_memlet in candidate_memlets:
                    add_memlet(cand_memlet, conditional._certain_reads, use_intersection=True)
            # Gather the certain writes (= Intersection of certain writes for each branch)
            for container in conditional._possible_writes.keys():
                candidate_memlets = []
                skip = False
                for _, branch in conditional.branches:
                    if container in branch._certain_writes:
                        candidate_memlets.append(branch._certain_writes[container])
                    else:
                        skip = True
                        break
                if skip:
                    continue
                for cand_memlet in candidate_memlets:
                    add_memlet(cand_memlet, conditional._certain_writes, use_intersection=True)

        # Ensure the first condition's reads are part of the certain reads.
        first_cond = conditional.branches[0][0]
        if first_cond is not None:
            read_memlets = memlets_in_ast(first_cond.code[0], conditional.sdfg.arrays)
            for read_memlet in read_memlets:
                add_memlet(read_memlet, conditional._certain_reads)

    def _propagate_loop(self, loop: LoopRegion) -> None:
        # First propagate the contents of the loop for one iteration.
        self._propagate_cfg(loop)

        # TODO: Remove loop-carried dependencies from the writes (i.e., only the first read would be a true read)

        # Propagate memlets from inside the loop through the loop ranges.
        # Collect loop information and form the loop variable range first.
        itvar = loop.loop_variable
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if itvar and start is not None and end is not None and stride is not None:
            loop_range = Range([(start, end, stride)])
        else:
            loop_range = None

        # Collect all symbols defined at this point, particularly by looking at defined loop variables in the parent
        # chain up the control flow tree.
        symbols_at_loop = OrderedDict(loop.sdfg.symbols)
        pivot = loop
        while pivot is not None:
            if isinstance(pivot, LoopRegion):
                new_symbols = pivot.new_symbols(loop.sdfg.symbols)
                symbols_at_loop.update(new_symbols)
            pivot = pivot.parent_graph
        defined_symbols = [symbolic.pystr_to_symbolic(s) for s in symbols_at_loop.keys()]
        repos_to_propagate = [(loop._certain_reads, False),
                            (loop._certain_writes, True),
                            (loop._possible_reads, False),
                            (loop._possible_writes, True)]
        # Propagate memlet subsets through the loop variable and its range.
        for (memlet_repo, use_dst) in repos_to_propagate:
            for dat in memlet_repo.keys():
                memlet = memlet_repo[dat]
                arr = loop.sdfg.data(dat)
                new_memlet = propagate_subset([memlet], arr, [itvar], loop_range, defined_symbols, use_dst)
                memlet_repo[dat] = new_memlet

    def _propagate_cfg(self, cfg: ControlFlowRegion) -> None:
        cfg._possible_reads = {}
        cfg._possible_writes = {}
        cfg._certain_reads = {}
        cfg._certain_writes = {}

        alldoms = cfg_analysis.all_dominators(cfg)

        # For each node in the CFG, check what reads are covered by exactly covering writes in dominating nodes. If such
        # a dominating write is found, the read is contained to read data originating from within the same CFG, and thus
        # is not counted as an input to the CFG.
        for nd in cfg.nodes():
            # For each node, also determine possible reads from interstate edges. For this, any read from any outgoing
            # interstate edge is counted as a possible read. The only time it is NOT counted as a read, is when there
            # is a certain write in the block itself tha covers the read.
            for oedge in cfg.out_edges(nd):
                for read_memlet in oedge.data.get_read_memlets(cfg.sdfg.arrays):
                    covered = False
                    if (read_memlet.data not in nd._certain_writes or
                        not nd._certain_writes[read_memlet.data].subset.covers_precise(read_memlet.subset)):
                        if read_memlet.data in nd._possible_reads:
                            existing_memlet = nd._possible_reads[read_memlet.data]
                            if isinstance(existing_memlet.subset, SubsetUnion):
                                existing_memlet.subset.union(read_memlet.subset)
                            else:
                                subset = SubsetUnion(read_memlet.subset)
                                subset.union(existing_memlet.subset)
                                existing_memlet.subset = subset
                        else:
                            nd._possible_reads[read_memlet.data] = read_memlet
            for read_data in nd._possible_reads:
                read_memlet = nd._possible_reads[read_data]
                covered = False
                for dom in alldoms[nd]:
                    if read_data in dom._certain_writes:
                        write_memlet = dom._certain_writes[read_data]
                        if write_memlet.subset.covers_precise(read_memlet.subset):
                            covered = True
                            break
                if not covered:
                    cfg._possible_reads[read_data] = copy.deepcopy(read_memlet)
                    cfg._certain_reads[read_data] = cfg._possible_reads[read_data]
        for nd in cfg.nodes():
            for cont in nd._possible_writes:
                if cont in cfg._possible_writes:
                    union = SubsetUnion(cfg._possible_writes[cont].subset)
                    union.union(nd._possible_writes[cont].subset)
                    cfg._possible_writes[cont] = Memlet(data=cont, subset=union)
                else:
                    cfg._possible_writes[cont] = copy.deepcopy(nd._possible_writes[cont])
            for cont in nd._certain_writes:
                if cont in cfg._certain_writes:
                    union = SubsetUnion(cfg._certain_writes[cont].subset)
                    union.union(nd._certain_writes[cont].subset)
                    cfg._certain_writes[cont] = Memlet(data=cont, subset=union)
                else:
                    cfg._certain_writes[cont] = copy.deepcopy(nd._certain_writes[cont])

    def apply(self, region: ControlFlowRegion, _) -> None:
        for nd in region.nodes():
            if isinstance(nd, SDFGState):
                self._propagate_state(nd)
        if isinstance(region, ConditionalBlock):
            self._propagate_conditional(region)
        elif isinstance(region, LoopRegion):
            self._propagate_loop(region)
        else:
            self._propagate_cfg(region)
