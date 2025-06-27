# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop detection transformation """

import sympy as sp
import networkx as nx
from typing import AnyStr, Generator, Iterable, Optional, Tuple, List, Set

from dace import sdfg as sd, symbolic
from dace.sdfg import graph as gr, utils as sdutil, InterstateEdge
from dace.sdfg.state import ControlFlowRegion, ControlFlowBlock
from dace.transformation import transformation


# NOTE: This class extends PatternTransformation directly in order to not show up in the matches
@transformation.explicit_cf_compatible
class DetectLoop(transformation.PatternTransformation):
    """ Detects a for-loop construct from an SDFG. """

    # Always available
    loop_begin = transformation.PatternNode(ControlFlowBlock)
    exit_state = transformation.PatternNode(ControlFlowBlock)

    # Available for natural loops
    loop_guard = transformation.PatternNode(ControlFlowBlock)

    # Available for rotated loops
    loop_latch = transformation.PatternNode(ControlFlowBlock)

    # Available for rotated and self loops
    entry_state = transformation.PatternNode(ControlFlowBlock)

    # Available for explicit-latch rotated loops
    loop_break = transformation.PatternNode(ControlFlowBlock)

    break_edges: Set[gr.Edge[InterstateEdge]] = set()
    continue_edges: Set[gr.Edge[InterstateEdge]] = set()

    @classmethod
    def expressions(cls):
        # Case 1: Loop with one state
        sdfg = gr.OrderedDiGraph()
        sdfg.add_nodes_from([cls.loop_guard, cls.loop_begin, cls.exit_state])
        sdfg.add_edge(cls.loop_guard, cls.loop_begin, sd.InterstateEdge())
        sdfg.add_edge(cls.loop_guard, cls.exit_state, sd.InterstateEdge())
        sdfg.add_edge(cls.loop_begin, cls.loop_guard, sd.InterstateEdge())

        # Case 2: Loop with multiple states (no back-edge from state)
        # The reason for the second case is that subgraph isomorphism requires accounting for every involved edge
        msdfg = gr.OrderedDiGraph()
        msdfg.add_nodes_from([cls.loop_guard, cls.loop_begin, cls.exit_state])
        msdfg.add_edge(cls.loop_guard, cls.loop_begin, sd.InterstateEdge())
        msdfg.add_edge(cls.loop_guard, cls.exit_state, sd.InterstateEdge())

        # Case 3: Rotated single-state loop
        # Here the loop latch (like guard) is the last state in the loop
        rsdfg = gr.OrderedDiGraph()
        rsdfg.add_nodes_from([cls.entry_state, cls.loop_latch, cls.loop_begin, cls.exit_state])
        rsdfg.add_edge(cls.entry_state, cls.loop_begin, sd.InterstateEdge())
        rsdfg.add_edge(cls.loop_begin, cls.loop_latch, sd.InterstateEdge())
        rsdfg.add_edge(cls.loop_latch, cls.loop_begin, sd.InterstateEdge())
        rsdfg.add_edge(cls.loop_latch, cls.exit_state, sd.InterstateEdge())

        # Case 4: Rotated multi-state loop
        # The reason for this case is also that subgraph isomorphism requires accounting for every involved edge
        rmsdfg = gr.OrderedDiGraph()
        rmsdfg.add_nodes_from([cls.entry_state, cls.loop_latch, cls.loop_begin, cls.exit_state])
        rmsdfg.add_edge(cls.entry_state, cls.loop_begin, sd.InterstateEdge())
        rmsdfg.add_edge(cls.loop_latch, cls.loop_begin, sd.InterstateEdge())
        rmsdfg.add_edge(cls.loop_latch, cls.exit_state, sd.InterstateEdge())

        # Case 5: Self-loop
        ssdfg = gr.OrderedDiGraph()
        ssdfg.add_nodes_from([cls.entry_state, cls.loop_begin, cls.exit_state])
        ssdfg.add_edge(cls.entry_state, cls.loop_begin, sd.InterstateEdge())
        ssdfg.add_edge(cls.loop_begin, cls.loop_begin, sd.InterstateEdge())
        ssdfg.add_edge(cls.loop_begin, cls.exit_state, sd.InterstateEdge())

        # Case 6: Rotated multi-state loop with explicit exiting and latch states
        mlrmsdfg = gr.OrderedDiGraph()
        mlrmsdfg.add_nodes_from([cls.entry_state, cls.loop_break, cls.loop_latch, cls.loop_begin, cls.exit_state])
        mlrmsdfg.add_edge(cls.entry_state, cls.loop_begin, sd.InterstateEdge())
        mlrmsdfg.add_edge(cls.loop_latch, cls.loop_begin, sd.InterstateEdge())
        mlrmsdfg.add_edge(cls.loop_break, cls.exit_state, sd.InterstateEdge())
        mlrmsdfg.add_edge(cls.loop_break, cls.loop_latch, sd.InterstateEdge())

        # Case 7: Rotated single-state loop with explicit exiting and latch states
        mlrsdfg = gr.OrderedDiGraph()
        mlrsdfg.add_nodes_from([cls.entry_state, cls.loop_latch, cls.loop_begin, cls.exit_state])
        mlrsdfg.add_edge(cls.entry_state, cls.loop_begin, sd.InterstateEdge())
        mlrsdfg.add_edge(cls.loop_latch, cls.loop_begin, sd.InterstateEdge())
        mlrsdfg.add_edge(cls.loop_begin, cls.exit_state, sd.InterstateEdge())
        mlrsdfg.add_edge(cls.loop_begin, cls.loop_latch, sd.InterstateEdge())

        # Case 8: Guarded rotated multi-state loop with explicit exiting and latch states (modification of case 6)
        gmlrmsdfg = gr.OrderedDiGraph()
        gmlrmsdfg.add_nodes_from([cls.entry_state, cls.loop_break, cls.loop_latch, cls.loop_begin, cls.exit_state])
        gmlrmsdfg.add_edge(cls.entry_state, cls.loop_begin, sd.InterstateEdge())
        gmlrmsdfg.add_edge(cls.loop_latch, cls.loop_begin, sd.InterstateEdge())
        gmlrmsdfg.add_edge(cls.loop_begin, cls.loop_break, sd.InterstateEdge())
        gmlrmsdfg.add_edge(cls.loop_break, cls.exit_state, sd.InterstateEdge())
        gmlrmsdfg.add_edge(cls.loop_break, cls.loop_latch, sd.InterstateEdge())

        return [sdfg, msdfg, rsdfg, rmsdfg, ssdfg, mlrmsdfg, mlrsdfg, gmlrmsdfg]

    @property
    def inverted(self) -> bool:
        """
        Whether the loop matched a pattern of an inverted (do-while style) loop.
        """
        return self.expr_index in (2, 3, 5, 6, 7)

    @property
    def first_loop_block(self) -> ControlFlowBlock:
        """
        The first control flow block executed in each loop iteration.
        """
        return self.loop_guard if self.expr_index <= 1 else self.loop_begin

    def can_be_applied(self,
                       graph: ControlFlowRegion,
                       expr_index: int,
                       sdfg: sd.SDFG,
                       permissive: bool = False) -> bool:
        if expr_index == 0:
            return self.detect_loop(graph, multistate_loop=False, accept_missing_itvar=permissive) is not None
        elif expr_index == 1:
            return self.detect_loop(graph, multistate_loop=True, accept_missing_itvar=permissive) is not None
        elif expr_index == 2:
            return self.detect_rotated_loop(graph, multistate_loop=False, accept_missing_itvar=permissive) is not None
        elif expr_index == 3:
            return self.detect_rotated_loop(graph, multistate_loop=True, accept_missing_itvar=permissive) is not None
        elif expr_index == 4:
            return self.detect_self_loop(graph, accept_missing_itvar=permissive) is not None
        elif expr_index in (5, 7):
            return self.detect_rotated_loop(graph,
                                            multistate_loop=True,
                                            accept_missing_itvar=permissive,
                                            separate_latch=True) is not None
        elif expr_index == 6:
            return self.detect_rotated_loop(graph,
                                            multistate_loop=False,
                                            accept_missing_itvar=permissive,
                                            separate_latch=True) is not None

        raise ValueError(f'Invalid expression index {expr_index}')

    def detect_loop(self,
                    graph: ControlFlowRegion,
                    multistate_loop: bool,
                    accept_missing_itvar: bool = False) -> Optional[str]:
        """
        Detects a loop of the form:

        .. code-block:: text

                       ----------------
                       |              v
            entry -> guard -> body    exit
                       ^        |
                       ----------


        :param graph: The graph to look for the loop.
        :param multistate_loop: Whether the loop contains multiple states.
        :return: The loop variable or ``None`` if not detected.
        """
        guard = self.loop_guard
        begin = self.loop_begin

        # A for-loop guard only has two incoming edges (init and increment)
        guard_inedges = graph.in_edges(guard)
        if len(guard_inedges) < 2:
            return None
        # A for-loop guard only has two outgoing edges (loop and exit-loop)
        guard_outedges = graph.out_edges(guard)
        if len(guard_outedges) != 2:
            return None

        # All incoming edges to the guard must set the same variable
        itvar: Optional[Set[str]] = None
        for iedge in guard_inedges:
            if itvar is None:
                itvar = set(iedge.data.assignments.keys())
            else:
                itvar &= iedge.data.assignments.keys()
        if itvar is None:
            return None

        # Outgoing edges must be a negation of each other
        if guard_outedges[0].data.condition_sympy() != (sp.Not(guard_outedges[1].data.condition_sympy())):
            return None

        # All nodes inside loop must be dominated by loop guard
        dominators = nx.dominance.immediate_dominators(graph.nx, graph.start_block)
        postdominators = sdutil.postdominators(graph, True)
        loop_nodes = self.loop_body()
        # If the exit state is in the loop nodes, this is not a valid loop
        if self.exit_state in loop_nodes:
            return None
        elif any(self.exit_state not in postdominators[1][n] for n in loop_nodes):
            # The loop exit must post-dominate all loop nodes
            return None
        backedge = None
        for node in loop_nodes:
            for e in graph.out_edges(node):
                if e.dst == guard:
                    backedge = e
                    break

            # Traverse the dominator tree upwards, if we reached the guard,
            # the node is in the loop. If we reach the starting state
            # without passing through the guard, fail.
            dom = node
            while dom != dominators[dom]:
                if dom == guard:
                    break
                dom = dominators[dom]
            else:
                return None

        if backedge is None:
            return None

        # The backedge must reassign the iteration variable
        itvar &= backedge.data.assignments.keys()
        if len(itvar) != 1:
            if not accept_missing_itvar:
                # Either no consistent iteration variable found, or too many consistent iteration variables found
                return None
            else:
                if len(itvar) == 0:
                    return ''
                else:
                    return None

        return next(iter(itvar))

    def detect_rotated_loop(self,
                            graph: ControlFlowRegion,
                            multistate_loop: bool,
                            accept_missing_itvar: bool = False,
                            separate_latch: bool = False) -> Optional[str]:
        """
        Detects a loop of the form:

        .. code-block:: text

            entry -> body -> latch -> exit
                       ^        |
                       ----------


        :param graph: The graph to look for the loop.
        :param multistate_loop: Whether the loop contains multiple states.
        :return: The loop variable or ``None`` if not detected.
        """
        latch = self.loop_latch
        ltest = self.loop_latch
        if separate_latch:
            ltest = self.loop_break if multistate_loop else self.loop_begin
        begin = self.loop_begin

        # A for-loop start has at least two incoming edges (init and increment)
        begin_inedges = graph.in_edges(begin)
        if len(begin_inedges) < 2:
            return None
        # A for-loop latch only has two outgoing edges (loop condition and exit-loop)
        latch_outedges = graph.out_edges(ltest)
        if len(latch_outedges) != 2:
            return None

        # A for-loop latch can further only have one incoming edge (the increment edge). A while-loop, i.e., a loop
        # with no explicit iteration variable, may have more than that.
        latch_inedges = graph.in_edges(latch)
        if not accept_missing_itvar and len(latch_inedges) != 1:
            return None

        # Outgoing edges must be a negation of each other
        if latch_outedges[0].data.condition_sympy() != (sp.Not(latch_outedges[1].data.condition_sympy())):
            return None

        # Make sure the backedge (i.e, one of the condition edges) goes from the latch to the beginning state.
        if latch_outedges[0].dst is not self.loop_begin and latch_outedges[1].dst is not self.loop_begin:
            return None

        # All nodes inside loop must be dominated by loop start
        dominators = nx.dominance.immediate_dominators(graph.nx, graph.start_block)
        if begin is ltest:
            loop_nodes = [begin]
        else:
            loop_nodes = self.loop_body()
        loop_nodes.append(latch)
        if ltest is not latch and ltest is not begin:
            loop_nodes.append(ltest)
        postdominators = sdutil.postdominators(graph, True)
        if any(self.exit_state not in postdominators[1][n] for n in loop_nodes):
            # The loop exit must post-dominate all loop nodes
            return None
        backedge = None
        for node in loop_nodes:
            for e in graph.out_edges(node):
                if e.dst == begin:
                    backedge = e
                    break

            # Traverse the dominator tree upwards, if we reached the beginning,
            # the node is in the loop. If we reach any node in the loop
            # without passing through the loop start, fail.
            dom = node
            while dom != dominators[dom]:
                if dom == begin:
                    break
                dom = dominators[dom]
            else:
                return None

        if backedge is None:
            return None

        return rotated_loop_find_itvar(begin_inedges, latch_inedges, backedge, ltest, accept_missing_itvar)[0]

    def detect_self_loop(self, graph: ControlFlowRegion, accept_missing_itvar: bool = False) -> Optional[str]:
        """
        Detects a loop of the form:

        .. code-block:: text

            entry -> body state -> exit
                       ^    |
                       ------


        :param graph: The graph to look for the loop.
        :return: The loop variable or ``None`` if not detected.
        """
        body = self.loop_begin

        # A self-loop body must have only two incoming edges (initialize, increment)
        body_inedges = graph.in_edges(body)
        if len(body_inedges) != 2:
            return None
        # A self-loop body must have only two outgoing edges (condition success + increment, condition fail)
        body_outedges = graph.out_edges(body)
        if len(body_outedges) != 2:
            return None

        # All incoming edges to the body must set the same variable
        itvar = None
        for iedge in body_inedges:
            if itvar is None:
                itvar = set(iedge.data.assignments.keys())
            else:
                itvar &= iedge.data.assignments.keys()
        if itvar is None:
            return None

        # Outgoing edges must be a negation of each other
        if body_outedges[0].data.condition_sympy() != (sp.Not(body_outedges[1].data.condition_sympy())):
            return None

        # Backedge is the self-edge
        edges = graph.edges_between(body, body)
        if len(edges) != 1:
            return None
        backedge = edges[0]

        # The backedge must reassign the iteration variable
        itvar &= backedge.data.assignments.keys()
        if len(itvar) != 1:
            if not accept_missing_itvar:
                # Either no consistent iteration variable found, or too many consistent iteration variables found
                return None
            else:
                if len(itvar) == 0:
                    return ''
                else:
                    return None

        return next(iter(itvar))

    def apply(self, _, sdfg):
        pass

    ############################################
    # Functionality that provides loop metadata

    def loop_information(
        self,
        itervar: Optional[str] = None
    ) -> Optional[Tuple[AnyStr, Tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType], Tuple[
            List[sd.SDFGState], sd.SDFGState]]]:

        entry = self.loop_begin
        if self.expr_index <= 1:
            guard = self.loop_guard
            return find_for_loop(guard.parent_graph, guard, entry, itervar)
        elif self.expr_index in (2, 3, 5, 6, 7):
            latch = self.loop_latch
            return find_rotated_for_loop(latch.parent_graph,
                                         latch,
                                         entry,
                                         itervar,
                                         separate_latch=(self.expr_index in (5, 6, 7)))
        elif self.expr_index == 4:
            return find_rotated_for_loop(entry.parent_graph, entry, entry, itervar)

        raise ValueError(f'Invalid expression index {self.expr_index}')

    def _loop_body_dfs(self, terminator: ControlFlowBlock) -> Iterable[ControlFlowBlock]:
        self.break_edges.clear()
        visited = set()
        start = self.loop_begin
        graph = start.parent_graph
        exit_state = self.exit_state
        yield start
        visited.add(start)
        stack = [(start, iter(graph.successors(start)))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    if child == exit_state:
                        # If the exit state is reachable from the loop body, that counts as a break edge.
                        for e in graph.edges_between(parent, child):
                            self.break_edges.add(e)
                    elif child != terminator:
                        try:
                            yield child
                            stack.append((child, iter(graph.successors(child))))
                        except sdutil.StopTraversal:
                            pass
                    else:
                        # If we reached the terminator, we do not traverse further. All edges reaching the terminator
                        # are marked as continue edges. If there is only one continue edge int the end, it can be
                        # discarded (not actually a continue, simply the edge closing the loop).
                        for e in graph.edges_between(parent, child):
                            self.continue_edges.add(e)
            except StopIteration:
                stack.pop()

    def loop_body(self) -> List[ControlFlowBlock]:
        """
        Returns a list of all control flow blocks (or states) contained in the loop.
        """
        if self.expr_index in (0, 1):
            guard = self.loop_guard
            return list(self._loop_body_dfs(guard))
        elif self.expr_index in (2, 3):
            latch = self.loop_latch
            loop_nodes = list(self._loop_body_dfs(latch))
            loop_nodes += [latch]
            return loop_nodes
        elif self.expr_index == 4:
            return [self.loop_begin]
        elif self.expr_index in (5, 7):
            ltest = self.loop_break
            latch = self.loop_latch
            loop_nodes = list(self._loop_body_dfs(ltest))
            loop_nodes += [ltest, latch]
            return loop_nodes
        elif self.expr_index == 6:
            return [self.loop_begin, self.loop_latch]

        return []

    def loop_meta_states(self) -> List[ControlFlowBlock]:
        """
        Returns the non-body control-flow blocks of this loop (e.g., guard, latch).
        """
        if self.expr_index in (0, 1):
            return [self.loop_guard]
        if self.expr_index in (2, 3, 6):
            return [self.loop_latch]
        if self.expr_index in (5, 7):
            return [self.loop_break, self.loop_latch]
        return []

    def loop_init_edge(self) -> gr.Edge[InterstateEdge]:
        """
        Returns the initialization edge of the loop (assignment to the beginning of the loop range).
        """
        begin = self.loop_begin
        graph = begin.parent_graph
        if self.expr_index in (0, 1):
            guard = self.loop_guard
            body = self.loop_body()
            return next(e for e in graph.in_edges(guard) if e.src not in body)
        elif self.expr_index in (2, 3, 5, 6, 7):
            latch = self.loop_latch
            return next(e for e in graph.in_edges(begin) if e.src is not latch)
        elif self.expr_index == 4:
            return next(e for e in graph.in_edges(begin) if e.src is not begin)

        raise ValueError(f'Invalid expression index {self.expr_index}')

    def loop_exit_edge(self) -> gr.Edge[InterstateEdge]:
        """
        Returns the negative condition edge that exits the loop.
        """
        exitstate = self.exit_state
        graph = exitstate.parent_graph
        if self.expr_index in (0, 1):
            guard = self.loop_guard
            return graph.edges_between(guard, exitstate)[0]
        elif self.expr_index in (2, 3):
            latch = self.loop_latch
            return graph.edges_between(latch, exitstate)[0]
        elif self.expr_index in (4, 6):
            begin = self.loop_begin
            return graph.edges_between(begin, exitstate)[0]
        elif self.expr_index in (5, 7):
            ltest = self.loop_break
            return graph.edges_between(ltest, exitstate)[0]

        raise ValueError(f'Invalid expression index {self.expr_index}')

    def loop_condition_edge(self) -> gr.Edge[InterstateEdge]:
        """
        Returns the positive condition edge that (re-)enters the loop after the bound check.
        """
        begin = self.loop_begin
        graph = begin.parent_graph
        if self.expr_index in (0, 1):
            guard = self.loop_guard
            return graph.edges_between(guard, begin)[0]
        elif self.expr_index in (2, 3):
            latch = self.loop_latch
            return graph.edges_between(latch, begin)[0]
        elif self.expr_index == 4:
            begin = self.loop_begin
            return graph.edges_between(begin, begin)[0]
        elif self.expr_index in (5, 6, 7):
            latch = self.loop_latch
            ltest = self.loop_break if self.expr_index in (5, 7) else self.loop_begin
            return graph.edges_between(ltest, latch)[0]

        raise ValueError(f'Invalid expression index {self.expr_index}')

    def loop_increment_edge(self) -> gr.Edge[InterstateEdge]:
        """
        Returns the back-edge that increments the loop induction variable.
        """
        begin = self.loop_begin
        graph = begin.parent_graph
        if self.expr_index in (0, 1):
            guard = self.loop_guard
            body = self.loop_body()
            return next(e for e in graph.in_edges(guard) if e.src in body)
        elif self.expr_index in (2, 3, 5, 6, 7):
            _, step_edge = rotated_loop_find_itvar(graph.in_edges(begin), graph.in_edges(self.loop_latch),
                                                   graph.edges_between(self.loop_latch, begin)[0], self.loop_latch)
            return step_edge
        elif self.expr_index == 4:
            return graph.edges_between(begin, begin)[0]

        raise ValueError(f'Invalid expression index {self.expr_index}')


def rotated_loop_find_itvar(
        begin_inedges: List[gr.Edge[InterstateEdge]],
        latch_inedges: List[gr.Edge[InterstateEdge]],
        backedge: gr.Edge[InterstateEdge],
        latch: ControlFlowBlock,
        accept_missing_itvar: bool = False) -> Tuple[Optional[str], Optional[gr.Edge[InterstateEdge]]]:
    # The iteration variable must be assigned (initialized) on all edges leading into the beginning block, which
    # are not the backedge. Gather all variabes for which that holds - they are all candidates for the iteration
    # variable (Phase 1). Said iteration variable must then be incremented:
    # EITHER: On the backedge, in which case the increment is only executed if the loop does not exit. This
    #         corresponds to a while(true) loop that checks the condition at the end of the loop body and breaks
    #         if it does not hold before incrementing. (Scenario 1)
    # OR:     On the edge(s) leading into the latch, in which case the increment is executed BEFORE the condition is
    #         checked - which corresponds to a do-while loop. (Scenario 2)
    # For either case, the iteration variable may only be incremented on one of these places. Filter the candidates
    # down to each variable for which this condition holds (Phase 2). If there is exactly one candidate remaining,
    # that is the iteration variable. Otherwise it cannot be determined.

    # Phase 1: Gather iteration variable candidates.
    itvar_candidates = None
    for e in begin_inedges:
        if e is backedge:
            continue
        if itvar_candidates is None:
            itvar_candidates = set(e.data.assignments.keys())
        else:
            itvar_candidates &= set(e.data.assignments.keys())

    # Phase 2: Filter down the candidates according to incrementation edges.
    step_edge = None
    filtered_candidates = set()
    backedge_incremented = set(backedge.data.assignments.keys())
    latch_incremented = None
    if backedge.src is not backedge.dst:
        # If this is a self loop, there are no edges going into the latch to be considered. The only incoming edges are
        # from outside the loop.
        for e in latch_inedges:
            if e is backedge:
                continue
            if latch_incremented is None:
                latch_incremented = set(e.data.assignments.keys())
            else:
                latch_incremented &= set(e.data.assignments.keys())
    if latch_incremented is None:
        latch_incremented = set()
    for cand in itvar_candidates:
        if cand in backedge_incremented:
            # Scenario 1.

            # Note, only allow this scenario if the backedge leads directly from the latch to the entry, i.e., there is
            # no intermediate block on the backedge path.
            if backedge.src is not latch:
                continue

            if cand not in latch_incremented:
                filtered_candidates.add(cand)
        elif cand in latch_incremented:
            # Scenario 2.
            if cand not in backedge_incremented:
                filtered_candidates.add(cand)
    if len(filtered_candidates) != 1:
        if not accept_missing_itvar:
            # Either no consistent iteration variable found, or too many consistent iteration variables found
            return None, None
        else:
            if len(filtered_candidates) == 0:
                return '', None
            else:
                return None, None
    else:
        itvar = next(iter(filtered_candidates))
        if itvar in backedge_incremented:
            step_edge = backedge
        elif len(latch_inedges) == 1:
            step_edge = latch_inedges[0]
        return itvar, step_edge


def find_for_loop(
    graph: ControlFlowRegion,
    guard: sd.SDFGState,
    entry: sd.SDFGState,
    itervar: Optional[str] = None
) -> Optional[Tuple[AnyStr, Tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType], Tuple[
        List[sd.SDFGState], sd.SDFGState]]]:
    """
    Finds loop range from state machine.

    :param guard: State from which the outgoing edges detect whether to exit
                  the loop or not.
    :param entry: First state in the loop body.
    :param itervar: An optional field that overrides the analyzed iteration variable.
    :return: (iteration variable, (start, end, stride),
             (start_states, last_loop_state)), or None if proper
             for-loop was not detected. ``end`` is inclusive.
    """

    # Extract state transition edge information
    guard_inedges = graph.in_edges(guard)
    condition_edge = graph.edges_between(guard, entry)[0]

    # All incoming edges to the guard must set the same variable
    if itervar is None:
        itervars = None
        for iedge in guard_inedges:
            if itervars is None:
                itervars = set(iedge.data.assignments.keys())
            else:
                itervars &= iedge.data.assignments.keys()
        if itervars and len(itervars) == 1:
            itervar = next(iter(itervars))
        else:
            # Ambiguous or no iteration variable
            return None

    condition = condition_edge.data.condition_sympy()

    # Find the stride edge. All in-edges to the guard except for the stride edge
    # should have exactly the same assignment, since a valid for loop can only
    # have one assignment.
    init_edges = []
    init_assignment = None
    step_edge = None
    itersym = symbolic.symbol(itervar)
    for iedge in guard_inedges:
        assignment = iedge.data.assignments[itervar]
        if itersym in symbolic.pystr_to_symbolic(assignment).free_symbols:
            if step_edge is None:
                step_edge = iedge
            else:
                # More than one edge with the iteration variable as a free
                # symbol, which is not legal. Invalid for loop.
                return None
        else:
            if init_assignment is None:
                init_assignment = assignment
                init_edges.append(iedge)
            elif init_assignment != assignment:
                # More than one init assignment variations mean that this for
                # loop is not valid.
                return None
            else:
                init_edges.append(iedge)
    if step_edge is None or len(init_edges) == 0 or init_assignment is None:
        # Less than two assignment variations, can't be a valid for loop.
        return None

    # Get the init expression and the stride.
    start = symbolic.pystr_to_symbolic(init_assignment)
    stride = (symbolic.pystr_to_symbolic(step_edge.data.assignments[itervar]) - itersym)

    # Get a list of the last states before the loop and a reference to the last
    # loop state.
    start_states = []
    for init_edge in init_edges:
        start_state = init_edge.src
        if start_state not in start_states:
            start_states.append(start_state)
    last_loop_state = step_edge.src

    # Find condition by matching expressions
    end: Optional[symbolic.SymbolicType] = None
    a = sp.Wild('a')
    match = condition.match(itersym < a)
    if match:
        end = match[a] - 1
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            end = match[a] + 1
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(sp.Ne(itersym + stride, a))
        if match:
            end = match[a] - stride

    if end is None:  # No match found
        return None

    return itervar, (start, end, stride), (start_states, last_loop_state)


def find_rotated_for_loop(
    graph: ControlFlowRegion,
    latch: sd.SDFGState,
    entry: sd.SDFGState,
    itervar: Optional[str] = None,
    separate_latch: bool = False,
) -> Optional[Tuple[AnyStr, Tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType], Tuple[
        List[sd.SDFGState], sd.SDFGState]]]:
    """
    Finds rotated loop range from state machine.

    :param latch: State from which the outgoing edges detect whether to reenter the loop or not.
    :param entry: First state in the loop body.
    :param itervar: An optional field that overrides the analyzed iteration variable.
    :return: (iteration variable, (start, end, stride),
             (start_states, last_loop_state)), or None if proper
             for-loop was not detected. ``end`` is inclusive.
    """
    # Extract state transition edge information
    entry_inedges = graph.in_edges(entry)
    if separate_latch:
        condition_edge = graph.in_edges(latch)[0]
        backedge = graph.edges_between(latch, entry)[0]
    else:
        condition_edge = graph.edges_between(latch, entry)[0]
        backedge = condition_edge
    latch_inedges = graph.in_edges(latch)

    self_loop = latch is entry
    step_edge = None
    if itervar is None:
        itervar, step_edge = rotated_loop_find_itvar(entry_inedges, latch_inedges, backedge, latch)
        if itervar is None:
            return None

    condition = condition_edge.data.condition_sympy()

    # Find the stride edge. All in-edges to the entry except for the stride edge
    # should have exactly the same assignment, since a valid for loop can only
    # have one assignment.
    init_edges = []
    init_assignment = None
    itersym = symbolic.symbol(itervar)
    for iedge in entry_inedges:
        if iedge is condition_edge:
            continue
        assignment = iedge.data.assignments[itervar]
        if itersym not in symbolic.pystr_to_symbolic(assignment).free_symbols:
            if init_assignment is None:
                init_assignment = assignment
                init_edges.append(iedge)
            elif init_assignment != assignment:
                # More than one init assignment variations mean that this for
                # loop is not valid.
                return None
            else:
                init_edges.append(iedge)
    if len(init_edges) == 0 or init_assignment is None:
        # Less than two assignment variations, can't be a valid for loop.
        return None

    if self_loop:
        step_edge = condition_edge
    else:
        if step_edge is None:
            return None

    # Get the init expression and the stride.
    start = symbolic.pystr_to_symbolic(init_assignment)
    stride = (symbolic.pystr_to_symbolic(step_edge.data.assignments[itervar]) - itersym)

    # Get a list of the last states before the loop and a reference to the last
    # loop state.
    start_states = []
    for init_edge in init_edges:
        start_state = init_edge.src
        if start_state not in start_states:
            start_states.append(start_state)
    last_loop_state = step_edge.src

    # Find condition by matching expressions
    end: Optional[symbolic.SymbolicType] = None
    a = sp.Wild('a')
    match = condition.match(itersym < a)
    if match:
        end = match[a] - 1
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            end = match[a] + 1
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            end = match[a]
    if end is None:
        match = condition.match(sp.Ne(itersym + stride, a))
        if match:
            end = match[a] - stride

    if end is None:  # No match found
        return None

    return itervar, (start, end, stride), (start_states, last_loop_state)
