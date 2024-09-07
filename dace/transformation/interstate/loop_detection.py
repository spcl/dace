# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop detection transformation """

import sympy as sp
import networkx as nx
from typing import AnyStr, Optional, Tuple, List, Set

from dace import sdfg as sd, symbolic
from dace.sdfg import graph as gr, utils as sdutil, InterstateEdge
from dace.sdfg.state import ControlFlowRegion, ControlFlowBlock
from dace.transformation import transformation


# NOTE: This class extends PatternTransformation directly in order to not show up in the matches
@transformation.experimental_cfg_block_compatible
class DetectLoop(transformation.PatternTransformation):
    """ Detects a for-loop construct from an SDFG. """

    # Always available
    loop_begin = transformation.PatternNode(sd.SDFGState)
    exit_state = transformation.PatternNode(sd.SDFGState)

    # Available for natural loops
    loop_guard = transformation.PatternNode(sd.SDFGState)

    # Available for rotated loops
    loop_latch = transformation.PatternNode(sd.SDFGState)

    # Available for rotated and self loops
    entry_state = transformation.PatternNode(sd.SDFGState)

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

        return [sdfg, msdfg, rsdfg, rmsdfg, ssdfg]

    def can_be_applied(self,
                       graph: ControlFlowRegion,
                       expr_index: int,
                       sdfg: sd.SDFG,
                       permissive: bool = False) -> bool:
        if expr_index == 0:
            return self.detect_loop(graph, False) is not None
        elif expr_index == 1:
            return self.detect_loop(graph, True) is not None
        elif expr_index == 2:
            return self.detect_rotated_loop(graph, False) is not None
        elif expr_index == 3:
            return self.detect_rotated_loop(graph, True) is not None
        elif expr_index == 4:
            return self.detect_self_loop(graph) is not None

        raise ValueError(f'Invalid expression index {expr_index}')

    def detect_loop(self, graph: ControlFlowRegion, multistate_loop: bool) -> Optional[str]:
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
        loop_nodes = sdutil.dfs_conditional(graph, sources=[begin], condition=lambda _, child: child != guard)
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
            # Either no consistent iteration variable found, or too many
            # consistent iteration variables found
            return None

        return next(iter(itvar))

    def detect_rotated_loop(self, graph: ControlFlowRegion, multistate_loop: bool) -> Optional[str]:
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
        begin = self.loop_begin

        # A for-loop start has at least two incoming edges (init and increment)
        begin_inedges = graph.in_edges(begin)
        if len(begin_inedges) < 2:
            return None
        # A for-loop latch only has two outgoing edges (loop condition and exit-loop)
        latch_outedges = graph.out_edges(latch)
        if len(latch_outedges) != 2:
            return None

        # All incoming edges to the start of the loop must set the same variable
        itvar = None
        for iedge in begin_inedges:
            if itvar is None:
                itvar = set(iedge.data.assignments.keys())
            else:
                itvar &= iedge.data.assignments.keys()
        if itvar is None:
            return None

        # Outgoing edges must be a negation of each other
        if latch_outedges[0].data.condition_sympy() != (sp.Not(latch_outedges[1].data.condition_sympy())):
            return None

        # All nodes inside loop must be dominated by loop start
        dominators = nx.dominance.immediate_dominators(graph.nx, graph.start_block)
        loop_nodes = list(sdutil.dfs_conditional(graph, sources=[begin], condition=lambda _, child: child != latch))
        loop_nodes += [latch]
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

        # The backedge must reassign the iteration variable
        itvar &= backedge.data.assignments.keys()
        if len(itvar) != 1:
            # Either no consistent iteration variable found, or too many
            # consistent iteration variables found
            return None

        return next(iter(itvar))

    def detect_self_loop(self, graph: ControlFlowRegion) -> Optional[str]:
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
            # Either no consistent iteration variable found, or too many
            # consistent iteration variables found
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
        elif self.expr_index in (2, 3):
            latch = self.loop_latch
            return find_rotated_for_loop(latch.parent_graph, latch, entry, itervar)
        elif self.expr_index == 4:
            return find_rotated_for_loop(entry.parent_graph, entry, entry, itervar)

        raise ValueError(f'Invalid expression index {self.expr_index}')

    def loop_body(self) -> List[ControlFlowBlock]:
        """
        Returns a list of all control flow blocks (or states) contained in the loop.
        """
        begin = self.loop_begin
        graph = begin.parent_graph
        if self.expr_index in (0, 1):
            guard = self.loop_guard
            return list(sdutil.dfs_conditional(graph, sources=[begin], condition=lambda _, child: child != guard))
        elif self.expr_index in (2, 3):
            latch = self.loop_latch
            loop_nodes = list(sdutil.dfs_conditional(graph, sources=[begin], condition=lambda _, child: child != latch))
            loop_nodes += [latch]
            return loop_nodes
        elif self.expr_index == 4:
            return [begin]

        return []

    def loop_meta_states(self) -> List[ControlFlowBlock]:
        """
        Returns the non-body control-flow blocks of this loop (e.g., guard, latch).
        """
        if self.expr_index in (0, 1):
            return [self.loop_guard]
        if self.expr_index in (2, 3):
            return [self.loop_latch]
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
        elif self.expr_index in (2, 3):
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
        elif self.expr_index == 4:
            begin = self.loop_begin
            return graph.edges_between(begin, exitstate)[0]

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
        elif self.expr_index in (2, 3):
            body = self.loop_body()
            return next(e for e in graph.in_edges(begin) if e.src in body)
        elif self.expr_index == 4:
            return graph.edges_between(begin, begin)[0]

        raise ValueError(f'Invalid expression index {self.expr_index}')


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

    if end is None:  # No match found
        return None

    return itervar, (start, end, stride), (start_states, last_loop_state)


def find_rotated_for_loop(
    graph: ControlFlowRegion,
    latch: sd.SDFGState,
    entry: sd.SDFGState,
    itervar: Optional[str] = None
) -> Optional[Tuple[AnyStr, Tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType], Tuple[
        List[sd.SDFGState], sd.SDFGState]]]:
    """
    Finds rotated loop range from state machine.
    
    :param latch: State from which the outgoing edges detect whether to exit
                  the loop or not.
    :param entry: First state in the loop body.
    :param itervar: An optional field that overrides the analyzed iteration variable.
    :return: (iteration variable, (start, end, stride),
             (start_states, last_loop_state)), or None if proper
             for-loop was not detected. ``end`` is inclusive.
    """
    # Extract state transition edge information
    entry_inedges = graph.in_edges(entry)
    condition_edge = graph.edges_between(latch, entry)[0]

    # All incoming edges to the loop entry must set the same variable
    if itervar is None:
        itervars = None
        for iedge in entry_inedges:
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

    # Find the stride edge. All in-edges to the entry except for the stride edge
    # should have exactly the same assignment, since a valid for loop can only
    # have one assignment.
    init_edges = []
    init_assignment = None
    step_edge = None
    itersym = symbolic.symbol(itervar)
    for iedge in entry_inedges:
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

    if end is None:  # No match found
        return None

    return itervar, (start, end, stride), (start_states, last_loop_state)
