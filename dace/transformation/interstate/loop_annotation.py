# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop annotation transformation """

import sympy as sp
import networkx as nx
from sympy.sets.sets import Interval
from sympy.sets.setexpr import SetExpr
from dace.transformation.interstate.loop_detection import DetectLoop
from dace import sdfg as sd, symbolic, registry
from dace.sdfg import graph as gr, utils as sdutil
from typing import List, Tuple, Dict, Optional

class AnnotateLoop(DetectLoop):
    """ Annotates states in loop constructs according to the loop range. """


    @staticmethod
    def _get_loop_nodes(
        guard_candidate: sd.SDFGState,
        last_loop_edge: sd.InterstateEdge,
        idom: Dict[sd.SDFGState, sd.SDFGState]
    ) -> Optional[List[sd.SDFGState]]:
        """
        Finds all states inside a loop based on its guard and exiting edge.

        This traverses the immediate dominators of the last loop state until
        either a dead end is reached (not a loop, return None), or the loop
        guard is reached, in which case it is a loop and all states on the path
        taken are returned.

        :param guard_candidate: State which is suspected to be a loop guard.
        :param last_loop_edge: Last edge in the loop, pointing back to the guard.
        :param idom: Dictionary of immediate dominators for each state.
        :return: A list of all nodes inside the loop, or None if no proper loop
                 was detected.
        """
        pivot = last_loop_edge.src
        state_list = []
        while True:
            state_list.append(pivot)
            dominator = idom[pivot]
            if dominator is None or dominator == pivot:
                # We reached a tail, this is not a loop.
                return None
            elif dominator == guard_candidate:
                # We looped back to the loop guard candidate, this is a loop.
                return state_list
            else:
                pivot = dominator


    @staticmethod
    def _annotate(sdfg: sd.SDFG) -> None:
        """
        Annotate the states of an SDFG with the number of executions.

        :param sdfg: The SDFG to annotate.
        """
        for e in list(sdfg.edges()):
            if e.data.assignments and not e.data.is_unconditional():
                tmpstate = sdfg.add_state()
                sdfg.add_edge(
                    e.src,
                    tmpstate,
                    sd.InterstateEdge(condition=e.data.condition)
                )
                sdfg.add_edge(
                    tmpstate,
                    e.dst,
                    sd.InterstateEdge(assignments=e.data.assignments)
                )
                sdfg.remove_edge(e)
        for v in sdfg.nodes():
            v.ranges = {}

        state = sdfg.start_state
        state.executions = 1
        state.dynamic_executions = False

        out_degree = sdfg.out_degree(state)
        out_edges = sdfg.out_edges(state)
        idom = nx.immediate_dominators(sdfg.nx, state)
        if out_degree == 1:
            return AnnotateLoop._traverse_annotate(
                sdfg=sdfg,
                state=out_edges[0].dst,
                iedge=out_edges[0],
                proposed_executions=1,
                proposed_dynamic=False,
                idom=idom,
                visited_states=[state]
            )
        elif out_degree > 1:
            # XXX: Can this happen? If yes, conditional split here.
            pass
        return


    @staticmethod
    def _traverse_annotate(
        sdfg: sd.SDFG,
        state: sd.SDFGState,
        iedge: sd.InterstateEdge,
        proposed_executions: sp.Expr,
        proposed_dynamic: bool,
        idom: Dict[sd.SDFGState, sd.SDFGState],
        visited_states: List[sd.SDFGState],
        dominating_loop_guard: Optional[sd.SDFGState]=None
    ) -> None:
        """
        Recursively traverse the state machine and annotate each state.

        :param sdfg: The SDFG that's being annotated.
        :param state: State currently being annotated, pivot in the traversal.
        :param iedge: State over which this state was reached.
        :param proposed_executions: Number of times this state was determined to
                                    be executed.
        :param proposed_dynamic: Whether this state is proposed to have a
                                 dynamic number of executions.
        :param idom: Dictionary of immediate dominators for each state.
        :param visited_states: States that have already been traversed.
        :param dominating_loop_guard: If the current pivot state is part of a
                                      loop, this is the guard state of that
                                      loop.
        """
        in_degree = sdfg.in_degree(state)
        in_edges = sdfg.in_edges(state)
        out_degree = sdfg.out_degree(state)
        out_edges = sdfg.out_edges(state)

        if state in visited_states:
            if dominating_loop_guard is not None and dominating_loop_guard == state:
                # We just finished traversing a loop and this is our loop guard,
                # so we additively merge the executions.
                state.executions = state.executions + proposed_executions
                state.dynamic_executions = state.dynamic_executions or proposed_dynamic
            else:
                # This must be after a conditional branch, so we give an upper
                # bound on the number of executions by taking the maximum of
                # all branches.
                state.executions = sp.Max(state.executions, proposed_executions)
                state.dynamic_executions = True
            return
        else:
            visited_states.append(state)
            state.executions = proposed_executions
            state.dynamic_executions = proposed_dynamic

        if out_degree == 0:
            # There's no next state, this DFS traversal is done.
            return
        elif out_degree == 1:
            oedge = out_edges[0]
            return AnnotateLoop._traverse_annotate(
                sdfg,
                oedge.dst,
                oedge,
                state.executions,
                state.dynamic_executions,
                idom,
                visited_states,
                dominating_loop_guard
            )
        elif out_degree == 2:
            if in_degree == 2:
                # This may be a loop, check if that's true.
                assignment_edge = iedge
                step_edge = in_edges[0] if in_edges[1] == iedge else in_edges[1]
                loop_nodes = AnnotateLoop._get_loop_nodes(state, step_edge, idom)
                if loop_nodes is not None:
                    condition_edge = out_edges[0] if out_edges[0].dst == loop_nodes[-1] else out_edges[1]
                    condition = symbolic.pystr_to_symbolic(
                        condition_edge.data.condition.as_string
                    )
                    itvar = list(assignment_edge.data.assignments)[0]
                    loop_range = AnnotateLoop._loop_range(
                        itvar,
                        in_edges,
                        condition
                    )
                    if loop_range is not None:
                        start = loop_range[0]
                        stop = loop_range[1]
                        stride = loop_range[2]
                        loop_executions = (
                            (((stop + 1) - start) / stride) * state.executions
                        )

                        dynamic = False
                        if isinstance(loop_executions, sp.Basic):
                            dynamic = not loop_executions.is_number

                        end_node_executions = state.executions
                        end_node_dynamic = state.dynamic_executions

                        # Save the dominating loop guard in case we're currently
                        # in a loop, so we can restore it when traversing back
                        # out of the loop.
                        previous_dominating_loop_guard = dominating_loop_guard

                        # Traverse down the loop.
                        AnnotateLoop._traverse_annotate(
                            sdfg,
                            condition_edge.dst,
                            condition_edge,
                            loop_executions,
                            dynamic,
                            idom,
                            visited_states,
                            dominating_loop_guard=state
                        )

                        # Traverse down the non-loop side (loop end).
                        end_edge = out_edges[0] if out_edges[1] == condition_edge else out_edges[1]
                        AnnotateLoop._traverse_annotate(
                            sdfg,
                            end_edge.dst,
                            end_edge,
                            end_node_executions,
                            end_node_dynamic,
                            idom,
                            visited_states,
                            dominating_loop_guard=previous_dominating_loop_guard
                        )

                        return

        # This is a regular conditional split.
        for oedge in out_edges:
            AnnotateLoop._traverse_annotate(
                sdfg,
                oedge.dst,
                oedge,
                SetExpr(Interval(0, state.executions)),
                True, # XXX: should this instead be `state.dynamic_executions`?
                idom,
                visited_states,
                dominating_loop_guard
            )
        return


    @staticmethod
    def _loop_range(
        itervar: str,
        inedges: List[gr.Edge],
        condition: sp.Expr
    ) -> Optional[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
        """
        Finds loop range from state machine.

        :param itersym: String representing the iteration variable.
        :param inedges: Incoming edges into guard state (length must be 2).
        :param condition: Condition as sympy expression.
        :return: A three-tuple of (start, end, stride) expressions, or None if
                 proper for-loop was not detected. ``end`` is inclusive.
        """

        # Find starting expression and stride
        itersym = symbolic.symbol(itervar)

        if (itervar in inedges[0].data.assignments[itervar] and itervar not in inedges[1].data.assignments[itervar]):
            stride = (symbolic.pystr_to_symbolic(inedges[0].data.assignments[itervar]) - itersym)
            start = symbolic.pystr_to_symbolic(inedges[1].data.assignments[itervar])
        elif (itervar in inedges[1].data.assignments[itervar] and itervar not in inedges[0].data.assignments[itervar]):
            stride = (symbolic.pystr_to_symbolic(inedges[1].data.assignments[itervar]) - itersym)
            start = symbolic.pystr_to_symbolic(inedges[0].data.assignments[itervar])
        else:
            return None

        # Find condition by matching expressions
        end: Optional[sp.Expr] = None
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

        return start, end, stride


    def apply(self, sdfg):
        # Obtain loop information
        guard: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_guard])
        begin: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._loop_begin])
        after_state: sd.SDFGState = sdfg.node(self.subgraph[DetectLoop._exit_state])

        # Obtain iteration variable, range, and stride
        guard_inedges = sdfg.in_edges(guard)
        condition_edge = sdfg.edges_between(guard, begin)[0]
        itervar = list(guard_inedges[0].data.assignments.keys())[0]
        condition = condition_edge.data.condition_sympy()
        rng = AnnotateLoop._loop_range(itervar, guard_inedges, condition)

        # Find the state prior to the loop
        if rng[0] == symbolic.pystr_to_symbolic(
                guard_inedges[0].data.assignments[itervar]):
            before_state: sd.SDFGState = guard_inedges[0].src
            last_state: sd.SDFGState = guard_inedges[1].src
        else:
            before_state: sd.SDFGState = guard_inedges[1].src
            last_state: sd.SDFGState = guard_inedges[0].src

        # Get loop states
        loop_states = list(sdutil.dfs_conditional(
            sdfg,
            sources=[begin],
            condition=lambda _, child: child != guard
        ))
        first_id = loop_states.index(begin)
        last_id = loop_states.index(last_state)
        loop_subgraph = gr.SubgraphView(sdfg, loop_states)
        for v in loop_subgraph.nodes():
            v.ranges[itervar] = rng
