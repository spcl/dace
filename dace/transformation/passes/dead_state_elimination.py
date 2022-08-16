# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import collections
import sympy as sp
from typing import Optional, Set, Tuple, Union

from dace import SDFG, InterstateEdge, SDFGState, symbolic
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from dace.sdfg.validation import InvalidSDFGInterstateEdgeError
from dace.transformation import pass_pipeline as ppl


class DeadStateElimination(ppl.Pass):
    """
    Removes all unreachable states (e.g., due to a branch that will never be taken) from an SDFG.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If connectivity or any edges were changed, some more states might be dead
        return modified & (ppl.Modifies.InterstateEdges | ppl.Modifies.States)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[Union[SDFGState, Edge[InterstateEdge]]]]:
        """
        Removes unreachable states throughout an SDFG.
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :param initial_symbols: If not None, sets values of initial symbols.
        :return: A set of the removed states, or None if nothing was changed.
        """
        # Mark dead states and remove them
        dead_states, dead_edges, annotated = self.find_dead_states(sdfg, set_unconditional_edges=True)

        for e in dead_edges:
            sdfg.remove_edge(e)
        sdfg.remove_nodes_from(dead_states)

        result = dead_states | dead_edges

        if not annotated:
            return result or None
        else:
            return result or set()  # Return an empty set if edges were annotated

    def find_dead_states(
            self,
            sdfg: SDFG,
            set_unconditional_edges: bool = True) -> Tuple[Set[SDFGState], Set[Edge[InterstateEdge]], bool]:
        '''
        Finds "dead" (unreachable) states in an SDFG. A state is deemed unreachable if it is:
            * Unreachable from the starting state
            * Conditions leading to it will always evaluate to False
            * There is another unconditional (always True) inter-state edge that leads to another state

        :param sdfg: The SDFG to traverse.
        :param set_unconditional_edges: If True, conditions of edges evaluated as unconditional are removed.
        :return: A 3-tuple of (unreachable states, unreachable edges, were edges annotated).
        '''
        visited: Set[SDFGState] = set()
        dead_edges: Set[Edge[InterstateEdge]] = set()
        edges_annotated = False

        # Run a modified BFS where definitely False edges are not traversed, or if there is an
        # unconditional edge the rest are not. The inverse of the visited states is the dead set.
        queue = collections.deque([sdfg.start_state])
        while len(queue) > 0:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            # First, check for unconditional edges
            unconditional = None
            for e in sdfg.out_edges(node):
                # If an unconditional edge is found, ignore all other outgoing edges
                if self.is_definitely_taken(e.data, sdfg):
                    # If more than one unconditional outgoing edge exist, fail with Invalid SDFG
                    if unconditional is not None:
                        raise InvalidSDFGInterstateEdgeError('Multiple unconditional edges leave the same state', sdfg,
                                                             sdfg.edge_id(e))
                    unconditional = e
                    if set_unconditional_edges and not e.data.is_unconditional():
                        # Annotate edge as unconditional
                        e.data.condition = CodeBlock('1')
                        edges_annotated = True

                    # Continue traversal through edge
                    if e.dst not in visited:
                        queue.append(e.dst)
                        continue
            if unconditional is not None:  # Unconditional edge exists, skip traversal
                # Remove other (now never taken) edges from graph
                for e in sdfg.out_edges(node):
                    if e is not unconditional:
                        dead_edges.add(e)

                continue
            # End of unconditional check

            # Check outgoing edges normally
            for e in sdfg.out_edges(node):
                next_node = e.dst

                # Test for edges that definitely evaluate to False
                if self.is_definitely_not_taken(e.data, sdfg):
                    dead_edges.add(e)
                    continue

                # Continue traversal through edge
                if next_node not in visited:
                    queue.append(next_node)

        # Dead states are states that are not live (i.e., visited)
        return set(sdfg.nodes()) - visited, dead_edges, edges_annotated

    def report(self, pass_retval: Set[Union[SDFGState, Edge[InterstateEdge]]]) -> str:
        if pass_retval is not None and not pass_retval:
            return 'DeadStateElimination annotated new unconditional edges.'

        states = [p for p in pass_retval if isinstance(p, SDFGState)]
        return f'Eliminated {len(states)} states and {len(pass_retval) - len(states)} interstate edges.'

    def is_definitely_taken(self, edge: InterstateEdge, sdfg: SDFG) -> bool:
        """ Returns True iff edge condition definitely evaluates to True. """
        if edge.is_unconditional():
            return True

        # Evaluate condition
        scond = edge.condition_sympy()
        if scond == True or scond == sp.Not(sp.S.false, evaluate=False):
            return True

        # Evaluate non-optional arrays
        scond = symbolic.evaluate_optional_arrays(scond, sdfg)
        if scond == True:
            return True

        # Indeterminate or False condition
        return False

    def is_definitely_not_taken(self, edge: InterstateEdge, sdfg: SDFG) -> bool:
        """ Returns True iff edge condition definitely evaluates to False. """
        if edge.is_unconditional():
            return False

        # Evaluate condition
        scond = edge.condition_sympy()
        if scond == False or scond == sp.Not(sp.S.true, evaluate=False):
            return True

        # Evaluate non-optional arrays
        scond = symbolic.evaluate_optional_arrays(scond, sdfg)
        if scond == False:
            return True

        # Indeterminate or True condition
        return False
