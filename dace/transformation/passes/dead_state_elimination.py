# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import collections
from dace.properties import CodeBlock
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, SDFGState, InterstateEdge
from dace.sdfg.validation import InvalidSDFGInterstateEdgeError
from typing import Set, Optional


class DeadStateElimination(ppl.Pass):
    """
    Removes all unreachable states (e.g., due to a branch that will never be taken) from an SDFG.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If connectivity or any edges were changed, some more states might be dead
        return modified & (ppl.Modifies.InterstateEdges | ppl.Modifies.States)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[SDFGState]]:
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
        result = self.find_dead_states(sdfg, set_unconditional_edges=True)
        sdfg.remove_nodes_from(result)

        return result or None

    def find_dead_states(self, sdfg: SDFG, set_unconditional_edges: bool = True) -> Set[SDFGState]:
        '''
        Finds "dead" (unreachable) states in an SDFG. A state is deemed unreachable if it is:
            * Unreachable from the starting state
            * Conditions leading to it will always evaluate to False
            * There is another unconditional (always True) inter-state edge that leads to another state

        :param sdfg: The SDFG to traverse.
        :param set_unconditional_edges: If True, conditions of edges evaluated as unconditional are removed.
        :return: A set of unreachable states.
        '''
        visited: Set[SDFGState] = set()

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
                if self.is_definitely_taken(e.data):
                    # If more than one unconditional outgoing edge exist, fail with Invalid SDFG
                    if unconditional is not None:
                        raise InvalidSDFGInterstateEdgeError('Multiple unconditional edges leave the same state', sdfg,
                                                             sdfg.edge_id(e))
                    unconditional = e
                    if set_unconditional_edges and not e.data.is_unconditional():
                        # Annotate edge as unconditional
                        e.data.condition = CodeBlock('1')

                    # Continue traversal through edge
                    if e.dst not in visited:
                        queue.append(e.dst)
                        continue
            if unconditional is not None:  # Unconditional edge exists, skip traversal
                continue
            # End of unconditional check

            # Check outgoing edges normally
            for e in sdfg.out_edges(node):
                next_node = e.dst

                # Test for edges that definitely evaluate to False
                if self.is_definitely_not_taken(e.data):
                    continue

                # Continue traversal through edge
                if next_node not in visited:
                    queue.append(next_node)

        # Dead states are states that are not live (i.e., visited)
        return set(sdfg.nodes()) - visited

    def is_definitely_taken(self, edge: InterstateEdge) -> bool:
        """ Returns True iff edge condition definitely evaluates to True. """
        if edge.is_unconditional():
            return True

        # Evaluate condition
        scond = edge.condition_sympy()
        if scond == True:
            return True

        # Indeterminate or False condition
        return False

    def is_definitely_not_taken(self, edge: InterstateEdge) -> bool:
        """ Returns True iff edge condition definitely evaluates to False. """
        if edge.is_unconditional():
            return False

        # Evaluate condition
        scond = edge.condition_sympy()
        if scond == False:
            return True

        # Indeterminate or True condition
        return False
