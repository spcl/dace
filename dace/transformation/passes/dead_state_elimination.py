# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import collections
import sympy as sp
from typing import List, Optional, Set, Tuple, Union

from dace import SDFG, InterstateEdge, SDFGState, symbolic, properties
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion
from dace.sdfg.validation import InvalidSDFGInterstateEdgeError, InvalidSDFGNodeError
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class DeadStateElimination(ppl.Pass):
    """
    Removes all unreachable states (e.g., due to a branch that will never be taken) from an SDFG.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If connectivity or any edges were changed, some more states might be dead
        return modified & ppl.Modifies.CFG

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
        result: Set[Union[ControlFlowBlock, InterstateEdge]] = set()
        removed_regions: Set[ControlFlowRegion] = set()
        annotated = None
        for cfg in list(sdfg.all_control_flow_regions()):
            if cfg in removed_regions or isinstance(cfg, ConditionalBlock):
                continue

            # Mark dead blocks and remove them
            dead_blocks, dead_edges, annotated = self.find_dead_control_flow(cfg, set_unconditional_edges=True)
            for e in dead_edges:
                cfg.remove_edge(e)
            for block in dead_blocks:
                cfg.remove_node(block)
                if isinstance(block, ControlFlowRegion):
                    removed_regions.add(block)

            region_result = dead_blocks | dead_edges
            result |= region_result

            for node in cfg.nodes():
                if isinstance(node, ConditionalBlock):
                    dead_branches = self._find_dead_branches(node)
                    if len(dead_branches) < len(node.branches):
                        for _, b in dead_branches:
                            result.add(b)
                            node.remove_branch(b)
                        # If only one branch is left, and it is unconditionally executed, inline it.
                        if len(node.branches) == 1:
                            cond, branch = node.branches[0]
                            if cond is None or self._is_definitely_true(symbolic.pystr_to_symbolic(cond.as_string), sdfg):
                                node.parent_graph.add_node(branch)
                                for ie in cfg.in_edges(node):
                                    cfg.add_edge(ie.src, branch, ie.data)
                                for oe in cfg.out_edges(node):
                                    cfg.add_edge(branch, oe.dst, oe.data)
                                result.add(node)
                                cfg.remove_node(node)
                    else:
                        result.add(node)
                        is_start = node is cfg.start_block
                        replacement_pre = cfg.add_state_before(node, node.label + '_pre', is_start_block=is_start)
                        replacement_post = cfg.add_state_after(node, node.label + '_post')
                        cfg.add_edge(replacement_pre, replacement_post, InterstateEdge())
                        cfg.remove_node(node)

        if not annotated:
            return result or None
        else:
            return result or set()  # Return an empty set if edges were annotated

    def find_dead_control_flow(
            self,
            cfg: ControlFlowRegion,
            set_unconditional_edges: bool = True) -> Tuple[Set[ControlFlowBlock], Set[Edge[InterstateEdge]], bool]:
        """
        Finds "dead" (unreachable) control flow in a CFG. A block is deemed unreachable if it is:
        
            * Unreachable from the starting block
            * Conditions leading to it will always evaluate to False
            * There is another unconditional (always True) inter-state edge that leads to another block

        :param cfg: The CFG to traverse.
        :param set_unconditional_edges: If True, conditions of edges evaluated as unconditional are removed.
        :return: A 3-tuple of (unreachable blocks, unreachable edges, were edges annotated).
        """
        sdfg = cfg.sdfg if cfg.sdfg is not None else cfg
        visited: Set[ControlFlowBlock] = set()
        dead_edges: Set[Edge[InterstateEdge]] = set()
        edges_annotated = False

        # Run a modified BFS where definitely False edges are not traversed, or if there is an
        # unconditional edge the rest are not. The inverse of the visited blocks is the dead set.
        queue = collections.deque([cfg.start_block])
        while len(queue) > 0:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            # First, check for unconditional edges
            unconditional = None
            for e in cfg.out_edges(node):
                # If an unconditional edge is found, ignore all other outgoing edges
                if self.is_definitely_taken(e.data, sdfg):
                    # If more than one unconditional outgoing edge exist, fail with Invalid SDFG
                    if unconditional is not None:
                        raise InvalidSDFGInterstateEdgeError('Multiple unconditional edges leave the same block', cfg,
                                                             cfg.edge_id(e))
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
                for e in cfg.out_edges(node):
                    if e is not unconditional:
                        dead_edges.add(e)

                continue
            # End of unconditional check

            # Check outgoing edges normally
            for e in cfg.out_edges(node):
                next_node = e.dst

                # Test for edges that definitely evaluate to False
                if self.is_definitely_not_taken(e.data, sdfg):
                    dead_edges.add(e)
                    continue

                # Continue traversal through edge
                if next_node not in visited:
                    queue.append(next_node)

        # Dead states are states that are not live (i.e., visited)
        return set(cfg.nodes()) - visited, dead_edges, edges_annotated

    def _find_dead_branches(self, block: ConditionalBlock) -> List[Tuple[CodeBlock, ControlFlowRegion]]:
        dead_branches = []
        unconditional = None
        for i, (cond, branch) in enumerate(block.branches):
            if cond is None:
                if not i == len(block.branches) - 1:
                    raise InvalidSDFGNodeError('Conditional block detected, where else branch is not the last branch')
                break
            # If an unconditional branch is found, ignore all other branches that follow this one.
            if self._is_definitely_true(symbolic.pystr_to_symbolic(cond.as_string), block.sdfg):
                unconditional = branch
                break
        if unconditional is not None:
            # Remove other (now never taken) branches
            for cond, branch in block.branches:
                if branch is not unconditional:
                    dead_branches.append([cond, branch])
        else:
            # Check if any branches are certainly never taken.
            for cond, branch in block.branches:
                if cond is not None and self._is_definitely_false(symbolic.pystr_to_symbolic(cond.as_string), block.sdfg):
                    dead_branches.append([cond, branch])

        return dead_branches

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
        return self._is_definitely_true(edge.condition_sympy(), sdfg)

    def _is_definitely_true(self, cond: sp.Basic, sdfg: SDFG) -> bool:
        if cond == True or cond == sp.Not(sp.logic.boolalg.BooleanFalse(), evaluate=False):
            return True

        # Evaluate non-optional arrays
        cond = symbolic.evaluate_optional_arrays(cond, sdfg)
        if cond == True:
            return True

        # Indeterminate or False condition
        return False

    def is_definitely_not_taken(self, edge: InterstateEdge, sdfg: SDFG) -> bool:
        """ Returns True iff edge condition definitely evaluates to False. """
        if edge.is_unconditional():
            return False

        # Evaluate condition
        return self._is_definitely_false(edge.condition_sympy(), sdfg)

    def _is_definitely_false(self, cond: sp.Basic, sdfg: SDFG) -> bool:
        if cond == False or cond == sp.Not(sp.logic.boolalg.BooleanTrue(), evaluate=False):
            return True

        # Evaluate non-optional arrays
        cond = symbolic.evaluate_optional_arrays(cond, sdfg)
        if cond == False:
            return True

        # Indeterminate or True condition
        return False
