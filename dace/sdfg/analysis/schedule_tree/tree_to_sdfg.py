# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from collections import defaultdict
from dace.sdfg import nodes, memlet_utils as mmu, utils as sdfg_utils
from dace.sdfg.sdfg import SDFG, ControlFlowRegion, InterstateEdge
from dace.sdfg.state import SDFGState
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class StateBoundaryBehavior(Enum):
    STATE_TRANSITION = auto()  #: Creates multiple states with a state transition
    EMPTY_MEMLET = auto()  #: Happens-before empty memlet edges in the same state


def from_schedule_tree(stree: tn.ScheduleTreeRoot,
                       state_boundary_behavior: StateBoundaryBehavior = StateBoundaryBehavior.STATE_TRANSITION) -> SDFG:
    """
    Converts a schedule tree into an SDFG.
    
    :param stree: The schedule tree root to convert.
    :param state_boundary_behavior: Sets the behavior upon encountering a state boundary (e.g., write-after-write).
                                    See the ``StateBoundaryBehavior`` enumeration for more details.
    :return: An SDFG representing the schedule tree.
    """
    # Set SDFG descriptor repository
    result = SDFG(stree.name, propagate=False)
    result.arg_names = copy.deepcopy(stree.arg_names)
    for key, container in stree.containers.items():
        result._arrays[key] = copy.deepcopy(container)
    result.constants_prop = copy.deepcopy(stree.constants)
    result.symbols = copy.deepcopy(stree.symbols)

    # TODO: Fill SDFG contents
    stree = insert_state_boundaries_to_tree(stree)  # after WAW, before label, etc.

    class StreeToSDFG(tn.ScheduleNodeVisitor):

        def __init__(self) -> None:
            self._state_stack: List[SDFGState] = []
            self._current_state: Optional[SDFGState] = None
            self.access_cache: Dict[SDFGState, Dict[str, nodes.AccessNode]] = {}

        def _pop_state(self, label: Optional[str] = None) -> SDFGState:
            """Pops the last state from the stack.

            :param label: ensures the popped state's label starts with the given string

            :return: The popped state.
            """
            if not self._state_stack:
                raise ValueError("Can't pop state from empty stack.")

            popped = self._state_stack.pop()
            if label is not None:
                assert popped.label.startswith(label)

            return popped

        def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot, sdfg: SDFG) -> None:
            self._current_state = sdfg.add_state(label="tree_root", is_start_block=True)
            self.visit(node.children, sdfg=sdfg)

        def visit_StateBoundaryNode(self, node: tn.StateBoundaryNode, sdfg: SDFG) -> None:
            # TODO: When creating a state boundary, include all inter-state assignments that precede it.

            self._current_state = create_state_boundary(node, sdfg, self._current_state,
                                                        StateBoundaryBehavior.STATE_TRANSITION)

        def visit_GBlock(self, node: tn.GBlock, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_StateLabel(self, node: tn.StateLabel, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_GotoNode(self, node: tn.GotoNode, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_AssignNode(self, node: tn.AssignNode, sdfg: SDFG) -> None:
            # TODO: We'll need these symbol assignments
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_ForScope(self, node: tn.ForScope, sdfg: SDFG) -> None:
            before_state = self._current_state
            guard_state = sdfg.add_state(label="loop_guard")
            self._current_state = guard_state
            sdfg.add_edge(before_state, self._current_state,
                          InterstateEdge(assignments=dict({node.header.itervar: node.header.init})))

            body_state = sdfg.add_state(label="loop_body")
            self._current_state = body_state
            sdfg.add_edge(guard_state, body_state, InterstateEdge(condition=node.header.condition))

            # visit children inside the loop
            self.visit(node.children, sdfg=sdfg)
            sdfg.add_edge(self._current_state, guard_state,
                          InterstateEdge(assignments=dict({node.header.itervar: node.header.update})))

            after_state = sdfg.add_state(label="loop_after")
            self._current_state = after_state
            sdfg.add_edge(guard_state, after_state, InterstateEdge(condition=f"not {node.header.condition.as_string}"))

        def visit_WhileScope(self, node: tn.WhileScope, sdfg: SDFG) -> None:
            before_state = self._current_state
            guard_state = sdfg.add_state(label="guard_state")
            self._current_state = guard_state
            sdfg.add_edge(before_state, guard_state, InterstateEdge())

            body_state = sdfg.add_state(label="loop_body")
            self._current_state = body_state
            sdfg.add_edge(guard_state, body_state, InterstateEdge(condition=node.header.test))

            # visit children inside the loop
            self.visit(node.children, sdfg=sdfg)
            sdfg.add_edge(self._current_state, guard_state, InterstateEdge())

            after_state = sdfg.add_state(label="loop_after")
            self._current_state = after_state
            sdfg.add_edge(guard_state, after_state, InterstateEdge(f"not {node.header.test.as_string}"))

        def visit_DoWhileScope(self, node: tn.DoWhileScope, sdfg: SDFG) -> None:
            # AFAIK we don't support for do-while loops in the gt4py -> dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_GeneralLoopScope(self, node: tn.GeneralLoopScope, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_IfScope(self, node: tn.IfScope, sdfg: SDFG) -> None:
            before_state = self._current_state

            # add guard state
            guard_state = sdfg.add_state(label="guard_state")
            sdfg.add_edge(before_state, guard_state, InterstateEdge())

            # add true_state
            true_state = sdfg.add_state(label="true_state")
            sdfg.add_edge(guard_state, true_state, InterstateEdge(condition=node.condition))
            self._current_state = true_state

            # visit children in the true branch
            self.visit(node.children, sdfg=sdfg)

            # add merge_state
            merge_state = sdfg.add_state_after(self._current_state, label="merge_state")

            # Check if there's an `ElseScope` following this node (in the parent's children).
            # Filter StateBoundaryNodes, which we inserted earlier, for this analysis.
            filtered = [n for n in node.parent.children if not isinstance(n, tn.StateBoundaryNode)]
            if_index = filtered.index(node)
            has_else_branch = len(filtered) > if_index + 1 and isinstance(filtered[if_index + 1], tn.ElseScope)

            if has_else_branch:
                # push merge_state on the stack for later usage in `visit_ElseScope`
                self._state_stack.append(merge_state)
                false_state = sdfg.add_state(label="false_state")

                sdfg.add_edge(guard_state, false_state, InterstateEdge(condition=f"not {node.condition.as_string}"))

                # push false_state on the stack for later usage in `visit_ElseScope`
                self._state_stack.append(false_state)
            else:
                sdfg.add_edge(guard_state, merge_state, InterstateEdge(condition=f"not {node.condition.as_string}"))
                self._current_state = merge_state

        def visit_StateIfScope(self, node: tn.StateIfScope, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_BreakNode(self, node: tn.BreakNode, sdfg: SDFG) -> None:
            # AFAIK we don't support for break statements in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_ContinueNode(self, node: tn.ContinueNode, sdfg: SDFG) -> None:
            # AFAIK we don't support for continue statements in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_ElifScope(self, node: tn.ElifScope, sdfg: SDFG) -> None:
            # AFAIK we don't support elif scopes in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_ElseScope(self, node: tn.ElseScope, sdfg: SDFG) -> None:
            # get false_state form stack
            false_state = self._pop_state("false_state")
            self._current_state = false_state

            # visit children
            self.visit(node.children, sdfg=sdfg)

            # merge false-branch into merge_state
            merge_state = self._pop_state("merge_state")
            sdfg.add_edge(self._current_state, merge_state, InterstateEdge())
            self._current_state = merge_state

        def visit_TaskletNode(self, node: tn.TaskletNode, sdfg: SDFG) -> None:
            # Add Tasklet to current state
            tasklet = node.node
            self._current_state.add_node(tasklet)

            # Manage access cache
            if not self._current_state in self.access_cache:
                self.access_cache[self._current_state] = {}
            cache = self.access_cache[self._current_state]

            # Connect inputs and outputs
            for name, memlet in node.in_memlets.items():
                # cache read access
                if memlet.data not in cache:
                    cache[memlet.data] = self._current_state.add_read(memlet.data)

                access_node = cache[memlet.data]
                self._current_state.add_memlet_path(access_node, tasklet, dst_conn=name, memlet=memlet)

            for name, memlet in node.out_memlets.items():
                # we always write to a new access_node
                access_node = self._current_state.add_write(memlet.data)
                self._current_state.add_memlet_path(tasklet, access_node, src_conn=name, memlet=memlet)

                # cache write access node (or update an existing one) for read after write cases
                cache[memlet.data] = access_node

    # TODO: create_dataflow_scope
    StreeToSDFG().visit(stree, sdfg=result)

    return result


def insert_state_boundaries_to_tree(stree: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
    """
    Inserts StateBoundaryNode objects into a schedule tree where more than one SDFG state would be necessary.
    Operates in-place on the given schedule tree.
    
    This happens when there is a:
      * write-after-write dependency;
      * write-after-read dependency that cannot be fulfilled via memlets;
      * control flow block (for/if); or
      * otherwise before a state label (which means a state transition could occur, e.g., in a gblock)

    :param stree: The schedule tree to operate on.
    """

    # Simple boundary node inserter for control flow blocks and state labels
    class SimpleStateBoundaryInserter(tn.ScheduleNodeTransformer):

        def visit_scope(self, scope: tn.ScheduleTreeScope):
            if isinstance(scope, tn.ControlFlowScope):
                return [tn.StateBoundaryNode(True), self.generic_visit(scope)]
            return self.generic_visit(scope)

        def visit_StateLabel(self, node: tn.StateLabel):
            return [tn.StateBoundaryNode(True), self.generic_visit(node)]

    # First, insert boundaries around labels and control flow
    stree = SimpleStateBoundaryInserter().visit(stree)

    # Then, insert boundaries after unmet memory dependencies or potential data races
    _insert_memory_dependency_state_boundaries(stree)

    return stree


def _insert_memory_dependency_state_boundaries(scope: tn.ScheduleTreeScope):
    """
    Helper function that inserts boundaries after unmet memory dependencies.
    """
    reads: mmu.MemletDict[List[tn.ScheduleTreeNode]] = mmu.MemletDict()
    writes: mmu.MemletDict[List[tn.ScheduleTreeNode]] = mmu.MemletDict()
    parents: Dict[int, Set[int]] = defaultdict(set)
    boundaries_to_insert: List[int] = []

    for i, n in enumerate(scope.children):
        if isinstance(n, (tn.StateBoundaryNode, tn.ControlFlowScope)):  # Clear state
            reads.clear()
            writes.clear()
            parents.clear()
            if isinstance(n, tn.ControlFlowScope):  # Insert memory boundaries recursively
                _insert_memory_dependency_state_boundaries(n)
            continue

        # If dataflow scope, insert state boundaries recursively and as a node
        if isinstance(n, tn.DataflowScope):
            _insert_memory_dependency_state_boundaries(n)

        inputs = n.input_memlets()
        outputs = n.output_memlets()

        # Register reads
        for inp in inputs:
            if inp not in reads:
                reads[inp] = [n]
            else:
                reads[inp].append(n)

            # Transitively add parents
            if inp in writes:
                for parent in writes[inp]:
                    parents[id(n)].add(id(parent))
                    parents[id(n)].update(parents[id(parent)])

        # Inter-state assignment nodes with reads necessitate a state transition if they were written to.
        if isinstance(n, tn.AssignNode) and any(inp in writes for inp in inputs):
            boundaries_to_insert.append(i)
            reads.clear()
            writes.clear()
            parents.clear()
            continue

        # Write after write or potential write/write data race, insert state boundary
        if any(o in writes and (o not in reads or any(id(r) not in parents for r in reads[o])) for o in outputs):
            boundaries_to_insert.append(i)
            reads.clear()
            writes.clear()
            parents.clear()
            continue

        # Potential read/write data race: if any read is not in the parents of this node, it might
        # be performed in parallel
        if any(o in reads and any(id(r) not in parents for r in reads[o]) for o in outputs):
            boundaries_to_insert.append(i)
            reads.clear()
            writes.clear()
            parents.clear()
            continue

        # Register writes after all hazards have been tested for
        for out in outputs:
            if out not in writes:
                writes[out] = [n]
            else:
                writes[out].append(n)

    # Insert memory dependency state boundaries in reverse in order to keep indices intact
    for i in reversed(boundaries_to_insert):
        scope.children.insert(i, tn.StateBoundaryNode())


#############################################################################
# SDFG content creation functions


def create_state_boundary(bnode: tn.StateBoundaryNode, sdfg_region: ControlFlowRegion, state: SDFGState,
                          behavior: StateBoundaryBehavior) -> SDFGState:
    """
    Creates a boundary between two states

    :param bnode: The state boundary node to generate.
    :param sdfg_region: The control flow block in which to generate the boundary (e.g., SDFG).
    :param state: The last state prior to this boundary.
    :param behavior: The state boundary behavior with which to create the boundary.
    :return: The newly created state.
    """
    if behavior != StateBoundaryBehavior.STATE_TRANSITION:
        # Only STATE_TRANSITION is supported as StateBoundaryBehavior in this prototype.
        raise NotImplementedError

    # TODO: Some boundaries (control flow, state labels with goto) could not be fulfilled with every
    #       behavior. Fall back to state transition in that case.

    label = "cf_state_boundary" if bnode.due_to_control_flow else "state_boundary"
    return sdfg_region.add_state_after(state, label=label)
