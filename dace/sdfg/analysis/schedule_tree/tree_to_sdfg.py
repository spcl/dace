# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from collections import defaultdict
from dace.memlet import Memlet
from dace.sdfg import nodes, memlet_utils as mmu, utils as sdfg_utils
from dace.sdfg.sdfg import SDFG, ControlFlowRegion, InterstateEdge
from dace.sdfg.state import SDFGState
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from enum import Enum, auto
from typing import Dict, Final, List, Optional, Set, Tuple


class StateBoundaryBehavior(Enum):
    STATE_TRANSITION = auto()  #: Creates multiple states with a state transition
    EMPTY_MEMLET = auto()  #: Happens-before empty memlet edges in the same state


PREFIX_PASSTHROUGH_IN: Final[str] = "IN_"
PREFIX_PASSTHROUGH_OUT: Final[str] = "OUT_"


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

    # after WAW, before label, etc.
    stree = insert_state_boundaries_to_tree(stree)

    class StreeToSDFG(tn.ScheduleNodeVisitor):

        def __init__(self, start_state: Optional[SDFGState] = None) -> None:
            # state management
            self._state_stack: List[SDFGState] = []
            self._current_state = start_state

            # inter-state symbol assignments
            self._interstate_symbols: List[tn.AssignNode] = []

            # dataflow scopes
            self._dataflow_stack: List[Tuple[nodes.EntryNode, nodes.ExitNode]] = []

            # caches
            self._access_cache: Dict[SDFGState, Dict[str, nodes.AccessNode]] = {}

        def _pop_state(self, label: Optional[str] = None) -> SDFGState:
            """Pops the last state from the state stack.

            :param str, optional label: Ensures the popped state's label starts with the given string.

            :return: The popped state.
            """
            if not self._state_stack:
                raise ValueError("Can't pop state from empty stack.")

            popped = self._state_stack.pop()
            if label is not None:
                assert popped.label.startswith(label)

            return popped

        def _ensure_access_cache(self, state: SDFGState) -> Dict[str, nodes.AccessNode]:
            """Ensure an access_cache entry for the given state.

            Checks if there exists an access_cache for `state`. Creates an empty one if it doesn't exist yet.

            :param SDFGState state: The state to check.

            :return: The state's access_cache.
            """
            if state not in self._access_cache:
                self._access_cache[state] = {}

            return self._access_cache[state]

        def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot, sdfg: SDFG) -> None:
            assert self._current_state is None, "Expected no 'current_state' at root."
            assert not self._state_stack, "Expected empty state stack at root."
            assert not self._dataflow_stack, "Expected empty dataflow stack at root."
            assert not self._interstate_symbols, "Expected empty list of symbols at root."

            self._current_state = sdfg.add_state(label="tree_root", is_start_block=True)
            self.visit(node.children, sdfg=sdfg)

            assert not self._state_stack, "Expected empty state stack."
            assert not self._dataflow_stack, "Expected empty dataflow stack."
            assert not self._interstate_symbols, "Expected empty list of symbols to add."

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
            # We just collect them here. They'll be added when state boundaries are added,
            # see `visit_StateBoundaryNode()` above.
            self._interstate_symbols.append(node)

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
            if_index = _list_index(filtered, node)
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

            # visit children inside the else branch
            self.visit(node.children, sdfg=sdfg)

            # merge false-branch into merge_state
            merge_state = self._pop_state("merge_state")
            sdfg.add_edge(self._current_state, merge_state, InterstateEdge())
            self._current_state = merge_state

        def _generate_MapScope(self, node: tn.MapScope, sdfg: SDFG) -> None:
            dataflow_stack_size = len(self._dataflow_stack)
            outer_map_entry, outer_map_exit = self._dataflow_stack[-1] if dataflow_stack_size else (None, None)
            cache = self._ensure_access_cache(self._current_state)

            # map entry
            map_entry = nodes.MapEntry(node.node.map)
            self._current_state.add_node(map_entry)

            for memlet in node.input_memlets():
                map_entry.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{memlet.data}")
                map_entry.add_out_connector(f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}")

                if outer_map_entry is not None:
                    # passthrough if we are inside another map
                    self._current_state.add_edge(outer_map_entry, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", map_entry,
                                                 f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", memlet)
                else:
                    # add access node "outside the map" and connect to it
                    if memlet.data not in cache:
                        # cache read access
                        cache[memlet.data] = self._current_state.add_read(memlet.data)

                    if not self._current_state.edges_between(cache[memlet.data], map_entry):
                        self._current_state.add_edge(cache[memlet.data], None, map_entry,
                                                     f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", memlet)

            # Add empty memlet if outer_map_entry has no out_connectors to connect to
            if outer_map_entry is not None and not outer_map_entry.out_connectors and self._current_state.out_degree(
                    outer_map_entry) < 1:
                self._current_state.add_edge(outer_map_entry, None, map_entry, None, memlet=Memlet())

            # map exit
            map_exit = nodes.MapExit(node.node.map)
            self._current_state.add_node(map_exit)

            for memlet in node.output_memlets():
                map_exit.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{memlet.data}")
                map_exit.add_out_connector(f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}")

                if outer_map_exit:
                    # passthrough if we are inside another map
                    self._current_state.add_edge(map_exit, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", outer_map_exit,
                                                 f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", memlet)
                else:
                    # add access nodes "outside the map" and connect to it
                    # we always write to a new access_node
                    access_node = self._current_state.add_write(memlet.data)
                    self._current_state.add_edge(map_exit, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", access_node, None,
                                                 memlet)

                    # cache write access node (or update an existing one) for read after write cases
                    cache[memlet.data] = access_node

            # Add empty memlet if outer_map_exit has no in_connectors to connect to
            if outer_map_exit is not None and not outer_map_exit.in_connectors and self._current_state.in_degree(
                    outer_map_exit) < 1:
                self._current_state.add_edge(map_exit, None, outer_map_exit, None, memlet=Memlet())

            self._dataflow_stack.append((map_entry, map_exit))

            # visit children inside the map
            self.visit(node.children, sdfg=sdfg)

            self._dataflow_stack.pop()
            assert len(self._dataflow_stack) == dataflow_stack_size  # sanity check

        def _generate_MapScope_with_nested_SDFG(self, node: tn.MapScope, sdfg: SDFG) -> None:
            inputs = node.input_memlets()
            outputs = node.output_memlets()

            # setup nested SDFG
            nsdfg = SDFG("nested_sdfg", parent=self._current_state)
            start_state = nsdfg.add_state("nested_root", is_start_block=True)
            for memlet in [*inputs, *outputs]:
                if memlet.data not in nsdfg.arrays:
                    nsdfg.add_datadesc(memlet.data, sdfg.arrays[memlet.data].clone())

            # visit children inside nested SDFG
            inner_visitor = StreeToSDFG(start_state)
            for child in node.children:
                inner_visitor.visit(child, sdfg=nsdfg)

            nested_SDFG = self._current_state.add_nested_sdfg(nsdfg,
                                                              sdfg,
                                                              inputs={memlet.data
                                                                      for memlet in node.input_memlets()},
                                                              outputs={memlet.data
                                                                       for memlet in node.output_memlets()})

            dataflow_stack_size = len(self._dataflow_stack)
            outer_map_entry, outer_map_exit = self._dataflow_stack[-1] if dataflow_stack_size else (None, None)
            cache = self._ensure_access_cache(self._current_state)

            # map entry
            map_entry = nodes.MapEntry(node.node.map)
            self._current_state.add_node(map_entry)

            for memlet in inputs:
                map_entry.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{memlet.data}")
                map_entry.add_out_connector(f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}")

                # connect nested SDFG to map scope
                self._current_state.add_edge(map_entry, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", nested_SDFG,
                                             memlet.data, Memlet.from_memlet(memlet))

                # connect map scope to "outer world"
                if outer_map_entry is not None:
                    # passthrough if we are inside another map
                    self._current_state.add_edge(outer_map_entry, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", map_entry,
                                                 f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", memlet)
                else:
                    # add access node "outside the map" and connect to it
                    if memlet.data not in cache:
                        # cache read access
                        cache[memlet.data] = self._current_state.add_read(memlet.data)

                    if not self._current_state.edges_between(cache[memlet.data], map_entry):
                        self._current_state.add_edge(cache[memlet.data], None, map_entry,
                                                     f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", memlet)

            # Add empty memlet if no explicit connection from map_entry to nested_SDFG has been done so far
            if not inputs:
                self._current_state.add_edge(map_entry, None, nested_SDFG, None, memlet=Memlet())

            # Add empty memlet if outer_map_entry has no out_connectors to connect to
            if outer_map_entry is not None and not outer_map_entry.out_connectors and self._current_state.out_degree(
                    outer_map_entry) < 1:
                self._current_state.add_edge(outer_map_entry, None, map_entry, None, memlet=Memlet())

            # map exit
            map_exit = nodes.MapExit(node.node.map)
            self._current_state.add_node(map_exit)

            for memlet in outputs:
                map_exit.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{memlet.data}")
                map_exit.add_out_connector(f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}")

                # connect nested SDFG to map scope
                self._current_state.add_edge(nested_SDFG, memlet.data, map_exit,
                                             f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", Memlet.from_memlet(memlet))

                # connect map scope to "outer world"
                if outer_map_exit:
                    # passthrough if we are inside another map
                    self._current_state.add_edge(map_exit, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", outer_map_exit,
                                                 f"{PREFIX_PASSTHROUGH_IN}{memlet.data}", memlet)
                else:
                    # add access nodes "outside the map" and connect to it
                    # we always write to a new access_node
                    access_node = self._current_state.add_write(memlet.data)
                    self._current_state.add_edge(map_exit, f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}", access_node, None,
                                                 memlet)

                    # cache write access node (or update an existing one) for read after write cases
                    cache[memlet.data] = access_node

            # Add empty memlet if no explicit connection from map_entry to nested_SDFG has been done so far
            if not outputs:
                self._current_state.add_edge(nested_SDFG, None, map_exit, None, memlet=Memlet())

            # Add empty memlet if outer_map_exit has no in_connectors to connect to
            if outer_map_exit is not None and not outer_map_exit.in_connectors and self._current_state.in_degree(
                    outer_map_exit) < 1:
                self._current_state.add_edge(map_exit, None, outer_map_exit, None, memlet=Memlet())

        def visit_MapScope(self, node: tn.MapScope, sdfg: SDFG) -> None:
            if any([isinstance(child, tn.StateBoundaryNode) for child in node.children]):
                # support multiple states within this map by inserting a nested SDFG
                return self._generate_MapScope_with_nested_SDFG(node, sdfg)

            self._generate_MapScope(node, sdfg)

        def visit_ConsumeScope(self, node: tn.ConsumeScope, sdfg: SDFG) -> None:
            # AFAIK we don't support consume scopes in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_PipelineScope(self, node: tn.PipelineScope, sdfg: SDFG) -> None:
            # AFAIK we don't support pipeline scopes in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_TaskletNode(self, node: tn.TaskletNode, sdfg: SDFG) -> None:
            # Add Tasklet to current state
            tasklet = node.node
            self._current_state.add_node(tasklet)

            cache = self._ensure_access_cache(self._current_state)
            map_entry, map_exit = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)

            # Connect input memlets
            for name, memlet in node.in_memlets.items():
                # connect to dataflow_stack (if applicable)
                connector_name = f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}"
                if map_entry is not None and connector_name in map_entry.out_connectors:
                    self._current_state.add_edge(map_entry, connector_name, tasklet, name, memlet)
                    continue

                # cache read access
                if memlet.data not in cache:
                    cache[memlet.data] = self._current_state.add_read(memlet.data)

                access_node = cache[memlet.data]
                self._current_state.add_memlet_path(access_node, tasklet, dst_conn=name, memlet=memlet)

            # Add empty memlet if map_entry has no out_connectors to connect to
            if map_entry is not None and not map_entry.out_connectors and self._current_state.out_degree(map_entry) < 1:
                self._current_state.add_edge(map_entry, None, tasklet, None, memlet=Memlet())

            # Connect output memlets
            for name, memlet in node.out_memlets.items():
                # connect to dataflow_stack (if applicable)
                connector_name = f"{PREFIX_PASSTHROUGH_IN}{memlet.data}"
                if map_exit is not None and connector_name in map_exit.in_connectors:
                    self._current_state.add_edge(tasklet, name, map_exit, connector_name, memlet)
                    continue

                # we always write to a new access_node
                access_node = self._current_state.add_write(memlet.data)
                self._current_state.add_memlet_path(tasklet, access_node, src_conn=name, memlet=memlet)

                # cache write access node (or update an existing one) for read after write cases
                cache[memlet.data] = access_node

            # Add empty memlet if map_exit has no in_connectors to connect to
            if map_exit is not None and not map_exit.in_connectors and self._current_state.in_degree(map_exit) < 1:
                self._current_state.add_edge(tasklet, None, map_exit, None, memlet=Memlet())

        def visit_LibraryCall(self, node: tn.LibraryCall, sdfg: SDFG) -> None:
            # AFAIK we expand all library calls in the gt4py/dace bridge before coming here.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_CopyNode(self, node: tn.CopyNode, sdfg: SDFG) -> None:
            # AFAIK we don't support copy nodes in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_DynScopeCopyNode(self, node: tn.DynScopeCopyNode, sdfg: SDFG) -> None:
            # AFAIK we don't support dyn scope copy nodes in the gt4py/dace bridge.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_ViewNode(self, node: tn.ViewNode, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_NView(self, node: tn.NView, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_RefSetNode(self, node: tn.RefSetNode, sdfg: SDFG) -> None:
            # Let's see if we need this for the first prototype ...
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_StateBoundaryNode(self, node: tn.StateBoundaryNode, sdfg: SDFG) -> None:
            # When creating a state boundary, include all inter-state assignments that precede it.
            assignments = {}
            for symbol in self._interstate_symbols:
                assignments[symbol.name] = symbol.value
            self._interstate_symbols.clear()

            self._current_state = create_state_boundary(node, sdfg, self._current_state,
                                                        StateBoundaryBehavior.STATE_TRANSITION, assignments)

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


def create_state_boundary(bnode: tn.StateBoundaryNode,
                          sdfg_region: ControlFlowRegion,
                          state: SDFGState,
                          behavior: StateBoundaryBehavior,
                          assignments: Optional[Dict] = None) -> SDFGState:
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
    return sdfg_region.add_state_after(state, label=label, assignments=assignments)


def _list_index(list: List[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode) -> int:
    """Check if node is in list with "is" operator."""
    index = 0
    for element in list:
        # compare with "is" to get memory comparison. ".index()" uses value comparison
        if element is node:
            return index
        index += 1

    raise StopIteration
