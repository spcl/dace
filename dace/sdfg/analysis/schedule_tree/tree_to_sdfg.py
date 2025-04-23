# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from collections import defaultdict
from dace import dtypes, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes, memlet_utils as mmu
from dace.sdfg.sdfg import SDFG, ControlFlowRegion, InterstateEdge
from dace.sdfg.state import SDFGState
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg import propagation
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
            # List[ (MapEntryNode, ToConnect) | (SDFG, {"inputs": set(), "outputs": set()}) ]
            self._dataflow_stack: List[Tuple[nodes.EntryNode, Dict[str, Tuple[nodes.AccessNode, Memlet]]]
                                       | Tuple[SDFG, Dict[str, Set[str]]]] = []

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
            # see visitors below.
            self._interstate_symbols.append(node)

        def visit_ForScope(self, node: tn.ForScope, sdfg: SDFG) -> None:
            before_state = self._current_state
            pending = self._pending_interstate_assignments()
            pending[node.header.itervar] = node.header.init

            guard_state = _insert_and_split_assignments(sdfg, before_state, label="loop_guard", assignments=pending)
            self._current_state = guard_state

            body_state = sdfg.add_state(label="loop_body")
            self._current_state = body_state
            sdfg.add_edge(guard_state, body_state, InterstateEdge(condition=node.header.condition))

            # visit children inside the loop
            self.visit(node.children, sdfg=sdfg)

            pending = self._pending_interstate_assignments()
            pending[node.header.itervar] = node.header.update
            _insert_and_split_assignments(sdfg, self._current_state, after_state=guard_state, assignments=pending)

            after_state = sdfg.add_state(label="loop_after")
            self._current_state = after_state
            sdfg.add_edge(guard_state, after_state, InterstateEdge(condition=f"not {node.header.condition.as_string}"))

        def visit_WhileScope(self, node: tn.WhileScope, sdfg: SDFG) -> None:
            before_state = self._current_state
            guard_state = _insert_and_split_assignments(sdfg,
                                                        before_state,
                                                        label="guard_state",
                                                        assignments=self._pending_interstate_assignments())
            self._current_state = guard_state

            body_state = sdfg.add_state(label="loop_body")
            self._current_state = body_state
            sdfg.add_edge(guard_state, body_state, InterstateEdge(condition=node.header.test))

            # visit children inside the loop
            self.visit(node.children, sdfg=sdfg)
            _insert_and_split_assignments(sdfg,
                                          before_state=self._current_state,
                                          after_state=guard_state,
                                          assignments=self._pending_interstate_assignments())

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
            guard_state = _insert_and_split_assignments(sdfg,
                                                        before_state,
                                                        label="guard_state",
                                                        assignments=self._pending_interstate_assignments())

            # add true_state
            true_state = sdfg.add_state(label="true_state")
            sdfg.add_edge(guard_state, true_state, InterstateEdge(condition=node.condition))
            self._current_state = true_state

            # visit children in the true branch
            self.visit(node.children, sdfg=sdfg)

            # add merge_state
            merge_state = _insert_and_split_assignments(sdfg,
                                                        self._current_state,
                                                        label="merge_state",
                                                        assignments=self._pending_interstate_assignments())

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
            _insert_and_split_assignments(sdfg,
                                          before_state=self._current_state,
                                          after_state=merge_state,
                                          assignments=self._pending_interstate_assignments())
            self._current_state = merge_state

        def _insert_nestedSDFG(self, node: tn.MapScope, sdfg: SDFG) -> None:
            dataflow_stack_size = len(self._dataflow_stack)
            state_stack_size = len(self._state_stack)

            # prepare inner SDFG
            inner_sdfg = SDFG("nested_sdfg", parent=self._current_state)
            start_state = inner_sdfg.add_state("nested_root", is_start_block=True)

            # update stacks and current state
            old_state_label = self._current_state.label
            self._state_stack.append(self._current_state)
            self._dataflow_stack.append((inner_sdfg, {"inputs": set(), "outputs": set()}))
            self._current_state = start_state

            # visit children
            for child in node.children:
                self.visit(child, sdfg=inner_sdfg)

            # restore current state and stacks
            self._current_state = self._pop_state(old_state_label)
            assert len(self._state_stack) == state_stack_size
            _, connectors = self._dataflow_stack.pop()
            assert len(self._dataflow_stack) == dataflow_stack_size

            # insert nested SDFG
            nsdfg = self._current_state.add_nested_sdfg(inner_sdfg,
                                                        sdfg,
                                                        inputs=connectors["inputs"],
                                                        outputs=connectors["outputs"])

            # connect nested SDFG to surrounding map scope
            assert self._dataflow_stack
            map_entry, to_connect = self._dataflow_stack[-1]

            # connect nsdfg input memlets (to be propagated upon completion of the SDFG)
            for name in nsdfg.in_connectors:
                out_connector = f"{PREFIX_PASSTHROUGH_OUT}{name}"
                new_in_connector = map_entry.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{name}")
                new_out_connector = map_entry.add_out_connector(out_connector)
                assert new_in_connector == True
                assert new_in_connector == new_out_connector

                self._current_state.add_edge(map_entry, out_connector, nsdfg, name,
                                             Memlet.from_array(name, nsdfg.sdfg.arrays[name]))

            # Add empty memlet if we didn't add any in the loop above
            if self._current_state.out_degree(map_entry) < 1:
                self._current_state.add_nedge(map_entry, nsdfg, memlet=Memlet())

            # connect nsdfg output memlets (to be propagated)
            for name in nsdfg.out_connectors:
                to_connect[name] = (nsdfg, Memlet.from_array(name, nsdfg.sdfg.arrays[name]))

        def visit_MapScope(self, node: tn.MapScope, sdfg: SDFG) -> None:
            dataflow_stack_size = len(self._dataflow_stack)

            # map entry
            # ---------
            map_entry = nodes.MapEntry(node.node.map)
            self._current_state.add_node(map_entry)
            self._dataflow_stack.append((map_entry, dict()))

            # Set a new access_cache before visiting children such that they have their
            # own access cache (per map scope).
            access_cache = self._ensure_access_cache(self._current_state)
            self._access_cache[self._current_state] = {}

            # visit children inside the map
            if [type(child) for child in node.children] == [tn.StateBoundaryNode, tn.MapScope]:
                # skip weirdly added StateBoundaryNode
                self.visit(node.children[1], sdfg=sdfg)
            elif any([isinstance(child, tn.StateBoundaryNode) for child in node.children]):
                self._insert_nestedSDFG(node, sdfg)
            else:
                self.visit(node.children, sdfg=sdfg)

            # reset the access_cache
            self._access_cache[self._current_state] = access_cache

            # dataflow stack management
            _, to_connect = self._dataflow_stack.pop()
            assert len(self._dataflow_stack) == dataflow_stack_size
            outer_map_entry, outer_to_connect = self._dataflow_stack[-1] if dataflow_stack_size else (None, None)

            # connect potential input connectors on map_entry
            for connector in map_entry.in_connectors:
                memlet_data = connector.removeprefix(PREFIX_PASSTHROUGH_IN)

                # connect to local access node (if available)
                if memlet_data in access_cache:
                    cached_access = access_cache[memlet_data]
                    self._current_state.add_memlet_path(cached_access,
                                                        map_entry,
                                                        dst_conn=connector,
                                                        memlet=Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]))
                    continue

                if outer_map_entry is not None:
                    assert isinstance(outer_map_entry, nodes.EntryNode)

                    # get it from outside the map
                    connector_name = f"{PREFIX_PASSTHROUGH_OUT}{memlet_data}"
                    if connector_name not in outer_map_entry.out_connectors:
                        new_in_connector = outer_map_entry.add_in_connector(connector)
                        new_out_connector = outer_map_entry.add_out_connector(connector_name)
                        assert new_in_connector == True
                        assert new_in_connector == new_out_connector

                    self._current_state.add_edge(outer_map_entry, connector_name, map_entry, connector,
                                                 Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]))
                else:
                    # cache local read access
                    assert memlet_data not in access_cache
                    access_cache[memlet_data] = self._current_state.add_read(memlet_data)
                    cached_access = access_cache[memlet_data]
                    self._current_state.add_memlet_path(cached_access,
                                                        map_entry,
                                                        dst_conn=connector,
                                                        memlet=Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]))

            if outer_map_entry is not None and self._current_state.out_degree(outer_map_entry) < 1:
                self._current_state.add_nedge(outer_map_entry, map_entry, memlet=Memlet())

            # map_exit
            # --------
            map_exit = nodes.MapExit(node.node.map)
            self._current_state.add_node(map_exit)

            # connect writes to map_exit node
            for name in to_connect:
                in_connector_name = f"{PREFIX_PASSTHROUGH_IN}{name}"
                out_connector_name = f"{PREFIX_PASSTHROUGH_OUT}{name}"
                new_in_connector = map_exit.add_in_connector(in_connector_name)
                new_out_connector = map_exit.add_out_connector(out_connector_name)
                assert new_in_connector == new_out_connector

                # connect "inside the map"
                access_node, memlet = to_connect[name]
                if isinstance(access_node, nodes.NestedSDFG):
                    self._current_state.add_edge(access_node, name, map_exit, in_connector_name, memlet)
                else:
                    assert isinstance(access_node, nodes.AccessNode)
                    if self._current_state.out_degree(access_node) == 0 and self._current_state.in_degree(
                            access_node) == 1:
                        # this access_node is not used for anything else.
                        # let's remove it and add a direct connection instead
                        edges = [edge for edge in self._current_state.edges() if edge.dst == access_node]
                        assert len(edges) == 1
                        self._current_state.add_memlet_path(edges[0].src,
                                                            map_exit,
                                                            src_conn=edges[0].src_conn,
                                                            dst_conn=in_connector_name,
                                                            memlet=edges[0].data)
                        self._current_state.remove_node(access_node)  # edge is remove automatically
                    else:
                        self._current_state.add_memlet_path(access_node,
                                                            map_exit,
                                                            dst_conn=in_connector_name,
                                                            memlet=memlet)

                # connect "outside the map"
                access_node = self._current_state.add_write(name)
                self._current_state.add_memlet_path(map_exit,
                                                    access_node,
                                                    src_conn=out_connector_name,
                                                    memlet=Memlet.from_array(name, sdfg.arrays[name]))

                # cache write access into access_cache
                access_cache[name] = access_node

                if outer_to_connect is not None:
                    outer_to_connect[name] = (access_node, Memlet.from_array(name, sdfg.arrays[name]))

            # TODO If nothing is connected at this point, figure out what's the last thing that
            #      we should connect to. Then, add an empty memlet from that last thing to this
            #      map_exit.
            assert len(self._current_state.in_edges(map_exit)) > 0

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
            scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)

            # Connect input memlets
            for name, memlet in node.in_memlets.items():
                # connect to local access node if possible
                if memlet.data in cache:
                    cached_access = cache[memlet.data]
                    self._current_state.add_memlet_path(cached_access, tasklet, dst_conn=name, memlet=memlet)
                    continue

                if isinstance(scope_node, nodes.MapEntry):
                    # get it from outside the map
                    connector_name = f"{PREFIX_PASSTHROUGH_OUT}{memlet.data}"
                    if connector_name not in scope_node.out_connectors:
                        new_in_connector = scope_node.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{memlet.data}")
                        new_out_connector = scope_node.add_out_connector(connector_name)
                        assert new_in_connector == True
                        assert new_in_connector == new_out_connector

                    self._current_state.add_edge(scope_node, connector_name, tasklet, name, memlet)
                    continue

                if isinstance(scope_node, SDFG):
                    # Copy data descriptor from parent SDFG and add input connector
                    if memlet.data not in sdfg.arrays:
                        parent_sdfg = sdfg.parent.parent
                        while memlet.data not in parent_sdfg.arrays:
                            parent_sdfg = parent_sdfg.parent.parent
                            assert isinstance(parent_sdfg, SDFG)
                        sdfg.add_datadesc(memlet.data, parent_sdfg.arrays[memlet.data].clone())

                        # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                        if parent_sdfg.arrays[memlet.data].transient:
                            sdfg.arrays[memlet.data].transient = False
                            # TODO
                            # ... unless they are only ever used inside the nested SDFG, in which case
                            # we should delete them from the parent SDFG's array list.
                            # NOTE This can probably be done automatically by a cleanup pass in the end.
                            #      Something like DDE should be able to do this.

                        assert memlet.data not in to_connect["inputs"]
                        to_connect["inputs"].add(memlet.data)
                else:
                    assert scope_node is None

                # cache local read access
                assert memlet.data not in cache
                cache[memlet.data] = self._current_state.add_read(memlet.data)
                cached_access = cache[memlet.data]
                self._current_state.add_memlet_path(cached_access, tasklet, dst_conn=name, memlet=memlet)

            # Add empty memlet if map_entry has no out_connectors to connect to
            if isinstance(scope_node, nodes.MapEntry) and self._current_state.out_degree(scope_node) < 1:
                self._current_state.add_nedge(scope_node, tasklet, Memlet())

            # Connect output memlets
            for name, memlet in node.out_memlets.items():
                # we always write to a new access_node
                access_node = self._current_state.add_write(memlet.data)
                self._current_state.add_memlet_path(tasklet, access_node, src_conn=name, memlet=memlet)

                # cache write access node (or update an existing one) for read after write cases
                cache[memlet.data] = access_node

                if isinstance(scope_node, nodes.MapEntry):
                    # copy the memlet since we already used it in the memlet path above
                    to_connect[memlet.data] = (access_node, copy.deepcopy(memlet))
                    continue

                if isinstance(scope_node, SDFG):
                    if memlet.data not in sdfg.arrays:
                        parent_sdfg = sdfg.parent.parent
                        sdfg.add_datadesc(memlet.data, parent_sdfg.arrays[memlet.data].clone())

                        # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                        if parent_sdfg.arrays[memlet.data].transient:
                            sdfg.arrays[memlet.data].transient = False

                    # Add out_connector in any case if not yet present, e.g. write after read
                    to_connect["outputs"].add(memlet.data)

                else:
                    assert scope_node is None

        def visit_LibraryCall(self, node: tn.LibraryCall, sdfg: SDFG) -> None:
            # AFAIK we expand all library calls in the gt4py/dace bridge before coming here.
            raise NotImplementedError(f"{type(node)} not implemented")

        def visit_CopyNode(self, node: tn.CopyNode, sdfg: SDFG) -> None:
            # apparently we need this for the first prototype
            self._ensure_access_cache(self._current_state)
            access_cache = self._access_cache[self._current_state]

            # assumption source access may or may not yet exist (in this state)
            src_name = node.memlet.data
            source = access_cache[src_name] if src_name in access_cache else self._current_state.add_read(src_name)

            # assumption: target access node doesn't exist yet
            assert node.target not in access_cache
            target = self._current_state.add_write(node.target)

            self._current_state.add_memlet_path(source, target, memlet=node.memlet)

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
            pending = self._pending_interstate_assignments()

            self._current_state = create_state_boundary(node,
                                                        sdfg,
                                                        self._current_state,
                                                        StateBoundaryBehavior.STATE_TRANSITION,
                                                        assignments=pending)

        def _pending_interstate_assignments(self) -> Dict:
            """
            Return currently pending interstate assignments. Clears the cache.
            """
            assignments = {}

            for symbol in self._interstate_symbols:
                assignments[symbol.name] = symbol.value.as_string
            self._interstate_symbols.clear()

            return assignments

    StreeToSDFG().visit(stree, sdfg=result)

    propagation.propagate_memlets_sdfg(result)

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
    return _insert_and_split_assignments(sdfg_region, state, label=label, assignments=assignments)


def _insert_and_split_assignments(sdfg_region: ControlFlowRegion,
                                  before_state: SDFGState,
                                  after_state: Optional[SDFGState] = None,
                                  label: Optional[str] = None,
                                  assignments: Optional[Dict] = None) -> SDFGState:
    """
    Insert given assignments splitting them in case of potential race conditions.

    DaCe validation (currently) won't let us add multiple assignment with read after
    write pattern on the same edge. We thus split them over multiple state transitions
    (inserting empty states in between) to be safe.

    NOTE (later) This should be double-checked since python dictionaries preserve
                 insertion order since python 3.7 (which we rely on in this function
                 too). Depending on code generation it could(TM) be that we can
                 weaken (best case remove) the corresponding check from the sdfg
                 validator.
    """
    has_potential_race = False
    for key, value in assignments.items():
        syms = symbolic.free_symbols_and_functions(value)
        also_assigned = (syms & assignments.keys()) - {key}
        if also_assigned:
            has_potential_race = True
            break

    if not has_potential_race:
        if after_state is not None:
            sdfg_region.add_edge(before_state, after_state, InterstateEdge(assignments=assignments))
            return after_state
        return sdfg_region.add_state_after(before_state, label=label, assignments=assignments)

    last_state = before_state
    for index, assignment in enumerate(assignments.items()):
        key, value = assignment
        is_last_state = index == len(assignments) - 1
        if is_last_state and after_state is not None:
            sdfg_region.add_edge(last_state, after_state, InterstateEdge(assignments={key: value}))
            last_state = after_state
        else:
            last_state = sdfg_region.add_state_after(last_state, label=label, assignments={key: value})

    return last_state


def _list_index(list: List[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode) -> int:
    """Check if node is in list with "is" operator."""
    index = 0
    for element in list:
        # compare with "is" to get memory comparison. ".index()" uses value comparison
        if element is node:
            return index
        index += 1

    raise StopIteration
