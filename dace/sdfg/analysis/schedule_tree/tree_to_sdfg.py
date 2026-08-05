# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from types import TracebackType
from typing import Final, Optional, Sequence

from dace import data, dtypes, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes, memlet_utils as mmu
from dace.sdfg.sdfg import SDFG, ControlFlowRegion, InterstateEdge
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, LoopRegion, NamedRegion,
                             ReturnBlock, SDFGState)
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg import propagation
from dace.transformation.passes.write_conflict_resolution import ResolveWriteConflicts


class StateBoundaryBehavior(Enum):
    STATE_TRANSITION = auto()  #: Creates multiple states with a state transition
    EMPTY_MEMLET = auto()  #: Happens-before empty memlet edges in the same state


PREFIX_PASSTHROUGH_IN: Final[str] = "IN_"
PREFIX_PASSTHROUGH_OUT: Final[str] = "OUT_"


@dataclass
class _Context:
    """Context information for transforming a schedule tree into an SDFG."""

    root: tn.ScheduleTreeRoot
    current_scope: tn.ScheduleTreeScope | None

    access_cache: dict[tuple[SDFGState, int], dict[str, nodes.AccessNode]]
    """Per scope (hashed by id(scope_node) access_cache."""


class _TreeScope:
    """Automatically set the current scope on the context to the given node."""

    def __init__(self, node: tn.ScheduleTreeScope, ctx: _Context, state: SDFGState) -> None:
        if ctx.current_scope is None and not isinstance(node, tn.ScheduleTreeRoot):
            raise ValueError("ctx.current_scope is only allowed to be 'None' when node it tree root.")

        self._ctx = ctx
        self._parent_scope = ctx.current_scope
        self._node = node
        self._state = state

        cache_key = (state, id(node))
        assert cache_key not in self._ctx.access_cache
        self._ctx.access_cache[cache_key] = {}

    def __enter__(self) -> None:
        assert not self._ctx.access_cache[(self._state, id(
            self._node))], "Expecting an empty access_cache when entering the context."

        self._ctx.current_scope = self._node

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        cache_key = (self._state, id(self._node))
        assert cache_key in self._ctx.access_cache

        self._ctx.current_scope = self._parent_scope


class _StreeToSDFG(tn.ScheduleNodeVisitor):

    def __init__(
        self,
        *,
        start_state: SDFGState | None = None,
        boundary_behavior: StateBoundaryBehavior = StateBoundaryBehavior.STATE_TRANSITION,
        max_nested_sdfg: int = 1000,
    ) -> None:
        if boundary_behavior != StateBoundaryBehavior.STATE_TRANSITION:
            raise NotImplementedError("Only STATE_TRANSITION is currently supported as StateBoundaryBehavior.")

        self._ctx: _Context
        """Context information like tree root and current scope."""

        self._current_state = start_state
        """Current SDFGState in the SDFG that we are building."""

        self._current_nestedSDFG: int | None = None
        """Id of the current nested SDFG if we are inside one."""

        self._interstate_symbols: list[tn.AssignNode] = []
        """Interstate symbol assignments. Will be assigned with the next state transition."""

        self._view_bindings: dict[str, tn.ViewNode] = {}
        """View container name -> its ViewNode binding; resolved to viewing
        edges per state by _connect_view_edges after traversal."""

        self._nviews_free: list[tn.NView] = []
        """Keep track of NView (nested SDFG view) nodes that are "free" to be used."""

        self._nviews_bound_per_scope: dict[int, list[tn.NView]] = {}
        """Mapping of id(SDFG) -> list of active NView nodes in that SDFG."""

        self._nviews_deferred_removal: dict[int, list[tn.NView]] = {}
        """"Mapping of id(SDFG) -> list of NView nodes to be removed once we exit this nested SDFG."""

        # state management
        self._state_stack: list[SDFGState] = []

        # dataflow scopes
        # list[ (MapEntryNode, ToConnect) | (SDFG, {"inputs": set(), "outputs": set()}) ]
        self._dataflow_stack: list[tuple[nodes.EntryNode, dict[str, tuple[nodes.AccessNode, Memlet]]]
                                   | tuple[SDFG, dict[str, set[str]]]] = []

        self._consume_streams: dict[int, str] = {}
        """Consumed stream container per ConsumeEntry (keyed by id(entry)),
        so tasklet inputs reading the stream route through OUT_stream."""

        self._pending_dynamic_inputs: dict[str, Memlet] = {}
        """Dynamic map-range input memlets (from DynScopeCopyNode siblings
        emitted right before a MapScope) that are still waiting to be wired to
        a map entry's dynamic (unprefixed) input connector, keyed by target
        symbol name."""

        self._max_nested_sdfg = max_nested_sdfg

        self._body_local: list[set[str]] = []
        """Per nested-SDFG body: the transients used only within it, which stay
        local to that SDFG instead of becoming connectors (see
        ``_body_local_transients``)."""

    def _apply_nview_array_override(self, array_name: str, sdfg: SDFG) -> bool:
        """
        Apply an NView override if applicable. Returns true if the NView was applied.

        See `visit_NView()` for how we keep track of nested SDFG view nodes.
        """
        length = len(self._nviews_free)
        for index, nview in enumerate(reversed(self._nviews_free), start=1):
            if nview.target == array_name and nview not in self._nviews_deferred_removal[id(sdfg)]:
                # Add the "override" data descriptor
                sdfg.add_datadesc(nview.target, nview.view_desc.clone())
                if nview.src_desc.transient:
                    sdfg.arrays[nview.target].transient = False

                # Keep track of used NViews per scope (to "free" them again once the scope ends)
                self._nviews_bound_per_scope[id(sdfg)].append(nview)

                # This NView is in use now, remove it from the free NViews.
                del self._nviews_free[length - index]
                return True

        return False

    def _parent_sdfg_with_array(self, name: str, sdfg: SDFG) -> SDFG:
        """Find the closest parent SDFG containing an array with the given name."""
        parent_sdfg = sdfg.parent.sdfg
        sdfg_counter = 1
        while name not in parent_sdfg.arrays and sdfg_counter < self._max_nested_sdfg:
            parent_sdfg = parent_sdfg.parent.sdfg
            assert isinstance(parent_sdfg, SDFG)
            sdfg_counter += 1
        assert sdfg_counter < self._max_nested_sdfg, f"Array '{name}' not found in any parent of SDFG '{sdfg.name}'."
        return parent_sdfg

    def _pop_state(self, label: str | None = None) -> SDFGState:
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

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot, sdfg: SDFG) -> None:
        assert self._current_state is None, "Expected no 'current_state' at root."
        assert not self._state_stack, "Expected empty state stack at root."
        assert not self._dataflow_stack, "Expected empty dataflow stack at root."
        assert not self._interstate_symbols, "Expected empty list of symbols at root."

        self._current_state = sdfg.add_state(label="tree_root", is_start_block=True)
        self._ctx = _Context(root=node, access_cache={}, current_scope=None)
        with _TreeScope(node, self._ctx, self._current_state):
            self.visit(node.children, sdfg=sdfg)

        assert not self._state_stack, "Expected empty state stack."
        assert not self._dataflow_stack, "Expected empty dataflow stack."
        assert not self._interstate_symbols, "Expected empty list of symbols to add."

    def visit_GBlock(self, node: tn.GBlock, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for {type(node)} not yet implemented.")

    def visit_StateLabel(self, node: tn.StateLabel, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for {type(node)} not yet implemented.")

    def visit_GotoNode(self, node: tn.GotoNode, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for{type(node)} not yet implemented.")

    def visit_AssignNode(self, node: tn.AssignNode, sdfg: SDFG) -> None:
        # We just collect them here. They'll be added when state boundaries are added,
        # see visitors below.
        self._interstate_symbols.append(node)

        # If AssignNode depends on arrays, e.g. `my_sym = my_array[__k] > 0`, make sure array accesses can be resolved.
        input_memlets = node.input_memlets()
        if not input_memlets:
            return

        for entry in reversed(self._dataflow_stack):
            scope_node, to_connect = entry
            if isinstance(scope_node, SDFG):
                # In case we are inside a nested SDFG, make sure memlet data can be
                # resolved by explicitly adding inputs.
                for memlet in input_memlets:
                    # Copy data descriptor from parent SDFG and add input connector
                    if memlet.data not in sdfg.arrays:
                        parent_sdfg = self._parent_sdfg_with_array(memlet.data, sdfg)

                        # Support for NView nodes
                        use_nview = self._apply_nview_array_override(memlet.data, sdfg)
                        if not use_nview:
                            sdfg.add_datadesc(memlet.data, parent_sdfg.arrays[memlet.data].clone())

                            # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                            if parent_sdfg.arrays[memlet.data].transient:
                                sdfg.arrays[memlet.data].transient = False

                        # Dev note: nview.target and memlet.data are identical
                        assert memlet.data not in to_connect["inputs"]
                        to_connect["inputs"].add(memlet.data)
                return

        for memlet in input_memlets:
            # If we aren't inside a nested SDFG, make sure all memlets can be resolved.
            # Imo, this should always be the case. It not, raise an error.
            if memlet.data not in sdfg.arrays:
                raise ValueError(f"Parsing AssignNode {node} failed. Can't find {memlet.data} in {sdfg}.")

    def _loop_state_name_prefix(self, node: tn.ForScope | tn.WhileScope) -> str:
        if isinstance(node, tn.ForScope):
            return "for"

        if isinstance(node, tn.WhileScope):
            return "while"

        raise NotImplementedError(f"Loop state name prefix not implemented for loop of type {type(node)}.")

    def _add_loop_region(self, node: tn.ForScope | tn.WhileScope, sdfg: SDFG) -> None:
        current_state = self._current_state
        assert current_state is not None  # just to keep pyright happy
        cf_region = current_state.parent_graph

        loop_region = LoopRegion(
            label=node.loop.label,
            condition_expr=node.loop.loop_condition,
            loop_var=node.loop.loop_variable,
            initialize_expr=node.loop.init_statement,
            update_expr=node.loop.update_statement,
            unroll=node.loop.unroll,
            unroll_factor=node.loop.unroll_factor,
            inverted=node.loop.inverted,
            update_before_condition=node.loop.update_before_condition,
        )

        memlets = loop_region.get_meta_read_memlets(self._ctx.root.containers, include_scalars=True)
        self._ensure_data_descriptors(memlets, sdfg)

        cf_region.add_node(loop_region, ensure_unique_name=True)
        prefix = self._loop_state_name_prefix(node)
        loop_state = loop_region.add_state(f"{prefix}_loop_state_{id(node)}", is_start_block=True)

        _insert_and_split_assignments(current_state, loop_region)

        self._current_state = loop_state
        self.visit(node.children, sdfg=sdfg)

        after_state = _insert_and_split_assignments(loop_region, label=f"{prefix}_loop_after")
        self._current_state = after_state

    def visit_ForScope(self, node: tn.ForScope, sdfg: SDFG) -> None:
        self._add_loop_region(node, sdfg)

    def visit_WhileScope(self, node: tn.WhileScope, sdfg: SDFG) -> None:
        self._add_loop_region(node, sdfg)

    def visit_DoWhileScope(self, node: tn.DoWhileScope, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for {type(node)} not yet implemented.")

    def visit_LoopScope(self, node: tn.LoopScope, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for {type(node)} not yet implemented.")

    def _ensure_data_descriptors(self, memlets: Sequence[Memlet], sdfg: SDFG) -> None:
        scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)
        if isinstance(scope_node, SDFG):
            for memlet in memlets:
                # Copy data descriptor from parent SDFG and add input connector
                if memlet.data not in sdfg.arrays:
                    parent_sdfg = self._parent_sdfg_with_array(memlet.data, sdfg)

                    # Support for  NView nodes
                    use_nview = self._apply_nview_array_override(memlet.data, sdfg)
                    if not use_nview:
                        sdfg.add_datadesc(memlet.data, parent_sdfg.arrays[memlet.data].clone())

                        # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                        if parent_sdfg.arrays[memlet.data].transient:
                            sdfg.arrays[memlet.data].transient = False

                    # Dev note: memlet.data and nview.target are identical
                    assert memlet.data not in to_connect["inputs"]
                    to_connect["inputs"].add(memlet.data)

    def visit_IfScope(self, node: tn.IfScope, sdfg: SDFG) -> None:
        before_state = self._current_state
        assert before_state is not None
        cf_region = before_state.parent_graph

        conditional_block = ConditionalBlock(f"if_scope_{id(node)}")
        cf_region.add_node(conditional_block)
        _insert_and_split_assignments(
            before_state,
            conditional_block,
            assignments=self._pending_interstate_assignments(),
        )

        if_body = ControlFlowRegion("if_body", sdfg=sdfg)
        conditional_block.add_branch(node.condition, if_body)

        memlets = conditional_block.get_meta_read_memlets(self._ctx.root.containers, include_scalars=True)
        self._ensure_data_descriptors(memlets, sdfg)

        if_state = if_body.add_state("if_state", is_start_block=True)
        self._current_state = if_state

        # visit children of that branch
        self.visit(node.children, sdfg=sdfg)

        self._current_state = conditional_block

        # add merge_state
        merge_state = _insert_and_split_assignments(
            conditional_block,
            label="merge_state",
            assignments=self._pending_interstate_assignments(),
        )

        # Check if there's an `ElifScope`/`ElseScope` following this node (in the parent's children).
        # Filter StateBoundaryNodes, which we inserted earlier, for this analysis.
        if _has_branch_continuation(node):
            # push merge_state on the stack for later usage in `visit_ElifScope`/`visit_ElseScope`
            self._state_stack.append(merge_state)
            # push condition_block on the stack for later usage in `visit_ElifScope`/`visit_ElseScope`
            self._state_stack.append(conditional_block)
        else:
            self._current_state = merge_state

    def visit_StateIfScope(self, node: tn.StateIfScope, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for {type(node)} not yet implemented.")

    def visit_BreakNode(self, node: tn.BreakNode, sdfg: SDFG) -> None:
        self._insert_exit_block(BreakBlock(f"break_{id(node)}"))

    def visit_ContinueNode(self, node: tn.ContinueNode, sdfg: SDFG) -> None:
        self._insert_exit_block(ContinueBlock(f"continue_{id(node)}"))

    def _insert_exit_block(self, block: ControlFlowBlock) -> None:
        """Insert a control-flow exit block (break/continue/return) after the
        current state. Statements following the exit on the same path are dead
        code; they attach to a fresh successor of the block."""
        current_state = self._current_state
        assert current_state is not None
        cf_region = current_state.parent_graph
        cf_region.add_node(block)
        cf_region.add_edge(current_state, block, InterstateEdge(assignments=self._pending_interstate_assignments()))
        after = cf_region.add_state(f"after_{block.label}")
        cf_region.add_edge(block, after, InterstateEdge())
        self._current_state = after

    def visit_ElifScope(self, node: tn.ElifScope, sdfg: SDFG) -> None:
        # An additional conditional branch of the preceding if-chain
        conditional_block: ConditionalBlock = self._pop_state("if_scope")

        elif_body = ControlFlowRegion(f"elif_body_{id(node)}", sdfg=sdfg)
        conditional_block.add_branch(node.condition, elif_body)

        elif_state = elif_body.add_state("elif_state", is_start_block=True)
        self._current_state = elif_state

        self.visit(node.children, sdfg=sdfg)

        if self._pending_interstate_assignments():
            raise NotImplementedError("TODO: update edge with new assignments")

        if _has_branch_continuation(node):
            # Another elif/else follows: keep the block available (merge_state stays below it)
            self._state_stack.append(conditional_block)
        else:
            merge_state = self._pop_state("merge_state")
            self._current_state = merge_state

    def visit_ElseScope(self, node: tn.ElseScope, sdfg: SDFG) -> None:
        # get ConditionalBlock from stack
        conditional_block: ConditionalBlock = self._pop_state("if_scope")

        else_body = ControlFlowRegion("else_body", sdfg=sdfg)
        conditional_block.add_branch(None, else_body)

        else_state = else_body.add_state("else_state", is_start_block=True)
        self._current_state = else_state

        # visit children inside the else branch
        self.visit(node.children, sdfg=sdfg)

        # merge false-branch into merge_state
        merge_state = self._pop_state("merge_state")
        self._current_state = merge_state

        if self._pending_interstate_assignments():
            raise NotImplementedError("TODO: update edge with new assignments")

    def _insert_nestedSDFG_in_MapScope(self, node: tn.MapScope, sdfg: SDFG) -> None:
        dataflow_stack_size = len(self._dataflow_stack)
        state_stack_size = len(self._state_stack)
        outer_nestedSDFG = self._current_nestedSDFG

        # prepare inner SDFG
        inner_sdfg = SDFG("nested_sdfg", parent=self._current_state)
        start_state = inner_sdfg.add_state("nested_root", is_start_block=True)

        # update stacks and current state
        old_state_label = self._current_state.label
        self._state_stack.append(self._current_state)
        self._dataflow_stack.append((inner_sdfg, {"inputs": set(), "outputs": set()}))
        self._nviews_bound_per_scope[id(inner_sdfg)] = []
        self._nviews_deferred_removal[id(inner_sdfg)] = []
        self._current_nestedSDFG = id(inner_sdfg)
        self._current_state = start_state

        # Transients this body alone uses stay INSIDE it rather than becoming
        # connectors: routing a per-statement staging temporary through the map
        # entry and exit would make every iteration share one instance of it.
        self._body_local.append(_body_local_transients(node, self._ctx.root, sdfg))

        # visit children
        with _TreeScope(node, self._ctx, self._current_state):
            self.visit(node.children, sdfg=inner_sdfg)

        self._body_local.pop()

        # restore current state and stacks
        self._current_state = self._pop_state(old_state_label)
        assert len(self._state_stack) == state_stack_size
        _, connectors = self._dataflow_stack.pop()
        assert len(self._dataflow_stack) == dataflow_stack_size

        # Views bound inside this body are resolved here, while the body is
        # still the SDFG holding them: the top-level pass at the end of
        # conversion only walks the outermost SDFG's own states.
        _connect_view_edges(inner_sdfg, self._view_bindings)

        # Every non-transient the body actually touches must be a connector.
        # Deriving that from the emitted access nodes covers containers the
        # per-node visitors could not have registered themselves: the source of
        # a view resolved just above, and whatever a replacement expansion
        # wrote (``dace.reduce(f, X_in, X_out)`` writes an ARGUMENT).
        for state in inner_sdfg.all_states():
            for access in state.data_nodes():
                if inner_sdfg.arrays[access.data].transient:
                    continue
                if state.in_degree(access) > 0:
                    connectors["outputs"].add(access.data)
                if state.out_degree(access) > 0:
                    connectors["inputs"].add(access.data)

        # insert nested SDFG
        nsdfg = self._current_state.add_nested_sdfg(
            sdfg=inner_sdfg,
            inputs=connectors["inputs"],
            outputs=connectors["outputs"],
        )
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

            # Add Memlet for NView node (if applicable)
            edge_added = False
            for nview in self._nviews_bound_per_scope[id(inner_sdfg)]:
                if name == nview.target:
                    self._current_state.add_edge(map_entry, out_connector, nsdfg, name,
                                                 Memlet.from_memlet(nview.memlet))
                    edge_added = True
                    break

            if not edge_added:
                self._current_state.add_edge(map_entry, out_connector, nsdfg, name,
                                             Memlet.from_array(name, nsdfg.sdfg.arrays[name]))

        # Add empty memlet if we didn't add any in the loop above
        if self._current_state.out_degree(map_entry) < 1:
            self._current_state.add_nedge(map_entry, nsdfg, Memlet())

        # connect nsdfg output memlets (to be propagated)
        for name in nsdfg.out_connectors:
            # Add memlets for NView node (if applicable)
            edge_added = False
            for nview in self._nviews_bound_per_scope[id(inner_sdfg)]:
                if name == nview.target:
                    to_connect[name] = (nsdfg, Memlet.from_memlet(nview.memlet))
                    edge_added = True
                    break

            if not edge_added:
                to_connect[name] = (nsdfg, Memlet.from_array(name, nsdfg.sdfg.arrays[name]))

        # Move NViews back to "free" NViews for usage in a sibling scope.
        for nview in self._nviews_bound_per_scope[id(inner_sdfg)]:
            # If this NView ended in the current nested SDFG, don't add it back to the
            # "free NView" nodes. We need to keep it alive until here to make sure that
            # we can add the memlets above.
            if nview in self._nviews_deferred_removal[id(inner_sdfg)]:
                continue
            self._nviews_free.append(nview)

        del self._nviews_bound_per_scope[id(inner_sdfg)]
        del self._nviews_deferred_removal[id(inner_sdfg)]

        # Restore current nested SDFG
        self._current_nestedSDFG = outer_nestedSDFG

    def _connect_map_input(self, map_entry: nodes.MapEntry, connector: str, memlet_data: str, memlet: Memlet,
                           outer_map_entry, outer_to_connect, access_cache: dict, sdfg: SDFG) -> None:
        """
        Source one input connector of a map entry: from a local access node
        when the data was produced in the enclosing scope, through the
        enclosing map entry's IN_/OUT_ passthrough connectors when nested, or
        from a (cached) state-level read otherwise — registering the read on a
        nested-SDFG boundary when one encloses the map.
        """
        # connect to local access node (if available)
        if memlet_data in access_cache:
            self._current_state.add_memlet_path(access_cache[memlet_data], map_entry, dst_conn=connector, memlet=memlet)
            return

        if isinstance(outer_map_entry, nodes.EntryNode):
            # get it from outside the map
            connector_name = f"{PREFIX_PASSTHROUGH_OUT}{memlet_data}"
            if connector_name not in outer_map_entry.out_connectors:
                new_in_connector = outer_map_entry.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{memlet_data}")
                new_out_connector = outer_map_entry.add_out_connector(connector_name)
                assert new_in_connector == True
                assert new_in_connector == new_out_connector

            self._current_state.add_edge(outer_map_entry, connector_name, map_entry, connector, memlet)
            return

        if isinstance(outer_map_entry, SDFG):
            # Make the container resolvable in the nested SDFG and add an input
            # connector for it, unless it stays local to the body.
            if memlet_data not in sdfg.arrays:
                if self._crosses_nested_boundary(memlet_data, sdfg):
                    # Dev note: nview.target and memlet_data are identical
                    assert memlet_data not in outer_to_connect["inputs"]
                    outer_to_connect["inputs"].add(memlet_data)
        else:
            assert outer_map_entry is None

        # cache local read access
        assert memlet_data not in access_cache
        access_cache[memlet_data] = self._current_state.add_read(memlet_data)
        self._current_state.add_memlet_path(access_cache[memlet_data], map_entry, dst_conn=connector, memlet=memlet)

    def _connect_scope_exit(self, exit_node, to_connect, outer_map_entry, outer_to_connect, access_cache,
                            sdfg: SDFG) -> None:
        """
        Connect the writes recorded in a dataflow scope's ``to_connect`` to its
        exit node (map or consume): IN_/OUT_ passthrough connectors on the
        exit, single-use in-scope access nodes collapsed into direct edges,
        outside write access nodes (re)cached, and nested-SDFG/outer-scope
        registration mirrored from the input side.
        """
        # connect writes to the scope exit node
        for name in to_connect:
            access_node, memlet = to_connect[name]
            # Special case: connect tasklets without outputs via an empty Memlet to the exit node.
            if isinstance(memlet, Memlet) and memlet.is_empty():
                self._current_state.add_nedge(access_node, exit_node, memlet)
                continue

            in_connector_name = f"{PREFIX_PASSTHROUGH_IN}{name}"
            out_connector_name = f"{PREFIX_PASSTHROUGH_OUT}{name}"
            new_in_connector = exit_node.add_in_connector(in_connector_name)
            new_out_connector = exit_node.add_out_connector(out_connector_name)
            assert new_in_connector == new_out_connector

            # connect inside the scope
            if isinstance(access_node, nodes.NestedSDFG):
                self._current_state.add_edge(access_node, name, exit_node, in_connector_name, memlet)
            else:
                assert isinstance(access_node, nodes.AccessNode)
                if self._current_state.out_degree(access_node) == 0 and self._current_state.in_degree(access_node) == 1:
                    # this access_node is not used for anything else.
                    # let's remove it and add a direct connection instead
                    edges = [edge for edge in self._current_state.edges() if edge.dst == access_node]
                    assert len(edges) == 1
                    self._current_state.add_memlet_path(edges[0].src,
                                                        exit_node,
                                                        src_conn=edges[0].src_conn,
                                                        dst_conn=in_connector_name,
                                                        memlet=edges[0].data)
                    self._current_state.remove_node(access_node)  # edge is remove automatically
                else:
                    self._current_state.add_memlet_path(access_node,
                                                        exit_node,
                                                        dst_conn=in_connector_name,
                                                        memlet=memlet)

            if isinstance(outer_map_entry, SDFG):
                # Add out_connector in any case if not yet present, e.g. write
                # after read -- but only for containers that actually cross the
                # body's boundary.
                # Dev note: name and nview.target are identical
                if self._crosses_nested_boundary(name, sdfg):
                    outer_to_connect["outputs"].add(name)

            # connect outside the scope
            # only re-use cached write-only nodes, e.g. don't create a cycle for
            # map i=0:20:
            #  A[i] = tasklet(A[i])
            if name not in access_cache or self._current_state.out_degree(access_cache[name]) > 0:
                # cache write access into access_cache
                write_access_node = self._current_state.add_write(name)
                access_cache[name] = write_access_node

            access_node = access_cache[name]
            # The container may live only in an enclosing SDFG (a write that
            # passes straight through this nested one).
            if name in sdfg.arrays:
                data_descriptor = sdfg.arrays[name]
            else:
                data_descriptor = self._parent_sdfg_with_array(name, sdfg).arrays[name]
            self._current_state.add_memlet_path(exit_node,
                                                access_node,
                                                src_conn=out_connector_name,
                                                memlet=Memlet.from_array(name, data_descriptor))

            if isinstance(outer_map_entry, nodes.EntryNode):
                outer_to_connect[name] = (access_node, Memlet.from_array(name, data_descriptor))
            else:
                assert isinstance(outer_map_entry, SDFG) or outer_map_entry is None

    def visit_MapScope(self, node: tn.MapScope, sdfg: SDFG) -> None:
        dataflow_stack_size = len(self._dataflow_stack)
        cache_state = self._current_state

        # map entry
        # ---------
        map_entry = nodes.MapEntry(node.node.map)
        self._current_state.add_node(map_entry)

        # Claim any dynamic map-range inputs (DynScopeCopyNode siblings emitted
        # right before this scope) whose target symbol appears in this map's
        # range. They get raw (unprefixed) input connectors carrying their
        # element memlets; the actual wiring happens with the other input
        # connectors after the children are visited, so the source routes
        # through enclosing scopes like any other read.
        dynamic_inputs: dict[str, Memlet] = {}
        range_symbols = {str(s) for s in node.node.map.range.free_symbols}
        for target in list(self._pending_dynamic_inputs.keys()):
            if target in range_symbols:
                dynamic_inputs[target] = self._pending_dynamic_inputs.pop(target)
                map_entry.add_in_connector(target)

        self._dataflow_stack.append((map_entry, dict()))

        # visit children inside the map
        type_of_children = [type(child) for child in node.children]
        last_child_is_MapScope = type_of_children[-1] == tn.MapScope
        all_others_are_Boundaries = type_of_children.count(tn.StateBoundaryNode) == len(type_of_children) - 1
        if last_child_is_MapScope and all_others_are_Boundaries:
            # skip weirdly added StateBoundaryNode
            # tmp: use this - for now - to "backprop-insert" extra state boundaries for nested SDFGs
            with _TreeScope(node, self._ctx, self._current_state):
                self.visit(node.children[-1], sdfg=sdfg)
        elif any([isinstance(child, tn.StateBoundaryNode) for child in node.children]):
            self._insert_nestedSDFG_in_MapScope(node, sdfg)
        else:
            with _TreeScope(node, self._ctx, self._current_state):
                self.visit(node.children, sdfg=sdfg)

        cache_key = (cache_state, id(self._ctx.current_scope))
        if cache_key not in self._ctx.access_cache:
            self._ctx.access_cache[cache_key] = {}
        access_cache = self._ctx.access_cache[cache_key]

        # dataflow stack management
        _, to_connect = self._dataflow_stack.pop()
        assert len(self._dataflow_stack) == dataflow_stack_size
        outer_map_entry, outer_to_connect = self._dataflow_stack[-1] if dataflow_stack_size else (None, None)

        # connect potential input connectors on map_entry
        for connector in map_entry.in_connectors:
            if connector in dynamic_inputs:
                # Dynamic map-range input: an unprefixed connector carrying an
                # element memlet, sourced like any other read (local access
                # node, enclosing scope passthrough, or state-level read).
                memlet = copy.deepcopy(dynamic_inputs[connector])
                self._connect_map_input(map_entry, connector, memlet.data, memlet, outer_map_entry, outer_to_connect,
                                        access_cache, sdfg)
                continue
            memlet_data = connector.removeprefix(PREFIX_PASSTHROUGH_IN)
            if memlet_data not in sdfg.arrays:
                # This map scope's OWN sdfg parameter may itself be a nested
                # SDFG (e.g. a dynamic-range inner map nested inside a static
                # outer one, via _insert_nestedSDFG_in_MapScope): a connector
                # can reference a container that exists only in an ENCLOSING
                # sdfg and was never cloned in here, because nothing has read
                # or written it directly at this level yet -- only referenced
                # it as a pass-through connector. Same "clone the descriptor
                # from the nearest parent that has it, and register it as a
                # nested-SDFG connector" idiom _connect_map_input's own
                # ``isinstance(outer_map_entry, SDFG)`` branch already applies
                # -- duplicated (not deferred to that call) because
                # ``Memlet.from_array`` below needs the descriptor to exist
                # BEFORE _connect_map_input ever runs, so its normal
                # "not yet present" gate never fires for this case.
                if isinstance(outer_map_entry, SDFG):
                    if self._crosses_nested_boundary(memlet_data, sdfg):
                        outer_to_connect["inputs"].add(memlet_data)
                elif not self._apply_nview_array_override(memlet_data, sdfg):
                    self._import_nested_datadesc(memlet_data, sdfg)
            self._connect_map_input(map_entry, connector, memlet_data,
                                    Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]), outer_map_entry,
                                    outer_to_connect, access_cache, sdfg)

        if isinstance(outer_map_entry, nodes.EntryNode) and self._current_state.out_degree(outer_map_entry) < 1:
            self._current_state.add_nedge(outer_map_entry, map_entry, Memlet())

        # map_exit
        # --------
        map_exit = nodes.MapExit(node.node.map)
        self._current_state.add_node(map_exit)

        self._connect_scope_exit(map_exit, to_connect, outer_map_entry, outer_to_connect, access_cache, sdfg)

        # TODO If nothing is connected at this point, figure out what's the last thing that
        #      we should connect to. Then, add an empty memlet from that last thing to this
        #      map_exit.
        assert len(self._current_state.in_edges(map_exit)) > 0

    def visit_ConsumeScope(self, node: tn.ConsumeScope, sdfg: SDFG) -> None:
        """
        Lower a consume scope: a fresh ConsumeEntry/ConsumeExit pair around the
        scope body. The consumed stream feeds the entry's fixed ``IN_stream``
        connector and reaches popped-element reads through ``OUT_stream`` (see
        visit_TaskletNode); all other reads and writes use the same IN_/OUT_
        passthrough machinery as map scopes.
        """
        # Leading boundaries reflect hazards against nodes BEFORE the scope
        # (e.g. the producer pushing into the consumed stream) and impose no
        # ordering within it; boundaries between scope children would require
        # nesting (as maps do) and are not supported yet.
        children = list(node.children)
        while children and isinstance(children[0], tn.StateBoundaryNode):
            children.pop(0)
        if any(isinstance(child, tn.StateBoundaryNode) for child in children):
            raise NotImplementedError('State boundaries inside consume scopes are not supported yet.')

        dataflow_stack_size = len(self._dataflow_stack)
        cache_state = self._current_state

        consume = copy.deepcopy(node.node.consume)
        entry = nodes.ConsumeEntry(consume)  # Adds the fixed IN_stream/OUT_stream connectors
        self._current_state.add_node(entry)

        stream_name = self._consumed_stream(node)
        self._consume_streams[id(entry)] = stream_name

        self._dataflow_stack.append((entry, dict()))
        with _TreeScope(node, self._ctx, self._current_state):
            self.visit(children, sdfg=sdfg)

        cache_key = (cache_state, id(self._ctx.current_scope))
        if cache_key not in self._ctx.access_cache:
            self._ctx.access_cache[cache_key] = {}
        access_cache = self._ctx.access_cache[cache_key]

        _, to_connect = self._dataflow_stack.pop()
        assert len(self._dataflow_stack) == dataflow_stack_size
        outer_map_entry, outer_to_connect = self._dataflow_stack[-1] if dataflow_stack_size else (None, None)

        # Source the entry's input connectors: the stream through IN_stream,
        # everything else through the shared per-connector machinery.
        for connector in list(entry.in_connectors):
            if connector == 'IN_stream':
                memlet = Memlet.from_array(stream_name, sdfg.arrays[stream_name])
                self._connect_map_input(entry, connector, stream_name, memlet, outer_map_entry, outer_to_connect,
                                        access_cache, sdfg)
                continue
            memlet_data = connector.removeprefix(PREFIX_PASSTHROUGH_IN)
            self._connect_map_input(entry, connector, memlet_data,
                                    Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]), outer_map_entry,
                                    outer_to_connect, access_cache, sdfg)

        if isinstance(outer_map_entry, nodes.EntryNode) and self._current_state.out_degree(outer_map_entry) < 1:
            self._current_state.add_nedge(outer_map_entry, entry, Memlet())

        consume_exit = nodes.ConsumeExit(consume)
        self._current_state.add_node(consume_exit)
        self._connect_scope_exit(consume_exit, to_connect, outer_map_entry, outer_to_connect, access_cache, sdfg)
        assert len(self._current_state.in_edges(consume_exit)) > 0

    def _consumed_stream(self, node: tn.ConsumeScope) -> str:
        """The stream container a consume scope pops from: the (single) stream
        read by the scope's tasklets."""
        for child in node.preorder_traversal():
            if not isinstance(child, tn.TaskletNode):
                continue
            for memlet in child.in_memlets.values():
                if isinstance(self._ctx.root.containers[memlet.data], data.Stream):
                    return memlet.data
        raise NotImplementedError('Consume scope without a stream read is not supported.')

    def visit_TaskletNode(self, node: tn.TaskletNode, sdfg: SDFG) -> None:
        # Add Tasklet to current state
        tasklet = node.node
        self._current_state.add_node(tasklet)

        cache_key = (self._current_state, id(self._ctx.current_scope))
        if cache_key not in self._ctx.access_cache:
            self._ctx.access_cache[cache_key] = {}
        cache = self._ctx.access_cache[cache_key]
        scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)

        # Connect input memlets
        for name, memlet in node.in_memlets.items():
            # connect to local access node if possible
            if memlet.data in cache:
                cached_access = cache[memlet.data]
                self._current_state.add_memlet_path(cached_access, tasklet, dst_conn=name, memlet=memlet)
                continue

            if isinstance(scope_node, nodes.ConsumeEntry) and memlet.data == self._consume_streams.get(id(scope_node)):
                # The consumed stream enters through the entry's fixed
                # OUT_stream connector (popped-element reads).
                self._current_state.add_edge(scope_node, 'OUT_stream', tasklet, name, memlet)
                continue

            if isinstance(scope_node, nodes.EntryNode):
                # get it from outside the scope
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
                    self._ensure_nested_container(memlet.data, sdfg)

                # A body-local transient stays inside this SDFG; only shared
                # containers become connectors.
                if not sdfg.arrays[memlet.data].transient:
                    to_connect["inputs"].add(memlet.data)
            else:
                assert scope_node is None

            # cache local read access
            assert memlet.data not in cache
            cache[memlet.data] = self._current_state.add_read(memlet.data)
            cached_access = cache[memlet.data]
            self._current_state.add_memlet_path(cached_access, tasklet, dst_conn=name, memlet=memlet)

        # Add an empty (control-only) memlet from the map entry if this tasklet has no data
        # inputs of its own (e.g. it only reads scope symbols, like a hoisted "i + 1"
        # computation). Without this, the tasklet would have in-degree zero and, despite
        # being nested inside the map, would be misclassified as a graph-level source node by
        # SDFGState.source_nodes() -- breaking scope_dict()/memlet propagation. Keying this off
        # the map_entry's own out-degree (rather than this tasklet's in-degree) would only catch
        # the case where this happens to be the very first child connected to the scope.
        if isinstance(scope_node, nodes.EntryNode) and self._current_state.in_degree(tasklet) < 1:
            self._current_state.add_nedge(scope_node, tasklet, Memlet())

        # Connect output memlets
        for name, memlet in node.out_memlets.items():
            # only re-use cached write-only nodes, e.g. don't create a cycle for
            # A[1] = tasklet(A[1])
            if memlet.data not in cache or self._current_state.out_degree(cache[memlet.data]) > 0:
                # cache write access node
                write_access_node = self._current_state.add_write(memlet.data)
                cache[memlet.data] = write_access_node

            access_node = cache[memlet.data]
            self._current_state.add_memlet_path(tasklet, access_node, src_conn=name, memlet=memlet)

            if isinstance(scope_node, nodes.EntryNode):
                # copy the memlet since we already used it in the memlet path above
                to_connect[memlet.data] = (access_node, copy.deepcopy(memlet))
                continue

            if isinstance(scope_node, SDFG):
                if memlet.data not in sdfg.arrays:
                    self._ensure_nested_container(memlet.data, sdfg)

                # Add out_connector in any case if not yet present, e.g. write
                # after read -- unless the container is body-local, in which
                # case it stays inside this SDFG.
                if not sdfg.arrays[memlet.data].transient:
                    to_connect["outputs"].add(memlet.data)

            else:
                assert scope_node is None

        # Add empty memlet if this tasklet is a sink node
        if isinstance(scope_node, nodes.MapEntry) and not node.out_memlets:
            to_connect[f"tasklet_{id(tasklet)}"] = (tasklet, Memlet())

    def visit_LibraryCall(self, node: tn.LibraryCall, sdfg: SDFG) -> None:
        raise NotImplementedError(f"Support for {type(node)} not yet implemented.")

    def visit_CopyNode(self, node: tn.CopyNode, sdfg: SDFG) -> None:
        # ensure we have an access_cache and fetch it
        cache_key = (self._current_state, id(self._ctx.current_scope))
        if cache_key not in self._ctx.access_cache:
            self._ctx.access_cache[cache_key] = {}
        access_cache = self._ctx.access_cache[cache_key]

        # both, source and target nodes, may or may not exist (in this state)
        src_name = node.memlet.data
        # Inside a nested SDFG (a map body) both containers may live in an
        # enclosing SDFG; an access node needs the descriptor to be present here.
        self._ensure_nested_container(src_name, sdfg)
        self._ensure_nested_container(node.target, sdfg)
        if src_name not in access_cache:
            # cache new read access
            access_cache[src_name] = self._current_state.add_read(src_name)
        source = access_cache[src_name]

        # Reuse a cached write-only access node for the target, the same
        # write-after-write-avoidance idiom used in _connect_scope_exit above
        # ("only re-use cached write-only nodes, e.g. don't create a cycle").
        # Without this, a LATER CopyNode reading this same target within the
        # same state (e.g. a computed intermediate immediately copied
        # elsewhere) creates a fresh, disconnected access node instead of
        # reading from the one just written -- invisible to execution order
        # (a data dependency is still established via the shared name and
        # per-state sequencing), but it makes the write look like a dead end
        # to node-local dataflow analyses such as DeadDataflowElimination,
        # which found and removed the "unread" producer entirely.
        # A self-copy (``field[5, 0, 0:73] = copy field[0, 0, 0:73]``) always
        # needs a fresh write node, or source and target would be one node.
        if (node.target not in access_cache or self._current_state.out_degree(access_cache[node.target]) > 0
                or src_name == node.target):
            # cache new write access node
            access_cache[node.target] = self._current_state.add_write(node.target)
        target = access_cache[node.target]

        self._current_state.add_memlet_path(source, target, memlet=node.memlet)

    def visit_DynScopeCopyNode(self, node: tn.DynScopeCopyNode, sdfg: SDFG) -> None:
        # A dynamic map-range input: emitted as a sibling immediately before
        # the scope (typically a MapScope) whose range uses ``node.target`` as
        # a symbol. We can't wire anything yet -- the scope's entry node
        # doesn't exist until that sibling is visited -- so stash the memlet,
        # keyed by the target symbol; the entry node's visitor sources it like
        # any other map input (see visit_MapScope/_connect_map_input).
        self._pending_dynamic_inputs[node.target] = node.memlet

    def visit_ViewNode(self, node: tn.ViewNode, sdfg: SDFG) -> None:
        # Views are aliasing bindings, not dataflow: record the binding here;
        # the viewing edges ('views' connector) are attached per state in a
        # post-pass (_connect_view_edges), mirroring the classic frontend.
        existing = self._view_bindings.get(node.target)
        if existing is not None and (existing.source != node.source or str(existing.memlet) != str(node.memlet)):
            raise NotImplementedError(f"Re-binding view '{node.target}' to a different subset is not supported yet.")
        self._view_bindings[node.target] = node

    def visit_NView(self, node: tn.NView, sdfg: SDFG) -> None:
        # Basic working principle:
        #
        # - NView and (artificial) NViewEnd nodes are added in parallel to mark the region where the view applies.
        # - Keep a stack of NView nodes (per name) that is pushed/popped when NView and NViewEnd nodes are visited.
        # - In between, when going "down into" a NestedSDFG, use the current NView (if it applies)
        # - In between, when "coming back up" from a NestedSDFG, pop the NView from the stack.
        # - AccessNodes will automatically pick up the right name (from the NestedSDFG's array list)
        self._nviews_free.append(node)

    def visit_NViewEnd(self, node: tn.NViewEnd, sdfg: SDFG) -> None:
        # If bound to the current nested SDFG, defer cleanup
        if self._current_nestedSDFG is not None:
            currently_bound = self._nviews_bound_per_scope[self._current_nestedSDFG]
            for index, nview in enumerate(reversed(currently_bound)):
                if node.target == nview.target:
                    # Bound to current nested SDFG. Slate for deferred removal once we exit that nested SDFG.
                    self._nviews_deferred_removal[self._current_nestedSDFG].append(nview)
                    return

        length = len(self._nviews_free)
        for index, nview in enumerate(reversed(self._nviews_free), start=1):
            if node.target == nview.target:
                # Stack semantics: remove from the back of the list
                del self._nviews_free[length - index]
                return

        raise RuntimeError(f"No matching NView found for target {node.target} in {self._nviews_free}.")

    def visit_RefSetNode(self, node: tn.RefSetNode, sdfg: SDFG) -> None:
        # A reference set is an access node of the Reference container with an
        # incoming edge on the 'set' connector, pointing to the referenced
        # subset. References persist across states once set.
        if isinstance(node.src_desc, nodes.CodeNode):
            raise NotImplementedError("Reference sets from code nodes are not yet supported.")
        if node.memlet is None:
            raise NotImplementedError("Reference sets without a source memlet are not yet supported.")
        if self._dataflow_stack:
            raise NotImplementedError("Reference sets inside dataflow scopes are not yet supported.")

        cache_key = (self._current_state, id(self._ctx.current_scope))
        cache = self._ctx.access_cache.setdefault(cache_key, {})
        source_name = node.memlet.data
        source = cache[source_name] if source_name in cache else self._current_state.add_read(source_name)
        target = self._current_state.add_write(node.target)
        self._current_state.add_edge(source, None, target, 'set', copy.deepcopy(node.memlet))
        cache.setdefault(source_name, source)
        # Later reads of the reference in this state must go through the set
        # node, so the set-then-read order is preserved by the dataflow.
        cache[node.target] = target

    def visit_StateBoundaryNode(self, node: tn.StateBoundaryNode, sdfg: SDFG) -> None:
        # When creating a state boundary, include all inter-state assignments that precede it.
        pending = self._pending_interstate_assignments()

        self._current_state = _create_state_boundary(
            node,
            self._current_state,
            assignments=pending,
        )

    def visit_NamedRegionScope(self, node: tn.NamedRegionScope, sdfg: SDFG) -> None:
        # A labeled grouping: it becomes a real NamedRegion so the label
        # survives for profiling and transformation targeting, but it
        # constrains nothing, so the body lowers into it unchanged.
        current_state = self._current_state
        assert current_state is not None
        cf_region = current_state.parent_graph

        named_region = NamedRegion(node.label)
        cf_region.add_node(named_region, ensure_unique_name=True)
        _insert_and_split_assignments(current_state, named_region, assignments=self._pending_interstate_assignments())

        self._current_state = named_region.add_state(f'named_region_state_{id(node)}', is_start_block=True)
        self.visit(node.children, sdfg=sdfg)

        self._current_state = _insert_and_split_assignments(named_region, label=f'named_region_after_{id(node)}')

    def visit_FunctionCallScope(self, node: tn.FunctionCallScope, sdfg: SDFG) -> None:
        # An inlined nested-program body: its contents lower transparently in
        # place (the frontend already resolved arguments to shared containers).
        # Early returns inside the scope are rejected by visit_ReturnNode.
        self.visit(node.children, sdfg=sdfg)

    def visit_ReturnNode(self, node: tn.ReturnNode, sdfg: SDFG) -> None:
        # Frontends materialize return values into their (non-transient)
        # return containers before this node, so a tail return at the end of
        # the program is a no-op, and an early return is a plain control-flow
        # exit (ReturnBlock). Returns inside FunctionCallScope mean "exit the
        # inlined callee", which has no direct control-flow equivalent yet.
        parent = node.parent
        if parent is not None and isinstance(parent, tn.ScheduleTreeRoot):
            index = next(i for i, child in enumerate(parent.children) if child is node)
            if all(isinstance(sibling, tn.StateBoundaryNode) for sibling in parent.children[index + 1:]):
                return
        ancestor = parent
        while ancestor is not None:
            if isinstance(ancestor, tn.FunctionCallScope):
                raise NotImplementedError(
                    "Returns from inlined nested programs are not yet supported in tree-to-SDFG conversion.")
            ancestor = ancestor.parent
        self._insert_exit_block(ReturnBlock(f"return_{id(node)}"))

    def visit_PythonCallbackNode(self, node: tn.PythonCallbackNode, sdfg: SDFG) -> None:
        """
        Lower a Python callback to the stable callback ABI: a callback-typed
        SDFG symbol (registered in ``sdfg.callback_mapping``) invoked from a
        tasklet with ``side_effects=True``, serialized against other callbacks
        through the ``__pystate`` container.
        """
        function_name = node.outlined_function_name
        if function_name is None:
            raise NotImplementedError("PythonCallbackNode without outlined scaffolding cannot be lowered.")
        if self._dataflow_stack:
            raise NotImplementedError("Python callbacks inside dataflow scopes are not supported.")

        input_types = [sdfg.arrays[name] for name in node.input_names]
        output_types = [sdfg.arrays[name] for name in node.output_names]
        if not output_types:
            return_type = None
        elif len(output_types) == 1:
            return_type = output_types[0]
        else:
            return_type = list(output_types)
        callback_type = dtypes.callback(return_type, *input_types)

        if function_name not in sdfg.symbols:
            sdfg.add_symbol(function_name, callback_type)
        root = self._ctx.root
        sdfg.callback_mapping.setdefault(function_name, root.callback_mapping.get(function_name, function_name))

        if '__pystate' not in sdfg.arrays:
            sdfg.add_scalar('__pystate', dtypes.int32, transient=True)

        # Callback ordering is enforced by state transitions around the call
        # in addition to the __pystate edges.
        self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state,
                                                     self._pending_interstate_assignments())

        input_connectors = {f'__in_{name}' for name in node.input_names} | {'__istate'}
        output_connectors = {f'__out_{name}' for name in node.output_names} | {'__ostate'}
        input_arguments = ', '.join(f'__in_{name}' for name in node.input_names)
        if callback_type.is_scalar_function() and len(callback_type.return_types) > 0:
            code = f'__out_{node.output_names[0]} = {function_name}({input_arguments})'
        else:
            all_arguments = [f'__in_{name}' for name in node.input_names]
            all_arguments.extend(f'__out_{name}' for name in node.output_names)
            code = f'{function_name}({", ".join(all_arguments)})'

        # The tasklet must survive dead-code elimination and never reorder,
        # even when it has no data outputs: side effects are external.
        tasklet = nodes.Tasklet(f'callback_{id(node)}', input_connectors, output_connectors, code, side_effects=True)
        tasklet.add_in_connector('__istate', dtypes.int32, force=True)
        tasklet.add_out_connector('__ostate', dtypes.int32, force=True)
        # Avoid casting output pointers to scalars in code generation
        for name in node.output_names:
            if tuple(sdfg.arrays[name].shape) == (1, ):
                tasklet._out_connectors[f'__out_{name}'] = dtypes.pointer(sdfg.arrays[name].dtype)

        in_memlets = {f'__in_{name}': Memlet.from_array(name, sdfg.arrays[name]) for name in node.input_names}
        in_memlets['__istate'] = Memlet.from_array('__pystate', sdfg.arrays['__pystate'])
        out_memlets = {f'__out_{name}': Memlet.from_array(name, sdfg.arrays[name]) for name in node.output_names}
        out_memlets['__ostate'] = Memlet.from_array('__pystate', sdfg.arrays['__pystate'])
        self.visit_TaskletNode(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets), sdfg)

        self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state, {})

    def visit_ReplacementCallNode(self, node: tn.ReplacementCallNode, sdfg: SDFG) -> None:
        """
        Expand a deferred frontend replacement call by invoking the registered
        replacement (:class:`~dace.frontend.common.op_repository.Replacements`)
        on the current state and copying its result into the target container.
        """
        from dace.frontend.common import op_repository as oprepo  # Deferred: frontend import from IR module
        import dace.frontend.python.replacements  # noqa: F401 -- importing populates the registry

        # An expansion builds states, so it can only run where states exist: at
        # the top level, or inside a body that was emitted as a nested SDFG.
        #
        # This is a DEFENSIVE check, not a capability limit: the state boundary
        # ``insert_state_boundaries`` puts before every replacement call is what
        # makes ``visit_MapScope`` emit the body as a nested SDFG in the first
        # place, and it cascades outward through nested maps
        # (``NestedSDFGStateBoundaryInserter``). The decision has to be made
        # there rather than here, because by the time this node is visited the
        # map entry and everything before it in the body are already emitted
        # into the outer state -- nesting them retroactively would mean undoing
        # that. Consume scopes reject the boundary earlier
        # (``visit_ConsumeScope``), so they never reach this either.
        if self._dataflow_stack and not isinstance(self._dataflow_stack[-1][0], SDFG):
            raise NotImplementedError("Replacement calls directly inside a dataflow scope are not supported.")
        self._import_replacement_data(node, sdfg)
        self._release_declared_descriptor(node.target, sdfg)

        # The expansion adds its own access nodes for the call's data
        # arguments. Those must not land in a state that already writes them
        # (the two subgraphs would be unordered, so the replacement would read
        # an uninitialized container), so start a fresh state first -- the
        # classic frontend's Call visitor does the same before invoking a
        # replacement.
        self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state,
                                                     self._pending_interstate_assignments())
        # States existing before the expansion, so the view bindings it records
        # can be materialized in exactly the states it creates (below).
        states_before = set(sdfg.all_states()) - {self._current_state}
        shim = ReplacementVisitorShim(sdfg, self._current_state, node.target, node.target_preexisting)
        if node.receiver_object is not None:
            # METHOD-family replacement on a compile-time OBJECT receiver
            # (``commworld.Bcast(A)``): keyed on the object's class, and the
            # implementation resolves the object back through the visitor's
            # globals under the name it was recorded with.
            shim.globals[node.receiver] = node.receiver_object
            receiver_type = type(node.receiver_object)
            function = oprepo.Replacements.get_method(receiver_type, node.qualname)
            if function is None:
                raise NotImplementedError(
                    f"No method replacement registered for '{node.qualname}' on '{receiver_type.__name__}'.")
            result = function(shim, sdfg, self._current_state, *node.arguments, **node.keyword_arguments)
        elif node.receiver is not None:
            # METHOD-family replacement (e.g. ``A.copy()``, see
            # ``lowering.dispatch._lower_replacement_call``'s method-call arm).
            receiver_type = type(sdfg.arrays[node.receiver])
            function = oprepo.Replacements.get_method(receiver_type, node.qualname)
            if function is None:
                raise NotImplementedError(
                    f"No method replacement registered for '{node.qualname}' on '{receiver_type.__name__}'.")
            result = function(shim, sdfg, self._current_state, *node.arguments, **node.keyword_arguments)
        elif node.ufunc_name is not None:
            # NumPy universal functions live in a separate registry keyspace
            # (keyed on the reduce/accumulate/outer method) with their own
            # calling convention: a single positional (ast_node, ufunc_name,
            # args, kwargs) rather than the generic *args/**kwargs unpacking.
            function = oprepo.Replacements.get_ufunc(node.ufunc_method)
            if function is None:
                raise NotImplementedError(f"No ufunc replacement registered for '{node.qualname}'.")
            result = function(shim, None, sdfg, self._current_state, node.ufunc_name, list(node.arguments),
                              dict(node.keyword_arguments))
        elif node.qualname.startswith(oprepo.OPERATOR_QUALNAME_MARKER):
            # OPERATOR-family replacement (``A @ B``, see
            # ``lowering.dispatch._lower_registry_operator``): looked up by the
            # AST operator and the operand CLASSES, which come from the
            # operands themselves rather than from the recorded name.
            from dace.frontend.python.nextgen.semantics.inference import operator_lookup_arguments
            optype = oprepo.decode_operator_qualname(node.qualname)
            values = [
                sdfg.arrays[argument] if argument in node.data_arguments else argument for argument in node.arguments
            ]
            function = oprepo.Replacements.getop(*operator_lookup_arguments(optype, values))
            if function is None:
                raise NotImplementedError(f"No operator replacement registered for '{node.qualname}'.")
            result = function(shim, sdfg, self._current_state, *node.arguments)
        else:
            function = oprepo.Replacements.get(node.qualname)
            if function is None and oprepo.ATTRIBUTE_QUALNAME_MARKER in node.qualname:
                # ATTRIBUTE-family replacement (e.g. ``Array.@T``, see
                # ``lowering.dispatch.resolve_attribute_data``): the free-function
                # keyspace lookup above never finds these, so decode the qualname
                # and look the implementation up by (classname, attr_name) instead.
                classname, _, attr_name = node.qualname.partition(oprepo.ATTRIBUTE_QUALNAME_MARKER)
                function = oprepo.Replacements.get_attribute(classname, attr_name)
            if function is None:
                raise NotImplementedError(f"No replacement registered for '{node.qualname}'.")
            result = function(shim, sdfg, self._current_state, *node.arguments, **node.keyword_arguments)

        # Multi-state replacements return a (NestedCall, result) pair and/or
        # advance the shim's last block; follow them.
        if isinstance(result, tuple) and len(result) == 2 and type(result[0]).__name__ == 'NestedCall':
            shim.last_block = result[0].last_state or shim.last_block
            result = result[1]
        created_states = [s for s in sdfg.all_states() if s not in states_before]
        # Some replacements chain states through a ``NestedCall`` but discard
        # the ``NestedCall`` object itself (e.g. ``_ndarray_argmax``, which
        # keeps only the result name), leaving ``shim.last_block`` at the HEAD
        # of the chain they built. Continuing from there would order everything
        # that follows before the replacement's own trailing writes -- an
        # uninitialized read. Follow the linear chain of states this expansion
        # created to its end instead.
        self._current_state = _end_of_expansion_chain(shim.last_block, set(created_states))

        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], str):
            # Ufunc implementations always return a single-element list of
            # output datanames (List[UfuncOutput]); normalize to the same
            # bare-string form the generic replacement convention uses below.
            result = result[0]
        elif node.extra_targets:
            # A multi-output replacement (``numpy.split``, ``numpy.divmod``):
            # one result container per declared target, in result order. The
            # frontend already checked the arity on a scratch expansion
            # (``dispatch._multi_output_viable``), so a mismatch here is a
            # frontend bug rather than a user-facing limitation.
            if not isinstance(result, (list, tuple)) or len(result) != len(node.targets):
                raise NotImplementedError(f"Replacement '{node.qualname}' returned {result!r}, which does not match "
                                          f'its {len(node.targets)} declared result containers.')
            if shim.views:
                _materialize_view_bindings(shim.views, created_states)
            for produced, declared in zip(result, node.targets):
                if produced == declared:
                    continue
                self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state, {})
                state = self._current_state
                state.add_nedge(state.add_read(produced), state.add_write(declared),
                                Memlet.from_array(produced, sdfg.arrays[produced]))
            self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state,
                                                         self._pending_interstate_assignments())
            return

        if shim.views:
            if isinstance(result, str) and result in shim.views:
                # The replacement's RESULT is a view of one of its inputs (e.g.
                # ``reshape``/``ravel`` on a contiguous array). Binding that as
                # a Python name is frontend state -- writes through the name
                # must reach the source array -- which a deferred call cannot
                # express; the frontend has its own view-binding path for these
                # (``lowering.dispatch._lower_reshape_call``).
                raise NotImplementedError(
                    f"Replacement '{node.qualname}' returns a view binding, which deferred expansion "
                    f"does not support.")
            # Views recorded on the way to a freshly computed result (e.g.
            # ``argmax``/``flatten``, which flatten their input through a view
            # first) are internal to the expansion: connect each one to its
            # source here, exactly as the classic frontend does at the end of
            # parsing, so the expansion's reads see real data.
            _materialize_view_bindings(shim.views, created_states)

        if isinstance(result, str) and result in sdfg.arrays:
            if result != node.target:
                # The replacement allocated its own result container: copy it
                # into the frontend-declared target (simplify collapses the
                # copy). The copy goes into a fresh state so it orders after
                # the replacement's own writes.
                self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state, {})
                state = self._current_state
                state.add_nedge(state.add_read(result), state.add_write(node.target),
                                Memlet.from_array(result, sdfg.arrays[result]))
            # else: an in-place-mutating replacement (e.g. a pure-side-effect
            # method call) returned its own target unchanged — the mutation
            # already landed in the right place, so there's nothing to copy.
        elif result is not None and result != []:
            raise NotImplementedError(
                f"Replacement '{node.qualname}' returned an unsupported result form: {type(result).__name__}")

        self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state,
                                                     self._pending_interstate_assignments())

    def _release_declared_descriptor(self, target: Optional[str], sdfg: SDFG) -> None:
        """
        Free a DECLARED descriptor's name before its replacement runs, so the
        replacement installs the real one under that name.

        A communicator (``ProcessGrid``, from ``MPI.COMM_WORLD.Create_cart``)
        is not storage a frontend can allocate: ``SDFG.add_pgrid`` is what
        creates it, together with its ``name`` property and its init/exit code.
        The frontend still registers a descriptor of the right TYPE under the
        target name, because that is what later method calls on the name
        resolve through (``pgrid.Bcast(A)`` is registered on ``ProcessGrid``,
        not on a scalar). Leaving that declaration in place would make
        ``add_pgrid`` find the name taken and install the real grid beside it,
        wiring the collectives to the uninitialized declaration.

        A descriptor whose ``name`` is set was installed by a replacement, or
        came from a real SDFG through ``sdfg_to_tree``, and is left alone.

        An opaque Python-object SCALAR is the same situation one step further
        out: it is how a frontend declares "a handle this replacement creates"
        when the real descriptor is not even a data descriptor
        (``dace.comm.Subarray``, whose result lives in ``sdfg.subarrays``).
        Leaving it in place made ``add_subarray`` find the name taken and
        install the subarray beside it, after which the ``Redistribute`` that
        consumes the handle by name could not find it.
        """
        if target is None or target not in sdfg.arrays:
            return
        descriptor = sdfg.arrays[target]
        if isinstance(descriptor, data.DistributedDescriptor) and not descriptor.name:
            sdfg.remove_data(target, validate=False)
        elif isinstance(descriptor, data.Scalar) and isinstance(descriptor.dtype, dtypes.pyobject):
            sdfg.remove_data(target, validate=False)

    def _import_replacement_data(self, node: tn.ReplacementCallNode, sdfg: SDFG) -> None:
        """
        Make a replacement call's containers resolvable in the SDFG it expands
        into: the replacement looks every argument up in ``sdfg.arrays``.
        """
        names = set(node.data_arguments) | set(node.targets)
        if node.receiver is not None:
            names.add(node.receiver)
        for name in sorted(names):
            self._ensure_nested_container(name, sdfg)

    def _ensure_nested_container(self, name: str, sdfg: SDFG) -> None:
        """
        Make a container resolvable inside a nested SDFG (a map body), cloning
        it in from the enclosing SDFG that owns it.

        Connectors are NOT registered here — the nested-SDFG builder derives
        them from the access nodes the body ends up with, which is the only way
        to know which containers it actually reads and writes. A no-op at the
        top level, where the containers already exist.
        """
        if not self._dataflow_stack or not isinstance(self._dataflow_stack[-1][0], SDFG):
            return
        if name in sdfg.arrays or self._apply_nview_array_override(name, sdfg):
            return
        binding = self._view_bindings.get(name)
        if binding is None and self._body_local and name in self._body_local[-1]:
            # Used only inside this body: keep it a transient of the nested
            # SDFG, so each scope instance gets its own copy instead of all
            # iterations sharing one through a connector.
            parent_sdfg = self._parent_sdfg_with_array(name, sdfg)
            descriptor = parent_sdfg.arrays[name].clone()
            descriptor.transient = True
            sdfg.add_datadesc(name, descriptor)
            return
        if binding is not None:
            # A view binding is a relationship between two containers, and its
            # subset generally depends on the enclosing map parameters
            # (``__anf0 = view A[i, 0:8]``). Passing the VIEW in would put its
            # viewing edge outside the map, where ``i`` does not exist, so
            # import the viewed SOURCE and rebuild the view inside, where the
            # map parameters are in scope (``_connect_view_edges`` attaches the
            # edge per state).
            self._import_nested_datadesc(binding.source, sdfg)
            sdfg.add_datadesc(name, binding.view_desc.clone())
            sdfg.arrays[name].transient = True
            return
        self._import_nested_datadesc(name, sdfg)

    def _crosses_nested_boundary(self, name: str, sdfg: SDFG) -> bool:
        """
        Make a container resolvable inside the nested SDFG a map body became,
        and report whether it crosses that body's boundary as a connector.

        Not everything a body touches does. A body-local transient must not:
        routing it through the scope's connectors would allocate ONE instance
        shared by every concurrent iteration. Neither must a view whose subset
        depends on the map parameters (``__anf0 = view A[0:20, i]``): its
        viewing edge would then be attached outside the map as well, where
        ``i`` does not exist. :meth:`_ensure_nested_container` keeps the first
        as a transient of the body and rebuilds the second inside it from its
        (imported) source, and a transient is never a connector -- so both
        report False.

        :param name: The container the body reads or writes.
        :param sdfg: The nested SDFG the body is being emitted into.
        :return: True if ``name`` must be registered on the body's boundary.
        """
        assert self._dataflow_stack and self._dataflow_stack[-1][0] is sdfg
        if name not in sdfg.arrays and self._apply_nview_array_override(name, sdfg):
            # An NView is a boundary contract by construction: the descriptor
            # it installs is exactly what the connector carries.
            return True
        self._ensure_nested_container(name, sdfg)
        return not sdfg.arrays[name].transient

    def _import_nested_datadesc(self, name: str, sdfg: SDFG) -> None:
        """Clone a container from the closest enclosing SDFG that has it into
        ``sdfg``, as the non-transient a nested-SDFG connector requires."""
        if name in sdfg.arrays:
            return
        parent_sdfg = self._parent_sdfg_with_array(name, sdfg)
        descriptor = parent_sdfg.arrays[name].clone()
        descriptor.transient = False
        sdfg.add_datadesc(name, descriptor)

    def visit_SDFGCallNode(self, node: tn.SDFGCallNode, sdfg: SDFG) -> None:
        """
        Lower an explicit SDFG-valued call to a nested SDFG node, connecting
        data arguments and return containers with full-range memlets and
        passing non-data arguments through the symbol mapping.
        """
        if self._dataflow_stack:
            raise NotImplementedError("SDFG calls inside dataflow scopes are not supported.")

        inner = copy.deepcopy(node.sdfg)
        connections: list[tuple[str, str]] = []  # (inner connector, outer container)
        symbol_mapping: dict[str, object] = {}
        for parameter, expression in node.call.arguments.items():
            if expression in sdfg.arrays and parameter in inner.arrays:
                connections.append((parameter, expression))
            else:
                symbol_mapping[parameter] = symbolic.pystr_to_symbolic(expression)

        return_arrays = sorted(name for name in inner.arrays if name.startswith('__return'))
        if len(return_arrays) < len(node.return_targets):
            raise NotImplementedError("SDFG call with more return targets than callee return containers.")

        # Without dataflow analysis of the callee, arguments conservatively
        # connect as both inputs and outputs; returns are outputs only.
        input_connectors = {parameter for parameter, _ in connections}
        output_connectors = input_connectors | set(return_arrays[:len(node.return_targets)])

        state = self._current_state
        nested = state.add_nested_sdfg(inner,
                                       inputs=input_connectors,
                                       outputs=output_connectors,
                                       symbol_mapping=symbol_mapping)
        for parameter, container in connections:
            state.add_edge(state.add_read(container), None, nested, parameter,
                           Memlet.from_array(container, sdfg.arrays[container]))
            state.add_edge(nested, parameter, state.add_write(container), None,
                           Memlet.from_array(container, sdfg.arrays[container]))
        for inner_name, target in zip(return_arrays, node.return_targets):
            state.add_edge(nested, inner_name, state.add_write(target), None,
                           Memlet.from_array(target, sdfg.arrays[target]))

        self._current_state = _create_state_boundary(tn.StateBoundaryNode(), self._current_state,
                                                     self._pending_interstate_assignments())

    def _pending_interstate_assignments(self) -> dict[str, str]:
        """
        Return currently pending interstate assignments. Clears the cache.
        """
        assignments = {}

        for symbol in self._interstate_symbols:
            assignments[symbol.name] = symbol.value.as_string
        self._interstate_symbols.clear()

        return assignments


def from_schedule_tree(
    stree: tn.ScheduleTreeRoot,
    state_boundary_behavior: StateBoundaryBehavior = StateBoundaryBehavior.STATE_TRANSITION,
    max_nested_sdfgs: int = 1000,
) -> SDFG:
    """
    Converts a schedule tree into an SDFG.

    :param stree: The schedule tree root to convert.
    :param state_boundary_behavior: Sets the behavior upon encountering a state boundary (e.g., write-after-write).
                                    See the ``StateBoundaryBehavior`` enumeration for more details.
    :return: An SDFG representing the schedule tree.
    """
    # Setup SDFG descriptor repository
    result = SDFG(stree.name, propagate=False)
    result.arg_names = copy.deepcopy(stree.arg_names)
    for key, container in stree.containers.items():
        result._arrays[key] = copy.deepcopy(container)
    # Opaque Python-object constants (callback namespace entries) have no
    # code-generatable representation and may not even be deep-copyable
    # (modules, callables); they stay on the tree root only. The same goes for
    # constants keyed by a non-identifier qualname (e.g. 'self.parameter'):
    # preprocessing folds their values inline, so nothing references them by
    # name, and the dotted name is not a valid C identifier.
    result.constants_prop = copy.deepcopy({
        name: entry
        for name, entry in stree.constants.items()
        if name.isidentifier() and not (isinstance(entry, tuple) and isinstance(entry[0], data.Data)
                                        and isinstance(entry[0].dtype, dtypes.pyobject))
    })
    result.callback_mapping = copy.deepcopy(stree.callback_mapping)
    # Callbacks the tree carries the source of become live callables here, so
    # the converted SDFG is callable without the caller re-deriving them. Live
    # objects, hence shared rather than copied.
    result.callback_objects = dict(stree.materialize_callbacks())
    # Frontend-produced trees store symbol *objects*; the SDFG symbol
    # repository stores their dtypes.
    result.symbols = {
        name: (value.dtype if isinstance(value, symbolic.symbol) else copy.deepcopy(value))
        for name, value in stree.symbols.items()
    }

    # Insert artificial state boundaries after WAW, before label, etc.
    stree = _insert_state_boundaries_to_tree(stree)

    # Traverse tree and incrementally build SDFG, finally propagate memlets
    visitor = _StreeToSDFG(boundary_behavior=state_boundary_behavior, max_nested_sdfg=max_nested_sdfgs)
    visitor.visit(stree, sdfg=result)
    _connect_view_edges(result, visitor._view_bindings)

    # Decide conflict resolution on the finished dataflow, before propagation
    # widens the per-iteration subsets the decision reads. Any producer of
    # parallel dataflow -- a frontend, a library node's expansion, a
    # transformation -- lands here, which is the point: whether a write
    # collides is a property of the graph, not of the syntax that produced it.
    ResolveWriteConflicts().apply_pass(result, {})

    propagation.propagate_memlets_sdfg(result)

    return result


def _connect_view_edges(sdfg: SDFG, bindings: 'dict[str, tn.ViewNode]') -> None:
    """
    Attach the viewing edge (``'views'`` connector) for every state-level
    access to a view container, mirroring the classic frontend's per-state
    view resolution: view reads get an incoming edge from the viewed source,
    view writes an outgoing edge into it. Iterates to support views of views.
    """
    if not bindings:
        return
    for state in sdfg.all_states():
        scopes = state.scope_dict()
        to_process = list(state.data_nodes())
        while to_process:
            # New source/target access nodes may themselves be views
            next_round = []
            for view_node in to_process:
                binding = bindings.get(view_node.data)
                if binding is None or scopes.get(view_node) is not None:
                    continue
                if sdfg.parent is not None and not sdfg.arrays[view_node.data].transient:
                    # A non-transient inside a nested SDFG is one of its
                    # connectors, and a view that crosses a boundary was
                    # already bound on the other side of it -- either as the
                    # view itself or, when it is the SOURCE another view was
                    # rebuilt from, as the container that source names. Binding
                    # it again in here would import ITS source as a second
                    # connector, so the same memory would reach the body twice
                    # under two names, and the two aliasing paths would be
                    # unioned into one nonsensical memlet on the way out.
                    continue
                if (any(e.dst_conn == 'views' for e in state.in_edges(view_node))
                        or any(e.src_conn == 'views' for e in state.out_edges(view_node))):
                    continue
                if not _import_view_source(sdfg, binding.source):
                    continue
                memlet = copy.deepcopy(binding.memlet)
                if state.in_degree(view_node) == 0:
                    source = _produced_in_state(state, scopes, binding.source, bindings, view_node)
                    if source is None:
                        source = state.add_read(binding.source)
                    state.add_edge(source, None, view_node, 'views', memlet)
                    next_round.append(source)
                elif state.out_degree(view_node) == 0:
                    target = state.add_write(binding.source)
                    state.add_edge(view_node, 'views', target, None, memlet)
                    next_round.append(target)
                else:
                    raise NotImplementedError(f"View '{view_node.data}' is both read and written in one state; "
                                              "cannot determine the viewing direction.")
            to_process = next_round


def _produced_in_state(state: SDFGState, scopes: dict, name: str, bindings: 'dict[str, tn.ViewNode]',
                       view_node: nodes.AccessNode) -> Optional[nodes.AccessNode]:
    """
    The access node a view read must attach to when the viewed container is
    produced earlier in the same state, or None if there is no such node.

    ``a = np.arange(10); b = a.reshape(10, 1)`` puts the producing map and the
    view into one state -- a read-after-write is expressible with memlets and
    needs no state boundary. Reading the source through a *fresh* access node
    there would leave the produced data unconnected to the view: an
    uninitialized transient, which simplification is then free to eliminate
    along with the map that filled it.

    :param state: The state the view is being connected in.
    :param scopes: The state's scope dictionary.
    :param name: Name of the viewed container.
    :param bindings: All view bindings, to leave views of views to the
                     iterative resolution in :func:`_connect_view_edges`.
    :param view_node: The view access node the source would be attached to.
    :return: The unique top-level access node that finishes writing ``name`` in
             this state, or None when there is no unambiguous safe one.
    """
    if name in bindings:
        return None
    candidates = [
        node for node in state.data_nodes()
        if node.data == name and scopes.get(node) is None and state.in_degree(node) > 0 and state.out_degree(node) == 0
    ]
    if len(candidates) != 1:
        return None

    # A view that FEEDS the computation writing its own source -- ``nester(A[:,
    # i])`` inside a map, where the callee writes back into A -- reaches that
    # write, so the viewing edge would point back into itself. Such a view has
    # to read through its own access node.
    if _reaches(state, view_node, candidates[0]):
        return None
    return candidates[0]


def _reaches(state: SDFGState, source: nodes.Node, target: nodes.Node) -> bool:
    """Whether ``target`` is reachable from ``source`` along the state's edges."""
    seen = {id(source)}
    frontier = [source]
    while frontier:
        node = frontier.pop()
        for edge in state.out_edges(node):
            if edge.dst is target:
                return True
            if id(edge.dst) not in seen:
                seen.add(id(edge.dst))
                frontier.append(edge.dst)
    return False


def _import_view_source(sdfg: SDFG, name: str) -> bool:
    """
    Make a view's SOURCE resolvable in the SDFG the view is being connected in.

    A view bound inside a scope body (``__anf0 = view A[4 * i:4 * i + 4]``
    within a map) is materialized in the nested SDFG that body becomes, where
    the source array does not exist yet -- it lives in an enclosing SDFG and
    has to be imported as the non-transient a connector requires. The connector
    itself follows from the access node this then allows (see the connector
    scan in ``visit_MapScope``, which runs after view resolution).

    :return: Whether the source is available (False when no enclosing SDFG has
             it, in which case the caller leaves the view unconnected).
    """
    if name in sdfg.arrays:
        return True
    parent = sdfg.parent.sdfg if sdfg.parent is not None else None
    while parent is not None and name not in parent.arrays:
        parent = parent.parent.sdfg if parent.parent is not None else None
    if parent is None:
        return False
    descriptor = parent.arrays[name].clone()
    descriptor.transient = False
    sdfg.add_datadesc(name, descriptor)
    return True


def _insert_state_boundaries_to_tree(stree: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
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
            if isinstance(scope, tn.ControlFlowScope) and not isinstance(scope, (tn.ElifScope, tn.ElseScope)):
                return [tn.StateBoundaryNode(True), self.generic_visit(scope)]
            return self.generic_visit(scope)

        def visit_StateLabel(self, node: tn.StateLabel):
            return [tn.StateBoundaryNode(True), self.generic_visit(node)]

    # First, insert boundaries around labels and control flow
    stree = SimpleStateBoundaryInserter().visit(stree)

    # Then, insert boundaries after unmet memory dependencies or potential data races
    _insert_memory_dependency_state_boundaries(stree)

    # Insert a state boundary after every symbol assignment to ensure symbols are assigned before usage
    class SymbolAssignmentBoundaryInserter(tn.ScheduleNodeTransformer):

        def visit_AssignNode(self, node: tn.AssignNode):
            # We can assume that assignment nodes are at least contained in the root scope.
            assert node.parent, "Expected assignment nodes live a parent scope."

            # Find this node in the parent's children.
            node_index = _list_index(node.parent.children, node)

            # Don't add boundary if there's already one or for immediately following assignment nodes.
            if node_index < len(node.parent.children) - 1 and isinstance(node.parent.children[node_index + 1],
                                                                         (tn.StateBoundaryNode, tn.AssignNode)):
                return self.generic_visit(node)

            return [self.generic_visit(node), tn.StateBoundaryNode()]

    stree = SymbolAssignmentBoundaryInserter().visit(stree)

    # A replacement expansion builds its own states, which a map body cannot
    # hold directly -- but a map body containing a state boundary is emitted as
    # a NESTED SDFG (see ``_insert_nestedSDFG_in_MapScope``), which can. Forcing
    # the boundary here is what lets registry calls (``dace.reduce``,
    # ``numpy.mean``, ...) lower inside a map instead of falling back to the
    # interpreter.
    class ReplacementCallBoundaryInserter(tn.ScheduleNodeTransformer):

        def visit_ReplacementCallNode(self, node: tn.ReplacementCallNode):
            assert node.parent is not None, 'Expected replacement calls to live in a parent scope.'
            node_index = _list_index(node.parent.children, node)
            if node_index > 0 and isinstance(node.parent.children[node_index - 1], tn.StateBoundaryNode):
                return self.generic_visit(node)
            return [tn.StateBoundaryNode(), self.generic_visit(node)]

    stree = ReplacementCallBoundaryInserter().visit(stree)

    # Hack: "backprop-insert" state boundaries from nested SDFGs
    class NestedSDFGStateBoundaryInserter(tn.ScheduleNodeTransformer):

        def visit_MapScope(self, scope: tn.MapScope):
            visited = self.generic_visit(scope)
            if any([isinstance(child, tn.StateBoundaryNode) for child in scope.children]):
                # We can assume that map nodes are at least contained in the root scope.
                assert scope.parent is not None

                # Find this scope in its parent's children
                node_index = _list_index(scope.parent.children, scope)

                # If there's already a state boundary before the map, don't add another one
                if node_index > 0 and isinstance(scope.parent.children[node_index - 1], tn.StateBoundaryNode):
                    return visited

                return [tn.StateBoundaryNode(), visited]
            return visited

    stree = NestedSDFGStateBoundaryInserter().visit(stree)

    return stree


def _insert_memory_dependency_state_boundaries(scope: tn.ScheduleTreeScope):
    """
    Helper function that inserts boundaries after unmet memory dependencies.
    """
    reads: mmu.MemletDict[list[tn.ScheduleTreeNode]] = mmu.MemletDict()
    writes: mmu.MemletDict[list[tn.ScheduleTreeNode]] = mmu.MemletDict()
    parents: dict[int, set[int]] = defaultdict(set)
    boundaries_to_insert: list[int] = []

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

        def _restart_state(index: int) -> None:
            # The node at ``index`` becomes the first node of a new state: its
            # reads and writes must seed the fresh tables, so hazards against
            # it (e.g. a write-after-read within the new state) stay visible.
            boundaries_to_insert.append(index)
            reads.clear()
            writes.clear()
            parents.clear()
            for inp in inputs:
                reads[inp] = [n]
            for out in outputs:
                writes[out] = [n]

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
            _restart_state(i)
            continue

        # Write after write or potential write/write data race, insert state boundary
        if any(o in writes and (o not in reads or any(id(r) not in parents for r in reads[o])) for o in outputs):
            _restart_state(i)
            continue

        # Potential read/write data race: if any read is not in the parents of this node, it might
        # be performed in parallel
        if any(o in reads and any(id(r) not in parents for r in reads[o]) for o in outputs):
            _restart_state(i)
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


def _create_state_boundary(
    boundary_node: tn.StateBoundaryNode,
    state: SDFGState,
    assignments: dict[str, str] | None = None,
) -> SDFGState:
    """
    Creates a boundary between two states

    :param boundary_node: The state boundary node to generate.
    :param state: The last state prior to this boundary.
    :return: The newly created state.
    """
    label = "cf_state_boundary" if boundary_node.due_to_control_flow else "state_boundary"
    assignments = assignments if assignments is not None else {}
    return _insert_and_split_assignments(state, label=label, assignments=assignments)


def _body_local_transients(scope: tn.ScheduleTreeScope, root: tn.ScheduleTreeRoot, sdfg: SDFG) -> set[str]:
    """
    The transients a scope's body uses that nothing outside it references.

    Such a container is per-iteration storage — a staging temporary a frontend
    emitted for one statement, most typically — and must stay local to the
    nested SDFG the body becomes. Passing it through the scope's connectors
    instead would allocate ONE instance shared by every concurrent iteration,
    which is a data race: iteration ``i`` could read the value iteration ``j``
    had just staged there.

    :param scope: The scope whose body is being emitted as a nested SDFG.
    :param root: The whole tree, to find references from outside the body.
    """
    inside = {id(node) for node in scope.preorder_traversal()}
    # A scope node's memlet sets aggregate its whole subtree, so an ANCESTOR of
    # this body reports the body's own containers as if they were used outside
    # it. Ancestors are neither inside nor outside; skip them.
    ancestors: set[int] = set()
    parent = scope.parent
    while parent is not None:
        ancestors.add(id(parent))
        parent = getattr(parent, 'parent', None)

    used_inside: set[str] = set()
    used_outside: set[str] = set()
    for node in root.preorder_traversal():
        if id(node) in ancestors:
            continue
        target = used_inside if id(node) in inside else used_outside
        for memlet in list(node.input_memlets()) + list(node.output_memlets()):
            if memlet.data is not None:
                target.add(memlet.data)
        for attribute in ('target', 'source'):
            name = getattr(node, attribute, None)
            if isinstance(name, str):
                target.add(name)
    return {
        name
        for name in used_inside - used_outside
        if name in sdfg.arrays and sdfg.arrays[name].transient and name not in root.arg_names
    }


def _end_of_expansion_chain(block: ControlFlowBlock, created: set) -> ControlFlowBlock:
    """
    Follow a linear chain of states created by a replacement expansion to its
    end, so that whatever the caller emits next orders after everything the
    expansion wrote.

    Only states the expansion itself created are followed, and only while the
    chain is unambiguous (exactly one successor): anything else is left to the
    caller's own state handling.
    """
    while True:
        successors = [edge.dst for edge in block.parent_graph.out_edges(block) if edge.dst in created]
        if len(successors) != 1:
            return block
        block = successors[0]


def _materialize_view_bindings(views: dict[str, tuple[str, Memlet]], states: list[SDFGState]) -> None:
    """
    Connect view containers recorded by a replacement expansion to the data
    they view.

    A replacement that reshapes/flattens its input allocates a
    :class:`~dace.data.View` and records ``views[view_name] = (source_name,
    memlet)`` on the visitor instead of adding the edge itself; the classic
    frontend materializes those bindings once, at the end of parsing
    (``newast.py``'s ``_views_to_data``). Deferred expansion has no such final
    pass, so an unconnected view access node would read uninitialized memory.
    This performs the same materialization over the states an expansion just
    created: a view node with no incoming edge gets a read of its source, one
    with no outgoing edge gets a write to it. Views of views are handled by
    iterating until no new access nodes appear.

    :param views: The ``{view name: (source name, memlet)}`` bindings recorded
                  by the expansion.
    :param states: The states to materialize bindings in.
    """
    for state in states:
        nodes = list(state.data_nodes())
        while nodes:
            new_nodes = []
            for view_node in nodes:
                if view_node.data not in views:
                    continue
                source_name, memlet = views[view_node.data]
                if state.in_degree(view_node) == 0:
                    read = state.add_read(source_name)
                    state.add_edge(read, None, view_node, 'views', copy.deepcopy(memlet))
                    new_nodes.append(read)
                elif state.out_degree(view_node) == 0:
                    write = state.add_write(source_name)
                    state.add_edge(view_node, 'views', write, None, copy.deepcopy(memlet))
                    new_nodes.append(write)
                else:
                    raise NotImplementedError(
                        f'View "{view_node.data}" recorded by a replacement expansion already has both '
                        f'incoming and outgoing edges')
            nodes = new_nodes


class ReplacementVisitorShim:
    """
    Stand-in for the classic ``ProgramVisitor`` interface consumed by frontend
    replacements invoked at tree-to-SDFG time (see
    ``visit_ReplacementCallNode``). Covers the surface replacements actually
    use: naming/allocation helpers, state chaining, and inert bookkeeping
    dictionaries. View recordings are checked after expansion (dropping them
    would miscompile); anything else is deliberately absent so unsupported
    replacements fail loudly instead of miscompiling.
    """

    def __init__(self, sdfg: SDFG, state: SDFGState, target_name: str, target_preexisting: bool = False):
        self.sdfg = sdfg
        self._target_name = target_name
        #: Whether the target is a container the program already had, which a
        #: replacement may write into directly (see
        #: :meth:`get_assignment_destination`).
        self._target_preexisting = target_preexisting
        #: Most recently added control flow block (replacements chain states
        #: through ``_add_state``/``last_block``).
        self.last_block = state
        #: Python-name registrations; the expansion connects results through
        #: the returned container name instead, so entries are inert.
        self.variables: dict[str, str] = {}
        #: View bindings a replacement may record; must stay empty (checked
        #: by the caller after expansion).
        self.views: dict = {}
        #: Program globals a replacement may resolve by name (an MPI
        #: communicator receiver, see ``ReplacementCallNode.receiver_object``).
        #: Populated per expansion with exactly the objects that call records;
        #: any other lookup fails loudly.
        self.globals: dict = {}
        self.current_lineinfo = None

    @property
    def cfg_target(self):
        return self.last_block.parent_graph

    def get_target_name(self, output_index=None, default=None) -> str:
        return self._target_name or default or self.sdfg.temp_data_name()

    def get_assignment_destination(self) -> Optional[str]:
        """
        The existing container the call's result is written into, or None when
        the result needs a container of its own -- the same distinction
        ``ProgramVisitor.get_assignment_destination`` draws from the
        assignment's syntax, carried here by the frontend on
        :attr:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode.target_preexisting`
        (every target exists by the time a tree is expanded, so the
        descriptor alone cannot tell the two apart).
        """
        if not self._target_preexisting or self._target_name not in self.sdfg.arrays:
            return None
        return self._target_name

    def add_temp_transient(self, *args, output_index=None, **kwargs):
        kwargs['find_new_name'] = True
        return self.sdfg.add_transient(self.get_target_name(output_index), *args, **kwargs)

    def _add_state(self, label=None) -> SDFGState:
        state = self.cfg_target.add_state(label)
        self.cfg_target.add_edge(self.last_block, state, InterstateEdge())
        self.last_block = state
        return state


def _insert_and_split_assignments(
    before_state: ControlFlowBlock,
    after_state: ControlFlowBlock | None = None,
    *,
    label: str | None = None,
    assignments: dict[str, str] | None = None,
) -> ControlFlowBlock:
    """
    Insert given assignments splitting them in case of potential race conditions.

    The semantics of the SDFG dictates that we can not assume any order in the application
    of inter-state edge assignments. The only order is that conditions precede assignments.

    Since we just collect all inter-state assignments while parsing the schedule tree, we
    need to make sure to split problematic assignments over multiple state transitions.
    """
    assignments = assignments if assignments is not None else {}
    cf_region = before_state.parent_graph
    if after_state is not None and after_state.parent_graph != cf_region:
        raise ValueError("Expected before_state and after_state to be in the same control flow region.")

    has_potential_race = False
    for key, value in assignments.items():
        syms = symbolic.free_symbols_and_functions(value)
        also_assigned = (syms & assignments.keys()) - {key}
        if also_assigned:
            has_potential_race = True
            break

    if not has_potential_race:
        if after_state is not None:
            cf_region.add_edge(before_state, after_state, InterstateEdge(assignments=assignments))
            return after_state

        return cf_region.add_state_after(before_state, label=label, assignments=assignments)

    last_state = before_state
    for index, assignment in enumerate(assignments.items()):
        key, value = assignment
        is_last_state = index == len(assignments) - 1
        if is_last_state and after_state is not None:
            cf_region.add_edge(last_state, after_state, InterstateEdge(assignments={key: value}))
            last_state = after_state
        else:
            last_state = cf_region.add_state_after(last_state, label=label, assignments={key: value})

    return last_state


def _has_branch_continuation(node: tn.ScheduleTreeNode) -> bool:
    """Whether an if/elif branch scope is followed by another elif/else branch
    (ignoring inserted state boundaries)."""
    filtered = [sibling for sibling in node.parent.children if not isinstance(sibling, tn.StateBoundaryNode)]
    index = _list_index(filtered, node)
    return len(filtered) > index + 1 and isinstance(filtered[index + 1], (tn.ElifScope, tn.ElseScope))


def _list_index(list: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode) -> int:
    """Check if node is in list with "is" operator."""
    index = 0
    for element in list:
        # compare with "is" to get memory comparison. ".index()" uses value comparison
        if element is node:
            return index
        index += 1

    raise StopIteration
