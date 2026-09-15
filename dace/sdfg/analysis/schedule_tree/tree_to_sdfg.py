# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from types import TracebackType
from typing import Final, Sequence

from dace import data, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes, memlet_utils as mmu
from dace.sdfg.sdfg import SDFG, ControlFlowRegion, InterstateEdge
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, LoopRegion, NamedRegion,
                             ReturnBlock, SDFGState, UnstructuredControlFlow)
from dace.sdfg.analysis.schedule_tree import passes as stpasses, treenodes as tn
from dace.sdfg import propagation


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


class _NestedSDFGScope(tn.ControlFlowScope):
    """
    A scope whose children are converted into a nested SDFG. Only used internally during the conversion, e.g., to
    bound the effect of return blocks that lower gotos to the end of this scope.
    """

    def as_string(self, indent: int = 0):
        result = indent * tn.INDENTATION + 'nested sdfg:\n'
        return result + super().as_string(indent)


class _StreeToSDFG(tn.ScheduleNodeVisitor):

    def __init__(
        self,
        *,
        start_state: SDFGState | None = None,
        boundary_behavior: StateBoundaryBehavior = StateBoundaryBehavior.STATE_TRANSITION,
        max_nested_sdfg: int = 1000,
    ) -> None:
        self._boundary_behavior = boundary_behavior
        """How state boundaries that do not require a state transition are converted."""

        self._barriers: list[tuple[SDFGState, set[nodes.Node]]] = []
        """State boundaries converted into happens-before edges: the state and the nodes that precede the boundary."""

        self._frozen_access_nodes: set[int] = set()
        """IDs of access nodes before such a state boundary, which must not be written to after it."""

        self._ctx: _Context
        """Context information like tree root and current scope."""

        self._current_state = start_state
        """Current SDFGState in the SDFG that we are building."""

        self._current_nestedSDFG: int | None = None
        """Id of the current nested SDFG if we are inside one."""

        self._known_data_outside_nestedSDFG: set[str] | None = None
        """In case we are inside a nested SDFG, this list previously accessed data (arrays and scalars) outside the nestedSDFG."""

        self._interstate_symbols: list[tn.AssignNode] = []
        """Interstate symbol assignments. Will be assigned with the next state transition."""

        self._nviews_free: list[tn.NView] = []
        """Keep track of NView (nested SDFG view) nodes that are "free" to be used."""

        self._nviews_bound_per_scope: dict[int, list[tn.NView]] = {}
        """Mapping of id(SDFG) -> list of active NView nodes in that SDFG."""

        self._nviews_deferred_removal: dict[int, list[tn.NView]] = {}
        """"Mapping of id(SDFG) -> list of NView nodes to be removed once we exit this nested SDFG."""

        self._views: dict[str, tn.ViewNode] = {}
        """Mapping of view container name -> ViewNode that defines the view."""

        self._dynamic_scope_inputs: list[tn.DynScopeCopyNode] = []
        """Dynamic scope inputs (e.g., dynamic map ranges) of the next dataflow scope."""

        self._code_reference_sets: dict[tuple[int, str | None], tn.RefSetNode] = {}
        """Mapping of (id(code node), output connector) -> the reference set node that the output sets."""

        self._consume_streams: dict[int, str] = {}
        """Mapping of id(ConsumeEntry) -> name of the stream it consumes."""

        # state management
        self._state_stack: list[SDFGState] = []

        # dataflow scopes
        # list[ (MapEntryNode, ToConnect) | (SDFG, {"inputs": set(), "outputs": set()}) ]
        self._dataflow_stack: list[tuple[nodes.EntryNode, dict[str, tuple[nodes.AccessNode, Memlet]]]
                                   | tuple[SDFG, dict[str, set[str]]]] = []

        self._max_nested_sdfg = max_nested_sdfg

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
        self._code_reference_sets = _code_reference_sets(node)
        with _TreeScope(node, self._ctx, self._current_state):
            self.visit(node.children, sdfg=sdfg)

        assert not self._state_stack, "Expected empty state stack."
        assert not self._dataflow_stack, "Expected empty dataflow stack."
        assert not self._interstate_symbols, "Expected empty list of symbols to add."

    def visit_GBlock(self, node: tn.GBlock, sdfg: SDFG) -> None:
        """
        Converts a general block into an unstructured control flow region. Every label starts a new state, gotos to
        labels of this block become (conditional) inter-state edges, and falling off the end of a labeled segment exits
        the region.
        """
        before_state = self._current_state
        assert before_state is not None
        cf_region = before_state.parent_graph

        region = UnstructuredControlFlow(f"gblock_{id(node)}", sdfg=sdfg)
        cf_region.add_node(region, ensure_unique_name=True)
        _insert_and_split_assignments(before_state, region, assignments=self._pending_interstate_assignments())

        # Split children into labeled segments, the first of which is the entry of the region
        segments: list[tuple[str | None, list[tn.ScheduleTreeNode]]] = [(None, [])]
        for child in node.children:
            if isinstance(child, tn.StateLabel):
                segments.append((child.name, []))
            else:
                segments[-1][1].append(child)
        for _, children in segments:
            # State boundaries before labels only mark the (already given) state transition
            while children and isinstance(children[-1], tn.StateBoundaryNode):
                children.pop()
        if not segments[0][1] and len(segments) > 1:
            segments.pop(0)
        labels = {name for name, _ in segments if name is not None}
        states = [
            region.add_state(f"gblock_{name or 'entry'}", is_start_block=(i == 0))
            for i, (name, _) in enumerate(segments)
        ]
        label_states = {name: state for (name, _), state in zip(segments, states) if name is not None}

        # Inter-state edges are added after all segments were converted, since gotos may jump forward
        jumps: list[tuple[SDFGState, str, str | None, dict[str, str]]] = []

        for (_, children), state in zip(segments, states):
            self._current_state = state
            conditions: list[str] = []  # Conditions of the preceding gotos out of the current state

            def fall_through_condition() -> str | None:
                return ' and '.join(f'(not ({c}))' for c in conditions) if conditions else None

            def continue_after_conditions() -> None:
                # Statements after conditional gotos only run if none of the conditions hold
                if conditions:
                    continuation = region.add_state("gblock_continuation")
                    region.add_edge(self._current_state, continuation, InterstateEdge(fall_through_condition()))
                    self._current_state = continuation
                    conditions.clear()

            for index, child in enumerate(children):
                is_goto = isinstance(child, tn.GotoNode) and child.target in labels
                is_conditional_goto = (isinstance(child, tn.StateIfScope) and len(child.children) == 1
                                       and isinstance(child.children[0], tn.GotoNode)
                                       and child.children[0].target in labels)

                if isinstance(child, tn.StateBoundaryNode):
                    # State boundaries before transitions would add unconditional transitions to the current state
                    following = next((c for c in children[index + 1:] if not isinstance(c, tn.StateBoundaryNode)), None)
                    if isinstance(following, (tn.GotoNode, tn.StateIfScope)):
                        continue
                elif is_goto:
                    jumps.append((self._current_state, child.target, fall_through_condition(),
                                  self._pending_interstate_assignments()))
                    break  # The rest of the segment is unreachable
                elif is_conditional_goto:
                    if self._interstate_symbols:
                        # Assignments happen before evaluating the condition, i.e., in a transition before
                        continue_after_conditions()
                        self._current_state = _insert_and_split_assignments(
                            self._current_state,
                            label="gblock_assignments",
                            assignments=self._pending_interstate_assignments())
                    jumps.append((self._current_state, child.children[0].target, child.condition.as_string, {}))
                    conditions.append(child.condition.as_string)
                    continue

                continue_after_conditions()
                self.visit(child, sdfg=sdfg)
            else:
                # Falling off the end of the segment exits the region, after pending assignments
                if self._interstate_symbols:
                    exit_state = region.add_state("gblock_exit")
                    region.add_edge(
                        self._current_state, exit_state,
                        InterstateEdge(fall_through_condition(), assignments=self._pending_interstate_assignments()))

        for source, target, condition, assignments in jumps:
            region.add_edge(source, label_states[target], InterstateEdge(condition=condition, assignments=assignments))

        # Remove segments that no goto jumps to
        reachable = set(region.bfs_nodes(region.start_block))
        for block in [block for block in region.nodes() if block not in reachable]:
            region.remove_node(block)

        self._current_state = _insert_and_split_assignments(region, label="gblock_after")

    def visit_StateLabel(self, node: tn.StateLabel, sdfg: SDFG) -> None:
        # Outside of general blocks, labels only mark the target of forward gotos, which are lowered by
        # ``_lower_forward_gotos`` and ``visit_GotoNode``.
        pass

    def visit_GotoNode(self, node: tn.GotoNode, sdfg: SDFG) -> None:
        if node.target is not None:
            label = _find_label(node)
            if label is None or not _jumps_to_sdfg_end(node, label):
                raise ValueError(f"Cannot convert '{node.as_string().strip()}': the target label is not at the end of "
                                 "the SDFG being built.")
        elif any(isinstance(scope, (tn.DataflowScope, _NestedSDFGScope)) for scope in _ancestors(node)):
            raise ValueError("Exit gotos inside dataflow scopes or nested SDFGs are not supported.")

        # Jumping to the end of the SDFG that is being built is equivalent to returning from it
        self._insert_exit_block(ReturnBlock(f"return_{id(node)}"))

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

                    # Add in_connector in case of read after write of "outside" data
                    if memlet.data in self._known_data_outside_nestedSDFG:
                        to_connect["inputs"].add(memlet.data)
                return

        for memlet in input_memlets:
            # If we aren't inside a nested SDFG, make sure all memlets can be resolved.
            # Imo, this should always be the case. It not, raise an error.
            if memlet.data not in sdfg.arrays:
                raise ValueError(f"Parsing AssignNode {node} failed. Can't find {memlet.data} in {sdfg}.")

    def _loop_state_name_prefix(self, node: tn.LoopScope) -> str:
        if isinstance(node, tn.ForScope):
            return "for"

        if isinstance(node, tn.WhileScope):
            return "while"

        if isinstance(node, tn.DoWhileScope):
            return "do_while"

        return "loop"

    def _add_loop_region(self, node: tn.LoopScope, sdfg: SDFG) -> None:
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
        self._flush_pending_assignments(f"{prefix}_loop_body_end")

        after_state = _insert_and_split_assignments(loop_region, label=f"{prefix}_loop_after")
        self._current_state = after_state

    def visit_ForScope(self, node: tn.ForScope, sdfg: SDFG) -> None:
        self._add_loop_region(node, sdfg)

    def visit_WhileScope(self, node: tn.WhileScope, sdfg: SDFG) -> None:
        self._add_loop_region(node, sdfg)

    def visit_DoWhileScope(self, node: tn.DoWhileScope, sdfg: SDFG) -> None:
        self._add_loop_region(node, sdfg)

    def visit_LoopScope(self, node: tn.LoopScope, sdfg: SDFG) -> None:
        # General loops (e.g., do-for loops) are fully described by their loop region properties
        self._add_loop_region(node, sdfg)

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

                # Add in_connector in case of read after write in case of "outside data"
                if memlet.data in self._known_data_outside_nestedSDFG:
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
        self._flush_pending_assignments("if_body_end")

        self._current_state = conditional_block

        # add merge_state
        merge_state = _insert_and_split_assignments(conditional_block, label="merge_state")

        self._process_next_conditional_block(node, conditional_block, merge_state)

    def _process_next_conditional_block(self, node: tn.ControlFlowScope, conditional_block: ConditionalBlock,
                                        merge_state: SDFGState) -> None:
        """
        Prepares the next branch of a conditional block after visiting the branch of ``node``.

        If an ``ElifScope`` or ``ElseScope`` follows ``node`` in its parent's children, the merge state and the
        conditional block are pushed onto the state stack for that branch. Otherwise, the merge state becomes the
        current state.

        :param node: The if or elif scope whose branch was visited.
        :param conditional_block: The conditional block of the branch.
        :param merge_state: The state after the conditional block.
        """
        # Filter StateBoundaryNodes, which we inserted earlier, for this analysis.
        filtered = [n for n in node.parent.children if not isinstance(n, tn.StateBoundaryNode)]
        index = _list_index(filtered, node)
        has_next_branch = len(filtered) > index + 1 and isinstance(filtered[index + 1], (tn.ElifScope, tn.ElseScope))

        if has_next_branch:
            # push merge_state and condition_block on the stack for later usage in `visit_ElifScope`/`visit_ElseScope`
            self._state_stack.append(merge_state)
            self._state_stack.append(conditional_block)
        else:
            self._current_state = merge_state

    def visit_NamedRegionScope(self, node: tn.NamedRegionScope, sdfg: SDFG) -> None:
        # A labeled grouping becomes a named region, such that the label survives for profiling and transformation
        # targeting. It constrains nothing, so the body is converted into it unchanged.
        current_state = self._current_state
        assert current_state is not None
        cf_region = current_state.parent_graph

        named_region = NamedRegion(node.label)
        cf_region.add_node(named_region, ensure_unique_name=True)
        _insert_and_split_assignments(current_state, named_region, assignments=self._pending_interstate_assignments())

        self._current_state = named_region.add_state(f"named_region_state_{id(node)}", is_start_block=True)
        self.visit(node.children, sdfg=sdfg)
        self._flush_pending_assignments(f"named_region_end_{id(node)}")

        self._current_state = _insert_and_split_assignments(named_region, label=f"named_region_after_{id(node)}")

    def visit_StateIfScope(self, node: tn.StateIfScope, sdfg: SDFG) -> None:
        # Outside of general blocks, a state transition condition is a regular conditional (e.g., around a goto)
        self.visit_IfScope(node, sdfg)

    def _insert_exit_block(self, block: BreakBlock | ContinueBlock | ReturnBlock) -> None:
        """
        Adds a control flow exit block (break, continue, or return) after the current state. Statements that follow it
        in the same scope are unreachable and are placed into a new state after the block.

        :param block: The exit block to add.
        """
        cf_region = self._current_state.parent_graph
        cf_region.add_node(block, ensure_unique_name=True)
        _insert_and_split_assignments(self._current_state, block, assignments=self._pending_interstate_assignments())
        self._current_state = _insert_and_split_assignments(block, label=f"after_{block.label}")

    def visit_BreakNode(self, node: tn.BreakNode, sdfg: SDFG) -> None:
        self._insert_exit_block(BreakBlock(f"break_{id(node)}"))

    def visit_ContinueNode(self, node: tn.ContinueNode, sdfg: SDFG) -> None:
        self._insert_exit_block(ContinueBlock(f"continue_{id(node)}"))

    def visit_ElifScope(self, node: tn.ElifScope, sdfg: SDFG) -> None:
        # get ConditionalBlock and merge state from stack
        conditional_block: ConditionalBlock = self._pop_state("if_scope")
        merge_state = self._pop_state("merge_state")

        elif_body = ControlFlowRegion("elif_body", sdfg=sdfg)
        conditional_block.add_branch(node.condition, elif_body)

        memlets = conditional_block.get_meta_read_memlets(self._ctx.root.containers, include_scalars=True)
        self._ensure_data_descriptors(memlets, sdfg)

        self._current_state = elif_body.add_state("elif_state", is_start_block=True)

        # visit children inside the elif branch
        self.visit(node.children, sdfg=sdfg)
        self._flush_pending_assignments("elif_body_end")

        self._process_next_conditional_block(node, conditional_block, merge_state)

    def visit_ElseScope(self, node: tn.ElseScope, sdfg: SDFG) -> None:
        # get ConditionalBlock from stack
        conditional_block: ConditionalBlock = self._pop_state("if_scope")

        else_body = ControlFlowRegion("else_body", sdfg=sdfg)
        conditional_block.add_branch(None, else_body)

        else_state = else_body.add_state("else_state", is_start_block=True)
        self._current_state = else_state

        # visit children inside the else branch
        self.visit(node.children, sdfg=sdfg)
        self._flush_pending_assignments("else_body_end")

        # merge false-branch into merge_state
        merge_state = self._pop_state("merge_state")
        self._current_state = merge_state

    def _insert_nested_sdfg(self, node: tn.ScheduleTreeScope, sdfg: SDFG, scope_body: bool) -> None:
        """
        Converts the children of a scope into a nested SDFG and connects it in the current state.

        :param node: The scope whose children form the body of the nested SDFG.
        :param sdfg: The SDFG that is currently being built.
        :param scope_body: True if the nested SDFG is the entire body of the surrounding dataflow scope, in which case
                           it is connected directly to the pass-through connectors of the scope. Otherwise, it is
                           connected to the accessed data containers like any other node.
        """
        dataflow_stack_size = len(self._dataflow_stack)
        state_stack_size = len(self._state_stack)
        outer_nestedSDFG = self._current_nestedSDFG
        outer_known_data = self._known_data_outside_nestedSDFG

        self._known_data_outside_nestedSDFG = set()
        for access_dict in self._ctx.access_cache.values():
            for name in access_dict:
                self._known_data_outside_nestedSDFG.add(name)

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

        # visit children
        with _TreeScope(node, self._ctx, self._current_state):
            self.visit(node.children, sdfg=inner_sdfg)

        # restore current state and stacks
        self._current_state = self._pop_state(old_state_label)
        assert len(self._state_stack) == state_stack_size
        _, connectors = self._dataflow_stack.pop()
        assert len(self._dataflow_stack) == dataflow_stack_size

        # insert nested SDFG
        nsdfg = self._current_state.add_nested_sdfg(
            sdfg=inner_sdfg,
            inputs={name: None
                    for name in connectors["inputs"]},
            outputs={name: None
                     for name in connectors["outputs"]},
        )
        if scope_body:
            self._connect_nested_sdfg_in_map(nsdfg, inner_sdfg)
        else:
            self._connect_nested_sdfg(nsdfg, inner_sdfg, sdfg)

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
        self._known_data_outside_nestedSDFG = outer_known_data

    def _connect_nested_sdfg_in_map(self, nsdfg: nodes.NestedSDFG, inner_sdfg: SDFG) -> None:
        """
        Connects the inputs and outputs of a nested SDFG to the pass-through connectors of the surrounding map scope.

        :param nsdfg: The nested SDFG node.
        :param inner_sdfg: The SDFG of the nested SDFG node.
        """
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

    def _connect_nested_sdfg(self, nsdfg: nodes.NestedSDFG, inner_sdfg: SDFG, sdfg: SDFG) -> None:
        """
        Connects the inputs and outputs of a nested SDFG outside of a map scope to the accessed data containers.

        :param nsdfg: The nested SDFG node.
        :param inner_sdfg: The SDFG of the nested SDFG node.
        :param sdfg: The SDFG that is currently being built.
        """
        nview_memlets = {nview.target: nview.memlet for nview in self._nviews_bound_per_scope[id(inner_sdfg)]}

        def memlet(name: str) -> Memlet:
            if name in nview_memlets:
                return Memlet.from_memlet(nview_memlets[name])
            return Memlet.from_array(name, inner_sdfg.arrays[name])

        for name in nsdfg.in_connectors:
            source, source_conn = self._read_source(name, sdfg)
            self._current_state.add_edge(source, source_conn, nsdfg, name, memlet(name))
        for name in nsdfg.out_connectors:
            self._connect_output(nsdfg, name, name, memlet(name), sdfg)

        # Within a dataflow scope, connect the nested SDFG to the scope even if it has no inputs or outputs
        scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)
        if isinstance(scope_node, nodes.EntryNode):
            if not nsdfg.in_connectors:
                self._current_state.add_nedge(scope_node, nsdfg, Memlet())
            if not nsdfg.out_connectors:
                to_connect[f"nested_sdfg_{id(nsdfg)}"] = (nsdfg, Memlet())

    def visit__NestedSDFGScope(self, node: _NestedSDFGScope, sdfg: SDFG) -> None:
        self._insert_nested_sdfg(node, sdfg, scope_body=False)

        # Start a new state such that subsequent accesses do not share access nodes with the nested SDFG's outputs.
        # Within dataflow scopes, nested SDFG scopes span the rest of the scope.
        if not (self._dataflow_stack and isinstance(self._dataflow_stack[-1][0], nodes.EntryNode)):
            self._current_state = _insert_and_split_assignments(self._current_state,
                                                                label="nested_sdfg_after",
                                                                assignments=self._pending_interstate_assignments())

    def visit_MapScope(self, node: tn.MapScope, sdfg: SDFG) -> None:
        self._visit_dataflow_scope(node, nodes.MapEntry(node.node.map), nodes.MapExit(node.node.map), sdfg)

    def visit_ConsumeScope(self, node: tn.ConsumeScope, sdfg: SDFG) -> None:
        entry = nodes.ConsumeEntry(node.node.consume)
        exit_node = nodes.ConsumeExit(node.node.consume)

        # The consumed stream is given as a dynamic scope input to the entry's stream connector
        stream_inputs = [inp for inp in self._dynamic_scope_inputs if inp.target == 'IN_stream']
        if stream_inputs:
            self._consume_streams[id(entry)] = stream_inputs[-1].memlet.data
        else:
            stream = self._consumed_stream(node)
            self._consume_streams[id(entry)] = stream
            self._dynamic_scope_inputs.append(
                tn.DynScopeCopyNode(target='IN_stream',
                                    memlet=Memlet.from_array(stream, self._ctx.root.containers[stream])))

        # Code generation supports only one read of the consumed element, and it cannot be passed into nested SDFGs
        stream = self._consume_streams[id(entry)]
        has_boundaries = any(isinstance(child, tn.StateBoundaryNode) for child in node.children)
        if has_boundaries or len(_reads_of(node, stream)) > 1:
            self._hoist_consumed_element(node, stream, sdfg, nest=has_boundaries)

        self._visit_dataflow_scope(node, entry, exit_node, sdfg)

    def _hoist_consumed_element(self, node: tn.ConsumeScope, stream: str, sdfg: SDFG, nest: bool) -> None:
        """
        Copies the consumed element into a new scalar at the beginning of a consume scope, and replaces reads of the
        stream in the scope with reads of that scalar. Operates in-place.

        Within a consume scope, reading the consumed stream provides the current element. Code generation only supports
        one such read, and the element cannot be passed into a nested SDFG under the name of the stream (which is still
        used to push into it).

        :param node: The consume scope.
        :param stream: The name of the consumed stream.
        :param sdfg: The SDFG that is currently being built.
        :param nest: If True, the rest of the scope is converted into a nested SDFG (e.g., due to state boundaries).
        """
        if node.node.consume.chunksize != 1:
            raise NotImplementedError("Consume scopes with chunks that read the element more than once or require "
                                      "multiple states are not supported.")

        # The element is registered in the top-level SDFG, from which nested SDFGs obtain their descriptors
        root_sdfg = sdfg
        while root_sdfg.parent is not None:
            root_sdfg = root_sdfg.parent.sdfg
        containers = self._ctx.root.containers
        element = data.find_new_name(f"__{stream}_element", set(containers.keys()) | set(root_sdfg.arrays.keys()))
        containers[element] = data.Scalar(containers[stream].dtype, transient=True)
        root_sdfg.add_datadesc(element, containers[element].clone())

        body = list(node.children)
        for child in body:
            _rename_reads(child, stream, element)
        pop = tn.CopyNode(target=element, memlet=Memlet(f"{stream}[0]"))
        pop.parent = node
        if nest:
            node.children = [pop, _NestedSDFGScope(children=body, parent=node)]
        else:
            node.children = [pop] + body

    def _consumed_stream(self, node: tn.ConsumeScope) -> str:
        """
        Returns the stream a consume scope pops from, if it is not given as a dynamic scope input: the only stream
        read by the code nodes directly in the scope.

        :param node: The consume scope.
        :return: The name of the consumed stream container.
        """
        streams = {
            memlet.data
            for child in node.children if isinstance(child, (tn.TaskletNode, tn.LibraryCall))
            for memlet in child.input_memlets() if isinstance(self._ctx.root.containers.get(memlet.data), data.Stream)
        }
        if len(streams) != 1:
            raise ValueError(f"Cannot determine the consumed stream of '{node.as_string().splitlines()[0].strip()}' "
                             f"(candidates: {sorted(streams)}).")
        return next(iter(streams))

    def _visit_dataflow_scope(self, node: tn.DataflowScope, entry: nodes.EntryNode, exit_node: nodes.ExitNode,
                              sdfg: SDFG) -> None:
        """
        Converts a dataflow scope (map or consume) with the given entry and exit nodes. Inputs and outputs of the scope
        are connected through pass-through connectors on the entry and exit nodes.

        :param node: The dataflow scope to convert.
        :param entry: The entry node of the scope.
        :param exit_node: The exit node of the scope.
        :param sdfg: The SDFG that is currently being built.
        """
        dataflow_stack_size = len(self._dataflow_stack)
        cache_state = self._current_state
        map_entry = entry

        # scope entry
        # -----------
        self._current_state.add_node(map_entry)

        # connect dynamic scope inputs (e.g., map ranges), which are read outside of the scope
        for dynamic_input in self._dynamic_scope_inputs:
            map_entry.add_in_connector(dynamic_input.target)
            source, source_conn = self._read_source(dynamic_input.memlet.data, sdfg)
            self._current_state.add_edge(source, source_conn, map_entry, dynamic_input.target,
                                         copy.deepcopy(dynamic_input.memlet))
        self._dynamic_scope_inputs.clear()

        self._dataflow_stack.append((map_entry, dict()))

        # visit children inside the map
        type_of_children = [type(child) for child in node.children]
        last_child_is_MapScope = bool(type_of_children) and type_of_children[-1] == tn.MapScope
        all_others_are_Boundaries = type_of_children.count(tn.StateBoundaryNode) == len(type_of_children) - 1
        if last_child_is_MapScope and all_others_are_Boundaries:
            # skip weirdly added StateBoundaryNode
            # tmp: use this - for now - to "backprop-insert" extra state boundaries for nested SDFGs
            with _TreeScope(node, self._ctx, self._current_state):
                self.visit(node.children[-1], sdfg=sdfg)
        elif any([isinstance(child, tn.StateBoundaryNode) for child in node.children]):
            self._insert_nested_sdfg(node, sdfg, scope_body=True)
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
        connected = {e.dst_conn for e in self._current_state.in_edges(map_entry)}
        for connector in map_entry.in_connectors:
            if not connector.startswith(PREFIX_PASSTHROUGH_IN) or connector in connected:
                continue  # dynamic scope inputs are already connected
            memlet_data = connector.removeprefix(PREFIX_PASSTHROUGH_IN)

            # connect to local access node (if available)
            if memlet_data in access_cache:
                cached_access = access_cache[memlet_data]
                self._current_state.add_memlet_path(
                    cached_access,
                    map_entry,
                    dst_conn=connector,
                    memlet=Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]),
                )
                if isinstance(outer_map_entry, SDFG) and memlet_data in self._known_data_outside_nestedSDFG:
                    # in case of read after write of memory that comes from an "outside" SDFG,
                    # make sure that we register the read in the nested SDFG.
                    outer_to_connect["inputs"].add(memlet_data)
                continue

            if isinstance(outer_map_entry, nodes.EntryNode):
                # get it from outside the map
                connector_name = f"{PREFIX_PASSTHROUGH_OUT}{memlet_data}"
                if connector_name not in outer_map_entry.out_connectors:
                    new_in_connector = outer_map_entry.add_in_connector(connector)
                    new_out_connector = outer_map_entry.add_out_connector(connector_name)
                    assert new_in_connector == True
                    assert new_in_connector == new_out_connector

                if memlet_data in sdfg.arrays:
                    data_descriptor = sdfg.arrays[memlet_data]
                else:
                    _sdfg = self._parent_sdfg_with_array(memlet_data, sdfg)
                    data_descriptor = _sdfg.arrays[memlet_data]
                self._current_state.add_edge(outer_map_entry, connector_name, map_entry, connector,
                                             Memlet.from_array(memlet_data, data_descriptor))
            else:
                if isinstance(outer_map_entry, SDFG):
                    # Copy data descriptor from parent SDFG and add input connector
                    if memlet_data not in sdfg.arrays:
                        parent_sdfg = self._parent_sdfg_with_array(memlet_data, sdfg)

                        # Add support for NView nodes
                        use_nview = self._apply_nview_array_override(memlet_data, sdfg)
                        if not use_nview:
                            sdfg.add_datadesc(memlet_data, parent_sdfg.arrays[memlet_data].clone())
                            # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                            if parent_sdfg.arrays[memlet_data].transient:
                                sdfg.arrays[memlet_data].transient = False

                        # Dev note: nview.target and memlet_data are identical
                        outer_to_connect["inputs"].add(memlet_data)

                    # Add in_connector in case of read after write of "outside data"
                    if memlet_data in self._known_data_outside_nestedSDFG:
                        outer_to_connect["inputs"].add(memlet_data)
                else:
                    assert outer_map_entry is None

                # cache local read access
                assert memlet_data not in access_cache
                access_cache[memlet_data] = self._current_state.add_read(memlet_data)
                cached_access = access_cache[memlet_data]
                self._current_state.add_memlet_path(
                    cached_access,
                    map_entry,
                    dst_conn=connector,
                    memlet=Memlet.from_array(memlet_data, sdfg.arrays[memlet_data]),
                )

        if isinstance(outer_map_entry, nodes.EntryNode) and self._current_state.out_degree(outer_map_entry) < 1:
            self._current_state.add_nedge(outer_map_entry, map_entry, Memlet())

        # scope exit
        # ----------
        map_exit = exit_node
        self._current_state.add_node(map_exit)

        # connect writes to map_exit node
        for name in to_connect:
            access_node, memlet = to_connect[name]
            # Special case: connect tasklets without outputs via an empty Memlet to the MapExit node.
            if isinstance(memlet, Memlet) and memlet.is_empty():
                self._current_state.add_nedge(access_node, map_exit, memlet)
                continue

            in_connector_name = f"{PREFIX_PASSTHROUGH_IN}{name}"
            out_connector_name = f"{PREFIX_PASSTHROUGH_OUT}{name}"
            new_in_connector = map_exit.add_in_connector(in_connector_name)
            new_out_connector = map_exit.add_out_connector(out_connector_name)
            assert new_in_connector == new_out_connector

            # connect "inside the map"
            if isinstance(access_node, nodes.NestedSDFG):
                self._current_state.add_edge(access_node, name, map_exit, in_connector_name, memlet)
            else:
                assert isinstance(access_node, nodes.AccessNode)
                if self._current_state.out_degree(access_node) == 0 and self._current_state.in_degree(access_node) == 1:
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

            if isinstance(outer_map_entry, SDFG):
                if name not in sdfg.arrays:
                    parent_sdfg = self._parent_sdfg_with_array(name, sdfg)

                    # Support for NView nodes
                    use_nview = self._apply_nview_array_override(name, sdfg)
                    if not use_nview:
                        sdfg.add_datadesc(name, parent_sdfg.arrays[name].clone())
                        # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                        if parent_sdfg.arrays[name].transient:
                            sdfg.arrays[name].transient = False

                # Add out connector in any case because we don't know who (if anyone)
                # is gonna read from it down the line.
                # Dev not: name and nview.target are identical
                outer_to_connect["outputs"].add(name)

            # connect "outside the map"
            # only re-use cached write-only nodes, e.g. don't create a cycle for
            # map i=0:20:
            #  A[i] = tasklet(A[i])
            if (name not in access_cache or self._current_state.out_degree(access_cache[name]) > 0
                    or id(access_cache[name]) in self._frozen_access_nodes):
                # cache write access into access_cache
                write_access_node = self._current_state.add_write(name)
                access_cache[name] = write_access_node

            access_node = access_cache[name]
            if name in sdfg.arrays:
                data_descriptor = sdfg.arrays[name]
            else:
                _sdfg = self._parent_sdfg_with_array(name, sdfg)
                data_descriptor = _sdfg.arrays[name]
            self._current_state.add_memlet_path(map_exit,
                                                access_node,
                                                src_conn=out_connector_name,
                                                memlet=Memlet.from_array(name, data_descriptor))

            if isinstance(outer_map_entry, nodes.EntryNode):
                outer_to_connect[name] = (access_node, Memlet.from_array(name, data_descriptor))
            else:
                assert isinstance(outer_map_entry, SDFG) or outer_map_entry is None

        # Connect empty scopes
        if self._current_state.in_degree(map_exit) == 0:
            self._current_state.add_nedge(map_entry, map_exit, Memlet())

    def _local_access_cache(self) -> dict[str, nodes.AccessNode]:
        """Returns the access node cache of the current state and tree scope."""
        cache_key = (self._current_state, id(self._ctx.current_scope))
        if cache_key not in self._ctx.access_cache:
            self._ctx.access_cache[cache_key] = {}
        return self._ctx.access_cache[cache_key]

    def _read_source(self, name: str, sdfg: SDFG) -> tuple[nodes.Node, str | None]:
        """
        Returns the node and output connector that provide the data container ``name`` for reading in the current
        state and dataflow scope. Creates access nodes, map pass-through connectors, and nested SDFG inputs as
        necessary.

        :param name: The name of the data container to read.
        :param sdfg: The SDFG that is currently being built.
        :return: A tuple of the source node and its output connector.
        """
        cache = self._local_access_cache()
        scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)

        # Views are materialized in the scope they are used in, viewing their (pass-through) source
        if name in self._views:
            return self._add_view_access(name, sdfg, is_write=False), None

        # connect to local access node if possible
        if name in cache:
            return cache[name], None

        if isinstance(scope_node, nodes.ConsumeEntry) and self._consume_streams.get(id(scope_node)) == name:
            # the consumed stream provides the current element
            return scope_node, 'OUT_stream'

        if isinstance(scope_node, nodes.EntryNode):
            # get it from outside the map
            connector_name = f"{PREFIX_PASSTHROUGH_OUT}{name}"
            if connector_name not in scope_node.out_connectors:
                new_in_connector = scope_node.add_in_connector(f"{PREFIX_PASSTHROUGH_IN}{name}")
                new_out_connector = scope_node.add_out_connector(connector_name)
                assert new_in_connector == True
                assert new_in_connector == new_out_connector
            return scope_node, connector_name

        if isinstance(scope_node, SDFG):
            # Copy data descriptor from parent SDFG and add input connector
            if name not in sdfg.arrays:
                parent_sdfg = self._parent_sdfg_with_array(name, sdfg)

                # Support for  NView nodes
                use_nview = self._apply_nview_array_override(name, sdfg)
                if not use_nview:
                    sdfg.add_datadesc(name, parent_sdfg.arrays[name].clone())

                    # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                    if parent_sdfg.arrays[name].transient:
                        sdfg.arrays[name].transient = False

                # Dev note: name and nview.target are identical
                to_connect["inputs"].add(name)

            # Add in_connector in case of read after (partial) write of "outside data"
            if name in self._known_data_outside_nestedSDFG:
                to_connect["inputs"].add(name)
        else:
            assert scope_node is None

        # cache local read access
        cache[name] = self._current_state.add_read(name)
        return cache[name], None

    def _connect_output(self, src: nodes.Node, src_conn: str | None, name: str, memlet: Memlet, sdfg: SDFG) -> None:
        """
        Connects a node in the current state to the data container ``name`` it writes to, creating access nodes and
        registering map / nested SDFG outputs as necessary.

        :param src: The node writing the data.
        :param src_conn: The output connector of ``src``.
        :param name: The name of the written data container.
        :param memlet: The memlet on the edge from ``src`` to the written container.
        :param sdfg: The SDFG that is currently being built.
        """
        cache = self._local_access_cache()
        scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)

        # Writes to views go through a new view access node that writes into the view's source
        if name in self._views:
            view_access = self._add_view_access(name, sdfg, is_write=True)
            self._current_state.add_edge(src, src_conn, view_access, None, memlet)
            return

        # only re-use cached write-only nodes, e.g. don't create a cycle for
        # A[1] = tasklet(A[1]) or A[1] = copy A[0]
        if (name not in cache or cache[name] is src or self._current_state.out_degree(cache[name]) > 0
                or id(cache[name]) in self._frozen_access_nodes):
            # cache write access node
            write_access_node = self._current_state.add_write(name)
            cache[name] = write_access_node

        access_node = cache[name]
        self._current_state.add_memlet_path(src, access_node, src_conn=src_conn, memlet=memlet)

        if isinstance(scope_node, nodes.EntryNode):
            # copy the memlet since we already used it in the memlet path above
            if memlet.data == name:
                to_connect[name] = (access_node, copy.deepcopy(memlet))
            else:
                # Copy memlets refer to their source container, describe the written subset instead
                if memlet.other_subset is not None:
                    subset = copy.deepcopy(memlet.other_subset)
                else:
                    subset = subsets.Range.from_array(self._ctx.root.containers[name])
                to_connect[name] = (access_node, Memlet(data=name, subset=subset))
            return

        if isinstance(scope_node, SDFG):
            if name not in sdfg.arrays:
                parent_sdfg: SDFG = self._parent_sdfg_with_array(name, sdfg)

                # Support for NView nodes
                use_nview = self._apply_nview_array_override(name, sdfg)
                if not use_nview:
                    sdfg.add_datadesc(name, parent_sdfg.arrays[name].clone())

                    # Transients passed into a nested SDFG become non-transient inside that nested SDFG
                    if parent_sdfg.arrays[name].transient:
                        sdfg.arrays[name].transient = False

            # Add out connector in any case because we don't know who (if anyone)
            # is gonna read from it down the line.
            # Dev note: name and nview.target are identical
            to_connect["outputs"].add(name)
        else:
            assert scope_node is None

    def _add_view_access(self, name: str, sdfg: SDFG, is_write: bool) -> nodes.AccessNode:
        """
        Adds an access node of a view to the current state and connects it to the viewed container.

        Reading views get an incoming ``views`` edge, writing views an outgoing one. The viewed container is accessed
        like any other container, which resolves views of views and passes the container through surrounding maps
        and nested SDFGs. View access nodes are not cached, such that every access observes the latest write to the
        viewed container.

        :param name: The name of the view container.
        :param sdfg: The SDFG that is currently being built.
        :param is_write: True if the view is written to, False if it is read.
        :return: The new view access node.
        """
        view = self._views[name]
        if name not in sdfg.arrays:
            # The view is defined outside of the nested SDFG that is being built, re-create it in there
            sdfg.add_datadesc(name, view.view_desc.clone())
        view_access = self._current_state.add_access(name)
        if is_write:
            self._connect_output(view_access, "views", view.source, copy.deepcopy(view.memlet), sdfg)
        else:
            source, source_conn = self._read_source(view.source, sdfg)
            self._current_state.add_edge(source, source_conn, view_access, "views", copy.deepcopy(view.memlet))
        return view_access

    def _add_code_node(self, code_node: nodes.CodeNode, in_memlets: dict[str, Memlet] | set[Memlet],
                       out_memlets: dict[str, Memlet] | set[Memlet], sdfg: SDFG) -> None:
        """
        Adds a code node (tasklet or library node) to the current state and connects its inputs and outputs.

        :param code_node: The code node to add.
        :param in_memlets: The input memlets per connector, or a set of memlets for connector-less nodes.
        :param out_memlets: The output memlets per connector, or a set of memlets for connector-less nodes.
        :param sdfg: The SDFG that is currently being built.
        """
        self._current_state.add_node(code_node)
        scope_node, to_connect = self._dataflow_stack[-1] if self._dataflow_stack else (None, None)
        inputs = in_memlets.items() if isinstance(in_memlets, dict) else [(None, memlet) for memlet in in_memlets]
        outputs = out_memlets.items() if isinstance(out_memlets, dict) else [(None, memlet) for memlet in out_memlets]

        for name, memlet in inputs:
            source, source_conn = self._read_source(memlet.data, sdfg)
            self._current_state.add_edge(source, source_conn, code_node, name, memlet)

        # Add empty memlet if this code node is a source node
        if isinstance(scope_node, nodes.EntryNode) and not in_memlets:
            self._current_state.add_nedge(scope_node, code_node, Memlet())

        for name, memlet in outputs:
            reference_set = self._code_reference_sets.get((id(code_node), name))
            if reference_set is not None:
                self._set_reference(code_node, name, reference_set, sdfg)
            else:
                self._connect_output(code_node, name, memlet.data, memlet, sdfg)

        # Add empty memlet if this code node is a sink node
        if isinstance(scope_node, nodes.EntryNode) and not out_memlets:
            to_connect[f"tasklet_{id(code_node)}"] = (code_node, Memlet())

    def visit_TaskletNode(self, node: tn.TaskletNode, sdfg: SDFG) -> None:
        self._add_code_node(node.node, node.in_memlets, node.out_memlets, sdfg)

    def visit_LibraryCall(self, node: tn.LibraryCall, sdfg: SDFG) -> None:
        self._add_code_node(node.node, node.in_memlets, node.out_memlets, sdfg)

    def visit_CopyNode(self, node: tn.CopyNode, sdfg: SDFG) -> None:
        source, source_conn = self._read_source(node.memlet.data, sdfg)
        memlet = node.memlet
        target_desc = self._ctx.root.containers.get(node.target)
        if isinstance(target_desc, data.Stream) and memlet.data != node.target:
            # Code generation pushes into streams only if the copy is described on the stream
            memlet = Memlet(data=node.target,
                            subset=copy.deepcopy(memlet.other_subset)
                            if memlet.other_subset is not None else subsets.Range.from_array(target_desc),
                            other_subset=copy.deepcopy(memlet.subset))
        self._connect_output(source, source_conn, node.target, memlet, sdfg)

    def visit_DynScopeCopyNode(self, node: tn.DynScopeCopyNode, sdfg: SDFG) -> None:
        # Connected to the entry node of the following dataflow scope, see ``visit_MapScope``
        self._dynamic_scope_inputs.append(node)

    def visit_ViewNode(self, node: tn.ViewNode, sdfg: SDFG) -> None:
        # Views are connected lazily: every read or write of ``node.target`` creates a view access node that is
        # connected to ``node.source`` in the state and scope of that access (see ``_add_view_access``).
        self._views[node.target] = node

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
        if isinstance(node.src_desc, nodes.CodeNode):
            if not any(ref is node for ref in self._code_reference_sets.values()):
                raise ValueError(f"Cannot find the code node that sets reference '{node.target}'.")
            return  # Connected when the code node is added, see ``_add_code_node``

        source, source_conn = self._read_source(node.memlet.data, sdfg)
        self._set_reference(source, source_conn, node, sdfg)

    def _set_reference(self, source: nodes.Node, source_conn: str | None, node: tn.RefSetNode, sdfg: SDFG) -> None:
        """
        Connects the source of a reference set to a new access node of the reference.

        :param source: The node that provides the value of the reference.
        :param source_conn: The output connector of ``source``.
        :param node: The reference set node.
        :param sdfg: The SDFG that is currently being built.
        """
        if node.target not in sdfg.arrays:
            # The reference is set inside the nested SDFG that is being built
            sdfg.add_datadesc(node.target, self._ctx.root.containers[node.target].clone())

        ref_access = self._current_state.add_access(node.target)
        self._current_state.add_edge(source, source_conn, ref_access, "set", copy.deepcopy(node.memlet))

        # Subsequent reads of the reference in this state and scope depend on the reference set
        self._local_access_cache()[node.target] = ref_access

    def visit_StateBoundaryNode(self, node: tn.StateBoundaryNode, sdfg: SDFG) -> None:
        # Boundaries that do not precede control flow or perform assignments may be expressed within the state
        if (self._boundary_behavior == StateBoundaryBehavior.EMPTY_MEMLET and not node.due_to_control_flow
                and not self._interstate_symbols):
            self._add_barrier()
            return

        # When creating a state boundary, include all inter-state assignments that precede it.
        pending = self._pending_interstate_assignments()

        self._current_state = _create_state_boundary(
            node,
            self._current_state,
            assignments=pending,
        )

    def _add_barrier(self) -> None:
        """
        Converts a state boundary into happens-before relations within the current state: every node added after this
        point executes after all nodes that already exist. The empty memlet edges are added in ``connect_barriers``,
        once all nodes are known.
        """
        state = self._current_state
        self._barriers.append((state, set(state.nodes())))

        # Writes after the boundary must use new access nodes, otherwise they would precede the nodes they follow
        self._frozen_access_nodes.update(id(n) for n in state.nodes() if isinstance(n, nodes.AccessNode))

    def connect_barriers(self) -> None:
        """
        Adds empty memlet edges for state boundaries that were converted into happens-before relations, from every
        top-level sink before a boundary to every top-level source after it. Scopes are represented by their exit node
        as a sink and by their entry node as a source.
        """
        # Later boundaries first, such that nodes after a later boundary are no longer sources for an earlier one
        for state, before in reversed(self._barriers):
            scope_dict = state.scope_dict()
            top_level = {n for n in state.nodes() if scope_dict[n] is None}
            top_level |= {state.exit_node(n) for n in top_level if isinstance(n, nodes.EntryNode)}
            preceding = top_level & before
            following = top_level - before

            sinks = [
                n for n in preceding
                if not isinstance(n, nodes.EntryNode) and not any(e.dst in preceding for e in state.out_edges(n))
            ]
            sources = [
                n for n in following
                if not isinstance(n, nodes.ExitNode) and not any(e.src in following for e in state.in_edges(n))
            ]
            for sink in sinks:
                for source in sources:
                    if not state.edges_between(sink, source):
                        state.add_nedge(sink, source, Memlet())

    def _flush_pending_assignments(self, label: str) -> None:
        """
        Performs pending inter-state assignments in a transition to a new state, e.g., at the end of a branch or loop
        body, where they would otherwise be lost or performed on the wrong path.

        :param label: The label of the new state.
        """
        if self._interstate_symbols:
            self._current_state = _insert_and_split_assignments(self._current_state,
                                                                label=label,
                                                                assignments=self._pending_interstate_assignments())

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
    result.constants_prop = copy.deepcopy(stree.constants)
    result.callback_mapping = copy.deepcopy(stree.callback_mapping)
    result.symbols = copy.deepcopy(stree.symbols)
    for code, tree_code in ((result.global_code, stree.global_code), (result.init_code, stree.init_code),
                            (result.exit_code, stree.exit_code)):
        code.update(copy.deepcopy(tree_code))

    # Restructure or nest the targets of forward gotos, such that they can be lowered to return blocks
    _lower_forward_gotos(stree)

    # Nested SDFG views are applied to the nested SDFGs that contain their accesses
    _nest_nview_regions(stree)

    # Insert artificial state boundaries after WAW, before label, etc.
    stree = _insert_state_boundaries_to_tree(stree, state_boundary_behavior)

    # Traverse tree and incrementally build SDFG, finally propagate memlets
    converter = _StreeToSDFG(boundary_behavior=state_boundary_behavior, max_nested_sdfg=max_nested_sdfgs)
    converter.visit(stree, sdfg=result)
    converter.connect_barriers()

    # Memlet directions (src/dst subsets) are determined when edges are added. Scope pass-through edges are
    # connected later than the edges inside the scope, so re-initialize them before propagation.
    for nested_sdfg in result.all_sdfgs_recursive():
        for state in nested_sdfg.states():
            for edge in state.edges():
                edge.data.try_initialize(nested_sdfg, state, edge)
    propagation.propagate_memlets_sdfg(result)

    return result


def _code_reference_sets(stree: tn.ScheduleTreeRoot) -> dict[tuple[int, str | None], tn.RefSetNode]:
    """
    Finds the outputs of code nodes that set references.

    :param stree: The schedule tree.
    :return: A mapping of (id(code node), output connector) to the reference set node of that output.
    """
    code_nodes = {id(n.node): n for n in stree.preorder_traversal() if isinstance(n, (tn.TaskletNode, tn.LibraryCall))}
    result = {}
    for ref in stree.preorder_traversal():
        if not isinstance(ref, tn.RefSetNode) or not isinstance(ref.src_desc, nodes.CodeNode):
            continue
        code = code_nodes.get(id(ref.src_desc))
        if code is None:
            raise ValueError(f"Reference '{ref.target}' is set by a code node that is not in the schedule tree.")
        outputs = code.out_memlets.items() if isinstance(code.out_memlets, dict) else [(None, m)
                                                                                       for m in code.out_memlets]
        connectors = [name for name, memlet in outputs if memlet is ref.memlet or memlet.data == ref.target]
        if len(connectors) != 1:
            raise ValueError(f"Cannot determine the output of '{ref.src_desc}' that sets reference '{ref.target}'.")
        result[(id(ref.src_desc), connectors[0])] = ref
    return result


def _reads_of(node: tn.ScheduleTreeNode, name: str) -> list[Memlet]:
    """Returns the memlets that read the given data container in a node and its descendants (one per connector)."""
    result = []
    for n in node.preorder_traversal():
        if isinstance(n, (tn.TaskletNode, tn.LibraryCall)):
            memlets = n.in_memlets.values() if isinstance(n.in_memlets, dict) else n.in_memlets
            result.extend(memlet for memlet in memlets if memlet.data == name)
        elif isinstance(n, (tn.CopyNode, tn.DynScopeCopyNode, tn.RefSetNode)) and n.memlet.data == name:
            result.append(n.memlet)
    return result


def _rename_reads(node: tn.ScheduleTreeNode, old: str, new: str) -> None:
    """
    Replaces reads of a data container in a node and its descendants with reads of another container. Memlets are
    replaced rather than modified, since they may be shared with other nodes.

    :param node: The node to operate on.
    :param old: The name of the container to replace.
    :param new: The name of the container to read instead.
    """

    def renamed(memlet: Memlet) -> Memlet:
        if memlet.data != old:
            return memlet
        result = copy.deepcopy(memlet)
        result.data = new
        return result

    for n in node.preorder_traversal():
        if isinstance(n, (tn.TaskletNode, tn.LibraryCall)):
            if isinstance(n.in_memlets, dict):
                n.in_memlets = {connector: renamed(memlet) for connector, memlet in n.in_memlets.items()}
            else:
                n.in_memlets = type(n.in_memlets)(renamed(memlet) for memlet in n.in_memlets)
        elif isinstance(n, (tn.CopyNode, tn.DynScopeCopyNode, tn.RefSetNode)):
            n.memlet = renamed(n.memlet)


def _ancestors(node: tn.ScheduleTreeNode) -> list[tn.ScheduleTreeScope]:
    """Returns the scopes containing the given node, from the innermost to the root."""
    result = []
    scope = node.parent
    while scope is not None:
        result.append(scope)
        scope = scope.parent
    return result


def _find_label(goto: tn.GotoNode) -> tn.StateLabel | None:
    """
    Returns the label a goto jumps to, if the label is contained in one of the scopes that contain the goto.
    """
    for scope in _ancestors(goto):
        for child in scope.children:
            if isinstance(child, tn.StateLabel) and child.name == goto.target:
                return child
    return None


def _is_sdfg_end(label: tn.StateLabel) -> bool:
    """
    Returns True if nothing follows the given label in the SDFG it is converted into. This is the case at the end of
    the tree root and at the end of scopes that are converted into nested SDFGs.

    Dataflow scopes that contain a label are always converted into nested SDFGs, as a state boundary is inserted
    before the label.
    """
    scope = label.parent
    if not isinstance(scope, (tn.ScheduleTreeRoot, tn.DataflowScope, _NestedSDFGScope)):
        return False
    index = _list_index(scope.children, label)
    return all(isinstance(child, (tn.StateLabel, tn.StateBoundaryNode)) for child in scope.children[index + 1:])


def _jumps_to_sdfg_end(goto: tn.GotoNode, label: tn.StateLabel) -> bool:
    """
    Returns True if the goto jumps to the end of the SDFG it is converted in, i.e., it can be lowered to a return block.
    """
    if not _is_sdfg_end(label):
        return False
    for scope in _ancestors(goto):
        if scope is label.parent:
            return True
        if isinstance(scope, (tn.DataflowScope, _NestedSDFGScope)):
            return False  # The goto is converted within another SDFG
    return False


def _lower_forward_gotos(stree: tn.ScheduleTreeRoot) -> None:
    """
    Prepares gotos outside of general blocks for their conversion into return blocks. Operates in-place.

    Exit gotos and gotos to labels that follow them in an enclosing scope (e.g., the end of an inlined nested SDFG)
    are handled in order of preference:

      1. Gotos in conditionals are removed by moving the subsequent statements into the other branches.
      2. Gotos that jump to the end of the SDFG being built are kept and later converted into return blocks.
      3. Otherwise, the statements from the first one containing a goto up to the label are wrapped in a scope that
         is converted into a nested SDFG, at whose end the label then is.

    :param stree: The schedule tree to operate on.
    """
    if _gotos_to(stree, None):
        stpasses.eliminate_forward_gotos(stree, len(stree.children), None)

    labels = [
        n for n in stree.preorder_traversal() if isinstance(n, tn.StateLabel) and not isinstance(n.parent, tn.GBlock)
    ]
    for label in labels:
        scope = label.parent
        index = _list_index(scope.children, label)
        gotos = _gotos_to(stree, label.name)
        if not gotos:
            continue

        preceding = {id(n) for child in scope.children[:index] for n in child.preorder_traversal()}
        for goto in gotos:
            if id(goto) not in preceding:
                raise ValueError(f"Cannot convert '{goto.as_string().strip()}': gotos may only jump forward to a "
                                 "label in an enclosing scope.")
            enclosed_by = _ancestors(goto)
            if any(isinstance(s, tn.DataflowScope) for s in enclosed_by[:_list_index(enclosed_by, scope)]):
                raise ValueError(f"Cannot convert '{goto.as_string().strip()}': gotos may not leave dataflow scopes.")

        if stpasses.eliminate_forward_gotos(scope, index, label.name):
            continue
        if all(_jumps_to_sdfg_end(goto, label) for goto in gotos):
            continue

        # Nest the statements from the first one containing a goto up to (and including) the label
        first = next(i for i, child in enumerate(scope.children) if _gotos_to(child, label.name))
        nested = [n for child in scope.children[first:index + 1] for n in child.preorder_traversal()]
        nested_labels = {n.name for n in nested if isinstance(n, tn.StateLabel)}
        for goto in (n for n in nested if isinstance(n, tn.GotoNode)):
            if goto.target is None or goto.target not in nested_labels:
                raise NotImplementedError(f"Cannot convert '{goto.as_string().strip()}' within the nested SDFG that "
                                          f"is required for the gotos to '{label.name}'.")
        scope.children[first:index + 1] = [_NestedSDFGScope(children=scope.children[first:index + 1], parent=scope)]


def _nest_nview_regions(stree: tn.ScheduleTreeRoot) -> None:
    """
    Wraps the nodes from every nested SDFG view (``NView``) to its end (``NViewEnd``) in a scope that is converted into
    a nested SDFG, whose data descriptor of the view target is the view. Operates in-place.

    :param stree: The schedule tree to operate on.
    """
    scopes = [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]
    for scope in scopes:
        index = 0
        while index < len(scope.children):
            child = scope.children[index]
            if type(child) is not tn.NView:
                index += 1
                continue
            end = next((i for i in range(index + 1, len(scope.children))
                        if isinstance(scope.children[i], tn.NViewEnd) and scope.children[i].target == child.target),
                       None)
            if end is None:
                raise ValueError(f"No end found for nested SDFG view '{child.as_string().strip()}'.")
            region = scope.children[index:end + 1]
            scope.children[index:end + 1] = [_NestedSDFGScope(children=region, parent=scope)]
            index += 1


def _gotos_to(node: tn.ScheduleTreeNode, target: str | None) -> list[tn.GotoNode]:
    """Returns the gotos to the given target (``None`` for exit gotos) in the given node and its descendants."""
    return [n for n in node.preorder_traversal() if isinstance(n, tn.GotoNode) and n.target == target]


def _insert_state_boundaries_to_tree(
    stree: tn.ScheduleTreeRoot,
    boundary_behavior: StateBoundaryBehavior = StateBoundaryBehavior.STATE_TRANSITION,
) -> tn.ScheduleTreeRoot:
    """
    Inserts StateBoundaryNode objects into a schedule tree where more than one SDFG state would be necessary.
    Operates in-place on the given schedule tree.

    This happens when there is a:
      * write-after-write dependency;
      * write-after-read dependency that cannot be fulfilled via memlets;
      * control flow block (for/if); or
      * otherwise before a state label (which means a state transition could occur, e.g., in a gblock)

    :param stree: The schedule tree to operate on.
    :param boundary_behavior: The behavior of the conversion upon state boundaries. With
                              ``StateBoundaryBehavior.EMPTY_MEMLET``, boundaries that must start a new state (e.g.,
                              before nested SDFGs) are marked as due to control flow.
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

    # Hack: "backprop-insert" state boundaries from nested SDFGs
    class NestedSDFGStateBoundaryInserter(tn.ScheduleNodeTransformer):

        def visit_MapScope(self, scope: tn.DataflowScope):
            visited = self.generic_visit(scope)
            if any([isinstance(child, tn.StateBoundaryNode) for child in scope.children]):
                # We can assume that map nodes are at least contained in the root scope.
                assert scope.parent is not None

                # Find this scope in its parent's children
                node_index = _list_index(scope.parent.children, scope)

                # If there's already a state boundary before the map, don't add another one
                if node_index > 0 and isinstance(scope.parent.children[node_index - 1], tn.StateBoundaryNode):
                    return visited

                # The nested SDFG starts a new state, even if other boundaries are converted within states
                return [tn.StateBoundaryNode(boundary_behavior == StateBoundaryBehavior.EMPTY_MEMLET), visited]
            return visited

        visit_ConsumeScope = visit_MapScope

    stree = NestedSDFGStateBoundaryInserter().visit(stree)

    return stree


def _view_source_memlets(memlets: Sequence[Memlet], views: dict[str, Memlet]) -> list[Memlet]:
    """
    Replaces memlets on views with memlets on the containers they view.

    :param memlets: The memlets to resolve.
    :param views: A mapping from view names to a memlet on the viewed container, covering the entire view.
    :return: The resolved memlets.
    """
    result = []
    for memlet in memlets:
        visited = set()
        while memlet.data in views and memlet.data not in visited:
            visited.add(memlet.data)
            memlet = views[memlet.data]
        result.append(memlet)
    return result


def _insert_memory_dependency_state_boundaries(scope: tn.ScheduleTreeScope, views: dict[str, Memlet] | None = None):
    """
    Helper function that inserts boundaries after unmet memory dependencies.

    Accesses to views are treated as accesses to the entire viewed subset of their source containers, such that
    aliasing reads and writes are detected.

    :param scope: The scope to insert state boundaries in.
    :param views: Views defined before this scope, mapping view names to memlets on the viewed containers.
    """
    reads: mmu.MemletDict[list[tn.ScheduleTreeNode]] = mmu.MemletDict()
    writes: mmu.MemletDict[list[tn.ScheduleTreeNode]] = mmu.MemletDict()
    parents: dict[int, set[int]] = defaultdict(set)
    boundaries_to_insert: list[int] = []
    views = dict(views) if views is not None else {}

    for i, n in enumerate(scope.children):
        # Views do not move data, but define an alias for subsequent nodes
        if type(n) is tn.ViewNode:
            views[n.target] = Memlet(data=n.source, subset=copy.deepcopy(n.memlet.subset))
            continue

        if isinstance(n, (tn.StateBoundaryNode, tn.ControlFlowScope)):  # Clear state
            reads.clear()
            writes.clear()
            parents.clear()
            if isinstance(n, tn.ControlFlowScope):  # Insert memory boundaries recursively
                _insert_memory_dependency_state_boundaries(n, views)
            continue

        # If dataflow scope, insert state boundaries recursively and as a node
        scope_views = views
        if isinstance(n, tn.DataflowScope):
            _insert_memory_dependency_state_boundaries(n, views)

            # Views defined inside the scope may depend on its parameters, assume the entire source is accessed
            scope_views = dict(views)
            containers = scope.get_root().containers
            for child in n.preorder_traversal():
                if type(child) is tn.ViewNode and child.source in containers:
                    scope_views[child.target] = Memlet.from_array(child.source, containers[child.source])

        inputs = _view_source_memlets(n.input_memlets(), scope_views)
        outputs = _view_source_memlets(n.output_memlets(), scope_views)

        def register_reads() -> None:
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

        def unordered_reads(o: Memlet) -> bool:
            """Returns True if another node read the output before, but is not guaranteed to run before this node."""
            return any(r is not n and id(r) not in parents[id(n)] for r in reads[o])

        register_reads()

        # Inter-state assignment nodes with reads necessitate a state transition if they were written to.
        needs_boundary = isinstance(n, tn.AssignNode) and any(inp in writes for inp in inputs)

        # Write after write or potential write/write data race, insert state boundary
        needs_boundary = needs_boundary or any(o in writes and (o not in reads or unordered_reads(o)) for o in outputs)

        # Potential read/write data race: if any read is not in the parents of this node, it might
        # be performed in parallel
        needs_boundary = needs_boundary or any(o in reads and unordered_reads(o) for o in outputs)

        if needs_boundary:
            boundaries_to_insert.append(i)
            reads.clear()
            writes.clear()
            parents.clear()

            # This node is the first one after the boundary
            register_reads()

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


def _list_index(list: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode) -> int:
    """Check if node is in list with "is" operator."""
    index = 0
    for element in list:
        # compare with "is" to get memory comparison. ".index()" uses value comparison
        if element is node:
            return index
        index += 1

    raise StopIteration
