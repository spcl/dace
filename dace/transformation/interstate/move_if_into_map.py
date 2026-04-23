# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transformation that pushes a conditional block through one level of map nesting.

When a ``Map`` whose body is a single ``NestedSDFG`` guards an inner ``Map`` via
a conditional block, the outer guard can be pushed inside the inner map's body.
This exposes the two maps to fusion/collapsing passes that otherwise refuse to
cross the conditional block.
"""
import copy
from typing import Dict, Optional, Set, Tuple

from dace import dtypes, memlet as mm, symbolic
from dace import sdfg as sd
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import MapEntry, MapExit, NestedSDFG, AccessNode
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import transformation


@transformation.explicit_cf_compatible
class MoveIfIntoMap(transformation.MultiStateTransformation):
    """
    Pushes a conditional block through a level of map nesting.

    Matches the following structure::

        outer_state (in some outer SDFG):
            ... -> outer_map_entry -> outer_nsdfg -> outer_map_exit -> ...

        outer_nsdfg.sdfg contents (possibly preceded by empty states):
            ConditionalBlock(<cond>):
                single branch with a control-flow region containing
                    (optional empty states) -> branch_state
                    branch_state contains:
                        ... -> inner_map_entry -> inner_nsdfg -> inner_map_exit -> ...

    After the transformation, the conditional block at the outer nested SDFG
    level is replaced by ``branch_state`` (so the two maps become neighbours),
    and a new conditional block with the original condition is placed *inside*
    ``inner_nsdfg.sdfg`` wrapping its original contents. The condition value
    is materialised as an ``int32`` symbol on the interstate edge leading to
    the branch state and passed through to ``inner_nsdfg`` via
    ``symbol_mapping`` so it is evaluated exactly once, at the outer level
    where all its free symbols are still in scope.
    """

    cond_block = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cond_block)]

    # ---- Helpers ---------------------------------------------------------

    @staticmethod
    def _single_meaningful_state(region: ControlFlowRegion) -> Optional[SDFGState]:
        """Returns the single non-empty SDFGState inside ``region`` if the
        region's other blocks are all empty SDFGStates, else None."""
        blocks = list(region.nodes())
        if not blocks:
            return None
        for b in blocks:
            if not isinstance(b, SDFGState):
                return None
        non_empty = [s for s in blocks if not s.is_empty()]
        if len(non_empty) != 1:
            return None
        return non_empty[0]

    @staticmethod
    def _find_inner_map_pieces(branch_state: SDFGState) -> Optional[Tuple[MapEntry, MapExit, NestedSDFG]]:
        """Returns (map_entry, map_exit, inner_nsdfg) if ``branch_state`` has
        exactly one top-level map whose body is a single NestedSDFG (with
        access-node taps allowed around it)."""
        map_entries = [n for n in branch_state.nodes() if isinstance(n, MapEntry)]
        if len(map_entries) != 1:
            return None
        map_entry = map_entries[0]
        map_exit = branch_state.exit_node(map_entry)

        for node in branch_state.nodes():
            if branch_state.entry_node(node) is not None:
                continue
            if node is map_entry or node is map_exit:
                continue
            if isinstance(node, AccessNode):
                continue
            return None

        body = list(branch_state.all_nodes_between(map_entry, map_exit))
        nsdfgs = [n for n in body if isinstance(n, NestedSDFG)]
        if len(nsdfgs) != 1:
            return None
        inner_nsdfg = nsdfgs[0]

        for node in body:
            if node is inner_nsdfg:
                continue
            if isinstance(node, AccessNode):
                continue
            return None

        return map_entry, map_exit, inner_nsdfg

    # ---- Pattern matching ------------------------------------------------

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        cond_block: ConditionalBlock = self.cond_block

        if len(cond_block.branches) != 1:
            return False
        branch_cond, branch_cfr = cond_block.branches[0]
        if branch_cond is None:
            return False

        parent_graph = cond_block.parent_graph
        if not isinstance(parent_graph, sd.SDFG):
            return False
        enclosing_sdfg: sd.SDFG = parent_graph

        if enclosing_sdfg.parent_nsdfg_node is None or enclosing_sdfg.parent is None:
            return False
        outer_state: SDFGState = enclosing_sdfg.parent
        outer_nsdfg: NestedSDFG = enclosing_sdfg.parent_nsdfg_node

        if outer_state.entry_node(outer_nsdfg) is None:
            return False

        for b in enclosing_sdfg.nodes():
            if b is cond_block:
                continue
            if not isinstance(b, SDFGState) or not b.is_empty():
                return False

        if not isinstance(branch_cfr, ControlFlowRegion):
            return False
        branch_state = self._single_meaningful_state(branch_cfr)
        if branch_state is None:
            return False

        if self._find_inner_map_pieces(branch_state) is None:
            return False

        return True

    # ---- Application -----------------------------------------------------

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        cond_block: ConditionalBlock = self.cond_block
        enclosing_sdfg: sd.SDFG = cond_block.parent_graph  # type: ignore[assignment]

        branch_cond: CodeBlock = cond_block.branches[0][0]
        branch_cfr: ControlFlowRegion = cond_block.branches[0][1]
        branch_state: SDFGState = self._single_meaningful_state(branch_cfr)  # type: ignore[assignment]

        inner_map_entry, _, inner_nsdfg = self._find_inner_map_pieces(branch_state)  # type: ignore[misc]
        inner_sdfg: sd.SDFG = inner_nsdfg.sdfg

        try:
            cond_free_syms = {str(s) for s in branch_cond.get_free_symbols()}
        except Exception:
            cond_free_syms = set()

        # Pull out any interstate-edge assignment feeding ``cond_block``
        # whose LHS is referenced by the guard condition. Those assignments
        # must move inside the inner NSDFG together with the condition they
        # define so that none of them is left orphaned on the outer edges
        # (they would otherwise add an always-true pre-state to the outer
        # map body, blocking collapse).
        in_edges = list(enclosing_sdfg.in_edges(cond_block))
        moved_assignments: Dict[str, str] = {}
        for e in in_edges:
            for k in list(e.data.assignments.keys()):
                if k in cond_free_syms:
                    moved_assignments[k] = e.data.assignments[k]
                    del e.data.assignments[k]

        # Identify arrays referenced by the moved assignments -- they must
        # be piped as inputs into the inner NSDFG so the moved-inside
        # assignments can still evaluate.
        arrays_to_pipe: Set[str] = set()
        for sym, rhs in moved_assignments.items():
            tmp_edge = InterstateEdge(assignments={sym: rhs})
            for mmlt in tmp_edge.get_read_memlets(enclosing_sdfg.arrays):
                arrays_to_pipe.add(mmlt.data)

        piped_array_shape_syms: Set[str] = set()
        for arr_name in arrays_to_pipe:
            if arr_name not in inner_sdfg.arrays:
                desc = copy.deepcopy(enclosing_sdfg.arrays[arr_name])
                desc.transient = False
                inner_sdfg.add_datadesc(arr_name, desc)
            if arr_name not in inner_nsdfg.in_connectors:
                inner_nsdfg.add_in_connector(arr_name)
            piped_array_shape_syms.update(str(s) for s in enclosing_sdfg.arrays[arr_name].free_symbols)

        # Pipe free symbols that the condition and moved RHS expressions
        # reference. Skip any names that now live as data descriptors OR
        # that are themselves defined inside (LHS of a moved assignment).
        syms_needed: Set[str] = set(cond_free_syms)
        for rhs in moved_assignments.values():
            try:
                for s in symbolic.pystr_to_symbolic(rhs).free_symbols:
                    syms_needed.add(str(s))
            except Exception:
                pass
        syms_needed |= piped_array_shape_syms
        syms_needed -= arrays_to_pipe
        syms_needed -= set(moved_assignments.keys())
        for sym_name in syms_needed:
            if sym_name in enclosing_sdfg.symbols:
                if sym_name not in inner_sdfg.symbols:
                    inner_sdfg.add_symbol(sym_name, enclosing_sdfg.symbols[sym_name])
                if sym_name not in inner_nsdfg.symbol_mapping:
                    inner_nsdfg.symbol_mapping[sym_name] = sym_name

        # Declare the moved-assignment LHS symbols inside the inner SDFG.
        # They're defined by the pre-state's interstate edge below.
        for sym in moved_assignments:
            if sym not in inner_sdfg.symbols:
                dtype = enclosing_sdfg.symbols.get(sym, dtypes.int32)
                inner_sdfg.add_symbol(sym, dtype)

        new_body_cfr = ControlFlowRegion(label=f"{cond_block.label}_body")
        body_blocks = list(inner_sdfg.nodes())
        original_start = inner_sdfg.start_block
        copy_mapping = {}
        for b in body_blocks:
            new_b = copy.deepcopy(b)
            copy_mapping[b] = new_b
            new_body_cfr.add_node(new_b, is_start_block=(b is original_start))
        for e in inner_sdfg.edges():
            new_body_cfr.add_edge(copy_mapping[e.src], copy_mapping[e.dst], copy.deepcopy(e.data))

        new_cond_block = ConditionalBlock(label=f"{cond_block.label}_moved")
        new_cond_block.add_branch(CodeBlock(branch_cond.as_string), new_body_cfr)

        inner_sdfg.remove_nodes_from(body_blocks)
        inner_sdfg.add_node(new_cond_block, is_start_block=True, ensure_unique_name=True)

        # If there are assignments to move, prepend a pre-state that
        # materialises them via an interstate edge to ``new_cond_block``.
        if moved_assignments:
            inner_pre = inner_sdfg.add_state(f"{cond_block.label}_materialize")
            inner_sdfg.add_edge(inner_pre, new_cond_block, InterstateEdge(assignments=dict(moved_assignments)))
            inner_sdfg.start_block = inner_sdfg.node_id(inner_pre)

        new_branch_state = copy.deepcopy(branch_state)
        enclosing_sdfg.add_node(new_branch_state, ensure_unique_name=True)

        # Wire the newly-needed array inputs through the copied inner map
        # to the copied inner NSDFG node. The copied ``new_branch_state``
        # has its own MapEntry/NestedSDFG; we look them up again.
        if arrays_to_pipe:
            copied_entry, _, copied_nsdfg = self._find_inner_map_pieces(new_branch_state)
            for arr_name in arrays_to_pipe:
                if arr_name not in copied_nsdfg.in_connectors:
                    copied_nsdfg.add_in_connector(arr_name)
                if "IN_" + arr_name not in copied_entry.in_connectors:
                    copied_entry.add_in_connector("IN_" + arr_name)
                if "OUT_" + arr_name not in copied_entry.out_connectors:
                    copied_entry.add_out_connector("OUT_" + arr_name)
                arr_read = new_branch_state.add_read(arr_name)
                new_branch_state.add_edge(arr_read, None, copied_entry, "IN_" + arr_name,
                                          mm.Memlet.from_array(arr_name, enclosing_sdfg.arrays[arr_name]))
                new_branch_state.add_edge(copied_entry, "OUT_" + arr_name, copied_nsdfg, arr_name,
                                          mm.Memlet.from_array(arr_name, enclosing_sdfg.arrays[arr_name]))

        out_edges = list(enclosing_sdfg.out_edges(cond_block))
        was_start = enclosing_sdfg.start_block is cond_block

        # Wire in_edges to new_branch_state. If an edge has been emptied
        # by the moved-inside step (no assignments, no condition), drop it
        # and also drop its source state when the source is an empty
        # placeholder that exists only to feed the ConditionalBlock.
        states_to_try_remove: Set[SDFGState] = set()
        for e in in_edges:
            if e.data.is_unconditional() and not e.data.assignments:
                states_to_try_remove.add(e.src) if isinstance(e.src, SDFGState) else None
                enclosing_sdfg.remove_edge(e)
            else:
                enclosing_sdfg.add_edge(e.src, new_branch_state, copy.deepcopy(e.data))
                enclosing_sdfg.remove_edge(e)

        if was_start or not in_edges:
            enclosing_sdfg.start_block = enclosing_sdfg.node_id(new_branch_state)

        for e in out_edges:
            enclosing_sdfg.add_edge(new_branch_state, e.dst, copy.deepcopy(e.data))
            enclosing_sdfg.remove_edge(e)

        enclosing_sdfg.remove_node(cond_block)

        # Drop the moved symbols from the enclosing SDFG's symbol table if
        # no interstate edge or nested SDFG symbol-mapping still uses them.
        # Otherwise the enclosing SDFG would appear to require them as free
        # symbols from its own parent, but they're now defined (and
        # consumed) purely inside the inner NSDFG.
        def _still_references(sdfg: sd.SDFG, name: str) -> bool:
            # Only scan the enclosing SDFG's own top-level edges and
            # direct-child NestedSDFG symbol_mappings; anything deeper is
            # encapsulated and doesn't pull ``name`` out of this scope.
            for e in sdfg.edges():
                if not isinstance(e.data, InterstateEdge):
                    continue
                for v in e.data.assignments.values():
                    try:
                        if name in {str(s) for s in symbolic.pystr_to_symbolic(v).free_symbols}:
                            return True
                    except Exception:
                        pass
                if e.data.condition is not None:
                    try:
                        if name in {str(s) for s in e.data.condition.get_free_symbols()}:
                            return True
                    except Exception:
                        pass
            for state in sdfg.states():
                for n in state.nodes():
                    if isinstance(n, NestedSDFG):
                        for v in n.symbol_mapping.values():
                            try:
                                if name in {str(s) for s in symbolic.pystr_to_symbolic(str(v)).free_symbols}:
                                    return True
                            except Exception:
                                pass
            return False

        for sym in list(moved_assignments.keys()):
            if sym in enclosing_sdfg.symbols and not _still_references(enclosing_sdfg, sym):
                enclosing_sdfg.remove_symbol(sym)

        # Drop empty placeholder pre-states that are no longer reachable.
        for s in states_to_try_remove:
            if (s in enclosing_sdfg.nodes() and s.is_empty() and enclosing_sdfg.in_degree(s) == 0
                    and enclosing_sdfg.out_degree(s) == 0):
                was_start_src = enclosing_sdfg.start_block is s
                enclosing_sdfg.remove_node(s)
                if was_start_src:
                    enclosing_sdfg.start_block = enclosing_sdfg.node_id(new_branch_state)

        set_nested_sdfg_parent_references(sdfg)
