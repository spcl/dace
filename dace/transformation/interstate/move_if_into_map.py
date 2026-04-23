# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transformation that pushes a conditional block through one level of map nesting.

When a ``Map`` whose body is a single ``NestedSDFG`` guards an inner ``Map`` via
a conditional block, the outer guard can be pushed inside the inner map's body.
This exposes the two maps to fusion/collapsing passes that otherwise refuse to
cross the conditional block.
"""
import copy
from typing import Optional, Tuple

from dace import dtypes
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
    def _find_inner_map_pieces(
            branch_state: SDFGState) -> Optional[Tuple[MapEntry, MapExit, NestedSDFG]]:
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

        _, _, inner_nsdfg = self._find_inner_map_pieces(branch_state)  # type: ignore[misc]
        inner_sdfg: sd.SDFG = inner_nsdfg.sdfg

        cond_sym = enclosing_sdfg.add_symbol(f"{cond_block.label}_cond",
                                             dtypes.int32,
                                             find_new_name=True)

        if cond_sym not in inner_sdfg.symbols:
            inner_sdfg.add_symbol(cond_sym, dtypes.int32)
        inner_nsdfg.symbol_mapping[cond_sym] = cond_sym

        new_body_cfr = ControlFlowRegion(label=f"{cond_block.label}_body")
        body_blocks = list(inner_sdfg.nodes())
        original_start = inner_sdfg.start_block
        copy_mapping = {}
        for b in body_blocks:
            new_b = copy.deepcopy(b)
            copy_mapping[b] = new_b
            new_body_cfr.add_node(new_b, is_start_block=(b is original_start))
        for e in inner_sdfg.edges():
            new_body_cfr.add_edge(copy_mapping[e.src], copy_mapping[e.dst],
                                  copy.deepcopy(e.data))

        new_cond_block = ConditionalBlock(label=f"{cond_block.label}_moved")
        new_cond_block.add_branch(CodeBlock(f"{cond_sym} == 1"), new_body_cfr)

        inner_sdfg.remove_nodes_from(body_blocks)
        inner_sdfg.add_node(new_cond_block, is_start_block=True, ensure_unique_name=True)

        new_branch_state = copy.deepcopy(branch_state)
        enclosing_sdfg.add_node(new_branch_state, ensure_unique_name=True)

        in_edges = list(enclosing_sdfg.in_edges(cond_block))
        out_edges = list(enclosing_sdfg.out_edges(cond_block))
        was_start = enclosing_sdfg.start_block is cond_block

        cond_assignment = {cond_sym: branch_cond.as_string}

        if len(in_edges) == 0:
            pre_state = enclosing_sdfg.add_state(label=f"{cond_block.label}_materialize")
            enclosing_sdfg.add_edge(pre_state, new_branch_state,
                                    InterstateEdge(assignments=cond_assignment))
            if was_start:
                enclosing_sdfg.start_block = enclosing_sdfg.node_id(pre_state)
        else:
            for e in in_edges:
                new_data = copy.deepcopy(e.data)
                for k, v in cond_assignment.items():
                    new_data.assignments[k] = v
                enclosing_sdfg.add_edge(e.src, new_branch_state, new_data)
                enclosing_sdfg.remove_edge(e)

        for e in out_edges:
            enclosing_sdfg.add_edge(new_branch_state, e.dst, copy.deepcopy(e.data))
            enclosing_sdfg.remove_edge(e)

        enclosing_sdfg.remove_node(cond_block)

        set_nested_sdfg_parent_references(sdfg)
