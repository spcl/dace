# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transformation that pushes a conditional block through one level of map nesting.

When a ``Map`` whose body is a single ``NestedSDFG`` guards an inner ``Map`` via
a conditional block, the outer guard can be pushed inside the inner map's body.
This exposes the two maps to fusion/collapsing passes that otherwise refuse to
cross the conditional block.
"""
import copy
from typing import Optional

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

        outer_state (in outer_sdfg):
            ... -> outer_map_entry -> outer_nsdfg -> outer_map_exit -> ...

        outer_nsdfg.sdfg contents:
            [optional empty state ->] ConditionalBlock(cond):
                branch (single, with condition `cond`):
                    branch_state:
                        ... -> inner_map_entry -> inner_nsdfg -> inner_map_exit -> ...

    After the transformation, ``outer_nsdfg.sdfg`` contains a single state
    holding the inner map, and ``inner_nsdfg.sdfg`` is wrapped in a new
    ``ConditionalBlock`` carrying the original condition. The condition value
    is materialized into a scalar symbol on the interstate edge so that it is
    evaluated once (at the outer level, where all free symbols are in scope)
    and then passed through to the inner nested SDFG via ``symbol_mapping``.
    """

    cond_block = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cond_block)]

    # ---- Pattern matching ------------------------------------------------

    def _find_inner_map_pieces(self, branch_state: SDFGState):
        """Returns (map_entry, map_exit, inner_nsdfg) if the branch state
        matches the expected pattern, else None."""
        map_entries = [n for n in branch_state.nodes() if isinstance(n, MapEntry)]
        if len(map_entries) != 1:
            return None
        map_entry = map_entries[0]
        map_exit = branch_state.exit_node(map_entry)

        # Only one top-level map allowed: every non-access-node at scope 0
        # must be the map entry/exit.
        for node in branch_state.nodes():
            if branch_state.entry_node(node) is not None:
                continue  # inside the map scope
            if node is map_entry or node is map_exit:
                continue
            if isinstance(node, AccessNode):
                continue
            return None

        # Map body must be exactly one NestedSDFG (access nodes allowed as
        # taps inside the scope).
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

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        cond_block: ConditionalBlock = self.cond_block

        # The conditional must have a single branch with a real condition.
        if len(cond_block.branches) != 1:
            return False
        branch_cond, branch_cfr = cond_block.branches[0]
        if branch_cond is None:
            return False

        # cond_block must sit directly in the SDFG of a NestedSDFG (no wrapping
        # control-flow regions, loops, etc.).
        parent_graph = cond_block.parent_graph
        if not isinstance(parent_graph, sd.SDFG):
            return False
        enclosing_sdfg: sd.SDFG = parent_graph

        # The enclosing SDFG must itself be nested inside an outer state.
        if enclosing_sdfg.parent_nsdfg_node is None or enclosing_sdfg.parent is None:
            return False
        outer_state: SDFGState = enclosing_sdfg.parent
        outer_nsdfg: NestedSDFG = enclosing_sdfg.parent_nsdfg_node

        # That outer NestedSDFG must live inside a Map scope (the "outer map").
        if outer_state.entry_node(outer_nsdfg) is None:
            return False

        # Inside enclosing_sdfg, only cond_block is allowed (plus optionally a
        # single empty state feeding it).
        nodes = list(enclosing_sdfg.nodes())
        if len(nodes) == 1:
            if nodes[0] is not cond_block:
                return False
        elif len(nodes) == 2:
            other = nodes[0] if nodes[1] is cond_block else nodes[1]
            if not isinstance(other, SDFGState) or not other.is_empty():
                return False
            # Other must be the predecessor of cond_block.
            in_edges = list(enclosing_sdfg.in_edges(cond_block))
            if len(in_edges) != 1 or in_edges[0].src is not other:
                return False
        else:
            return False

        # Branch must contain exactly one state.
        if not isinstance(branch_cfr, ControlFlowRegion):
            return False
        branch_blocks = list(branch_cfr.nodes())
        if len(branch_blocks) != 1 or not isinstance(branch_blocks[0], SDFGState):
            return False
        branch_state: SDFGState = branch_blocks[0]

        # Inner map + inner nsdfg pattern.
        if self._find_inner_map_pieces(branch_state) is None:
            return False

        return True

    # ---- Application -----------------------------------------------------

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        cond_block: ConditionalBlock = self.cond_block
        enclosing_sdfg: sd.SDFG = cond_block.parent_graph  # type: ignore[assignment]

        branch_cond: CodeBlock = cond_block.branches[0][0]
        branch_cfr: ControlFlowRegion = cond_block.branches[0][1]
        branch_state: SDFGState = [b for b in branch_cfr.nodes() if isinstance(b, SDFGState)][0]

        _, _, inner_nsdfg = self._find_inner_map_pieces(branch_state)  # type: ignore[misc]
        inner_sdfg: sd.SDFG = inner_nsdfg.sdfg

        # 1. Allocate a fresh int32 symbol at enclosing_sdfg level to carry the
        #    condition value into the inner nested SDFG.
        cond_sym = enclosing_sdfg.add_symbol(f"{cond_block.label}_cond",
                                             dtypes.int32,
                                             find_new_name=True)

        # Make the symbol visible inside inner_sdfg and pipe it via the NSDFG
        # symbol mapping.
        if cond_sym not in inner_sdfg.symbols:
            inner_sdfg.add_symbol(cond_sym, dtypes.int32)
        inner_nsdfg.symbol_mapping[cond_sym] = cond_sym

        # 2. Wrap the inner_sdfg's contents in a new ConditionalBlock.
        new_body_cfr = ControlFlowRegion(label=f"{cond_block.label}_body")
        original_start = inner_sdfg.start_block
        node_mapping = {}
        for node in list(inner_sdfg.nodes()):
            node_mapping[node] = node  # we'll move, not copy
        for node in list(inner_sdfg.nodes()):
            new_body_cfr.add_node(node, is_start_block=(node is original_start))
        for edge in list(inner_sdfg.edges()):
            new_body_cfr.add_edge(edge.src, edge.dst, edge.data)
        # Remove nodes from inner_sdfg (edges go with them).
        inner_sdfg.remove_nodes_from(list(node_mapping.keys()))

        new_cond_block = ConditionalBlock(label=f"{cond_block.label}_moved")
        new_cond_block.add_branch(CodeBlock(f"{cond_sym} == 1"), new_body_cfr)
        inner_sdfg.add_node(new_cond_block, is_start_block=True, ensure_unique_name=True)

        # 3. Replace cond_block in enclosing_sdfg with a copy of branch_state.
        #    The interstate edge feeding branch_state carries the
        #    ``cond_sym = <original condition>`` assignment.
        new_branch_state = copy.deepcopy(branch_state)
        # Re-home the deep-copied inner NestedSDFG node so its .sdfg reference
        # points to the *same* inner_sdfg we just rewrote (deepcopy on a state
        # containing a NestedSDFG duplicates inner_sdfg; we want to keep our
        # rewritten one).
        for node in list(new_branch_state.nodes()):
            if isinstance(node, NestedSDFG) and node.label == inner_nsdfg.label:
                node.sdfg = inner_sdfg
                inner_sdfg.parent_nsdfg_node = node
                inner_sdfg.parent = new_branch_state
                inner_sdfg.parent_sdfg = enclosing_sdfg.parent_sdfg if False else None  # fixed below
                # Ensure the moved symbol mapping is present on the live node too.
                node.symbol_mapping[cond_sym] = cond_sym
                break

        enclosing_sdfg.add_node(new_branch_state, ensure_unique_name=True)

        in_edges = list(enclosing_sdfg.in_edges(cond_block))
        out_edges = list(enclosing_sdfg.out_edges(cond_block))
        was_start = enclosing_sdfg.start_block is cond_block

        cond_assignment = {cond_sym: branch_cond.as_string}

        if len(in_edges) == 0:
            # cond_block was the start block -> add a predecessor state.
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

        # 4. Repair parent/sdfg references throughout the tree.
        set_nested_sdfg_parent_references(sdfg)
