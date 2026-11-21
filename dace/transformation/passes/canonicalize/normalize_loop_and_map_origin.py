# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rebase every Map range and LoopRegion counter to a 0-based begin, KEEPING the stride: ``b:e:s`` -> ``0:(e-b):s`` with
``p -> p + b`` substituted in the scope, so a subscript ``a[p]`` becomes ``a[p+b]`` -- affine with coefficient 1. Pure
subtraction: no trip count, no division."""

from collections.abc import Iterable
from typing import Any

import dace
from dace import SDFG, properties, subsets
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, LoopRegion
from dace.symbolic import pystr_to_symbolic, symstr
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis, map_scope
from dace.transformation.passes.offset_loop_and_maps import (
    add_to_rhs,
    process_memlets_in_edges,
    repl_recursive,
    repl_tasklets_on_node_list,
    tasklets_assign,
)


def rebinds_params(node_list: Iterable[nodes.Node], params: dict[str, None]) -> bool:
    """Whether a NestedSDFG under ``node_list`` binds one of ``params`` to a differently named inner symbol.

    :param node_list: Nodes of one scope / state.
    :param params: Names of the parameters about to be shifted.
    :returns: ``True`` if any binding of a shifted parameter is not the identity.
    """
    for node in node_list:
        if not isinstance(node, nodes.NestedSDFG):
            continue
        for inner, outer in node.symbol_mapping.items():
            if str(inner) == str(outer):
                continue
            outer_symbols = (str(s) for s in pystr_to_symbolic(str(outer)).free_symbols)
            if str(inner) in params or any(s in params for s in outer_symbols):
                return True
        # Every binding is the identity, so the same names carry through -- keep checking down.
        if rebinds_params([n for state in node.sdfg.states() for n in state.nodes()], params):
            return True
    return False


def shift_map_ranges(node_list: Iterable[nodes.Node], repldict: dict[str, str]) -> None:
    """Substitute ``repldict`` into the range of every Map under ``node_list``, nested SDFGs included.

    :param node_list: Nodes of one scope / state.
    :param repldict: Parameter name -> replacement expression, as source strings.
    """
    symrepl = {pystr_to_symbolic(name): pystr_to_symbolic(value) for name, value in repldict.items()}
    for node in node_list:
        if isinstance(node, nodes.MapEntry):
            node.map.range.replace(symrepl)
        elif isinstance(node, nodes.NestedSDFG):
            shift_map_ranges([n for state in node.sdfg.states() for n in state.nodes()], repldict)


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeLoopAndMapOrigin(ppl.Pass):
    """Rebase every Map range / ``LoopRegion`` counter to begin at 0, keeping the stride (``b:e:s`` -> ``0:(e-b):s``,
    ``p -> p + b`` substituted in the scope)."""

    CATEGORY: str = "Canonicalization"

    validate = properties.Property(
        dtype=bool,
        default=False,
        desc="Validate the SDFG once after the pass changed it. Off by default: the caller validates its result.",
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Rebase every Map/LoopRegion begin to 0 (recursively), keeping stride."""
        rebased = self.visit_region(sdfg)
        if rebased == 0:
            return None
        if self.validate:
            sdfg.validate()
        return rebased

    def visit_region(self, region: AbstractControlFlowRegion) -> int:
        """Visit every block of ``region`` in order."""
        rebased = 0
        for block in region.nodes():
            rebased += self.visit_block(block)
        return rebased

    def visit_block(self, block: ControlFlowBlock) -> int:
        """Visit one control flow block, rebasing it before anything it contains."""
        if isinstance(block, LoopRegion):
            return self.rebase_loop(block) + self.visit_region(block)
        if isinstance(block, AbstractControlFlowRegion):
            # ``ConditionalBlock.nodes()`` are its branch bodies, so this covers both.
            return self.visit_region(block)
        if isinstance(block, dace.SDFGState):
            return self.visit_state(block)
        return 0

    def visit_state(self, state: dace.SDFGState) -> int:
        """Visit one state's Maps outermost scope first, then the SDFGs nested in them."""
        # Materialized once: rebasing re-attaches edges, which invalidates the state's cached
        # scope tree, but node membership (all this walk needs) cannot change under it.
        children = {scope: list(scope_nodes) for scope, scope_nodes in state.scope_children().items()}
        rebased = 0
        level: list[nodes.Node | None] = [None]
        while level:
            deeper: list[nodes.Node] = []
            for scope in level:
                for node in children[scope]:
                    if isinstance(node, nodes.MapEntry):
                        rebased += self.rebase_map(state, node)
                    elif isinstance(node, nodes.NestedSDFG):
                        rebased += self.visit_region(node.sdfg)
                    if node in children:  # a scope entry of any kind -- its children come next
                        deeper.append(node)
            level = deeper
        return rebased

    def rebase_map(self, state: dace.SDFGState, entry: nodes.MapEntry) -> int:
        """Rebase every non-zero-begin dimension of ``entry``'s map, keeping its stride."""
        new_ranges = list(entry.map.range.ranges)
        repldict: dict[str, str] = {}
        for i, (param, (begin, end, stride)) in enumerate(zip(entry.map.params, entry.map.range.ranges)):
            if begin == 0:
                continue
            new_ranges[i] = (0, end - begin, stride)
            repldict[str(param)] = f"({param} + ({symstr(begin)}))"
        if not repldict:
            return 0

        # Scope membership, not a reachability walk: ``all_nodes_between`` returns EMPTY as soon as the
        # scope holds one node with no out-edge (a write-only scratch scalar), and the rebase would then
        # shift the map range while substituting the shift into nothing -- every body read off by ``begin``.
        scope_nodes = map_scope.map_body_nodes(state, entry)
        if rebinds_params(scope_nodes, repldict) or tasklets_assign(scope_nodes, repldict):
            return 0  # refuse before touching anything

        entry.map.range = subsets.Range(new_ranges)
        process_memlets_in_edges(state, list(state.all_edges(*scope_nodes)), repldict)
        repl_tasklets_on_node_list(scope_nodes, repldict)
        shift_map_ranges(scope_nodes, repldict)
        for node in scope_nodes:
            if isinstance(node, nodes.NestedSDFG):
                repl_recursive(node.sdfg, repldict)
        return 1

    def rebase_loop(self, loop: LoopRegion) -> int:
        """Rebase ``loop``'s counter to 0, keeping its stride."""
        var = loop.loop_variable
        if not var or loop.init_statement is None or loop.loop_condition is None or loop.update_statement is None:
            return 0
        start = loop_analysis.get_init_assignment(loop)
        if start is None or start == 0:
            return 0
        if loop_analysis.get_loop_end(loop) is None or loop_analysis.get_loop_stride(loop) is None:
            return 0
        # The body would read ``var + start`` with the CURRENT start, not the one the loop began at
        if loop_analysis.body_writes(loop, start):
            return 0
        try:
            new_condition = add_to_rhs(loop.loop_condition.as_string, -start)
        except (ValueError, SyntaxError, TypeError):
            return 0

        repldict = {str(var): f"({var} + ({symstr(start)}))"}
        body_nodes = [n for state in loop.states() for n in state.nodes()]
        if rebinds_params(body_nodes, dict.fromkeys([str(var)])) or tasklets_assign(body_nodes, [str(var)]):
            return 0  # refuse before touching anything

        loop.init_statement = CodeBlock(f"{var} = 0")
        loop.loop_condition = CodeBlock(new_condition)
        # ``repl_recursive`` skips the region it is rooted at, so the header just set stands.
        repl_recursive(loop, repldict)
        shift_map_ranges(body_nodes, repldict)
        return 1


__all__ = ["NormalizeLoopAndMapOrigin", "rebinds_params", "shift_map_ranges"]
