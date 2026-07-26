# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rebase every Map range and LoopRegion counter to a 0-based begin, KEEPING the
stride: ``b:e:s`` -> ``0:(e-b):s`` with ``p -> p + b`` substituted in the
scope, so a subscript ``a[p]`` becomes ``a[p+b]`` -- affine with coefficient 1.
Pure subtraction: no trip count, no division.

This is deliberately narrower than ``NormalizeLoopsAndMaps`` (``b:e:s`` ->
``0:(e-b)//s:1``, substituting ``p -> b + s*p``): that pass pushes the stride
into every subscript, and a corpus measurement (TSVC, 151 kernels) showed
``a[b+s*j]`` stops being recognised by ``LoopToMap`` as uniquely indexed by
``j`` -- 0 kernels gained, net -1 parallel map (``s172``). Keeping the stride
means a rewritten subscript stays ``a[j+b]``, the same affine-coefficient-1
shape ``LoopToMap`` already handles.

Two ranges of equal trip count but different origin -- a slice-vectorized Map
``0:N-3:1`` beside a sequential scan ``LoopRegion`` ``1:N-2`` (polybench
``seidel_2d``) -- become the identical ``0:N-3:1`` once both are rebased,
which is what lets a same-range fusion/skew pass treat them as one shape.

ONE lazy top-down traversal, outermost scope to innermost: each Map / LoopRegion
is visited once, its begin is read AT VISIT TIME and shifted immediately. Reading
lazily is what makes a single pass sufficient for a NESTED begin. Take a
triangular ``for i in 1:N: for j in i:N``, whose inner begin is the literal token
``i``. Rebasing the outer loop substitutes ``i -> i + 1`` into everything it
contains, including the inner loop's header, which becomes ``j = i + 1``. A
design that pre-collected the begins as strings would still be holding the stale
``"i"``, match nothing, and leave the inner loop unrebased -- the residual that
forced a reconvergence loop. Visiting the inner loop afterwards and reading its
CURRENT begin ``i + 1`` shifts it correctly, first try.

The shifting mechanics (subset / tasklet / interstate substitution) are
``OffsetLoopsAndMaps``' module-level helpers, driven here per scope rather than
over the whole SDFG.
"""
from typing import Any, Dict, Iterable, List, Optional

import dace
from dace import SDFG, properties, subsets
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, LoopRegion
from dace.symbolic import pystr_to_symbolic, symstr
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.offset_loop_and_maps import (add_to_rhs, process_memlets_in_edges, repl_recursive,
                                                             repl_tasklets_on_node_list)


def rebinds_params(node_list: Iterable[nodes.Node], params: Dict[str, None]) -> bool:
    """Whether a NestedSDFG under ``node_list`` binds one of ``params`` to a differently named
    inner symbol.

    The rebase substitutes ``p -> p + b`` INSIDE a nested SDFG, which is the right rewrite only
    for the identity binding ``{'p': 'p'}`` the frontend emits: there, the inner ``p`` is the
    outer (now shifted) parameter and adding ``b`` back recovers the original value. Under
    ``{'k': 'p'}`` the body reads ``k``, the substitution finds nothing to rewrite, and the map
    hands ``k`` the shifted value -- a silent off-by-``b``. Report it so the caller can refuse.

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
        if rebinds_params([n for state in node.sdfg.all_states() for n in state.nodes()], params):
            return True
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeLoopAndMapOrigin(ppl.Pass):
    """Rebase every Map range / ``LoopRegion`` counter to begin at 0, keeping the
    stride (``b:e:s`` -> ``0:(e-b):s``, ``p -> p + b`` substituted in the scope).
    See the module docstring for the rationale and the traversal order.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Dict:
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Rebase every Map/LoopRegion begin to 0 (recursively), keeping stride.

        :param sdfg: The SDFG to rewrite in place.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :returns: The number of Maps / ``LoopRegion``s rebased, or ``None`` if every begin was
                  already 0 or every non-zero one was refused (no mutation in that case).
        """
        rebased = self.visit_region(sdfg)
        if rebased == 0:
            return None
        sdfg.validate()
        return rebased

    def visit_region(self, region: AbstractControlFlowRegion) -> int:
        """Visit every block of ``region`` in order.

        :param region: The SDFG / control flow region to walk.
        :returns: The number of Maps / ``LoopRegion``s rebased below it.
        """
        rebased = 0
        for block in region.nodes():
            rebased += self.visit_block(block)
        return rebased

    def visit_block(self, block: ControlFlowBlock) -> int:
        """Visit one control flow block, rebasing it before anything it contains.

        The order falls out of this one rule: a ``LoopRegion`` is rebased, and only then are its
        contents visited -- so a nested Map or loop whose begin mentions the counter is read after
        the counter's own substitution has rewritten it. ``ConditionalBlock`` branches and plain
        regions carry no counter and are just descended into. Break / continue / return blocks hold
        neither.

        :param block: The block to visit.
        :returns: The number of Maps / ``LoopRegion``s rebased at or below it.
        """
        if isinstance(block, LoopRegion):
            return self.rebase_loop(block) + self.visit_region(block)
        if isinstance(block, AbstractControlFlowRegion):
            # ``ConditionalBlock.nodes()`` are its branch bodies, so this covers both.
            return self.visit_region(block)
        if isinstance(block, dace.SDFGState):
            return self.visit_state(block)
        return 0

    def visit_state(self, state: dace.SDFGState) -> int:
        """Visit one state's Maps outermost scope first, then the SDFGs nested in them.

        ``scope_children()`` restarts at every NestedSDFG, so this tree is exactly the Map nesting
        of THIS state; a nested SDFG is recursed into as its own root, from the scope level it sits
        at -- i.e. after every Map enclosing it has already been shifted.

        :param state: The state to walk.
        :returns: The number of Maps / ``LoopRegion``s rebased in it.
        """
        # Materialized once: rebasing re-attaches edges, which invalidates the state's cached
        # scope tree, but node membership (all this walk needs) cannot change under it.
        children = {scope: list(scope_nodes) for scope, scope_nodes in state.scope_children().items()}
        rebased = 0
        level: List[Optional[nodes.Node]] = [None]
        while level:
            deeper: List[nodes.Node] = []
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
        """Rebase every non-zero-begin dimension of ``entry``'s map, keeping its stride.

        :param state: The state owning the map.
        :param entry: The map entry to rebase.
        :returns: ``1`` if the map was rebased, ``0`` if it was already 0-based or was refused.
        """
        new_ranges = list(entry.map.range.ranges)
        repldict: Dict[str, str] = {}
        for i, (param, (begin, end, stride)) in enumerate(zip(entry.map.params, entry.map.range.ranges)):
            if begin == 0:
                continue
            new_ranges[i] = (0, end - begin, stride)
            repldict[str(param)] = f"({param} + ({symstr(begin)}))"
        if not repldict:
            return 0

        scope_nodes = list(state.all_nodes_between(entry, state.exit_node(entry)))
        if rebinds_params(scope_nodes, repldict):
            return 0  # refuse before touching anything

        entry.map.range = subsets.Range(new_ranges)
        process_memlets_in_edges(state, list(state.all_edges(*scope_nodes)), repldict)
        repl_tasklets_on_node_list(scope_nodes, repldict)
        for node in scope_nodes:
            if isinstance(node, nodes.NestedSDFG):
                repl_recursive(node.sdfg, repldict)
        return 1

    def rebase_loop(self, loop: LoopRegion) -> int:
        """Rebase ``loop``'s counter to 0, keeping its stride.

        Refuses (leaving the loop untouched) anything ``loop_analysis`` cannot read back as an
        affine ``var = start ; var <op> end`` counter -- a while-shaped region has no init statement
        at all, and a condition that is not a comparison against the bare counter cannot be shifted
        by adding to its right-hand side.

        :param loop: The loop region to rebase.
        :returns: ``1`` if the loop was rebased, ``0`` if it was already 0-based or was refused.
        """
        var = loop.loop_variable
        if not var or loop.init_statement is None or loop.loop_condition is None or loop.update_statement is None:
            return 0
        start = loop_analysis.get_init_assignment(loop)
        if start is None or start == 0:
            return 0
        if loop_analysis.get_loop_end(loop) is None or loop_analysis.get_loop_stride(loop) is None:
            return 0
        try:
            new_condition = add_to_rhs(loop.loop_condition.as_string, -start)
        except (ValueError, SyntaxError, TypeError):
            return 0

        repldict = {str(var): f"({var} + ({symstr(start)}))"}
        body_nodes = [n for state in loop.all_states() for n in state.nodes()]
        if rebinds_params(body_nodes, dict.fromkeys([str(var)])):
            return 0  # refuse before touching anything

        loop.init_statement = CodeBlock(f"{var} = 0")
        loop.loop_condition = CodeBlock(new_condition)
        # ``repl_recursive`` skips the region it is rooted at, so the header just set stands.
        repl_recursive(loop, repldict)
        return 1


__all__ = ['NormalizeLoopAndMapOrigin', 'rebinds_params']
