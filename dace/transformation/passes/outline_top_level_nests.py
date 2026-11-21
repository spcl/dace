# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrap every top-level loop nest of the root SDFG in its own nested SDFG.

Each outermost map-nest (a ``MapEntry`` at the top of a state's scope tree), each top-level CFG
``LoopRegion`` and each top-level ``ConditionalBlock`` holding a nest (a guard, or canonicalize's
parallel/sequential specialization) is outlined into a ``NestedSDFG`` marked ``no_inline`` and given a stable
``unique_name``. The rewrite is purely structural -- it moves subgraphs into nested SDFGs using the
same ``nest_state_subgraph`` / ``nest_sdfg_subgraph`` outliners the rest of DaCe uses, so both the
legacy and the experimental CPU code generators consume the result identically.

The point is to give downstream codegen a nest it can emit as its own function (and, with the
per-nest translation-unit path, its own ``.cpp``). ``no_inline`` is the contract: it already stops
simplification from inlining the nest back, and codegen treats a ``no_inline`` nested SDFG as a real
function rather than folding it into the parent. ``unique_name`` names that function deterministically
so two structurally identical nests are not deduplicated into one.

With ``library_nodes``, each top-level library node is outlined the same way, so a node whose expansion is
not a nest (a C++ or vendor implementation) still gets its own function; the split-file codegen leaves it off.

This is the codegen-agnostic half of the "outline top-level loop nests into their own translation
units" feature; the emission side (routing each ``no_inline`` nest to its own file) lives in the CPU
code generator.
"""

from typing import Any

from dace import SDFG, dtypes
from dace.libraries.standard.nodes import external_call
from dace.properties import Property, make_properties
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import helpers
from dace.transformation import pass_pipeline as ppl

NestNode = nodes.MapEntry | LoopRegion
NestRegion = LoopRegion | ConditionalBlock


def loop_defined_symbols(loop: NestRegion) -> set[str]:
    """Symbols DEFINED inside a loop region or conditional block: each loop variable (the region and any
    nested ``LoopRegion``) plus every inter-state-edge assignment target.

    ``nest_sdfg_subgraph`` emits a symbolic output for each such symbol that something outside the loop
    observes, and types an inter-state assignment target from the nested-or-parent ``sdfg.symbols``; a
    target DaCe never registered would raise there, so the caller pre-declares these on the parent
    first. (Loop counters type themselves via ``LoopRegion.new_symbols`` and need no pre-declaration,
    but they are cheap to include and a counter left declared is removed again by the outliner.)
    """
    syms: set[str] = set()
    for block in [loop, *loop.all_control_flow_blocks()]:
        if isinstance(block, LoopRegion) and block.loop_variable and block.init_statement:
            syms.add(block.loop_variable)
    for edge in loop.all_interstate_edges():
        syms.update(edge.data.assignments.keys())
    return syms


def holds_nest(block: ConditionalBlock) -> bool:
    """Whether a branch of ``block`` holds a map, a loop region or a library node: a guard around scalar work
    alone is no nest."""
    if any(isinstance(b, LoopRegion) for b in block.all_control_flow_blocks(recursive=True)):
        return True
    return any(isinstance(n, nodes.MapEntry | nodes.LibraryNode) for n, parent in block.all_nodes_recursive())


@make_properties
class OutlineTopLevelNests(ppl.Pass):
    """Outline each top-level map-nest, CFG loop region and nest-holding conditional block of the root SDFG
    into its own ``no_inline`` nested SDFG. See the module docstring."""

    CATEGORY: str = "Optimization Preparation"

    library_nodes = Property(dtype=bool, default=False, desc="Also outline each top-level library node.")

    def modifies(self) -> ppl.Modifies:
        # Creates NestedSDFG nodes and rewrites a state's dataflow (Nodes covers NestedSDFGs/Scopes);
        # the loop-region path replaces a CFG block with a state (States) and registers loop symbols
        # on the parent (Symbols); connectors/memlets at the new boundary change (Memlets).
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Idempotent by construction: after one run the top-level nests are single-nsdfg scopes, so
        # re-running would wrap nothing new. Never worth re-triggering.
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Outline the root SDFG's top-level nests. Returns the number outlined, or ``None`` if none."""
        # Snapshot the top-level nests BEFORE mutating: each outline rewrites the graph in place, so
        # collecting refs up front (as NestForge's `outer` strategy does) avoids iterating a mutating
        # container. Top-level nests are disjoint scopes, so sibling node objects stay valid across an
        # extraction as long as map subgraphs are recomputed at extraction time.
        loop_refs: list[NestRegion] = []
        map_refs: list[tuple[SDFGState, nodes.MapEntry]] = []
        library_refs: list[tuple[SDFGState, nodes.LibraryNode]] = []
        for block in sdfg.nodes():
            if isinstance(block, LoopRegion) or isinstance(block, ConditionalBlock) and holds_nest(block):
                loop_refs.append(block)
            elif isinstance(block, SDFGState):
                top = block.scope_children()[None]
                map_refs.extend((block, node) for node in top if isinstance(node, nodes.MapEntry))
                if self.library_nodes:
                    library_refs.extend(
                        (block, node)
                        for node in top
                        if isinstance(node, nodes.LibraryNode) and not isinstance(node, external_call.ExternalCall)
                    )

        if not loop_refs and not map_refs and not library_refs:
            return None

        count = 0

        for state, entry in map_refs:
            # Recompute the scope subgraph now: a sibling map extracted earlier in this same state
            # changed the node set, but this MapEntry object is still present and its scope intact.
            subgraph = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            name = self._unique_label(sdfg, count)
            nsdfg = helpers.nest_state_subgraph(sdfg, state, subgraph, name=name)
            self._mark(nsdfg, name)
            count += 1

        for state, libnode in library_refs:
            name = self._unique_label(sdfg, count)
            nsdfg = helpers.nest_state_subgraph(sdfg, state, SubgraphView(state, [libnode]), name=name)
            self._mark(nsdfg, name)
            count += 1

        for loop in loop_refs:
            loop_vars = {
                block.loop_variable
                for block in [loop, *loop.all_control_flow_blocks()]
                if isinstance(block, LoopRegion) and block.loop_variable and block.init_statement
            }
            # nest_sdfg_subgraph needs every nest-defined symbol registered on the parent to type its
            # symbolic outputs; a loop's own iterator is bound by the loop, never declared.
            for symbol in loop_defined_symbols(loop):
                if symbol not in sdfg.symbols and symbol not in loop_vars:
                    sdfg.add_symbol(symbol, dtypes.int64)
            inner_state = helpers.nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop]))
            nsdfg = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
            # A loop variable the nest re-initialises internally needs no INBOUND value. A counter that
            # nothing outside the loop observes is already classified strictly-internal by
            # nest_sdfg_subgraph and never mapped; one that IS observed outside stays mapped, and only
            # its outbound ``sym = __sym_out_sym`` assignment carries the value. Drop the inbound
            # mapping in that case too, or it leaks as a REQUIRED free-symbol argument of the root SDFG.
            for var in loop_vars:
                nsdfg.symbol_mapping.pop(var, None)
            self._mark(nsdfg, self._unique_label(sdfg, count))
            count += 1

        return count or None

    def _mark(self, nsdfg: nodes.NestedSDFG, name: str) -> None:
        nsdfg.no_inline = True
        nsdfg.unique_name = name

    def _unique_label(self, sdfg: SDFG, index: int) -> str:
        return "%s_nest_%d" % (sdfg.label, index)


def outline_top_level_nests(sdfg: SDFG, library_nodes: bool = False) -> int:
    """Outline the root SDFG's top-level loop nests (and, with ``library_nodes``, library nodes) in place; returns
    the number outlined."""
    outliner = OutlineTopLevelNests()
    outliner.library_nodes = library_nodes
    return outliner.apply_pass(sdfg, {}) or 0
