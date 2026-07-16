# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Wrap every top-level loop nest of the root SDFG in its own nested SDFG.

Each outermost map-nest (a ``MapEntry`` at the top of a state's scope tree) and each top-level CFG
``LoopRegion`` is outlined into a ``NestedSDFG`` marked ``no_inline`` and given a stable
``unique_name``. The rewrite is purely structural -- it moves subgraphs into nested SDFGs using the
same ``nest_state_subgraph`` / ``nest_sdfg_subgraph`` outliners the rest of DaCe uses, so both the
legacy and the experimental CPU code generators consume the result identically.

The point is to give downstream codegen a nest it can emit as its own function (and, with the
per-nest translation-unit path, its own ``.cpp``). ``no_inline`` is the contract: it already stops
simplification from inlining the nest back, and codegen treats a ``no_inline`` nested SDFG as a real
function rather than folding it into the parent. ``unique_name`` names that function deterministically
so two structurally identical nests are not deduplicated into one.

This is the codegen-agnostic half of the "outline top-level loop nests into their own translation
units" feature; the emission side (routing each ``no_inline`` nest to its own file) lives in the CPU
code generator.
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dace import SDFG, dtypes
from dace.properties import Property, make_properties
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import helpers
from dace.transformation import pass_pipeline as ppl

NestNode = Union[nodes.MapEntry, LoopRegion]


def loop_defined_symbols(loop: LoopRegion) -> Set[str]:
    """Symbols DEFINED inside a loop region: each loop variable (the region and any nested
    ``LoopRegion``) plus every inter-state-edge assignment target.

    ``nest_sdfg_subgraph`` emits a symbolic output for each such symbol and looks its dtype up in the
    nested-or-parent ``sdfg.symbols``; a loop index DaCe never registered as a symbol would raise a
    ``KeyError`` there, so the caller pre-declares these on the parent first.
    """
    syms: Set[str] = set()
    for block in [loop, *loop.all_control_flow_blocks()]:
        if isinstance(block, LoopRegion) and block.loop_variable and block.init_statement:
            syms.add(block.loop_variable)
    for edge in loop.all_interstate_edges():
        syms.update(edge.data.assignments.keys())
    return syms


@make_properties
class OutlineTopLevelNests(ppl.Pass):
    """Outline each top-level map-nest and CFG loop region of the root SDFG into its own
    ``no_inline`` nested SDFG. See the module docstring."""

    CATEGORY: str = 'Optimization Preparation'

    full_data = Property(dtype=bool,
                         default=False,
                         desc='Nest entire input/output arrays instead of only the accessed subrange. '
                         'The default (False) gives accurate accessed-subrange connectors; set True '
                         'to keep whole-array signatures at the nest boundary.')

    def __init__(self, full_data: bool = False):
        self.full_data = full_data

    def modifies(self) -> ppl.Modifies:
        # Creates NestedSDFG nodes and rewrites a state's dataflow (Nodes covers NestedSDFGs/Scopes);
        # the loop-region path replaces a CFG block with a state (States) and registers loop symbols
        # on the parent (Symbols); connectors/memlets at the new boundary change (Memlets).
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Idempotent by construction: after one run the top-level nests are single-nsdfg scopes, so
        # re-running would wrap nothing new. Never worth re-triggering.
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Outline the root SDFG's top-level nests. Returns the number outlined, or ``None`` if none."""
        # Snapshot the top-level nests BEFORE mutating: each outline rewrites the graph in place, so
        # collecting refs up front (as NestForge's `outer` strategy does) avoids iterating a mutating
        # container. Top-level nests are disjoint scopes, so sibling node objects stay valid across an
        # extraction as long as map subgraphs are recomputed at extraction time.
        loop_refs: List[LoopRegion] = []
        map_refs: List[Tuple[SDFGState, nodes.MapEntry]] = []
        for block in sdfg.nodes():
            if isinstance(block, LoopRegion):
                loop_refs.append(block)
            elif isinstance(block, SDFGState):
                for node in block.scope_children()[None]:
                    if isinstance(node, nodes.MapEntry):
                        map_refs.append((block, node))

        if not loop_refs and not map_refs:
            return None

        count = 0

        for state, entry in map_refs:
            # Recompute the scope subgraph now: a sibling map extracted earlier in this same state
            # changed the node set, but this MapEntry object is still present and its scope intact.
            subgraph = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            name = self._unique_label(sdfg, count)
            nsdfg = helpers.nest_state_subgraph(sdfg, state, subgraph, name=name, full_data=self.full_data)
            self._mark(nsdfg, name)
            count += 1

        for loop in loop_refs:
            # nest_sdfg_subgraph needs every nest-defined symbol registered on the parent to type its
            # symbolic outputs; a bare loop index need not be a declared symbol, so declare it.
            for symbol in loop_defined_symbols(loop):
                if symbol not in sdfg.symbols:
                    sdfg.add_symbol(symbol, dtypes.int64)
            loop_vars = {
                block.loop_variable
                for block in [loop, *loop.all_control_flow_blocks()]
                if isinstance(block, LoopRegion) and block.loop_variable and block.init_statement
            }
            inner_state = helpers.nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [loop]))
            nsdfg = next(n for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG))
            # A loop variable that the nest re-initialises internally needs no INBOUND value. The
            # outliner maps it (``sym: sym``) anyway -- because we pre-declared it on the parent to
            # type the outliner's symbolic outputs, it is no longer classified strictly-internal
            # (helpers.nest_sdfg_subgraph keys that on ``not in sdfg.symbols``), so it survives in the
            # node's symbol_mapping and would leak as a REQUIRED free-symbol argument of the root SDFG.
            # Drop those inbound mappings; the outbound ``sym = __sym_out_sym`` assignment the outliner
            # also emits still propagates the final value out.
            for var in loop_vars:
                nsdfg.symbol_mapping.pop(var, None)
            self._mark(nsdfg, self._unique_label(sdfg, count))
            count += 1

        return count or None

    def _mark(self, nsdfg: nodes.NestedSDFG, name: str) -> None:
        nsdfg.no_inline = True
        nsdfg.unique_name = name

    def _unique_label(self, sdfg: SDFG, index: int) -> str:
        return '%s_nest_%d' % (sdfg.label, index)


def outline_top_level_nests(sdfg: SDFG, full_data: bool = False) -> int:
    """Outline the root SDFG's top-level loop nests in place; returns the number outlined."""
    return OutlineTopLevelNests(full_data=full_data).apply_pass(sdfg, {}) or 0
