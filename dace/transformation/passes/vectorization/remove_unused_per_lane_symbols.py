# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-clean pass that removes unused per-lane symbols.

Per user direction 2026-06-10: the indirect-access (gather / scatter) lowering
emits MANY per-lane SDFG symbols as an intermediate stage (via the
``LaneIdScheme.make_multi`` naming scheme: ``<base>_lane<d>id_<l>`` chained per
dim). After the gather is materialised into a tile and downstream consumers
read from the tile, the per-lane symbols themselves may have no further uses --
they were only "named intermediates" for the fan-out.

This pass walks every SDFG (recursive), identifies symbols matching
:func:`LaneIdScheme.is_laneid` that have NO remaining references in any:

* memlet subset / volume / wcr,
* tasklet code body,
* interstate-edge condition / assignment RHS,
* loop-region condition / init / update statement,
* conditional-block branch guard,
* array descriptor shape / strides / offsets,
* (recursively) symbol RHSes in interstate-edge assignments.

For each such unused symbol, the pass:

1. Removes the symbol from ``sdfg.symbols``.
2. Removes any interstate-edge assignment whose LHS is the unused symbol (the
   defining assignment becomes dead too).
3. Removes any matching ``symbol_mapping`` entry on any NestedSDFG that references
   the unused symbol.

The pass is idempotent: a second invocation is a no-op once every detectable
per-lane symbol has been swept.

Design contract: this pass DOES NOT collapse contiguous per-lane symbol chains
into direct slice loads (the peephole optimisation mentioned in the design doc).
That's a separate follow-up slice; this pass is only the structural sweep.
"""
from typing import Any, Dict, Optional, Set

from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.nodes import NestedSDFG, Tasklet
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def _symbols_in_code_block(code_block) -> Set[str]:
    """Wrap :func:`dace.symbolic.symbols_in_code` for an SDFG ``CodeBlock``-like value.

    Accepts ``None`` (returns empty set), a ``CodeBlock`` instance (reads
    ``.as_string``), or any other value (str-coerced).
    """
    if code_block is None:
        return set()
    src = code_block.as_string if hasattr(code_block, "as_string") else str(code_block)
    return symbolic.symbols_in_code(src)


def _collect_referenced_symbols(sdfg: SDFG) -> Set[str]:
    """Walk every place a symbol can appear in ``sdfg`` and return the union.

    Uses :meth:`dace.data.Data.free_symbols`, :meth:`dace.memlet.Memlet.used_symbols`,
    :meth:`dace.sdfg.InterstateEdge.used_symbols` and
    :func:`dace.symbolic.symbols_in_code` instead of hand-rolled regex extraction,
    so any change in DaCe's symbol-tracking semantics flows through here uniformly.

    Conservative: over-reporting (flagging a symbol that isn't really used) keeps
    the symbol alive, which is safe; under-reporting would delete a still-live
    symbol and break the SDFG.
    """
    referenced: Set[str] = set()
    # Array descriptors (shape / strides / offset / start_offset).
    for desc in sdfg.arrays.values():
        referenced.update(str(s) for s in desc.free_symbols)
    # State-level memlets + tasklet bodies + NSDFG symbol_mapping values.
    for state in sdfg.states():
        for edge in state.edges():
            if edge.data is None:
                continue
            referenced.update(edge.data.free_symbols)
        for node in state.nodes():
            if isinstance(node, Tasklet):
                referenced.update(_symbols_in_code_block(node.code))
            elif isinstance(node, NestedSDFG):
                for value in node.symbol_mapping.values():
                    if hasattr(value, "free_symbols"):
                        referenced.update(str(s) for s in value.free_symbols)
                    else:
                        referenced.update(symbolic.symbols_in_code(str(value)))
    # Interstate-edge condition + assignment RHSes.
    for edge in sdfg.all_interstate_edges():
        referenced.update(edge.data.used_symbols(all_symbols=False, union_lhs_symbols=False))
    # LoopRegion + ConditionalBlock code blocks.
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion):
            referenced.update(_symbols_in_code_block(cfg.loop_condition))
            referenced.update(_symbols_in_code_block(cfg.init_statement))
            referenced.update(_symbols_in_code_block(cfg.update_statement))
        if isinstance(cfg, ConditionalBlock):
            for cond, _body in cfg.branches:
                referenced.update(_symbols_in_code_block(cond))
    return referenced


def _drop_assignment_in_iedges(sdfg: SDFG, sym: str) -> int:
    """Drop any ``sym = ...`` assignment from any interstate edge in ``sdfg``.
    Returns the number of assignments removed."""
    dropped = 0
    for edge in sdfg.all_interstate_edges():
        if sym in edge.data.assignments:
            del edge.data.assignments[sym]
            dropped += 1
    return dropped


def _drop_symbol_mapping_in_nsdfgs(sdfg: SDFG, sym: str) -> int:
    """Drop any entry whose key is ``sym`` from any NestedSDFG node's symbol_mapping.
    Returns the number of mappings removed."""
    dropped = 0
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, NestedSDFG) and sym in node.symbol_mapping:
                del node.symbol_mapping[sym]
                dropped += 1
    return dropped


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveUnusedPerLaneSymbols(ppl.Pass):
    """Remove every per-lane symbol from the SDFG that has no remaining use.

    Per-lane symbols are identified by :func:`LaneIdScheme.is_laneid` (both the
    canonical ``<base>_lane<d>id_<l>`` form and the legacy ``<base>_laneid_<l>``
    form). The pass walks recursively into NestedSDFGs.
    """

    CATEGORY: str = "Vectorization Cleanup"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _sweep(self, sdfg: SDFG) -> int:
        """Sweep unused per-lane symbols from ``sdfg`` (one level). Returns the
        number of symbols removed."""
        removed = 0
        referenced = _collect_referenced_symbols(sdfg)
        # Lane-encoded symbols that are NOT referenced are dead.
        candidates = [s for s in list(sdfg.symbols) if LaneIdScheme.is_laneid(s)]
        for sym in candidates:
            if sym in referenced:
                # Still referenced somewhere; keep.
                continue
            # Drop any defining iedge assignment.
            _drop_assignment_in_iedges(sdfg, sym)
            # Drop any symbol_mapping entry that binds this symbol on a child NSDFG.
            _drop_symbol_mapping_in_nsdfgs(sdfg, sym)
            # Drop from sdfg.symbols.
            sdfg.symbols.pop(sym, None)
            removed += 1
        return removed

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Optional[Dict[str, Any]]) -> Optional[int]:
        """Apply the sweep recursively to every NSDFG in ``sdfg``. Returns the
        total number of removed symbols, or ``None`` if zero."""
        total = self._sweep(sdfg)
        # Recurse into nested SDFGs.
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, NestedSDFG):
                    inner_result = self.apply_pass(node.sdfg, _pipeline_results)
                    if inner_result:
                        total += inner_result
        return total or None
