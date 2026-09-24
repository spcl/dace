# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Remove unused per-lane symbols.

The gather / scatter lowering emits per-lane symbols (``LaneIdScheme.make_multi``:
``<base>_lane<d>id_<l>``) that are dead once the gather is materialized into a tile. This pass
walks every SDFG and, for each :func:`LaneIdScheme.is_laneid` symbol with no remaining reference
(memlets, tasklet code, interstate edges, loop and branch conditions, descriptor shapes, or other
symbol definitions), removes it from ``sdfg.symbols``, drops its defining interstate assignments
and its ``symbol_mapping`` entries on NestedSDFGs. Idempotent.
"""
from typing import Any

import sympy

from dace import properties, symbolic
from dace.sdfg import SDFG
from dace.sdfg.nodes import NestedSDFG, Tasklet
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def _symbols_in_code_block(code_block: CodeBlock | str | None) -> set[str]:
    # Wrap :func:`dace.symbolic.symbols_in_code` for an SDFG ``CodeBlock``-like value.
    if code_block is None:
        return set()
    src = code_block.as_string if isinstance(code_block, CodeBlock) else str(code_block)
    return symbolic.symbols_in_code(src)


def _collect_referenced_symbols(sdfg: SDFG) -> set[str]:
    # Walk every place a symbol can appear in ``sdfg`` and return the union.
    referenced: set[str] = set()
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
                    if isinstance(value, sympy.Basic):
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
    # Drop any ``sym = ...`` assignment from any interstate edge in ``sdfg``.
    dropped = 0
    for edge in sdfg.all_interstate_edges():
        if sym in edge.data.assignments:
            del edge.data.assignments[sym]
            dropped += 1
    return dropped


def _drop_symbol_mapping_in_nsdfgs(sdfg: SDFG, sym: str) -> int:
    # Drop any entry whose key is ``sym`` from any NestedSDFG node's symbol_mapping.
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

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        return set()

    def _sweep(self, sdfg: SDFG) -> int:
        # Sweep unused per-lane symbols from ``sdfg`` (one level).
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

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any] | None) -> int | None:
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
