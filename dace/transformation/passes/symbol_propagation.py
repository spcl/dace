# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from dataclasses import dataclass
from dace.sdfg.state import (
    ControlFlowBlock,
    ControlFlowRegion,
    ConditionalBlock,
    LoopRegion,
)
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, properties, SDFGState
from typing import Any, Dict, Set, Optional
from dace import data as dt
from dace.symbolic import pystr_to_symbolic


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolPropagation(ppl.Pass):
    """
    Propagates symbols that were assigned to one value forward through the SDFG, reducing the number of overall symbols.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        # Assumption: Symbols can only change in InterStateEdges

        # Get all CFG blocks present in the SDFG
        all_cfg_blks = dict()
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowBlock):
                all_cfg_blks[node] = parent

        # For each CFG Block maintain a dict of incoming and outgoing symbols
        in_syms = {cfg_blk: {} for cfg_blk in all_cfg_blks.keys()}
        out_syms = {cfg_blk: {} for cfg_blk in all_cfg_blks.keys()}

        # Perform a forward fixed-point iteration to propagate symbols
        changed = True
        while changed:
            changed = False

            # Update incoming symbols
            for cfg_blk, parent in all_cfg_blks.items():
                new_in_syms = self._get_in_syms(sdfg, cfg_blk, parent, in_syms, out_syms)
                # Check if the incoming symbols have changed
                if new_in_syms != in_syms[cfg_blk]:
                    changed = True
                    in_syms[cfg_blk] = new_in_syms

            # Update outgoing symbols
            for cfg_blk, parent in all_cfg_blks.items():
                new_out_syms = self._get_out_syms(cfg_blk, parent, in_syms, out_syms)
                # Check if the outgoing symbols have changed
                if new_out_syms != out_syms[cfg_blk]:
                    changed = True
                    out_syms[cfg_blk] = new_out_syms

        # Update symbols in the cfg_blk
        for cfg_blk, parent in all_cfg_blks.items():
            self._update_syms(cfg_blk, parent, in_syms, out_syms)
        return set()

    # Given a cfg_blk, builds the incoming set of symbols
    def _get_in_syms(
        self,
        sdfg: SDFG,
        cfg_blk: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Combine the outgoing symbols of all incoming edges with their assignments to the cfg_blk
        new_in_syms = {}
        for i, edge in enumerate(parent.in_edges(cfg_blk)):
            sym_table = copy.deepcopy(out_syms[edge.src])
            sym_table.update(edge.data.assignments)

            # Filter out symbols containing arrays accesses as they cannot be safely propagated (nested array accesses are not supported)
            sym_table = {k: v for k, v in sym_table.items() if v is None or ("[" not in v and "]" not in v)}

            # Also filter out symbols containing views as they cannot be safely propagated (they are seen as pointers)
            sym_table = {
                k: v
                for k, v in sym_table.items() if v is None or not any([
                    str(s) in sdfg.arrays and isinstance(sdfg.arrays[str(s)], dt.View)
                    for s in pystr_to_symbolic(v).free_symbols
                ])
            }

            # Combine the symbols
            if i == 0:
                new_in_syms = sym_table
            else:
                self._combine_syms(new_in_syms, sym_table)

        # Nested starting CFBGs should inherit the symbols from their parent
        # Ignore SDFGs as nested SDFGs have symbol mappings
        if (parent.start_block == cfg_blk and not isinstance(parent, SDFG)) or (isinstance(parent, ConditionalBlock)
                                                                                and cfg_blk in parent.sub_regions()):
            assert new_in_syms == {}
            new_in_syms = in_syms[parent]

            # For LoopRegions, remove loop carried variables from the incoming symbols
            if isinstance(parent, LoopRegion):
                new_in_syms = copy.deepcopy(new_in_syms)
                all_syms = set([s for e in parent.all_interstate_edges() for s in e.data.assignments.keys()])
                for sym in all_syms:
                    if sym in new_in_syms:
                        new_in_syms[sym] = None

        return new_in_syms

    # Given a cfg_blk, builds the outgoing set of symbols
    def _get_out_syms(
        self,
        cfg_blk: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(cfg_blk, LoopRegion):
            # Any symbol that is assigned in the loop region is not propagated out
            new_out_syms = copy.deepcopy(in_syms[cfg_blk])
            for edge in cfg_blk.all_interstate_edges():
                for sym in edge.data.assignments.keys():
                    if sym in new_out_syms:
                        new_out_syms[sym] = None
            return new_out_syms

        elif isinstance(cfg_blk, ConditionalBlock):
            # Combine all outgoing symbols of the branches
            new_out_syms = copy.deepcopy(out_syms[cfg_blk.sub_regions()[0]])
            for b in cfg_blk.sub_regions():
                self._combine_syms(new_out_syms, out_syms[b])

            # If no else branch is present, also combine the incoming table (implicit else branch)
            has_non_conds = any([c is None for c, _ in cfg_blk.branches])
            if not has_non_conds:
                self._combine_syms(new_out_syms, in_syms[cfg_blk])

            return new_out_syms

        elif isinstance(cfg_blk, SDFGState):
            # Cannot change symbols in SDFGStates
            return in_syms[cfg_blk]

        else:
            # Use sink symbols as outgoing symbols
            sink_nodes = [n for n in cfg_blk.nodes() if cfg_blk.out_degree(n) == 0 and isinstance(n, ControlFlowBlock)]
            if len(sink_nodes) == 0:
                return in_syms[cfg_blk]

            new_out_syms = copy.deepcopy(out_syms[sink_nodes[0]])
            for n in sink_nodes:
                self._combine_syms(new_out_syms, out_syms[n])
            return new_out_syms

    # Given a cfg_blk, updates the symbols in the cfg_blk
    def _update_syms(
        self,
        cfg_blk: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> None:
        new_in_syms = copy.deepcopy(in_syms[cfg_blk])
        new_out_syms = copy.deepcopy(out_syms[cfg_blk])

        # Remove all symbols that are None
        new_in_syms = {sym: val for sym, val in new_in_syms.items() if val is not None}
        new_out_syms = {sym: val for sym, val in new_out_syms.items() if val is not None}

        changed = True
        while changed:
            changed = False
            free_sym = cfg_blk.free_symbols
            free_edge_sym = set([sym for edge in parent.out_edges(cfg_blk) for sym in edge.data.free_symbols])

            # Replace all symbols in the cfg_blk with their values
            if isinstance(cfg_blk, LoopRegion):
                cfg_blk.replace_meta_accesses(new_in_syms)
            elif isinstance(cfg_blk, ConditionalBlock):
                cfg_blk.replace_meta_accesses(new_in_syms)
            elif isinstance(cfg_blk, SDFGState):
                cfg_blk.replace_dict(new_in_syms)
            else:
                # Don't replace, as the nested CFBGs should inherit the symbols from their parent
                pass

            # Also replace all symbols in the outgoing edges with their values
            for edge in parent.out_edges(cfg_blk):
                edge.data.replace_dict(new_out_syms, replace_keys=False)

            # Check if the symbols have changed
            new_free_edge_sym = set([sym for edge in parent.out_edges(cfg_blk) for sym in edge.data.free_symbols])
            if free_sym != cfg_blk.free_symbols or free_edge_sym != new_free_edge_sym:
                changed = True

    # Combines two symbol dictionaries, setting the value to None if they don't agree. Directly modifies sym1
    def _combine_syms(self, sym1: Dict[str, Any], sym2: Dict[str, Any]) -> None:
        for sym, val in sym2.items():
            if sym not in sym1 or sym1[sym] != val:
                sym1[sym] = None
