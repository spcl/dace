# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from dataclasses import dataclass
from dace.sdfg.state import (
    ControlFlowBlock,
    ControlFlowRegion,
)
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, properties
from typing import Any, Dict, Set, Optional


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
        all_cfgb = dict()
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowBlock):
                all_cfgb[node] = parent

        # For each CFG Block maintain a dict of incoming and outgoing symbols
        in_syms = {cfgb: {} for cfgb in all_cfgb.keys()}
        out_syms = {cfgb: {} for cfgb in all_cfgb.keys()}

        # Perform a fixed-point iteration to propagate symbols
        changed = True
        while changed:
            changed = False

            # Update incoming symbols
            for cfgb, parent in all_cfgb.items():
                new_in_syms = self._get_in_syms(cfgb, parent, out_syms)
                # Check if the incoming symbols have changed
                if new_in_syms != in_syms[cfgb]:
                    changed = True
                    in_syms[cfgb] = new_in_syms

            # Update outgoing symbols
            for cfgb, parent in all_cfgb.items():
                new_out_syms = self._get_out_syms(cfgb, parent, in_syms)
                # Check if the outgoing symbols have changed
                if new_out_syms != out_syms[cfgb]:
                    changed = True
                    out_syms[cfgb] = new_out_syms

        # Update symbols in the CFGB
        for cfgb, parent in all_cfgb.items():
            self._update_syms(cfgb, parent, in_syms, out_syms)
        return set()

    # Given a CFGB, builds the incoming set of symbols
    def _get_in_syms(
        self,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Combine the outgoing symbols of all incoming edges with their assignments to the CFGB
        in_syms = {}
        for edge in parent.in_edges(cfgb):
            sym_table = copy.deepcopy(out_syms[edge.src])
            sym_table.update(edge.data.assignments)

            for sym, val in sym_table.items():
                # Add the symbol to the incoming symbols if it is not already present. Cannot propagate arrays accesses
                if sym not in in_syms:
                    in_syms[sym] = (
                        val if val and "[" not in val and "]" not in val else None
                    )

                # If multiple sources don't agree on the value of a symbol, set it to None
                if sym in in_syms and in_syms[sym] != val:
                    in_syms[sym] = None

        # Starting block CFBGs should inherit the symbols from their parent
        # Ignore SDFGs as nested SDFGs have symbol mappings
        if (
            parent.start_block == cfgb
            and not isinstance(parent, SDFG)
            and parent in out_syms
        ):
            assert in_syms == {}
            in_syms = copy.deepcopy(out_syms[parent])

        return in_syms

    # Given a CFGB, builds the outgoing set of symbols
    def _get_out_syms(
        self,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Since symbols cannot be changed in a CFGB, the outgoing symbols are the same as the incoming symbols
        return in_syms[cfgb]

    # Given a CFGB, updates the symbols in the CFGB
    def _update_syms(
        self,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> None:
        in_syms = copy.deepcopy(in_syms[cfgb])
        out_syms = copy.deepcopy(out_syms[cfgb])

        # Remove all symbols that are None
        in_syms = {sym: val for sym, val in in_syms.items() if val is not None}
        out_syms = {sym: val for sym, val in out_syms.items() if val is not None}

        changed = True
        while changed:
            changed = False
            free_sym = cfgb.free_symbols

            # Replace all symbols in the CFGB with their values
            cfgb.replace_dict(in_syms)

            # Also replace all symbols in the outgoing edges with their values
            for edge in parent.out_edges(cfgb):
                edge.data.replace_dict(out_syms, replace_keys=False)

            # Check if the symbols have changed
            if free_sym != cfgb.free_symbols:
                changed = True
