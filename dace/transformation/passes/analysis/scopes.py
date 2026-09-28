# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis pass tabulating the symbols defined at every scope of an SDFG."""
from typing import Any, Dict

from dace import SDFG, properties
from dace.sdfg.state import SymbolResolver
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolScopes(ppl.Pass):
    """
    The symbols defined at every scope of every state, in one top-down walk over the SDFG hierarchy.

    The result is a :class:`~dace.sdfg.state.SymbolResolver` with every state tabulated:
    ``result.defined_at(state, node)`` equals ``state.symbols_defined_at(node)``, and
    ``result.scopes(state)`` maps each scope entry (``None`` for the top level) to its table.
    Like any analysis result, it is valid until a pass modifies what ``should_reapply`` names.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified
                    & (ppl.Modifies.Descriptors | ppl.Modifies.Symbols | ppl.Modifies.CFG | ppl.Modifies.Nodes))

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict[str, Any]) -> SymbolResolver:
        resolver = SymbolResolver()
        for sdfg in top_sdfg.all_sdfgs_recursive():
            for state in sdfg.states():
                resolver.scopes(state)
        return resolver
