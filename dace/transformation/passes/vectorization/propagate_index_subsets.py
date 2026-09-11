# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PropagateIndexSubsets`` — inline promoted index symbols back into memlet subsets.

Frontend promotes computed index ``i + offset`` -> scalar (``i_plus_offset``, tasklet-written)
-> symbol (``__sym = i_plus_offset`` on iedge) used in subset ``A[__sym]``. Opaque symbol hides
iter-var ``i`` from tile-access classifier -> mis-staged access. Fix: inline promoted index back
to original arithmetic (``A[__sym]`` / ``A[i_plus_offset]`` -> ``A[i + offset]``) via
:func:`~dace.transformation.passes.vectorization.utils.tile_access.propagate_subset` -> direct
access, widens to dense load.

Data-dependent indices (gather ``A[idx[i]]``) left untouched: keep symbol, flow to gather
machinery. Run AFTER if-condition mask lowering (flat single-level body states -> reaching-def
walk complete), BEFORE tiling passes (widening sees direct subset). Pairs: ``SymbolPropagation``
(folds ``__sym`` layer) before, ``RemoveUnusedSymbols`` (sweeps dead promotion symbols) after.
"""
from typing import Any

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.tile_access import (build_symbol_definition_map, propagate_subset)


class PropagateIndexSubsets(ppl.Pass):
    """Inline promoted index symbols into memlet subsets (dense-access recovery)."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Rewrite every memlet ``subset`` / ``other_subset`` in ``sdfg`` + nested SDFGs by
        inlining promoted index symbols. ``propagate_subset`` = best-effort no-op on
        unresolvable / data-dependent bound.

        :param sdfg: SDFG to transform in place.
        :returns: Number of subsets rewritten, or ``None`` if none.
        """
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            # Hoisted: the map depends on (sd, state) only, and this rewrites subsets, not the
            # assignments and tasklets it is built from. Was rebuilt per subset.
            scan_cache: dict[int, Any] = {}
            # Same span: the gather-scalar test reads structure and descriptors, never a subset.
            dd_memo: dict[str, bool] = {}
            for state in sd.states():
                defs = build_symbol_definition_map(sd, state, scan_cache)
                if not defs:
                    continue
                for edge in state.edges():
                    mem = edge.data
                    if mem is None:
                        continue
                    if mem.subset is not None:
                        new = propagate_subset(mem.subset, sd, state, defs=defs, data_dep_memo=dd_memo)
                        if new is not None:
                            mem.subset = new
                            count += 1
                    if mem.other_subset is not None:
                        new_o = propagate_subset(mem.other_subset, sd, state, defs=defs, data_dep_memo=dd_memo)
                        if new_o is not None:
                            mem.other_subset = new_o
                            count += 1
        return count or None
