# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PropagateIndexSubsets`` — inline promoted index symbols back into memlet subsets.

The Python/Fortran frontend promotes a computed index ``i + offset`` to a scalar
(``i_plus_offset``, written by a tasklet) then to a symbol (``__sym = i_plus_offset``
on an interstate edge) used in a memlet subset ``A[__sym]``. That opaque symbol hides the
iter-var ``i`` from the tile-access classifier, which then mis-stages the access. This pass
rewrites every memlet subset by inlining the promoted index back to its original arithmetic
(``A[__sym]`` / ``A[i_plus_offset]`` -> ``A[i + offset]``) via
:func:`~dace.transformation.passes.vectorization.utils.tile_access.propagate_subset`, so the
access is direct and widens to a dense load.

Data-dependent indices (a gather ``A[idx[i]]``) are left untouched — they keep their symbol
and flow to the gather machinery. Run AFTER if-condition mask lowering (flat single-level body
states, so the reaching-def walk is complete) and BEFORE the tiling passes (so widening sees the
direct subset). Pairs with ``SymbolPropagation`` (folds the ``__sym`` layer) before it and
``RemoveUnusedSymbols`` (sweeps the now-dead promotion symbols) after it.
"""
from typing import Any, Dict, Optional

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.utils.tile_access import propagate_subset


class PropagateIndexSubsets(ppl.Pass):
    """Inline promoted index symbols into memlet subsets (dense-access recovery)."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Rewrite every memlet ``subset`` / ``other_subset`` in ``sdfg`` and its
        nested SDFGs by inlining promoted index symbols. ``propagate_subset`` is a
        best-effort no-op when a bound is unresolvable or data-dependent.

        :param sdfg: SDFG to transform in place.
        :returns: Number of subsets rewritten, or ``None`` if none.
        """
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for edge in state.edges():
                    mem = edge.data
                    if mem is None:
                        continue
                    if mem.subset is not None:
                        new = propagate_subset(mem.subset, sd, state)
                        if new is not None:
                            mem.subset = new
                            count += 1
                    if mem.other_subset is not None:
                        new_o = propagate_subset(mem.other_subset, sd, state)
                        if new_o is not None:
                            mem.other_subset = new_o
                            count += 1
        return count or None
