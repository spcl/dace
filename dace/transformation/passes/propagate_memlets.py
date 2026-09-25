# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rebuild every scope-summary memlet in an SDFG from its body."""
from typing import Any, Dict, Optional, Set

from dace.sdfg import SDFG
from dace.sdfg.propagation import propagate_memlets_scope
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@transformation.explicit_cf_compatible
class PropagateMemlets(ppl.Pass):
    """Recompute the memlets on every scope boundary from the memlets inside it.

    A map entry/exit edge is a SUMMARY of the body it encloses, so it is derived data: any pass
    that rewrites a body memlet without re-propagating leaves the summary describing a graph that
    no longer exists. The stale summary is always an over-approximation, never wrong output -- but
    the dependence analyses read it, and an over-approximated write is exactly what makes
    ``LoopToMap`` refuse a parallel loop.

    Two producers of stale summaries are known, and both are structural rather than accidental:

    * ``InlineSDFG`` replaces a NestedSDFG with its body and rewrites the body memlets into outer
      coordinates, but leaves the enclosing scope's edges at the whole-array approximation that the
      opaque nested node had forced. Polybench ``covariance`` writes ``cov[i, i:M]`` inside a map
      whose exit edge still reads ``cov[0:M, 0:M]`` long after the inline exposed the exact subset.
    * Unrolling and peeling change which symbols are defined at a scope, so summaries computed
      before them were built against a smaller ``symbols_defined_at`` (this is what widened
      cloudsc's ``tendency_loc_cld`` to its full extent).

    Only the SCOPE boundaries are rebuilt -- a nested SDFG's connector memlets are left alone.
    Re-deriving those means folding the body's frame back into the caller's, and where the body
    holds the whole array and indexes it absolutely the only sound result keeps the outer base,
    so ``a[it]`` widens to the growing range ``a[0:it + 1]``. That reads downstream as a
    triangular write and refuses the very parallelization this pass exists to enable.

    Propagation is value-preserving -- it only ever rewrites derived edges -- so this is safe to
    schedule anywhere. Cost is one walk per state, which is why it is scheduled at the few points
    that consume the summaries rather than after every rewrite.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set[Any]:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                propagate_memlets_scope(nested, state, state.scope_leaves())
        return 1
