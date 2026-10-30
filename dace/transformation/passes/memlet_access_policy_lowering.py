# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The code-generation-window pass that lowers every memlet access policy (see
:mod:`dace.sdfg.memlet_access_policy`) to ordinary SDFG constructs."""

from typing import Any

from dace import SDFG, SDFGState, properties
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.memlet_access_policy import MemletAccessPolicy
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.memlet_access_policies.helpers import leaf_edges


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerMemletAccessPolicies(ppl.Pass):
    """Lower every non-default memlet access policy to ordinary SDFG constructs by dispatching to the policy kind's
    :meth:`~dace.sdfg.memlet_access_policy.MemletAccessPolicy.lower`. Meant to run on the code-generation copy of the
    SDFG, after :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` and before
    library-node expansion; :func:`dace.codegen.codegen.generate_code` does so automatically. Idempotent: memlets
    that are already lowered are left alone."""

    assume_int32 = properties.Property(
        dtype=bool,
        default=False,
        desc='Treat "auto" cursors as int32 even when the array extent is not '
        "provably < 2**31 (caller guarantees 31-bit offsets).",
    )
    chain_outer_loops = properties.Property(
        dtype=bool,
        default=True,
        desc="Initialize an inner-loop cursor from an outer-loop cursor when "
        "its entry value is affine in the outer loop variable (one add per "
        "loop level, no multiplies), instead of recomputing it per entry.",
    )

    def __init__(self, assume_int32: bool = False, chain_outer_loops: bool = True):
        super().__init__()
        self.assume_int32 = assume_int32
        self.chain_outer_loops = chain_outer_loops

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> dict[str, int] | None:
        """
        :return: The summed counters of the policy kinds' lowerings (for loop cursors ``{'cursors': n,
                 'memlets': m, 'dropped': d}``), or ``None`` if there was nothing to lower.
        """
        totals: dict[str, int] = {}
        options = {"assume_int32": self.assume_int32, "chain_outer_loops": self.chain_outer_loops}
        # Policies are lowered per kind over the whole SDFG tree: a memlet in a nested SDFG may be addressed
        # relative to a loop of an enclosing SDFG along its memlet path.
        by_kind: dict[type[MemletAccessPolicy], list[tuple[SDFGState, MultiConnectorEdge[Memlet]]]] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for edge in leaf_edges(state):
                    if not edge.data.access_policy.is_default:
                        by_kind.setdefault(type(edge.data.access_policy), []).append((state, edge))
        for kind, entries in by_kind.items():
            for key, value in kind.lower(sdfg, entries, **options).items():
                totals[key] = totals.get(key, 0) + value
        return totals or None
