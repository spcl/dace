# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Set every transient array's allocation lifetime to ``SDFG``.

Walks the whole SDFG tree (top-level + every ``NestedSDFG.sdfg``) and, for
each descriptor with ``transient=True``, assigns
``dace.dtypes.AllocationLifetime.SDFG``. Non-transient descriptors are
left untouched.

Lifetime ``SDFG`` = allocated once when the innermost enclosing SDFG is
entered, freed when it exits. This is the right default for per-invocation
scratch; promoting to ``Persistent`` (carrying storage across SDFG calls)
is left to whatever codegen pass decides to pin that storage.
"""
from typing import Any, Dict, Optional

from dace import SDFG, dtypes
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl


class SetTransientSDFGLifetime(ppl.Pass):
    """Flip every transient descriptor's lifetime to ``AllocationLifetime.SDFG``."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Descriptors)

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        count = set_transient_sdfg_lifetime(sdfg)
        return count if count > 0 else None


def set_transient_sdfg_lifetime(sdfg: SDFG) -> int:
    """Functional entry point. Returns the number of descriptors flipped."""
    count = 0
    for g in _all_sdfgs(sdfg):
        for _name, desc in g.arrays.items():
            if desc.transient and desc.lifetime != dtypes.AllocationLifetime.SDFG:
                desc.lifetime = dtypes.AllocationLifetime.SDFG
                count += 1
    return count


def _all_sdfgs(sdfg: SDFG):
    yield sdfg
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            yield n.sdfg
