# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Verify that no ``NestedSDFG`` still carries a multi-element transient
array in its ``arrays`` table. Scalars (``shape == (1,)``) are allowed
because they're typically used as thread-local scratch.

This is the invariant ``LiftTransients`` is expected to establish once
it finishes. Useful as a standalone sanity check anywhere in a pipeline
after allocation-layout work -- for example, right before a GPU
offloading pass that assumes every multi-element transient allocation
happens at a known-scope SDFG.
"""
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl


class VerifyNoNestedTransients(ppl.Pass):
    """Assert no multi-element transient array lives inside any
    ``NestedSDFG``. Raises ``ValueError`` listing every offender."""

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
        """Return the list of offenders (as ``(nsdfg_label, array_name)``
        tuples) on violation so callers running inside a pipeline can
        inspect them; also raise ``ValueError`` so direct callers get
        the failure loudly. Returns ``None`` when the invariant holds."""
        offenders = _find_offenders(sdfg)
        if not offenders:
            return None
        details = "\n  - ".join(
            f"nested SDFG {label!r} declares transient {name!r} (shape={shape})"
            for label, name, shape in offenders)
        raise ValueError(
            f"VerifyNoNestedTransients failed: {len(offenders)} offender(s):\n  - {details}")


def verify_no_nested_transients(sdfg: SDFG):
    """Functional entry point. Raises ``ValueError`` on violation.
    No return value -- the check is boolean in spirit."""
    VerifyNoNestedTransients().apply_pass(sdfg, {})


def _find_offenders(sdfg: SDFG) -> List[Tuple[str, str, tuple]]:
    offenders: List[Tuple[str, str, tuple]] = []
    for n, _ in sdfg.all_nodes_recursive():
        if not isinstance(n, nodes.NestedSDFG):
            continue
        for name, desc in n.sdfg.arrays.items():
            if not desc.transient:
                continue
            shape = tuple(desc.shape)
            if shape == (1, ):
                continue
            offenders.append((n.label, name, shape))
    return offenders
