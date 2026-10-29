# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The canonicalization pipeline: brings an SDFG into the canonical form that later passes assume."""

from dataclasses import dataclass

from dace import properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.hoist_shared_transients import HoistSharedTransients

#: The passes of the canonicalization pipeline, in order.
CANONICALIZATION_PASSES = [
    HoistSharedTransients,
]


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class CanonicalizationPipeline(ppl.Pipeline):
    """
    Brings an SDFG into the canonical form that later passes, the code generator, and the Lean formal semantics in
    ``formal/`` assume. The pipeline runs after the frontend, after library node expansion, and before code
    generation, so it is idempotent and cheap on an SDFG that is already canonical.

    Invariants of the canonical form:

    1. No nested SDFG (at any depth) owns a transient whose storage outlives a single call, i.e., with
       ``Persistent``, ``External`` or ``Global`` lifetime (see
       :class:`~dace.transformation.passes.hoist_shared_transients.HoistSharedTransients`). Such containers live in
       the top-level SDFG and are passed to nested SDFGs as arguments.
    """

    CATEGORY: str = "Canonicalization"

    def __init__(self):
        super().__init__([p() for p in CANONICALIZATION_PASSES])
