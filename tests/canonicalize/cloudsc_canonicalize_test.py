# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Applies the canonicalization stages to a real CloudSC SDFG.

    Uses the in-repo Python-frontend-derived CloudSC SDFG (306 states, 139
    LoopRegions, 79 conditional blocks) as a realistic structural fixture --
    "copy parts from CloudSC and ensure we can apply". Every canonicalization
    stage through ``loop_to_map`` must apply and keep the SDFG valid. The final
    ``maximal_fusion`` stage currently invalidates CloudSC inside the existing
    ``FullMapFusion`` / ``TaskletFusion`` passes; that is a pre-existing fusion
    limitation, tracked as a strict xfail and out of the canonicalization
    sub-pass scope.
"""
import os

import pytest

import dace
from dace.transformation.passes.canonicalize import CanonicalizationPipeline
from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES

_CLOUDSC = os.path.join(os.path.dirname(__file__), os.pardir, "sdfg", "data", "sdfg_reconstruction",
                        "cloudsc_simplified.sdfgz")


def _load() -> dace.SDFG:
    return dace.SDFG.from_file(_CLOUDSC)


@pytest.mark.skipif(not os.path.exists(_CLOUDSC), reason="CloudSC fixture not present")
def test_canonicalization_stages_apply_to_cloudsc():
    sdfg = _load()
    sdfg.validate()  # fixture must start valid
    for name, factory in CANONICALIZE_STAGES:
        for unit in factory():
            unit.apply_pass(sdfg, {})
        sdfg.validate()  # each stage must preserve a valid SDFG
        if name == "loop_to_map":
            break  # proven-valid canonicalization-sub-pass scope


@pytest.mark.skipif(not os.path.exists(_CLOUDSC), reason="CloudSC fixture not present")
@pytest.mark.xfail(reason="The final maximal_fusion stage (existing FullMapFusion / "
                   "TaskletFusion) produces an InvalidSDFGNodeError on the "
                   "canonicalized CloudSC SDFG. Pre-existing fusion-pass "
                   "limitation, tracked as a follow-up; out of the "
                   "canonicalization sub-pass scope.",
                   strict=True)
def test_full_pipeline_on_cloudsc():
    sdfg = _load()
    CanonicalizationPipeline(validate=True).apply_pass(sdfg, {})
    sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
