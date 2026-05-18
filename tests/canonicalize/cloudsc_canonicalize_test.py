# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Applies the canonicalization stages to a real CloudSC SDFG.

    Uses the in-repo Python-frontend-derived CloudSC SDFG (306 states, 139
    LoopRegions, 79 conditional blocks) as a realistic structural fixture --
    "copy parts from CloudSC and ensure we can apply". In the loop-centric
    pipeline every stage through ``parallelize`` (``LoopToMap``) must apply
    and keep the SDFG valid. Canonicalizing the *full* CloudSC SDFG
    end-to-end is intentionally out of scope for now.
"""
import os

import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import _build_stages

_CLOUDSC = os.path.join(os.path.dirname(__file__), os.pardir, "sdfg", "data", "sdfg_reconstruction",
                        "cloudsc_simplified.sdfgz")


def _load() -> dace.SDFG:
    return dace.SDFG.from_file(_CLOUDSC)


@pytest.mark.skipif(not os.path.exists(_CLOUDSC), reason="CloudSC fixture not present")
def test_canonicalization_stages_apply_to_cloudsc():
    sdfg = _load()
    sdfg.validate()  # fixture must start valid
    prev = None
    for label, unit in _build_stages():
        if prev is not None and label != prev:
            sdfg.validate()  # each stage boundary must preserve a valid SDFG
            if prev == "parallelize":
                return  # proven-valid canonicalization-sub-pass scope (through LoopToMap)
        unit.apply_pass(sdfg, {})
        prev = label
    sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
