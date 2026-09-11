# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Applies the canonicalization stages to a real CloudSC SDFG.

    Uses the Python-frontend-derived CloudSC SDFG (306 states, 139
    LoopRegions, 79 conditional blocks) as a realistic structural fixture --
    "copy parts from CloudSC and ensure we can apply". In the loop-centric
    pipeline every stage through ``parallelize`` (``LoopToMap``) must apply
    and keep the SDFG valid. Canonicalizing the *full* CloudSC SDFG
    end-to-end is intentionally out of scope for now; that walk lives in
    ``cloudsc_canonicalize_staged_test.py``.

    The ``.sdfgz`` is a build artifact, not a committed fixture
    (``tests/.gitignore`` excludes ``data/``), so this file carries the
    ``integration`` marker: the unit gate deselects it by mark rather than
    reporting a green skip, and where the artifact exists its absence is an
    error.
"""
import os

import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES

CLOUDSC = os.path.join(os.path.dirname(__file__), os.pardir, "sdfg", "data", "sdfg_reconstruction",
                       "cloudsc_simplified.sdfgz")

pytestmark = pytest.mark.integration


def residual_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def map_entries(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def test_the_loop_centric_stages_expose_parallelism_in_cloudsc():
    """Every stage through ``parallelize`` applies, keeps CloudSC valid, and leaves it with fewer
    loops and more Maps than it started with -- which a recipe of no-ops cannot do."""
    assert os.path.exists(CLOUDSC), f"CloudSC artifact missing: {CLOUDSC}"
    sdfg = dace.SDFG.from_file(CLOUDSC)
    sdfg.validate()  # fixture must start valid
    loops_before, maps_before = residual_loops(sdfg), map_entries(sdfg)

    last_parallelize = max(i for i, (label, _) in enumerate(CANONICALIZE_STAGES) if label == 'parallelize')
    applied = []
    for label, factory in CANONICALIZE_STAGES[:last_parallelize + 1]:
        for unit in factory():
            if unit.apply_pass(sdfg, {}) is not None:
                applied.append((label, type(unit).__name__))
        sdfg.validate()  # each stage boundary must preserve a valid SDFG

    assert applied, "not one stage through parallelize rewrote anything"
    assert map_entries(sdfg) > maps_before, \
        f"LoopToMap exposed no parallelism: {maps_before} -> {map_entries(sdfg)} maps"
    assert residual_loops(sdfg) < loops_before, \
        f"no loop was lifted: {loops_before} -> {residual_loops(sdfg)} LoopRegions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
