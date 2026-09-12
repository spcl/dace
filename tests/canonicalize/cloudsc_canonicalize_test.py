# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Applies the canonicalization stages to a real CloudSC SDFG.

    Uses the Python-frontend-derived, simplified CloudSC SDFG as a realistic structural
    fixture -- "copy parts from CloudSC and ensure we can apply". In the loop-centric
    pipeline every stage through ``parallelize`` (``LoopToMap``) must apply and keep the
    SDFG valid. Canonicalizing the *full* CloudSC SDFG end-to-end is intentionally out of
    scope for now; that walk lives in ``cloudsc_canonicalize_staged_test.py``.

    The fixture comes from ``build_cloudsc_sdfg`` (same builder ``cloudsc_canonicalize_staged_test.py``
    uses): a fresh parse on a cache miss, a cached ``.sdfgz`` otherwise (see
    ``generate_data_for_cloudsc.cloudsc_cache_dir``). Never point this at a hand-picked file --
    self-contained is the whole point. The parse this falls back to is minutes long, so this stays
    ``integration``.
"""
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES
from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg

pytestmark = pytest.mark.integration


def residual_loops(sdfg: dace.SDFG) -> int:
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def map_entries(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def test_the_loop_centric_stages_expose_parallelism_in_cloudsc():
    """Every stage through ``parallelize`` applies, keeps CloudSC valid, and leaves it with fewer
    loops and more Maps than it started with -- which a recipe of no-ops cannot do."""
    sdfg = build_cloudsc_sdfg(simplify=True)
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
