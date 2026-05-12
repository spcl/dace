# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``SplitMapForVectorRemainder`` (P2 vectorization-prep).

For every innermost step-1 map whose trip count is not a multiple of
``vector_width``, the pass splits the map into a main map covering the
largest multiple-of-W prefix plus a trailing remainder map. End-to-end
numerical equivalence against the pre-pass SDFG output is the contract
for the rewrite, the split itself is structurally verified by counting
maps and comparing the leftover ranges.
"""
import numpy as np

import dace
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.split_map_for_vector_remainder import (
    SplitMapForVectorRemainder, )
from dace.transformation.passes.vectorization.vectorization_utils import is_innermost_map


N = dace.symbol("N")


@dace.program
def add_one(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0


def _count_innermost_maps(sdfg):
    return sum(1 for n, g in sdfg.all_nodes_recursive()
               if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n))


def _bake_trip(sdfg, trip: int):
    """Replace the ``N`` symbol with a literal so divisibility can be
    decided at analysis time, ``sdfg.specialize`` only updates symbol
    bindings without touching the map-range expressions."""
    sdfg.replace_dict({"N": trip})


def test_split_non_divisible_trip_produces_two_maps():
    """N=17, W=8 is not divisible, expect a split (1 -> 2 innermost maps)."""
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    _bake_trip(sdfg, 17)
    before = _count_innermost_maps(sdfg)
    applied = SplitMapForVectorRemainder(vector_width=8).apply_pass(sdfg, {})
    after = _count_innermost_maps(sdfg)
    assert applied == 1
    assert after == before + 1


def test_split_divisible_trip_is_noop():
    """N=16, W=8 is divisible, pass returns ``None``."""
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    _bake_trip(sdfg, 16)
    applied = SplitMapForVectorRemainder(vector_width=8).apply_pass(sdfg, {})
    assert applied is None


def test_split_idempotent_after_first_apply():
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    _bake_trip(sdfg, 17)
    SplitMapForVectorRemainder(vector_width=8).apply_pass(sdfg, {})
    second = SplitMapForVectorRemainder(vector_width=8).apply_pass(sdfg, {})
    # After the first split, the main map has provably divisible trip and the
    # remainder is too short to split further, so a second invocation is a no-op.
    assert second is None


def test_numerical_correctness_scalar_remainder():
    rng = np.random.default_rng(seed=0)
    for trip in (7, 16, 17, 31, 64):
        a = rng.standard_normal(trip).astype(np.float64)
        b_ref = np.zeros_like(a)
        b_post = np.zeros_like(a)

        # Reference, no split.
        ref = add_one.to_sdfg(simplify=True)
        ref(a=a, b=b_ref, N=trip)

        # Post-split SDFG.
        sdfg = add_one.to_sdfg(simplify=True)
        NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
        _bake_trip(sdfg, trip)
        SplitMapForVectorRemainder(vector_width=8).apply_pass(sdfg, {})
        sdfg(a=a, b=b_post)

        np.testing.assert_allclose(b_post, b_ref, err_msg=f"trip={trip}")


def test_invalid_mode_raises():
    import pytest
    sdfg = add_one.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    _bake_trip(sdfg, 17)
    with pytest.raises(ValueError, match="mode must be"):
        SplitMapForVectorRemainder(vector_width=8, mode="bogus").apply_pass(sdfg, {})
