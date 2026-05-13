# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pytest configuration for the topical vectorization test directory.

Provides the ``branch_mode`` fixture, parameterized over the two branch
lowering paths the M3.2 work exposes on ``VectorizeCPU``:

- ``"fp_factor"``: today's path, ``EliminateBranches`` collapses if/else
  to ``a = c*x + (1-c)*y``.
- ``"merge"``: the new M3 path, ``SameWriteSetIfElseToMergeCFG`` plus
  ``BranchNormalization`` rewrite arms into ``merge(cond, ..., ...)``
  tasklets that the vectorizer will later lower to a SIMD blend.

Tests that exercise conditionals consume ``branch_mode`` and forward it
to ``run_vectorization_test``. Tests with no branches do not need it.
Both modes must produce numerically identical results against the
unvectorized scalar reference, otherwise the two lowerings have drifted.
"""
import pytest


@pytest.fixture(params=["fp_factor", "merge"])
def branch_mode(request) -> str:
    return request.param


@pytest.fixture(params=["divides_evenly", "scalar"])
def remainder_strategy(request) -> str:
    """Parametrise tests over the remainder-handling strategies wired into
    VectorizeCPU. ``"divides_evenly"`` is today's default (assumes the
    range is %% W == 0). ``"scalar"`` enables P2(mode='scalar') so non-
    divisible ranges get a step-1 sequential postamble. ``"masked"`` and
    ``"full_loop_mask"`` are queued (R2 / R3) and not yet exercised.

    Tests that go through ``run_vectorization_test`` and want to cover
    both strategies declare a ``remainder_strategy`` parameter. Tests
    that pin a specific strategy can ignore the fixture and pass the
    knob directly to ``run_vectorization_test``."""
    return request.param
