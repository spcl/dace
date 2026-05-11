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
