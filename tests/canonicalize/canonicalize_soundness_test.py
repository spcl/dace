# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Kernels canonicalization once changed the value of: each must compute what its unchanged SDFG computes."""

import copy

import numpy as np

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)


def _assert_canon_keeps(program, name: str, arrays: dict, **symbols) -> None:
    """Run ``program`` unchanged and through CPU canonicalization on copies of ``arrays``; every array must agree."""
    results = []
    for tag in ("ref", "canon"):
        sdfg = program.to_sdfg(simplify=False)
        sdfg.name = f"{name}_{tag}"
        if tag == "canon":
            sdfg = canonicalize(sdfg, target="cpu")
            finalize_for_target(sdfg, "cpu")
        args = copy.deepcopy(arrays)
        sdfg(**args, **symbols)
        results.append(args)
    for key, value in results[0].items():
        assert np.allclose(value, results[1][key]), key


@dace.program
def diagonal_update(A: dace.float64[120], C: dace.float64[120]):
    for i in range(N):
        for j in range(M):
            A[i + j] = A[i + j] * 2 + C[i]


def test_a_write_offset_by_an_inner_iterator_stays_ordered():
    """``A[i + j]`` overlaps across ``i``: ``i`` stays a loop, and a tile of ``j`` does not move outside it."""
    rng = np.random.default_rng(0)
    _assert_canon_keeps(diagonal_update, "diagonal_update", {"A": rng.random(120), "C": rng.random(120)}, N=100, M=20)


if __name__ == "__main__":
    test_a_write_offset_by_an_inner_iterator_stays_ordered()
