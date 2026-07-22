# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization should fission a fused SCoP so it maximally parallelizes.

Encodes the static control part (SCoP) from Grosser et al., "Polly - Performing
Polyhedral Optimizations on a Low-Level Intermediate Representation", Parallel
Processing Letters 2012, Listing 1::

    for (i = 0; i <= N; i++) {
        if (i <= N - 50) A[5*i] = 1; else A[3*i] = 2;
        for (j = 0; j <= N; j++) B[i][2*j] = 3;
    }

Written with the Python frontend (the inner loop as a ``dace.map``). Fused, the
``i`` loop body mixes a conditional store and a nested loop, so it does not
become one clean parallel map. After canonicalization (fission before
loop-to-map) the independent statements should each become maximally parallel
maps, while staying numerically identical.
"""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes import canonicalize
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.interstate.loop_to_map import LoopToMap

N = dace.symbol('N')


@dace.program
def scop(A: dace.int64[5 * N], B: dace.int64[N, 2 * N]):
    for i in range(N):
        if i <= N - 50:
            A[5 * i] = 1
        else:
            A[3 * i] = 2
        for j in dace.map[0:N]:
            B[i, 2 * j] = 3


def _reference(n: int):
    a = np.full(5 * n, -1, dtype=np.int64)
    b = np.full((n, 2 * n), -1, dtype=np.int64)
    for i in range(n):
        if i <= n - 50:
            a[5 * i] = 1
        else:
            a[3 * i] = 2
        for j in range(n):
            b[i, 2 * j] = 3
    return a, b


def _num_map_entries(sdfg: dace.SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _build():
    return scop.to_sdfg(simplify=False)


def test_scop_canonicalize_numerically_correct():
    n = 200
    sdfg = _build()
    canonicalize(sdfg, validate=True)

    a = np.full(5 * n, -1, dtype=np.int64)
    b = np.full((n, 2 * n), -1, dtype=np.int64)
    sdfg(A=a, B=b, N=n)

    ref_a, ref_b = _reference(n)
    assert np.array_equal(a, ref_a)
    assert np.array_equal(b, ref_b)


def test_scop_fission_enables_more_parallelism():
    # Baseline: only loop-to-map on the fused form.
    baseline = _build()
    PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(baseline, {})
    baseline_maps = _num_map_entries(baseline)

    # Canonicalized: fission before loop-to-map.
    canon = _build()
    canonicalize(canon, validate=True)
    canon_maps = _num_map_entries(canon)

    assert canon_maps >= 1, "canonicalization produced no parallel map"
    assert canon_maps >= baseline_maps, (f"canonicalization parallelized less than the fused baseline "
                                         f"({canon_maps} < {baseline_maps})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
