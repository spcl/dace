# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` on a loop whose accesses are RANGES that mirror each other across the diagonal.

Polybench ``covariance`` is the shape: iteration ``i`` computes a row tail ``cov[i, i:M]`` and
copies it into the matching column tail ``cov[i:M, i]``. Every access is ragged -- its extent
depends on the iteration variable -- so no dimension separates on its own and nothing is a point
subset, which is exactly what the affine per-dimension rules and the point-only collision
certificate both abstain on. The loop is nevertheless a clean DOALL: the row and the column of one
iteration meet only at the diagonal element ``(i, i)``, inside that iteration.

The write/write pair and the read/write pair are separate refusals in ``can_be_applied``, and this
file pins both plus the negative that keeps the certificate honest.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.analysis import smt_dependence

M = dace.symbol('M')

pytestmark = pytest.mark.skipif(not smt_dependence.has_z3(), reason='the range certificate is SMT-backed')


def has_map(sdfg: dace.SDFG) -> bool:
    return any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


def loop_labels(sdfg: dace.SDFG):
    return [c.label for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


@dace.program
def mirror_triangle(cov: dace.float64[M, M], src: dace.float64[M, M]):
    for i in range(M):
        cov[i, i:M] = src[i, i:M]
        cov[i:M, i] = cov[i, i:M]


@dace.program
def mirror_full(cov: dace.float64[M, M], src: dace.float64[M, M]):
    # The same program with the triangular clamp dropped: row ``i`` and column ``i`` of DIFFERENT
    # iterations now genuinely share an element, so the loop carries a real dependence.
    for i in range(M):
        cov[i, :] = src[i, :]
        cov[:, i] = cov[i, :]


def test_mirrored_triangular_ranges_parallelize():
    """The whole point: no LoopRegion survives and a Map appears in its place. Asserting the
    structure rather than "applied == True" is what catches a lift that fires but leaves the loop
    standing."""
    sdfg = mirror_triangle.to_sdfg(simplify=True)
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated([LoopToMap]) > 0
    assert loop_labels(sdfg) == []
    assert has_map(sdfg)


def test_mirrored_full_ranges_are_refused():
    """Same shape, no clamp. Iteration ``p`` writes row ``p`` and iteration ``q`` writes column
    ``q``, which meet at ``(p, q)`` for every pair -- lifting it is a race. This is the negative
    that keeps the certificate from being read off the transpose alone."""
    sdfg = mirror_full.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg.apply_transformations_repeated([LoopToMap])
    assert loop_labels(sdfg) != []


def test_mirrored_triangular_ranges_keep_their_values():
    """The lift is only correct if the numbers do not move, and the diagonal is where a wrong
    ordering would show: it is the one element both statements of an iteration touch."""
    rng = np.random.default_rng(0)
    src = rng.random((64, 64))

    ref = np.zeros((64, 64))
    for i in range(64):
        ref[i, i:64] = src[i, i:64]
        ref[i:64, i] = ref[i, i:64]

    sdfg = mirror_triangle.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg.apply_transformations_repeated([LoopToMap])
    got = np.zeros((64, 64))
    sdfg(cov=got, src=src, M=64)
    assert np.allclose(got, ref)


if __name__ == '__main__':
    test_mirrored_triangular_ranges_parallelize()
    test_mirrored_full_ranges_are_refused()
    test_mirrored_triangular_ranges_keep_their_values()
