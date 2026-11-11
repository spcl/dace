# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map over a handful of species around plane-wide work unrolls, so each plane map is shared out on its own."""

import numpy as np

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.unroll_small_maps import UnrollSmallMaps

NB, KL = dace.symbol("NB"), dace.symbol("KL")


@dace.program
def species_sums(qx: dace.float64[NB, 5, KL], sol: dace.float64[NB, 5, 5, KL], qn: dace.float64[NB, 5, KL]):
    for jm in range(5):
        acc = np.zeros((NB, KL), dtype=np.float64)
        for jn in range(5):
            acc = acc + sol[:, jm, jn, :]
        qn[:, jm, :] = qx[:, jm, :] + acc


@dace.program
def plane_sums(sol: dace.float64[NB, 5, KL], out: dace.float64[NB, KL]):
    for jb in range(NB):
        out[jb, :] = sol[jb, 0, :] + sol[jb, 4, :]


def top_level_maps(sdfg: dace.SDFG) -> list:
    return [n for s in sdfg.states() for n in s.scope_children()[None] if isinstance(n, dace.nodes.MapEntry)]


def test_a_species_map_around_plane_work_unrolls_into_plane_maps():
    sdfg = species_sums.to_sdfg(simplify=True)
    canonicalize(sdfg)
    assert UnrollSmallMaps().apply_pass(sdfg, {})
    sdfg.validate()
    assert all(len(m.map.params) == 2 for m in top_level_maps(sdfg))

    rng = np.random.default_rng(0)
    qx, sol = rng.random((3, 5, 7)), rng.random((3, 5, 5, 7))
    qn = np.zeros((3, 5, 7))
    sdfg(qx=qx, sol=sol, qn=qn, NB=3, KL=7)
    np.testing.assert_allclose(qn, qx + sol.sum(axis=2))


def test_a_map_over_a_large_range_is_left_alone():
    sdfg = plane_sums.to_sdfg(simplify=True)
    canonicalize(sdfg)
    assert UnrollSmallMaps().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_a_species_map_around_plane_work_unrolls_into_plane_maps()
    test_a_map_over_a_large_range_is_left_alone()
