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


# Reproducer: Fix the untested 11-18 GPU and hint commits: all_states, SymExpr trip counts, stale tests (06214771fe)
def test_trip_count_reads_a_bound_that_carries_an_approximation():
    """A device map's bound can be an exact value with an approximation attached; the count is the exact one's, and a
    symbolic one is no literal count at all (the HIP rendering of a nested launch raised here)."""
    from dace import symbolic
    from dace.sdfg import nodes as nd
    from dace.transformation.passes.unroll_small_maps import constant_trips

    sdfg = dace.SDFG("trips_with_approximation")
    state = sdfg.add_state()
    literal, unused_exit = state.add_map("literal", {"i": "0:4"})
    literal.map.range = dace.subsets.Range([(0, symbolic.SymExpr(3, 7), 1)])
    assert isinstance(literal, nd.MapEntry) and constant_trips(literal) == 4
    symbolic_entry, unused_exit = state.add_map("symbolic", {"i": "0:N"})
    symbolic_entry.map.range = dace.subsets.Range([(0, symbolic.SymExpr(dace.symbol("N") - 1, 63), 1)])
    assert constant_trips(symbolic_entry) is None
