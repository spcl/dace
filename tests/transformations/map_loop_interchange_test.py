# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import MapLoopInterchange, MoveLoopIntoMap

N, T, K = (dace.symbol(s) for s in 'NTK')


@dace.program
def map_loop(A: dace.float64[T, N]):
    for i in dace.map[0:N]:
        for t in range(1, T):
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def map_triangle_loop(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for t in range(i + 1, N):
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def map_loop_break(A: dace.float64[T, N]):
    for i in dace.map[0:N]:
        for t in range(1, T):
            if A[t - 1, i] > K:
                break
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def loop_map(A: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + 1.0


def run(sdfg: dace.SDFG, **symbols) -> np.ndarray:
    A = np.random.default_rng(0).random((symbols['T'], symbols['N']))
    sdfg(A=A, **symbols)
    return A


def test_the_loop_moves_outside_the_map_and_computes_the_same_values():
    sdfg = map_loop.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)

    assert sdfg.apply_transformations(MapLoopInterchange) == 1

    (loop, ) = sdfg.nodes()
    (state, ) = loop.nodes()
    assert isinstance(loop, LoopRegion) and loop.loop_variable == 't'
    assert any(isinstance(n, nodes.MapEntry) for n in state.nodes())
    assert np.allclose(run(sdfg, N=5, T=4), run(reference, N=5, T=4), rtol=0, atol=0)


def test_the_interchange_undoes_move_loop_into_map():
    sdfg = loop_map.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations(MoveLoopIntoMap) == 1

    assert sdfg.apply_transformations(MapLoopInterchange) == 1

    assert [type(b) for b in sdfg.nodes()] == [LoopRegion]
    assert np.allclose(run(sdfg, N=5, T=4), run(reference, N=5, T=4), rtol=0, atol=0)


def test_a_loop_whose_bound_reads_the_map_parameter_stays_inside():
    """Each map iteration runs a different trip count, which one loop outside the map cannot express."""
    sdfg = map_triangle_loop.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(MapLoopInterchange) == 0


def test_a_loop_that_breaks_from_inside_a_branch_stays_inside():
    """A break ends one map iteration's loop early; outside the map it would end every iteration's."""
    sdfg = map_loop_break.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(MapLoopInterchange) == 0
