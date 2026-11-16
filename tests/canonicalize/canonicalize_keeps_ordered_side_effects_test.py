# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalization keeps every node with an ordered side effect (a communication, a print) in its loop and in the
order it had, and still parallelizes what runs beside it.

Before the fix a loop of per-row blocking ``Allreduce`` calls became a map (the ranks could then issue the
collectives in different orders), and a loop holding a ``Send`` and a ``Recv`` was distributed into all sends, then all
receives, which reorders the exchange and can deadlock. ``LoopToMap``, the loop distribution, ``LoopFission``,
``LoopFusion`` and ``WavefrontSkew`` now each refuse what would move such a node."""

import numpy as np
import pytest

import dace
from dace.libraries.comm.nodes.node import CommNode
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_fusion import LoopFusion
from dace.transformation.interstate.loop_to_map import LoopToMap, Refusal
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.perfect_loop_nesting import separates_ordered_side_effects
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.parallelization.wavefront.skew import WavefrontSkew
from dace.transformation.passes.parallelization.wavefront.statements import occurrence_index

# The machine-learning CI job collects every module and has no MPI
MPI = pytest.importorskip("mpi4py.MPI")

K = dace.symbol("K", dtype=dace.int64)
N = dace.symbol("N")


@dace.program
def per_row_allreduce(x: dace.float64[K, 4]):
    for i in range(K):
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, x[i], op=MPI.SUM)


@dace.program
def ring_loop(a: dace.float64[K, 4], b: dace.float64[K, 4], left: dace.int32, right: dace.int32):
    for i in range(K):
        MPI.COMM_WORLD.Send(a[i], right, 0)
        MPI.COMM_WORLD.Recv(b[i], left, 0)


@dace.program
def scale_then_allreduce(x: dace.float64[K, 4], y: dace.float64[K, 4], z: dace.float64[K, 4]):
    for i in range(K):
        y[i] = 2.0 * z[i]
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, x[i], op=MPI.SUM)


@dace.program
def two_logged_loops(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        a[i] = a[i] + 1.0
    for i in range(N):
        b[i] = b[i] + 1.0


@dace.program
def two_carried_loops(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + 1.0
    for i in range(1, N):
        b[i] = b[i - 1] + 1.0


@dace.program
def wavefront_2d(aa: dace.float64[N, N]):
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = (aa[i, j - 1] + aa[i - 1, j]) / 1.9


def inside_a_map(state: dace.SDFGState, node: nodes.Node) -> bool:
    """Whether a map encloses ``node``, in its state or around the nested SDFGs that hold it."""
    while state.entry_node(node) is None:
        nest = state.sdfg.parent_nsdfg_node
        if nest is None:
            return False
        node, state = nest, state.sdfg.parent
    return True


def comm_nodes(sdfg: dace.SDFG) -> list[tuple[str, bool]]:
    """Every communication node in program order: its type, and whether a map encloses it."""
    return [
        (type(node).__name__, inside_a_map(parent, node))
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, CommNode)
    ]


def loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [region for region in sdfg.all_control_flow_regions(recursive=True) if isinstance(region, LoopRegion)]


def maps(sdfg: dace.SDFG) -> int:
    return sum(isinstance(node, nodes.MapEntry) for node, parent in sdfg.all_nodes_recursive())


def logged(sdfg: dace.SDFG) -> dace.SDFG:
    """``sdfg`` with every tasklet marked as having an ordered side effect, as a print or a log call has."""
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            node.side_effects = True
    return sdfg


def test_a_loop_of_blocking_collectives_stays_a_loop():
    sdfg = per_row_allreduce.to_sdfg(simplify=True)

    canonicalize(sdfg)

    assert comm_nodes(sdfg) == [("Allreduce", False)]
    assert len(loops(sdfg)) == 1 and maps(sdfg) == 0


def test_a_send_and_a_receive_stay_in_one_loop_in_their_order():
    sdfg = ring_loop.to_sdfg(simplify=True)

    canonicalize(sdfg)

    assert comm_nodes(sdfg) == [("Send", False), ("Recv", False)]
    assert len(loops(sdfg)) == 1, "the exchange is not distributed into all sends, then all receives"


def test_the_computation_beside_a_collective_is_still_parallelized():
    sdfg = scale_then_allreduce.to_sdfg(simplify=True)

    canonicalize(sdfg)

    assert comm_nodes(sdfg) == [("Allreduce", False)]
    assert len(loops(sdfg)) == 1, "the collective keeps its loop"
    assert maps(sdfg) >= 1, "the scaling, distributed into a loop of its own, becomes a map"


def test_loop_to_map_refuses_a_loop_with_an_ordered_side_effect_and_says_why():
    sdfg = logged(two_logged_loops.to_sdfg(simplify=True))
    loop = loops(sdfg)[0]
    sut = LoopToMap()
    sut.setup_match(sdfg, loop.parent_graph.cfg_id, -1, {LoopToMap.loop: loop}, 0, override=True)

    assert not sut.can_be_applied(loop.parent_graph, 0, sdfg)
    assert sut.last_refusal is not None and sut.last_refusal.kind is Refusal.ORDERED_SIDE_EFFECT


def test_two_loops_with_ordered_side_effects_do_not_fuse_and_without_them_they_do():
    plain = two_carried_loops.to_sdfg(simplify=True)
    first, second = loops(plain)
    assert LoopFusion.can_be_applied_to(plain, first=first, second=second)

    sdfg = logged(two_carried_loops.to_sdfg(simplify=True))
    first, second = loops(sdfg)

    assert not LoopFusion.can_be_applied_to(sdfg, first=first, second=second)


def test_fission_and_distribution_keep_ordered_side_effects_in_one_loop():
    sdfg = ring_loop.to_sdfg(simplify=True)
    (loop,) = loops(sdfg)
    blocks = list(loop.nodes())

    assert len(loop_analysis.ordered_side_effects(loop)) == 2
    assert not LoopFission.can_fission(loop)
    assert separates_ordered_side_effects([[block] for block in blocks])
    assert not separates_ordered_side_effects([blocks])


def test_the_wavefront_skew_refuses_a_nest_with_an_ordered_side_effect():
    plain = wavefront_2d.to_sdfg(simplify=True)
    assert WavefrontSkew().plan(loops(plain)[0], plain, occurrence_index(plain)) is not None

    sdfg = logged(wavefront_2d.to_sdfg(simplify=True))
    reason = WavefrontSkew().plan_or_reason(loops(sdfg)[0], sdfg, occurrence_index(sdfg))

    assert isinstance(reason, str) and "ordered side effect" in reason


def test_a_loop_without_side_effects_still_parallelizes():
    sdfg = two_logged_loops.to_sdfg(simplify=True)

    canonicalize(sdfg)

    a, b = np.zeros(8), np.ones(8)
    sdfg(a=a, b=b, N=8)
    assert loops(sdfg) == [] and maps(sdfg) >= 1
    assert np.allclose(a, 1.0) and np.allclose(b, 2.0)


if __name__ == "__main__":
    test_a_loop_of_blocking_collectives_stays_a_loop()
    test_a_send_and_a_receive_stay_in_one_loop_in_their_order()
    test_the_computation_beside_a_collective_is_still_parallelized()
    test_loop_to_map_refuses_a_loop_with_an_ordered_side_effect_and_says_why()
    test_two_loops_with_ordered_side_effects_do_not_fuse_and_without_them_they_do()
    test_fission_and_distribution_keep_ordered_side_effects_in_one_loop()
    test_the_wavefront_skew_refuses_a_nest_with_an_ordered_side_effect()
    test_a_loop_without_side_effects_still_parallelizes()
