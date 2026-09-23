# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Small maps that share a state inside a serial host loop stay on the host together.

Loop pinning keeps a host loop's small maps over a container its host code also touches on the host,
but it asked each of them to be its state's ONLY device work. ls3df_scf's Jacobi ``eigh`` rotates a
row and a column of its matrix in one state per ``(p, q)`` pair, so both stayed kernels: two launches
and a host read of the matrix per rotation, most of the kernel's canon GPU time.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.offloading.host_maps import provably_at_least

K = dace.symbol('K')
ROWS, COLS, NSTATE = (dace.symbol(name) for name in ('rows', 'cols', 'nstate'))


@dace.program
def rotations_in_a_host_loop(C: dace.float64[K, K]):
    """The host tests one entry per ``(p, q)`` pair; two small maps in ONE state rotate a row and a column."""
    for p in range(K - 1):
        for q in range(p + 1, K):
            if C[p, q] > 0.5:
                t = C[p, q] * 0.25
                for j in dace.map[0:K]:
                    C[p, j] = C[p, j] + t * C[q, j]
                for j in dace.map[0:K]:
                    C[j, q] = C[j, q] - t * C[j, p]


def reference(C: np.ndarray) -> None:
    n = C.shape[0]
    for p in range(n - 1):
        for q in range(p + 1, n):
            if C[p, q] > 0.5:
                t = C[p, q] * 0.25
                C[p, :] = C[p, :] + t * C[q, :]
                C[:, q] = C[:, q] - t * C[:, p]


def offloaded() -> dace.SDFG:
    sdfg = rotations_in_a_host_loop.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    sdfg.validate()
    return sdfg


def copies_inside_loops(sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """``(source, destination)`` of every copy between host and device memory under a loop."""
    found = []
    for state in sdfg.all_states():
        region = state.parent_graph
        while region is not None and not isinstance(region, (LoopRegion, dace.SDFG)):
            region = region.parent_graph
        if not isinstance(region, LoopRegion):
            continue
        for src in state.data_nodes():
            for dst in state.successors(src):
                if isinstance(dst, dace.nodes.AccessNode):
                    sides = [n.desc(sdfg).storage == dtypes.StorageType.GPU_Global for n in (src, dst)]
                    if sides[0] != sides[1]:
                        found.append((src.data, dst.data))
    return found


def test_small_maps_sharing_a_state_in_a_host_loop_stay_on_the_host():
    """Both rotations are host maps, and nothing crosses the bus inside the loops."""
    sdfg = rotations_in_a_host_loop.to_sdfg(simplify=True)
    rotations = [(node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert len(rotations) == 2 and rotations[0][1] is rotations[1][1], 'the two rotations no longer share a state'
    sdfg = offloaded()
    rotations = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert rotations and all(entry.map.schedule == dtypes.ScheduleType.Sequential for entry in rotations)
    assert not copies_inside_loops(sdfg), copies_inside_loops(sdfg)


@pytest.mark.gpu
def test_rotations_kept_on_the_host_compute_what_numpy_computes():
    sdfg = offloaded()
    C = np.random.default_rng(7).random((6, 6))
    want = C.copy()
    reference(want)
    sdfg(C=C, K=6)
    np.testing.assert_allclose(C, want)


@pytest.mark.parametrize(
    'traffic, shared, moves_at_least_what_it_shares',
    [
        # srad's stencil against the J, iN, iS, jE, jW it shares with the loop's host code: a sum with
        # negative terms once the shared sizes are subtracted, which SymPy's sign inference leaves open.
        # Read as unproven it made the stencil a pinning candidate, and the whole program ran on the host.
        (2 * COLS**2 * ROWS + 8 * COLS * ROWS + 2 * ROWS**2, COLS * ROWS + 2 * ROWS + 2 * COLS, True),
        (3 * ROWS, ROWS + 2, True),
        (dace.symbolic.ipow(NSTATE, 2), NSTATE * NSTATE, True),
        # ls3df_scf's row rotation against the nstate x nstate matrix: smaller once nstate > 14.
        (14 * NSTATE, NSTATE**2, False),
        (2 * ROWS, 3 * ROWS - 5, False),
    ])
def test_a_map_moving_what_it_shares_is_proven_so(traffic, shared, moves_at_least_what_it_shares):
    assert provably_at_least(traffic, shared) is moves_at_least_what_it_shares
