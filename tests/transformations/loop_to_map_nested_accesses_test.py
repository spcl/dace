# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` checks each access a nested SDFG makes behind a map scope, not their bounding box.

CloudSC's species loop writes ``zsolqa[jm, nv, jl]`` and ``zsolqa[nv, jm, jl]`` under a branch in a map over the columns
``jl``. The branch makes the map body a nested SDFG, whose one connector for the array propagates to the box
``zsolqa[Min(nv, jm):Max(nv, jm) + 1, ..., 0:L]``: it meets the box of every other ``jm``, though no element is shared
between two iterations (both indices are ``nv`` only at ``jm == nv``).
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

M = dace.symbol("M")
L = dace.symbol("L")
c = dace.symbol("c")


@dace.program
def mirrored_in_a_map(a: dace.float64[M, M, L], x: dace.float64[M, L]):
    for jm in range(M):
        for jl in dace.map[0:L]:
            if x[jm, jl] < 0.0:
                a[jm, c, jl] = a[jm, c, jl] + x[jm, jl]
                a[c, jm, jl] = a[c, jm, jl] - x[jm, jl]


@dace.program
def shared_row_in_a_map(a: dace.float64[M, M, L], x: dace.float64[M, L]):
    for jm in range(M):
        for jl in dace.map[0:L]:
            if x[jm, jl] < 0.0:
                a[jm, c, jl] = a[jm, c, jl] + x[jm, jl]
                a[c, c, jl] = a[c, c, jl] - x[jm, jl]


def loops(sdfg: dace.SDFG) -> int:
    return sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def test_two_accesses_behind_one_nested_connector_that_never_meet_across_iterations_run_as_a_map():
    sdfg = mirrored_in_a_map.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    assert loops(sdfg) == 0
    assert sum(isinstance(n, nodes.MapEntry) for n, parent in sdfg.all_nodes_recursive()) == 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 4))
    a = rng.standard_normal((5, 5, 4))
    expected = a.copy()
    for jm in range(5):
        for jl in range(4):
            if x[jm, jl] < 0.0:
                expected[jm, 2, jl] += x[jm, jl]
                expected[2, jm, jl] -= x[jm, jl]
    sdfg(a=a, x=x, M=5, L=4, c=2)
    assert np.array_equal(a, expected)


def test_two_accesses_behind_one_nested_connector_that_meet_across_iterations_stay_a_loop():
    sdfg = shared_row_in_a_map.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    assert loops(sdfg) == 1


if __name__ == "__main__":
    test_two_accesses_behind_one_nested_connector_that_never_meet_across_iterations_run_as_a_map()
    test_two_accesses_behind_one_nested_connector_that_meet_across_iterations_stay_a_loop()
