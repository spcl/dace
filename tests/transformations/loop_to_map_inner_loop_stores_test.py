# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A store inside an inner loop, at an index the inner iterator does not name, is one box for the outer loop.

CloudSC's species loop (``jm``) stores ``zvqx[ncldqi - 1]`` inside its column loop under ``jm == ncldqi``, and reads
``zvqx[jm - 1]`` elsewhere in the body. Only the iteration ``jm == ncldqi`` stores, and no other iteration reads that
element, so the species iterations are independent. The proof used to give up on any store under an inner loop and to
accept only reads at the very subset a store writes.
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
def stage_one_row(v: dace.float64[M], x: dace.float64[M, L], out: dace.float64[M, L]):
    for jm in range(M):
        for jl in range(L):
            if jm == c:
                v[c] = x[jm, jl] * 2.0
            out[jm, jl] = v[jm]


@dace.program
def stage_the_row_every_iteration_reads(v: dace.float64[M], x: dace.float64[M, L], out: dace.float64[M, L]):
    for jm in range(M):
        for jl in range(L):
            if jm == c:
                v[c] = x[jm, jl] * 2.0
            out[jm, jl] = v[c]


def parallelized(program: dace.program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    return sdfg


def outer_loops(sdfg: dace.SDFG) -> int:
    return sum(
        isinstance(r, LoopRegion) and r.loop_variable == "jm" for r in sdfg.all_control_flow_regions(recursive=True)
    )


def maps(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, nodes.MapEntry) for n, parent in sdfg.all_nodes_recursive())


def test_a_store_in_an_inner_loop_that_no_other_iteration_reads_is_a_map_and_computes_the_same():
    sdfg = parallelized(stage_one_row)
    assert outer_loops(sdfg) == 0
    rng = np.random.default_rng(0)
    x, v = rng.standard_normal((5, 4)), rng.standard_normal(5)
    out = np.zeros((5, 4))
    expected_v, expected = v.copy(), np.zeros((5, 4))
    for jm in range(5):
        for jl in range(4):
            if jm == 2:
                expected_v[2] = x[jm, jl] * 2.0
            expected[jm, jl] = expected_v[jm]
    sdfg(v=v, x=x, out=out, M=5, L=4, c=2)
    assert np.array_equal(out, expected) and np.array_equal(v, expected_v)


def test_a_store_in_an_inner_loop_that_every_other_iteration_reads_stays_a_loop():
    assert outer_loops(parallelized(stage_the_row_every_iteration_reads)) == 1


if __name__ == "__main__":
    test_a_store_in_an_inner_loop_that_no_other_iteration_reads_is_a_map_and_computes_the_same()
    test_a_store_in_an_inner_loop_that_every_other_iteration_reads_stays_a_loop()
