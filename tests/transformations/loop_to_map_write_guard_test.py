# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A write at a fixed index under a branch that only one iteration satisfies is a unique write.

CloudSC's ``zvqx[ncldqi - 1]`` is written inside a loop over the species ``jm`` only under ``jm == ncldqi``. The
write-uniqueness proof used to ask whether the index differs between any two iterations, ignoring the branch, and
refused. The branch conditions that mean the same in every iteration (the iterator and symbols the loop never
changes, no array) now bound the two iterations the proof compares.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import array_facts
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

M = dace.symbol("M")
c = dace.symbol("c")


@dace.program
def write_where_the_iterator_is_c(z: dace.float64[M], x: dace.float64[M]):
    for jm in range(M):
        if jm == c:
            z[c] = x[jm] * 2.0


@dace.program
def write_from_c_on(z: dace.float64[M], x: dace.float64[M]):
    for jm in range(M):
        if jm >= c:
            z[c] = x[jm] * 2.0


@dace.program
def write_where_the_iterator_is_c_and_a_flag_holds(z: dace.float64[M], x: dace.float64[M], flag: dace.int32[1]):
    for jm in range(M):
        if jm == c and flag[0] > 0:
            z[c] = x[jm] * 2.0


@dace.program
def write_where_the_iterator_is_c_and_the_symbol_is_positive(z: dace.float64[M], x: dace.float64[M]):
    for jm in range(M):
        if jm == c and M > 3:
            z[c] = x[jm] * 2.0


@dace.program
def store_in_a_descending_loop_of_one_iteration(z: dace.float64[M], x: dace.float64[M]):
    for jm in range(M - 1, M - 2, -1):
        z[0] = x[jm] * 2.0


def parallelized(program: dace.program) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    ParallelizeLoops().apply_pass(sdfg, {})
    return sdfg


def loops(sdfg: dace.SDFG) -> int:
    return sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def maps(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, nodes.MapEntry) for n, parent in sdfg.all_nodes_recursive())


def test_a_write_under_the_iterator_equal_to_a_symbol_is_parallel_and_computes_the_same():
    sdfg = parallelized(write_where_the_iterator_is_c)
    assert (loops(sdfg), maps(sdfg)) == (0, 1)
    x, z = np.arange(1.0, 9.0), np.zeros(8)
    sdfg(z=z, x=x, M=8, c=3)
    expected = np.zeros(8)
    expected[3] = x[3] * 2.0
    assert np.array_equal(z, expected)


def test_a_write_under_a_conjunction_of_the_iterator_test_and_an_invariant_symbol_is_parallel():
    sdfg = parallelized(write_where_the_iterator_is_c_and_the_symbol_is_positive)
    assert (loops(sdfg), maps(sdfg)) == (0, 1)


def test_a_write_under_a_conjunction_with_an_array_test_still_uses_the_iterator_test():
    sdfg = parallelized(write_where_the_iterator_is_c_and_a_flag_holds)
    assert (loops(sdfg), maps(sdfg)) == (0, 1)


def test_a_write_under_a_branch_several_iterations_satisfy_stays_a_loop():
    sdfg = parallelized(write_from_c_on)
    assert (loops(sdfg), maps(sdfg)) == (1, 0)


def test_a_descending_loop_of_one_iteration_is_not_vouched_for_by_a_vacuous_uniqueness_proof():
    """No two iterations exist, so no solver proof about two of them says anything: CloudSC's last-level sweep stays a loop."""
    sdfg = store_in_a_descending_loop_of_one_iteration.to_sdfg(simplify=False)
    (loop,) = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert array_facts.runs_at_most_once(loop, array_facts.resolved_range(loop))
    assert array_facts.conflict_free_under_facts(sdfg, loop, "z") is None


if __name__ == "__main__":
    test_a_descending_loop_of_one_iteration_is_not_vouched_for_by_a_vacuous_uniqueness_proof()
    test_a_write_under_the_iterator_equal_to_a_symbol_is_parallel_and_computes_the_same()
    test_a_write_under_a_conjunction_of_the_iterator_test_and_an_invariant_symbol_is_parallel()
    test_a_write_under_a_conjunction_with_an_array_test_still_uses_the_iterator_test()
    test_a_write_under_a_branch_several_iterations_satisfy_stays_a_loop()
