# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the SHUFFLE layout family through the brute-force sweep.

A Shuffle renumbers a dimension's elements by a bijection but is TRANSPARENT (the physical reorder
plus inverse-composed consumers preserve the result), so every shuffle candidate -- like a
permutation -- must reproduce the oracle. This is the value-permutation analog of k14 (Eytzinger)
and k06 (gather): the sweep chooses the element layout, the algebra guarantees correctness.
"""
import numpy
import dace

from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.brute_force import sweep, best, shuffle_candidates

N = dace.symbol("N")


@dace.program
def saxpy(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = 2.5 * A[i] + B[i]


def test_shuffle_family_sweep_all_verify():
    """Renumbering A by an XOR swizzle or a cyclic shift is transparent, so every shuffle candidate
    reproduces C = 2.5*A + B."""
    register_shuffle("xor3", "i ^ 3", "i ^ 3")
    register_shuffle("cyc", "(i + 1) % N", "(i + N - 1) % N")

    _N = 8
    A = numpy.random.rand(_N)
    B = numpy.random.rand(_N)
    reference = {"C": 2.5 * A + B}

    def make_for(apply):

        def make():
            sdfg = saxpy.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    candidates = {name: make_for(apply) for name, apply in shuffle_candidates("A", 0, ["xor3", "cyc"])}
    assert set(candidates) == {"noshuffle_A", "shuffle_A_xor3", "shuffle_A_cyc"}

    def run(sdfg):
        C = numpy.zeros(_N)
        sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
        return {"C": C}

    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


if __name__ == "__main__":
    test_shuffle_family_sweep_all_verify()
    print("shuffle sweep tests PASS")
