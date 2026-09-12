# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the BLOCK layout family through the brute-force sweep.

Unlike a permutation (transparent -- the array keeps its logical shape), a Block lays an array out
as ``[N/b, b]``, so the caller must pass a correspondingly reshaped input. The sweep's ``run``
closure reads each candidate SDFG's descriptor and reshapes the (packed-C) logical input to match,
so every block factor is exercised end-to-end (SplitDimensions + normalize_schedule_for_layout) and
verified against the flat numpy oracle.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import sweep, best, block_candidates

N = dace.symbol("N")


@dace.program
def scale1d(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 2.0 + 1.0


def _eval_shape(desc, n):
    return tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in desc.shape)


def test_block_family_sweep_all_verify():
    """Block A by each of {8,16,32} (plus unblocked); every candidate reproduces C = A*2+1. Block is
    packed-C, so reshaping the flat logical A into [N/b, b] is the correct laid-out input."""
    n = 32  # divisible by every candidate factor
    A_logical = numpy.random.rand(n)
    reference = {"C": A_logical * 2.0 + 1.0}

    def make_for(apply):

        def make():
            sdfg = scale1d.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        return make

    candidates = {name: make_for(apply) for name, apply in block_candidates("A", 1, factors=(8, 16, 32))}
    assert set(candidates) == {"noblock_A", "block_A_d0_8", "block_A_d0_16", "block_A_d0_32"}

    def run(sdfg):
        a_shape = _eval_shape(sdfg.arrays["A"], n)
        c_shape = _eval_shape(sdfg.arrays["C"], n)
        A_in = A_logical.reshape(a_shape).copy()  # fresh contiguous array (not a view) for DaCe
        C = numpy.zeros(c_shape)
        sdfg(A=A_in, C=C, N=n)
        return {"C": numpy.asarray(C).reshape(n).copy()}

    results = sweep(candidates, run, reference, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


if __name__ == "__main__":
    test_block_family_sweep_all_verify()
    print("block sweep tests PASS")
