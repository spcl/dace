# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import numpy
import dace

from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.unblock_dimensions import UnblockDimensions

N = dace.symbol("N")


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])


@dace.program
def madd_blocked(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N:16, 0:N:4] @ dace.ScheduleType.Sequential:
        for ii, jj in dace.map[i:i + 16, j:j + 4] @ dace.ScheduleType.Sequential:
            C[ii, jj] = 0.5 * (A[ii, jj] + B[ii, jj])


def test_block_then_unblock_roundtrip():
    """Block an array then Unblock it with the same map: shapes and results
    return to the flat baseline (Unblock inverts Block)."""
    sdfg = madd.to_sdfg()
    split_map = {
        "A": ([True, True], [16, 4]),
        "B": ([True, True], [16, 4]),
    }
    SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    assert len(sdfg.arrays["A"].shape) == 4  # blocked: [N/16, N/4, 16, 4]

    UnblockDimensions(unblock_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    assert tuple(str(s) for s in sdfg.arrays["A"].shape) == ("N", "N")
    assert tuple(str(s) for s in sdfg.arrays["B"].shape) == ("N", "N")

    _N = 16 * 4 * 2
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C = numpy.zeros((_N, _N))
    ref = 0.5 * (A + B)
    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)
    assert numpy.allclose(C, ref)


def test_unblock_native_blocked_kernel():
    """Unblock a natively 5-loop blocked kernel down to flat arrays; run against a
    physically-blocked input and compare to the flat baseline."""
    baseline = madd.to_sdfg()

    sdfg = madd_blocked.to_sdfg()
    # Present A, B to the unblock pass in blocked (4D) form first.
    split_map = {
        "A": ([True, True], [16, 4]),
        "B": ([True, True], [16, 4]),
    }
    SplitDimensions(split_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    UnblockDimensions(unblock_map=split_map).apply_pass(sdfg, {})
    sdfg.validate()
    assert tuple(str(s) for s in sdfg.arrays["A"].shape) == ("N", "N")

    _N = 16 * 4 * 2
    A = numpy.random.rand(_N, _N)
    B = numpy.random.rand(_N, _N)
    C0 = numpy.zeros((_N, _N))
    C1 = numpy.zeros((_N, _N))
    baseline(A=A.copy(), B=B.copy(), C=C0, N=_N)
    sdfg(A=A.copy(), B=B.copy(), C=C1, N=_N)
    assert numpy.allclose(C1, C0)


if __name__ == "__main__":
    test_block_then_unblock_roundtrip()
    test_unblock_native_blocked_kernel()
    print("unblock tests PASS")
