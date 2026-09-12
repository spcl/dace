# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy
import dace

from dace.transformation.layout.pad_dimensions import PadDimensions

N = dace.symbol("N")


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] + B[i, j]


def test_pad_grows_dimension_and_keeps_live_region():
    sdfg = madd.to_sdfg()
    # Pad A's last dim by 3, C's first dim by 2; leave B unpadded.
    PadDimensions(pad_map={"A": [0, 3], "C": [2, 0]}).apply_pass(sdfg, {})
    sdfg.validate()

    a = sdfg.arrays["A"]
    c = sdfg.arrays["C"]
    assert tuple(str(s) for s in a.shape) == ("N", "N + 3")
    assert tuple(str(s) for s in c.shape) == ("N + 2", "N")
    # Packed-C strides for the grown shapes.
    assert [str(s) for s in a.strides] == ["N + 3", "1"]
    assert [str(s) for s in c.strides] == ["N", "1"]

    _N = 8
    A = numpy.random.rand(_N, _N + 3)
    B = numpy.random.rand(_N, _N)
    C = numpy.zeros((_N + 2, _N))
    C_guard = C.copy()

    ref = A[:, :_N] + B  # live region only

    sdfg(A=A.copy(), B=B.copy(), C=C, N=_N)

    assert numpy.allclose(C[:_N, :_N], ref)
    # Pad rows of C (last two) untouched.
    assert numpy.array_equal(C[_N:, :], C_guard[_N:, :])


if __name__ == "__main__":
    test_pad_grows_dimension_and_keeps_live_region()
    print("pad tests PASS")
