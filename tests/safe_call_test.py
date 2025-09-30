# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def indirect_access(A: dace.float64[5], B: dace.float64[5], ub: dace.int64):
    for i in range(ub):
        A[i] = B[i] + 1


def test_oob():
    sdfg = indirect_access.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A, B, 5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"

    # This should raise an exception, but not crash
    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    caught = False
    try:
        sdfg.safe_call(A, B, 6400)
        caught = False
    except Exception as e:
        caught = True
    assert caught, "Exception not raised!"


def test_instrumentation():
    sdfg = indirect_access.to_sdfg()
    sdfg.instrument = dace.InstrumentationType.Timer

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A, B, 5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"

    assert sdfg.get_latest_report() is not None, "Report not generated!"


def test_kwargs():
    sdfg = indirect_access.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A=A, B=B, ub=5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"


def test_symbols():
    N = dace.symbol('N')

    @dace.program
    def indirect_access_sym(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            A[i] = B[i] + 1

    sdfg = indirect_access_sym.to_sdfg()

    A = np.zeros((5, ), dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sdfg.safe_call(A=A, B=B, N=5)
    assert np.allclose(A, B + 1), "Output is not forwarded correctly!"


if __name__ == "__main__":
    test_oob()
    test_instrumentation()
    test_kwargs()
    test_symbols()
