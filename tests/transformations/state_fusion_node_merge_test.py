# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def prog1(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            input << A[i]
            out >> A[i]
            out = input + 42
    B[:] = A[:]
    for i in dace.map[0:N]:
        with dace.tasklet:
            input << A[i]
            out >> A[i]
            out = input + 43


@dace.program
def prog2(C: dace.float32[1], E: dace.float32[1], F: dace.float32[1]):
    with dace.tasklet:
        ci << C[0]
        co >> C[0]
        co = ci + 1

    with dace.tasklet:
        c << C[0]
        e >> E[0]
        e = c

    with dace.tasklet:
        c << C[0]
        f >> F[0]
        f = c


def relative_error(val, ref):
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


def test_one():
    N = 42
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    A_ref = A + 42 + 43
    B_ref = A + 42

    prog1(A, B)
    assert (relative_error(A, A_ref) < 1e-12)
    assert (relative_error(B, B_ref) < 1e-12)


def test_two():
    C = np.random.rand(1).astype(np.float32)
    E = np.random.rand(1).astype(np.float32)
    F = np.random.rand(1).astype(np.float32)
    C_ref = np.random.rand(1).astype(np.float32)
    C_ref[:] = C[:] + 1

    prog2(C, E, F)
    assert (C[0] == C_ref[0])
    assert (E[0] == C_ref[0])
    assert (F[0] == C_ref[0])


if __name__ == '__main__':
    test_one()
    test_two()
