# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def sftw(A: dace.float64[20]):
    B = dace.define_local([20], dace.float64)
    C = dace.define_local([20], dace.float64)
    D = dace.define_local([20], dace.float64)
    E = dace.define_local([20], dace.float64)
    dup = dace.define_local([20], dace.float64)

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << A[i]
            b >> B[i]
            b = a

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << B[i]
            b >> dup[i]
            b = a

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << dup[i]
            b >> D[i]
            b = a + 2

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << A[i]
            b >> C[i]
            b = a + 1

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << C[i]
            b >> dup[i]
            b = a + 1

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << dup[i]
            b >> E[i]
            b = a + 3

    for i in dace.map[0:20]:
        with dace.tasklet:
            d << D[i]
            e << E[i]
            a >> A[i]
            a = d + e


def test_sftw():
    A = np.random.rand(20)
    expected = 2 * A + 7
    sdfg = sftw.to_sdfg()
    sdfg.coarsen_dataflow()

    # Ensure almost all states were fused
    assert len(sdfg.nodes()) == 2

    sdfg(A=A)

    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_sftw()
