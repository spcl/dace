# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

symsym = dace.symbol('symsym', dace.float64)
value1 = dace.symbol('value1', dace.float64)
value2 = dace.symbol('value2', dace.float64)


@dace.program
def inner(A: dace.float64[10, 10, 10]):
    A[...] = symsym


@dace.program
def mid(A):
    tmp = value1 + value2
    inner(A=A, symsym=tmp)
    A[...] += 1


@dace.program
def outer(A, inp1: float, inp2: float):
    tmp = inp1 + inp2
    mid(A, value1=tmp, value2=1.0)


def test_symbol_mapping_replace():

    with dace.config.set_temporary('optimizer',
                                   'automatic_dataflow_coarsening',
                                   value=True):
        A = np.ones((10, 10, 10))
        ref = A.copy()
        b = 2.0
        c = 2.0
        outer(A, inp1=b, inp2=c)
        outer.f(ref, inp1=b, inp2=c)
        assert (np.allclose(A, ref))


if __name__ == '__main__':
    test_symbol_mapping_replace()
