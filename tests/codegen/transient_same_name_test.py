# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace.transformation.interstate import StateFusion


@dace.program
def nested_scope(A: dace.float64[3, 3], B: dace.float64[3, 3]):
    mytransient = dace.define_local([3, 3], dace.float64)
    mytransient[:] = A + 1
    B[:] = mytransient


@dace.program
def outer_scope(A: dace.float64[3, 3], B: dace.float64[3, 3]):
    mytransient = dace.define_local([3, 3], dace.float64)
    mytransient[:] = A
    nested_scope(mytransient, B)


def test_regression_transient_not_allocated():
    inp = np.zeros((3, 3)).astype(np.float64)

    sdfg: dace.SDFG = outer_scope.to_sdfg(simplify=False)
    result = np.zeros_like(inp)
    sdfg(A=inp.copy(), B=result)

    assert np.allclose(result, inp + 1)


if __name__ == '__main__':
    test_regression_transient_not_allocated()
