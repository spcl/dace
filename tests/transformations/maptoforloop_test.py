# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A test for the MapToForLoop transformation. """

import dace
import numpy as np
from dace.transformation.dataflow import MapExpansion, MapToForLoop


@dace.program
def map2for(A: dace.float64[20, 20, 20]):
    for k in range(1, 19):
        for i, j in dace.map[0:20, 0:20]:
            with dace.tasklet:
                inp << A[i, j, k]
                inp2 << A[i, j, k - 1]
                out >> A[i, j, k + 1]
                out = inp + inp2


def test_map2for_overlap():
    A = np.random.rand(20, 20, 20)
    expected = np.copy(A)
    for k in range(1, 19):
        expected[:, :, k + 1] = expected[:, :, k] + expected[:, :, k - 1]

    sdfg = map2for.to_sdfg()
    assert sdfg.apply_transformations([MapExpansion, MapToForLoop]) == 2
    sdfg(A=A)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_map2for_overlap()