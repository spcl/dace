# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import StripMining

N = dace.symbol('N')


@dace.program(dace.float64, dace.float64[N], dace.float64[N])
def axpy(A, X, Y):
    @dace.map(_[0:N])
    def multiplication(i):
        in_A << A
        in_X << X[i]
        in_Y << Y[i]
        out >> Y[i]

        out = in_A * in_X + in_Y


def test_tiling_number_of_tiles():
    size = 256

    np.random.seed(0)
    A = np.random.rand()
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = np.copy(Y)
    sdfg = axpy.to_sdfg()
    sdfg.coarsen_dataflow()
    sdfg.apply_transformations(StripMining, options=[{'tile_size': '16', 'tiling_type': dace.TilingType.NumberOfTiles}])
    sdfg(A=A, X=X, Y=Y, N=size)
    assert np.allclose(Y, A * X + Z)
    print('PASS')


if __name__ == "__main__":
    test_tiling_number_of_tiles()
