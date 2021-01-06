# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import StripMining

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N], dace.float64[N])
def multiply(X, Y, Z):
    @dace.map(_[0:N])
    def mult(i):
        x << X[i]
        y << Y[i]
        z >> Z[i]

        z = y * x


def test_tiling_number_of_tiles():
    size = 256

    np.random.seed(0)
    X = np.random.rand(size)
    Y = np.random.rand(size)
    Z = np.zeros_like(Y)

    sdfg = multiply.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations(StripMining,
                               options=[{
                                   'tile_size': '16',
                                   'tiling_type': 'number_of_tiles'
                               }])
    sdfg(X=X, Y=Y, Z=Z, N=size)

    assert np.allclose(Z, X * Y)
    print('PASS')


if __name__ == "__main__":
    test_tiling_number_of_tiles()
