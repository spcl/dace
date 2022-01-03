# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import StripMining, Vectorization
from dace.libraries.standard.memory import aligned_ndarray

N = dace.symbol('N')


@dace.program
def multiply(X: dace.float64[N], Y: dace.float64[N], Z: dace.float64[N]):
    @dace.map(_[0:N])
    def mult(i):
        x << X[i]
        y << Y[i]
        z >> Z[i]

        z = y * x


def test_tiling_vectorization():
    size = 256
    vector_len = 2  # Use 4 for AVX
    with dace.config.set_temporary('compiler',
                                   'allow_view_arguments',
                                   value=True):
        np.random.seed(0)
        X = aligned_ndarray(np.random.rand(size))
        Y = aligned_ndarray(np.random.rand(size))
        Z = aligned_ndarray(np.zeros_like(Y))

        sdfg = multiply.to_sdfg()
        sdfg.coarsen_dataflow()
        sdfg.apply_transformations([StripMining, Vectorization],
                                   options=[{
                                       'tile_size': str(vector_len)
                                   }, {
                                       'vector_len': vector_len
                                   }])
        sdfg(X=X, Y=Y, Z=Z, N=size)

        assert np.allclose(Z, X * Y)


if __name__ == "__main__":
    test_tiling_vectorization()
