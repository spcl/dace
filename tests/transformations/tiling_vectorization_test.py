import dace
import numpy as np
from dace.transformation.dataflow import StripMining, Vectorization

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N], dace.float64[N])
def multiply(X, Y, Z):
    @dace.map(_[0:N])
    def mult(i):
        x << X[i]
        y << Y[i]
        z >> Z[i]

        z = y * x


def aligned_ndarray(arr, alignment=64):
    """
    Allocates a and returns a copy of ``arr`` as an ``alignment``-byte aligned
    array. Useful for aligned vectorized access.
    
    Based on https://stackoverflow.com/a/20293172/6489142
    """
    if (arr.ctypes.data % alignment) == 0:
        return arr

    extra = alignment // arr.itemsize
    buf = np.empty(arr.size + extra, dtype=arr.dtype)
    ofs = (-buf.ctypes.data % alignment) // arr.itemsize
    result = buf[ofs:ofs + arr.size].reshape(arr.shape)
    np.copyto(result, arr)
    assert (result.ctypes.data % alignment) == 0
    return result


def test_tiling_vectorization():
    size = 256
    vector_len = 2  # Use 4 for AVX

    np.random.seed(0)
    X = aligned_ndarray(np.random.rand(size))
    Y = aligned_ndarray(np.random.rand(size))
    Z = aligned_ndarray(np.zeros_like(Y))

    sdfg = multiply.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations([StripMining, Vectorization],
                               options=[{
                                   'tile_size': str(vector_len)
                               }, {
                                   'vector_len': vector_len
                               }])
    sdfg(X=X, Y=Y, Z=Z, N=size)

    assert np.allclose(Z, X * Y)
    print('PASS')


if __name__ == "__main__":
    test_tiling_vectorization()
