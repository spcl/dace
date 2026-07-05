# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``stockham_fft`` (spectral_methods) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128


def rng_complex(shape, rng, datatype):
    return (rng.random(shape, dtype=datatype) + rng.random(shape, dtype=datatype) * 1j)


# N = R**K is derived (not an independent dataset symbol); keeping it out of SIZES
# avoids the size-cap clobbering it out of sync with the R**K-sized arrays.
SIZES = {'R': 2, 'K': 15}
INPUT_ARGS = ('R', 'K')
ARRAY_ARGS = ('x', 'y')
SCALARS = {}
OUTPUT_ARGS = ('y', )

R, K, M1, M2 = (dc.symbol(s, dtype=dc.int64, integer=True, positive=True) for s in ('R', 'K', 'M1', 'M2'))
N = R**K


def initialize(R, K, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)
    N = R**K
    X = rng_complex((N, ), rng, datatype)
    Y = np.zeros_like(X, dtype=X.dtype)
    return (X, Y)


def reference(R, K, x, y):
    N = R**K
    i_coord, j_coord = np.mgrid[0:R, 0:R]
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2j * np.pi * i_coord * j_coord / R)
    y[:] = x[:]
    ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]
    for i in range(K):
        yv = np.reshape(y, (R**i, R, R**(K - i - 1)))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        D = np.empty((R, R**i, R**(K - i - 1)), dtype=np.complex128)
        tmp = np.exp(-2j * np.pi * ii_coord[:, :R**i] * jj_coord[:, :R**i] / R**(i + 1))
        D[:] = np.repeat(np.reshape(tmp, (R, R**i, 1)), R**(K - i - 1), axis=2)
        tmp_twid = np.reshape(tmp_perm, (N, )) * np.reshape(D, (N, ))
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R**(K - 1))), (N, ))


@dc.program
def mgrid1(X: dc.uint32[R, R], Y: dc.uint32[R, R]):
    for i in range(R):
        X[i, :] = i
    for j in range(R):
        Y[:, j] = j


@dc.program
def mgrid2(X: dc.uint32[R, N], Y: dc.uint32[R, N]):
    for i in range(R):
        X[i, :] = i
    for j in range(R**K):
        Y[:, j] = j


@dc.program
def kernel(x: dc_complex_float[R**K], y: dc_complex_float[R**K]):
    i_coord = np.ndarray((R, R), dtype=np.uint32)
    j_coord = np.ndarray((R, R), dtype=np.uint32)
    mgrid1(i_coord, j_coord)
    dft_mat = np.empty((R, R), dtype=dc_complex_float)
    dft_mat[:] = np.exp(-2j * np.pi * i_coord * j_coord / R)
    y[:] = x[:]
    ii_coord = np.ndarray((R, N), dtype=np.uint32)
    jj_coord = np.ndarray((R, N), dtype=np.uint32)
    mgrid2(ii_coord, jj_coord)
    tmp_perm = np.empty_like(y)
    D = np.empty_like(y)
    tmp = np.empty_like(y)
    for i in range(K):
        yv = np.reshape(y, (R**i, R, R**(K - i - 1)))
        tmp_perm[:] = np.reshape(np.transpose(yv, axes=(1, 0, 2)), (N, ))
        Dv = np.reshape(D, (R, R**i, R**(K - i - 1)))
        tmpv = np.reshape(tmp, (R**(K - i - 1), R, R**i))
        tmpv[0] = np.exp(-2j * np.pi * ii_coord[:, :R**i] * jj_coord[:, :R**i] / R**(i + 1))
        for k in range(R**(K - i - 1)):
            Dv[:, :, k] = np.reshape(tmpv[0], (R, R**i, 1))
        tmp_twid = tmp_perm * D
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R**(K - 1))), (N, ))


CORPUS = dict(name='stockham_fft',
              dwarf='spectral_methods',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
