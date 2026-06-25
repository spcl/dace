# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``mandelbrot2`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'xmin': -2.0, 'xmax': 0.5, 'XN': 200, 'ymin': -1.25, 'ymax': 1.25, 'YN': 200, 'maxiter': 40, 'horizon': 2.0}
INPUT_ARGS = ('XN', 'YN')
ARRAY_ARGS = ('Z_out', 'N_out')
SCALARS = {}
OUTPUT_ARGS = ('Z_out', 'N_out')

XN, YN, M, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'M', 'N'])


def initialize(XN, YN, datatype=np.float64):
    cdtype = np.complex64 if np.dtype(datatype) == np.float32 else np.complex128
    Z_out = np.zeros((YN, XN), dtype=cdtype)
    N_out = np.zeros((YN, XN), dtype=np.int64)
    return (Z_out, N_out)


def reference(xmin, xmax, ymin, ymax, XN, YN, maxiter, horizon, Z_out, N_out):
    X = np.linspace(xmin, xmax, XN)
    Y = np.linspace(ymin, ymax, YN)
    C = X + Y[:, None] * 1j
    Z = np.zeros(C.shape, dtype=np.complex128)
    for i in range(maxiter):
        Z[abs(Z) < horizon] = Z[abs(Z) < horizon] * Z[abs(Z) < horizon] + C[abs(Z) < horizon]
        N_out[(abs(Z) > horizon) & (N_out == 0)] = i + 1
        Z_out[(abs(Z) > horizon) & (N_out == i + 1)] = Z[(abs(Z) > horizon) & (N_out == i + 1)]


@dc.program
def mgrid(X: dc.int64[M, N], Y: dc.int64[M, N]):
    for i in range(M):
        X[i, :] = i
    for j in range(N):
        Y[:, j] = j


@dc.program
def linspace(start: dc_float, stop: dc_float, X: dc_float[N]):
    dist = (stop - start) / (N - 1)
    for i in dace.map[0:N]:
        X[i] = start + i * dist


@dc.program
def kernel(xmin: dc_float, xmax: dc_float, ymin: dc_float, ymax: dc_float, maxiter: dc.int64, horizon: dc_float):
    Xi = np.ndarray((XN, YN), dtype=np.int64)
    Yi = np.ndarray((XN, YN), dtype=np.int64)
    mgrid(Xi, Yi)
    X = np.ndarray((XN, ), dtype=dc_float)
    Y = np.ndarray((YN, ), dtype=dc_float)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    C = np.ndarray((XN, YN), dtype=dc_complex_float)
    for i, j in dc.map[0:XN, 0:YN]:
        C[i, j] = X[i] + Y[j] * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=dc_complex_float)
    Xiv = np.reshape(Xi, (XN * YN, ))
    Yiv = np.reshape(Yi, (XN * YN, ))
    Cv = np.reshape(C, (XN * YN, ))
    Z = np.zeros(Cv.shape, dc_complex_float)
    I = np.ndarray((XN * YN, ), dtype=np.bool_)
    length = XN * YN
    k = 0
    while length > 0 and k < maxiter:
        Z[:length] = np.multiply(Z[:length], Z[:length])
        Z[:length] = np.add(Z[:length], Cv[:length])
        I[:length] = np.absolute(Z[:length]) > horizon
        for j in range(length):
            if I[j]:
                N_[Xiv[j], Yiv[j]] = k + 1
        for j in range(length):
            if I[j]:
                Z_[Xiv[j], Yiv[j]] = Z[j]
        I[:length] = np.logical_not(I[:length])
        count = 0
        for j in range(length):
            if I[j]:
                Z[count] = Z[j]
                Xiv[count] = Xiv[j]
                Yiv[count] = Yiv[j]
                Cv[count] = Cv[j]
                count += 1
        length = count
        k += 1
    return (Z_.T, N_.T)


CORPUS = dict(name='mandelbrot2',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
