# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``mandelbrot1`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

# numpy realizations matching the kernel precision (fp64 -> complex128); the npbench
# numpy reference reads these from the framework's precision module.
np_float = np.float64
np_complex = np.complex128

# The numpy reference reads lowercase ``xn``/``yn``; the dace kernel is parametrized
# over the uppercase symbols ``XN``/``YN``. Provide both (equal) so name-resolution
# works for the reference, ``initialize`` and the SDFG symbol binding alike.
SIZES = {
    'xmin': -1.75,
    'xmax': 0.25,
    'xn': 125,
    'XN': 125,
    'ymin': -1.0,
    'ymax': 1.0,
    'yn': 125,
    'YN': 125,
    'maxiter': 60,
    'horizon': 2.0
}
INPUT_ARGS = ('XN', 'YN')
ARRAY_ARGS = ('Z_out', 'N_out')
SCALARS = {}
OUTPUT_ARGS = ('Z_out', 'N_out')

XN, YN, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'N'])


def initialize(XN, YN, datatype=np.float64):
    cdtype = np.complex128 if np.dtype(datatype) == np.float64 else np.complex128
    Z_out = np.zeros((YN, XN), dtype=cdtype)
    N_out = np.zeros((YN, XN), dtype=np.int64)
    return (Z_out, N_out)


def reference(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon, Z_out, N_out):
    X = np.linspace(xmin, xmax, xn, dtype=np_float)
    Y = np.linspace(ymin, ymax, yn, dtype=np_float)
    C = X + Y[:, None] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np_complex)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter - 1] = 0
    Z_out[:] = Z
    N_out[:] = N


@dc.program
def linspace(start: dc_float, stop: dc_float, X: dc_float[N]):
    dist = (stop - start) / (N - 1)
    for i in dc.map[0:N]:
        X[i] = start + i * dist


@dc.program
def kernel(xmin: dc_float, xmax: dc_float, ymin: dc_float, ymax: dc_float, maxiter: dc.int64, horizon: dc_float,
           Z_out: dc_complex_float[YN, XN], N_out: dc.int64[YN, XN]):
    X = np.ndarray((XN, ), dtype=dc_float)
    Y = np.ndarray((YN, ), dtype=dc_float)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    C = np.ndarray((YN, XN), dtype=dc_complex_float)
    for i, j in dc.map[0:YN, 0:XN]:
        C[i, j] = X[j] + Y[i] * 1j
    Nc = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=dc_complex_float)
    for n in range(maxiter):
        I = np.less(np.absolute(Z), horizon)
        Nc[I] = n
        for j, k in dc.map[0:YN, 0:XN]:
            if I[j, k]:
                Z[j, k] = Z[j, k]**2 + C[j, k]
    Nc[Nc == maxiter - 1] = 0
    Z_out[:] = Z
    N_out[:] = Nc


CORPUS = dict(name='mandelbrot1',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
