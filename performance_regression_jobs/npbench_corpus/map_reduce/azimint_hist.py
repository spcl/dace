# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``azimint_hist`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'N': 400000, 'npt': 1000}
INPUT_ARGS = ('N', 'npt')
ARRAY_ARGS = ('data', 'radius', 'out')
SCALARS = {}
OUTPUT_ARGS = ('out', )

N, bins, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'bins', 'npt'))


def initialize(N, npt, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    data, radius = (rng.random((N, ), dtype=datatype), rng.random((N, ), dtype=datatype))
    out = np.zeros((npt, ), dtype=datatype)
    return (data, radius, out)


def reference(data, radius, npt, out):
    histu = np.histogram(radius, npt)[0]
    histw = np.histogram(radius, npt, weights=data)[0]
    out[:] = histw / histu


@dc.program
def get_bin_edges(a: dc_float[N], bin_edges: dc_float[bins + 1]):
    a_min = np.amin(a)
    a_max = np.amax(a)
    delta = (a_max - a_min) / bins
    for i in dc.map[0:bins]:
        bin_edges[i] = a_min + i * delta
    bin_edges[bins] = a_max


@dc.program
def compute_bin(x: dc_float, bin_edges: dc_float[bins + 1]):
    a_min = bin_edges[0]
    a_max = bin_edges[bins]
    return dc.int64(bins * (x - a_min) / (a_max - a_min))


@dc.program
def histogram(a: dc_float[N], bin_edges: dc_float[bins + 1]):
    hist = np.ndarray((bins, ), dtype=np.int64)
    hist[:] = 0
    get_bin_edges(a, bin_edges)
    for i in dc.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += 1
    return hist


@dc.program
def histogram_weights(a: dc_float[N], bin_edges: dc_float[bins + 1], weights: dc_float[N]):
    hist = np.ndarray((bins, ), dtype=weights.dtype)
    hist[:] = 0
    get_bin_edges(a, bin_edges)
    for i in dc.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += weights[i]
    return hist


@dc.program
def kernel(data: dc_float[N], radius: dc_float[N]):
    bin_edges_u = np.ndarray((npt + 1, ), dtype=dc_float)
    histu = histogram(radius, bin_edges_u)
    bin_edges_w = np.ndarray((npt + 1, ), dtype=dc_float)
    histw = histogram_weights(radius, bin_edges_w, data)
    return histw / histu


CORPUS = dict(name='azimint_hist',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
