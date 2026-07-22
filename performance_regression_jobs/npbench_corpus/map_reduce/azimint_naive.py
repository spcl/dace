# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``azimint_naive`` (map_reduce) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float32
dc_complex_float = dc.complex64

SIZES = {'N': 400000, 'npt': 1000}
INPUT_ARGS = ('N', 'npt')
ARRAY_ARGS = ('data', 'radius', 'res')
SCALARS = {}
OUTPUT_ARGS = ('res', )

N, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'npt'))


def initialize(N, npt, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)
    data, radius = (rng.random((N, ), dtype=datatype), rng.random((N, ), dtype=datatype))
    res = np.zeros((npt, ), dtype=datatype)
    return (data, radius, res)


def reference(data, radius, npt, res):
    rmax = radius.max()
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and(r1 <= radius, radius < r2)
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()


@dc.program
def kernel(data: dc_float[N], radius: dc_float[N]):
    rmax = np.amax(radius)
    res = np.zeros((npt, ), dtype=dc_float)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and(r1 <= radius, radius < r2)
        on_values = 0
        tmp = dc_float(0)
        for j in dc.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values
    return res


CORPUS = dict(name='azimint_naive',
              dwarf='map_reduce',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
