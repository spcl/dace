# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``hdiff`` (structured_grids) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'I': 64, 'J': 64, 'K': 60}
INPUT_ARGS = ('I', 'J', 'K')
ARRAY_ARGS = ('in_field', 'out_field', 'coeff')
SCALARS = {}
OUTPUT_ARGS = ('out_field', )

I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ('I', 'J', 'K'))


def initialize(I, J, K, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)
    in_field = rng.random((I + 4, J + 4, K), dtype=datatype)
    out_field = rng.random((I, J, K), dtype=datatype)
    coeff = rng.random((I, J, K), dtype=datatype)
    return (in_field, out_field, coeff)


def reference(in_field, out_field, coeff):
    I, J, K = (out_field.shape[0], out_field.shape[1], out_field.shape[2])
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
                                                       in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])
    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = np.where(res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :]) > 0, 0, res)
    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = np.where(res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :]) > 0, 0, res)
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] +
                                                                           fly_field[:, 1:, :] - fly_field[:, :-1, :])


@dc.program
def kernel(in_field: dc_float[I + 4, J + 4, K], out_field: dc_float[I, J, K], coeff: dc_float[I, J, K]):
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
                                                       in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])
    res1 = lap_field[1:, 1:J + 1, :] - lap_field[:I + 1, 1:J + 1, :]
    flx_field = np.where(res1 * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :]) > 0, 0, res1)
    res2 = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :J + 1, :]
    fly_field = np.where(res2 * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :]) > 0, 0, res2)
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] +
                                                                           fly_field[:, 1:, :] - fly_field[:, :-1, :])


CORPUS = dict(name='hdiff',
              dwarf='structured_grids',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
