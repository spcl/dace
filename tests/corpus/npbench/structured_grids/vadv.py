# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``vadv`` (structured_grids) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'I': 64, 'J': 64, 'K': 60}
INPUT_ARGS = ('I', 'J', 'K')
ARRAY_ARGS = ('utens_stage', 'u_stage', 'wcon', 'u_pos', 'utens')
SCALARS = {'dtr_stage': 1.0}
OUTPUT_ARGS = ('utens_stage', )

BET_M = 0.5
BET_P = 0.5
I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ('I', 'J', 'K'))


def initialize(I, J, K, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)
    dtr_stage = 3.0 / 20.0
    utens_stage = rng.random((I, J, K), dtype=datatype)
    u_stage = rng.random((I, J, K), dtype=datatype)
    wcon = rng.random((I + 1, J, K), dtype=datatype)
    u_pos = rng.random((I, J, K), dtype=datatype)
    utens = rng.random((I, J, K), dtype=datatype)
    return (utens_stage, u_stage, wcon, u_pos, utens, dtr_stage)


def reference(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = (utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2])
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)
    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided
    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        as_ = gav * BET_M
        cs = gcv * BET_M
        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - acol - ccol[:, :, k]
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                   u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    for k in range(K - 1, K):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    for k in range(K - 2, -1, -1):
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


@dc.program
def kernel(utens_stage: dc_float[I, J, K], u_stage: dc_float[I, J, K], wcon: dc_float[I + 1, J, K],
           u_pos: dc_float[I, J, K], utens: dc_float[I, J, K], dtr_stage: dc_float):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)
    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided
    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        as_ = gav * BET_M
        cs[:] = gcv * BET_M
        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol[:] = dtr_stage - acol - ccol[:, :, k]
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                      u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_[:] = gav * BET_M
        acol[:] = gav * BET_P
        bcol[:] = dtr_stage - acol
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    for k in range(K - 2, -1, -1):
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


CORPUS = dict(name='vadv',
              dwarf='structured_grids',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
