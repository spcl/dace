import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

# Sample constants
BET_M = 0.5
BET_P = 0.5

I, J, K = 60, 60, 40


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
@dc.program
def vadv(utens_stage: dc.float64[I, J, K], u_stage: dc.float64[I, J, K], wcon: dc.float64[I + 1, J, K],
         u_pos: dc.float64[I, J, K], utens: dc.float64[I, J, K], dtr_stage: dc.float64, S: dc.float64[1]):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        # cs = gcv * BET_M
        cs[:] = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        # bcol = dtr_stage - acol - ccol[:, :, k]
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] -
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                      u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        # as_ = gav * BET_M
        as_[:] = gav * BET_M
        # acol = gav * BET_P
        acol[:] = gav * BET_P
        # bcol = dtr_stage - acol
        bcol[:] = dtr_stage - acol

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        # datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    @dc.map(_[0:I, 0:J, 0:K])
    def summap(i, j, k):
        s >> S(1, lambda x, y: x + y)[0]
        z << utens_stage[i, j, k]
        s = z


vadv.use_experimental_cfg_blocks = True
sdfg = vadv.to_sdfg()
sdfg.save("log_sdfgs/vadv_forward_loops.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["utens"], outputs=["S"])

sdfg.save("log_sdfgs/vadv_backward_loops.sdfg")
sdfg.compile()
