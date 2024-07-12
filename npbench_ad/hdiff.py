import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

I, J, K = 32, 32, 32


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
@dc.program
def hdiff(in_field: dc.float64[I + 4, J + 4, K],
          out_field: dc.float64[I, J, K], coeff: dc.float64[I, J, K], S: dc.float64[1]):
    
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])


    res1 = lap_field[1:, 1:J + 1, :] - lap_field[:I + 1, 1:J + 1, :]
    flx_field = np.where(
        (res1 *
         (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res1,
    )
    
    res2 = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :J + 1, :]
    fly_field = np.where(
        (res2 *
         (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res2,
    )
    

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])

    @dc.map(_[0:I, 0:J, 0:K])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << out_field[i]
        s = z
        
sdfg = hdiff.to_sdfg()

sdfg.save("log_sdfgs/hdiff_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["in_field"], outputs=["S"])

sdfg.save("log_sdfgs/hdiff_backward.sdfg")