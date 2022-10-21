import numpy as np
import dace as dc
import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis, perform_soap_analysis_einsum
from dace.transformation.estimator.soap.utils import d2sp
import numpy as np
import sympy as sp

dim_m, dim_n = (dc.symbol(s, dtype=dc.int64) for s in ('dim_m', 'dim_n'))
@dc.program
def jacobi1d(TSTEPS: dc.int32, A: dc.float32[dim_m], B: dc.float32[dim_m]):
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@dc.program
def fdtd_2d(TSTEPS: dc.int32, ex: dc.float32[dim_m,dim_n], ey: dc.float32[dim_m,dim_n],
                hz: dc.float32[dim_m,dim_n], _fict_: dc.float32[dim_m]):
    for t in range(TSTEPS):
        # ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])



if __name__ == "__main__":
    sdfg = fdtd_2d.to_sdfg()
    decomp_params = [("p", 255), ("Ss", 102400)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))
    decomp_params.append(('TSTEPS', 20))
    decomp_params.append(('dim_m', 20000))
    decomp_params.append(('dim_n', 1000))
    soap_result = perform_soap_analysis(sdfg, decomp_params,
                    generate_schedule = True)
    soap_result.subgraphs[0].get_data_decomposition(0)
    print(soap_result.subgraphs[0].p_grid)
    a = 1
