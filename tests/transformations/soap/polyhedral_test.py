import numpy as np
import dace as dc
import sys
from dace.transformation.estimator.soap.einsum_to_sdfg import sdfg_gen
from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis, perform_soap_analysis_einsum
from dace.transformation.estimator.soap.utils import d2sp
import numpy as np
import sympy as sp

#M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))
M = 10
N = 5

def kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N],
           A: dc.float64[M, M], B: dc.float64[M, N]):

    temp2 = np.empty((N, ), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


if __name__ == "__main__":
    C = np.zeros([M,N])
    A = np.random.rand(M,N)
    B = np.random.rand(M,N)
    kernel(1,2, C, A, B)
