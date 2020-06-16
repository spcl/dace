import dace
from dace.transformation.heterogeneous.expansion import MultiExpansion
from dace.transformation.heterogeneous.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np

from dace.measure import Runner


N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]

@dace.program
def TEST(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O],
         D: dace.float64[M], E: dace.float64[N], F: dace.float64[P],
         G: dace.float64[M], H: dace.float64[P], I: dace.float64[N], J: dace.float64[R],
         X: dace.float64[N], Y: dace.float64[M], Z: dace.float64[P]):

    tmp1 = np.ndarray([N,M,O], dtype = dace.float64)
    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tmp1[i,j,k]

            out = in1 + in2 + in3

    tmp2 = np.ndarray([M,N,P], dtype = dace.float64)
    for j, k, l in dace.map[0:M, 0:N, 0:P]:
        with dace.tasklet:
            in1 << D[k]
            in2 << E[j]
            in3 << F[l]
            out >> tmp2[j,k,l]

            out = in1 + in2 + in3


    tmp3 = np.ndarray([P,N,R], dtype = dace.float64)
    for asdf1, asdf2, asdf3 in dace.map[0:P, 0:N, 0:R]:
        with dace.tasklet:
            in1 << H[asdf1]
            in2 << I[asdf2]
            in3 << J[asdf3]
            out >> tmp3[asdf1,asdf2,asdf3]

            out = in1 + in2 + in3


    tmp4 = np.ndarray([N,M,P], dtype = dace.float64)
    for i,j,k in dace.map[0:N, 0:M, 0:P]:
        with dace.tasklet:
            in1 << X[i]
            in2 << Y[j]
            in3 << Z[k]
            out >> tmp4[i,j,k]

            out = in1 + in2 + in3

if __name__ == "__main__":

    N.set(200)
    M.set(300)
    O.set(50)
    P.set(400)
    Q.set(420)
    R.set(250)

    sdfg = TEST.to_sdfg()
    state = sdfg.nodes()[0]

    runner = Runner()
    runner.go(sdfg, state, None, N, M, O, P, Q, R,
              output = [])
