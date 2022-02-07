# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.subgraph import MultiExpansion, SubgraphFusion
import dace.sdfg.nodes as nodes
import numpy as np

from typing import Union, List
from dace.sdfg.graph import SubgraphView

N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]


@dace.program
def subgraph_fusion_parallel(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O], D: dace.float64[M],
                             E: dace.float64[N], F: dace.float64[P], G: dace.float64[M], H: dace.float64[P],
                             I: dace.float64[N], J: dace.float64[R], X: dace.float64[N], Y: dace.float64[M],
                             Z: dace.float64[P]):

    tmp1 = np.ndarray([N, M, O], dtype=dace.float64)
    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tmp1[i, j, k]

            out = in1 + in2 + in3

    tmp2 = np.ndarray([M, N, P], dtype=dace.float64)
    for j, k, l in dace.map[0:M, 0:N, 0:P]:
        with dace.tasklet:
            in1 << D[k]
            in2 << E[j]
            in3 << F[l]
            out >> tmp2[j, k, l]

            out = in1 + in2 + in3

    tmp3 = np.ndarray([P, N, R], dtype=dace.float64)
    for asdf1, asdf2, asdf3 in dace.map[0:P, 0:N, 0:R]:
        with dace.tasklet:
            in1 << H[asdf1]
            in2 << I[asdf2]
            in3 << J[asdf3]
            out >> tmp3[asdf1, asdf2, asdf3]

            out = in1 + in2 + in3

    tmp4 = np.ndarray([N, M, P], dtype=dace.float64)
    for i, j, k in dace.map[0:N, 0:M, 0:P]:
        with dace.tasklet:
            in1 << X[i]
            in2 << Y[j]
            in3 << Z[k]
            out >> tmp4[i, j, k]

            out = in1 + in2 + in3


def test_p1():

    N.set(20)
    M.set(30)
    O.set(50)
    P.set(40)
    Q.set(42)
    R.set(25)

    sdfg = subgraph_fusion_parallel.to_sdfg()
    sdfg.simplify()
    state = sdfg.nodes()[0]

    A = np.random.rand(N.get()).astype(np.float64)
    B = np.random.rand(M.get()).astype(np.float64)
    C = np.random.rand(O.get()).astype(np.float64)
    D = np.random.rand(M.get()).astype(np.float64)
    E = np.random.rand(N.get()).astype(np.float64)
    F = np.random.rand(P.get()).astype(np.float64)
    G = np.random.rand(M.get()).astype(np.float64)
    H = np.random.rand(P.get()).astype(np.float64)
    I = np.random.rand(N.get()).astype(np.float64)
    J = np.random.rand(R.get()).astype(np.float64)
    X = np.random.rand(N.get()).astype(np.float64)
    Y = np.random.rand(M.get()).astype(np.float64)
    Z = np.random.rand(P.get()).astype(np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, X=X, Y=Y, Z=Z,\
          N=N, M=M, O=O, P=P, R=R,Q=Q)
    del csdfg

    subgraph = SubgraphView(state, [node for node in state.nodes()])
    expansion = MultiExpansion(subgraph)
    fusion = SubgraphFusion(subgraph)

    me = MultiExpansion(subgraph)
    assert me.can_be_applied(sdfg, subgraph)
    me.apply(sdfg)

    sf = SubgraphFusion(subgraph)
    assert sf.can_be_applied(sdfg, subgraph)
    sf.apply(sdfg)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, X=X, Y=Y, Z=Z,\
          N=N, M=M, O=O, P=P, R=R,Q=Q)
    print("PASS")


if __name__ == "__main__":
    test_p1()
