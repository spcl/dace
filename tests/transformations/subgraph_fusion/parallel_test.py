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
                             Z: dace.float64[P], o1: dace.float64[N, M, O], o2: dace.float64[M, N, P],
                             o3: dace.float64[P, N, R], o4: dace.float64[N, M, P]):

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> o1[i, j, k]

            out = in1 + in2 + in3

    for j, k, l in dace.map[0:M, 0:N, 0:P]:
        with dace.tasklet:
            in1 << D[k]
            in2 << E[j]
            in3 << F[l]
            out >> o2[j, k, l]

            out = in1 + in2 + in3

    for asdf1, asdf2, asdf3 in dace.map[0:P, 0:N, 0:R]:
        with dace.tasklet:
            in1 << H[asdf1]
            in2 << I[asdf2]
            in3 << J[asdf3]
            out >> o3[asdf1, asdf2, asdf3]

            out = in1 + in2 + in3

    for i, j, k in dace.map[0:N, 0:M, 0:P]:
        with dace.tasklet:
            in1 << X[i]
            in2 << Y[j]
            in3 << Z[k]
            out >> o4[i, j, k]

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
    o1 = np.random.rand(N.get(), M.get(), O.get())
    o2 = np.random.rand(M.get(), N.get(), P.get())
    o3 = np.random.rand(P.get(), N.get(), R.get())
    o4 = np.random.rand(N.get(), M.get(), P.get())

    csdfg = sdfg.compile()
    csdfg(A=A,
          B=B,
          C=C,
          D=D,
          E=E,
          F=F,
          G=G,
          H=H,
          I=I,
          J=J,
          X=X,
          Y=Y,
          Z=Z,
          N=N,
          M=M,
          O=O,
          P=P,
          R=R,
          Q=Q,
          o1=o1,
          o2=o2,
          o3=o3,
          o4=o4)
    del csdfg

    subgraph = SubgraphView(state, [node for node in state.nodes()])
    expansion = MultiExpansion()
    expansion.setup_match(subgraph)
    fusion = SubgraphFusion()
    fusion.setup_match(subgraph)

    me = MultiExpansion()
    me.setup_match(subgraph)
    assert me.can_be_applied(sdfg, subgraph)
    me.apply(sdfg)

    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert sf.can_be_applied(sdfg, subgraph)
    sf.apply(sdfg)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, X=X, Y=Y, Z=Z,\
          N=N, M=M, O=O, P=P, R=R,Q=Q)
    print("PASS")


if __name__ == "__main__":
    test_p1()
