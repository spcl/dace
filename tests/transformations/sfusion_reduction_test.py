import dace
from dace.transformation.subgraph.pipeline import expand_maps, expand_reduce, fusion
from dace.transformation.subgraph.helpers import *
from dace.transformation.subgraph import ReduceExpansion
import dace.sdfg.nodes as nodes
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')


@dace.program
def TEST(A: dace.float64[M,N], B:dace.float64[M,N],
          C: dace.float64[N]):

    tmp = np.ndarray(shape = [M,N], dtype = np.float64)
    tmp[:] = 2*A[:] + B[:]
    C[:] = dace.reduce(lambda a, b: a+b, tmp, axis = 0)


@dace.program
def TEST2(A: dace.float64[M,N], B:dace.float64[M,N],
          C: dace.float64[N]):

    tmp = np.ndarray(shape = [M,N], dtype = np.float64)
    C[:] = dace.reduce(lambda a, b: a+b, B, axis = 0)
    for i,j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << C[j]
            in2 << A[i,j]
            out1 >> tmp[i,j]
            out1 = in1 * in2
    C[:] = dace.reduce(lambda a, b: a+b, tmp, axis = 0)


def test1():
    sdfg = TEST.to_sdfg()
    state = sdfg.nodes()[0]
    for node in state.nodes():
        if isinstance(node, dace.libraries.standard.nodes.Reduce):
            reduce_node = node

    assert ReduceExpansion.can_be_applied(state, \
                                          {ReduceExpansion._reduce: state.nodes().index(reduce_node)}, \
                                          0, \
                                          sdfg) == True

    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype = np.float64)
    C2 = np.zeros([N.get()], dtype = np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C1,N=N,M=M)

    expand_reduce(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C2,N=N,M=M)

    assert np.allclose(C1, C2)
    print(np.linalg.norm(C1))
    print(np.linalg.norm(C2))
    print("PASS")

def test2():
    sdfg = TEST2.to_sdfg()
    state = sdfg.nodes()[0]
    A = np.random.rand(M.get(), N.get()).astype(np.float64)
    B = np.random.rand(M.get(), N.get()).astype(np.float64)
    C1 = np.zeros([N.get()], dtype = np.float64)
    C2 = np.zeros([N.get()], dtype = np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C1,N=N,M=M)

    expand_reduce(sdfg, state)
    csdfg = sdfg.compile()
    csdfg(A=A,B=B,C=C2,N=N,M=M)

    assert np.allclose(C1, C2)
    print(np.linalg.norm(C1))
    print(np.linalg.norm(C2))
    print("PASS")

if __name__ == "__main__":
    N.set(20)
    M.set(30)
    test1()
    test2()
