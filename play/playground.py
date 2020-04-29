import dace
import numpy as np

# TODO:
# Gemm = alpha * A @ B + beta *
# Implement Custom GEMM in different ways:
# - Using Python API
# - Using Custom Memlet
# - Construct SDFG bottom-up

# Test API
# - SDFG methods
# - Transformation / Optimization methods
# - Python parsing (read into that)

N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')

# method 1
@dace.program
def GEMM1(A:dace.float32[N,K], B:dace.float32[K,M], C:dace.float32[N,M], alpha:dace.float32, gamma:dace.float32 , R:dace.float32[N,M]):
    R[:] = alpha * A @ B + gamma * C
    #R[:] = (A@B)*alpha + C*gamma
    #R[:] = 0

# method 2
@dace.program
def GEMM2(A:dace.float32[N,K], B:dace.float32[K,M], C:dace.float32[N,M], alpha:dace.float32, gamma:dace.float32, R:dace.float32[N,M]):

    #tmp = dace.define_local()
    #tmp = dace.ndarray(shape=[N,M,K], dtype=dace.float32)
    tmp = dace.define_local(shape = [N,M,K], dtype = dace.float32)
    # TODO: see whether tmp = other assignment is valid as well
    for i,j,k in dace.map[0:N, 0:M, 0:K]:
        # for the sake of testing out a tasklet
        with dace.tasklet:
            a << A[i,k]
            b << B[k,j]
            alp << alpha
            result >> tmp[i,j,k]
            result = alpha * a * b

    dace.reduce(lambda a,b: a+b, tmp, R, 2)

    for i,j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            c << C[i,j]
            gam << gamma
            test << R[i,j]
            result >> R[i,j]
            result = test + gam * c


def GEMMRef(A, B, C, alpha, gamma):
    # numpy reference version
    return A.dot(B) * alpha + C * gamma


if __name__ == '__main__':
    '''
    n = 3
    m = 5
    k = 4
    A = np.random.rand(n,k).astype(np.float32)
    B = np.random.rand(k,m).astype(np.float32)
    C = np.random.rand(n,m).astype(np.float32)

    R1 = np.zeros((n,m)).astype(np.float32)
    R2 = np.zeros((n,m)).astype(np.float32)

    alpha = np.float32(0.85)
    gamma = np.float32(4.2)

    print("Compute Reference")
    reference = GEMMRef(A,B,C,alpha,gamma)

    # compile DaCe programs
    print("Compiling GEMM1")
    GEMM1c = GEMM1.compile()
    print("Compiling GEMM2")
    GEMM2c = GEMM2.compile()

    print("Computing GEMM1c")
    GEMM1c(A=A, B=B, C=C, alpha = alpha, gamma=gamma, R=R1, N=n, K=k, M=m)

    print("Computing GEMM2c")
    GEMM2c(A=A, B=B, C=C, alpha = alpha, gamma=gamma, R=R2, N=n, K=k, M=m)


    print("Matrices")
    print(reference)
    print(R1)
    print(R2)
    print("Frob Norm Reference")
    print(np.linalg.norm(reference))
    print(np.linalg.norm(R1))
    print(np.linalg.norm(R2))

    '''
    sdfg = GEMM1.to_sdfg(strict = True)

    print("Arrays")
    print(sdfg.arrays)

    print("Symbols")
    print(sdfg.symbols)

    print("Start State")
    print(sdfg.start_state)

    print("SDFG_List")
    print(sdfg._sdfg_list)

    print("Nodes")

    print(sdfg.nodes())

    print("Edges")
    print(sdfg.edges())

    print("Interstate Symbols")
    print(sdfg.interstate_symbols()[0], sdfg.interstate_symbols()[1])

    print("Data Symbols")
    print(sdfg.data_symbols(False))

    print("Scope Symbols")
    print(sdfg.scope_symbols())

    print("Scope_Dict")
    state = sdfg.nodes()[0]
    print("note_to_children = False:")
    print(state.scope_dict(node_to_children = False))
    print("node_to_children = True:")
    print(state.scope_dict(node_to_children = True))

    # this works so far...

    # Next up, try to test out some of the SDFG API functions
