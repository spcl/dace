import dace
import copy
import numpy as np
from scipy.sparse import csr_matrix

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")


@dace.program()
def gemm(A: dace.float64[N, M], B: dace.float64[M, K]):
    return A @ B


def test_gemm_expansion_pure_csr():
    sdfg_sparse = gemm.to_sdfg()
    sdfg_sparse.name = "sparse"

    sdfg_dense = copy.deepcopy(sdfg_sparse)
    sdfg_dense.name = "dense"
    sdfg_dense.expand_library_nodes()
    sdfg_dense.simplify()
    
    # SPMM
    for state in sdfg_sparse.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.LibraryNode):
                node.expand(sdfg_sparse, state)

    for state in sdfg_sparse.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.LibraryNode):
                node.data_format_type = dace.dtypes.DataFormatType.CSR
                node.expand(sdfg_sparse, state)

    sdfg_sparse.simplify()

    # Fuzzyflow-like verification
    for _ in range(3):
        n = 32
        m = 8
        k = 16

        A = np.random.random((n, m)).astype(np.float64)
        B = np.random.random((m, k)).astype(np.float64)

        C = sdfg_dense(A=A, B=B, N=n, M=m, K=k)

        A_sparse = csr_matrix(A)
        nnz = A_sparse.nnz
        A_val = np.copy(A_sparse.data)
        A_row = np.copy(A_sparse.indptr)
        A_col = np.copy(A_sparse.indices)

        C_sparse = sdfg_sparse(A_row=A_row, A_col=A_col, A_val=A_val, A_nnz=nnz, B=B, N=n, M=m, K=k)
        assert np.allclose(C, C_sparse)


if __name__ == "__main__":
    test_gemm_expansion_pure_csr()
