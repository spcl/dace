import dace
import copy
import pytest
import numpy as np

from dace.libraries.sparse import CSRMM
from scipy.sparse import csr_matrix

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")
NNZ = dace.symbol("NNZ")


def make_sdfg(transB: bool, alpha: float, beta: float, implementation: str, dtype) -> dace.SDFG:
    sdfg = dace.SDFG(name="CSRMM")
    sdfg.add_array("A_val", shape=(NNZ, ), dtype=dtype, transient=False)
    sdfg.add_array("A_row", shape=(N + 1, ), dtype=dace.int32, transient=False)
    sdfg.add_array("A_col", shape=(NNZ, ), dtype=dace.int32, transient=False)
    sdfg.add_array("C", shape=(N, K), dtype=dtype, transient=False)
    if transB:
        sdfg.add_array("B", shape=(K, M), dtype=dtype, transient=False)
    else:
        sdfg.add_array("B", shape=(M, K), dtype=dtype, transient=False)

    state = sdfg.add_state("state", is_start_state=True)
    a_row_node = state.add_access("A_row")
    a_col_node = state.add_access("A_col")
    a_val_node = state.add_access("A_val")
    B_node = state.add_access("B")
    C_node = state.add_access("C")

    library_node = CSRMM("csrmm", transB=transB, alpha=alpha, beta=beta)
    library_node.implementation = implementation

    state.add_node(library_node)

    state.add_edge(a_val_node, None, library_node, "_a_vals", dace.Memlet.from_array("A_val", sdfg.arrays["A_val"]))
    state.add_edge(a_row_node, None, library_node, "_a_rows", dace.Memlet.from_array("A_row", sdfg.arrays["A_row"]))
    state.add_edge(a_col_node, None, library_node, "_a_cols", dace.Memlet.from_array("A_col", sdfg.arrays["A_col"]))
    state.add_edge(B_node, None, library_node, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))

    state.add_edge(library_node, "_c", C_node, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))

    if beta != 0:
        cin_node = state.add_access("C")
        state.add_edge(cin_node, None, library_node, "_cin", dace.Memlet.from_array("C", sdfg.arrays["C"]))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("transB, alpha, beta, implementation, dtype", [
    pytest.param(False, 1.0, 0.0, "pure", dace.float32),
    pytest.param(False, 1.0, 0.0, "pure", dace.float64),
    pytest.param(False, 1.0, 1.0, "pure", dace.float32),
    pytest.param(False, 1.0, 1.0, "pure", dace.float64),
    pytest.param(False, 2.0, 2.0, "pure", dace.float32),
    pytest.param(False, 2.0, 2.0, "pure", dace.float64),
    pytest.param(True, 1.0, 0.0, "pure", dace.float32),
    pytest.param(True, 1.0, 0.0, "pure", dace.float64),
    pytest.param(True, 1.0, 1.0, "pure", dace.float32),
    pytest.param(True, 1.0, 1.0, "pure", dace.float64),
    pytest.param(True, 2.0, 2.0, "pure", dace.float32),
    pytest.param(True, 2.0, 2.0, "pure", dace.float64),
    pytest.param(False, 1.0, 0.0, "MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param(False, 1.0, 0.0, "MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param(False, 1.0, 1.0, "MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param(False, 1.0, 1.0, "MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param(False, 2.0, 1.0, "MKL", dace.float32, marks=pytest.mark.mkl),
    pytest.param(False, 2.0, 1.0, "MKL", dace.float64, marks=pytest.mark.mkl),
    pytest.param(False, 1.0, 0.0, "cuSPARSE", dace.float32, marks=pytest.mark.gpu),
    pytest.param(False, 1.0, 0.0, "cuSPARSE", dace.float64, marks=pytest.mark.gpu),
    pytest.param(False, 1.0, 1.0, "cuSPARSE", dace.float32, marks=pytest.mark.gpu),
    pytest.param(False, 1.0, 1.0, "cuSPARSE", dace.float64, marks=pytest.mark.gpu),
    pytest.param(False, 2.0, 1.0, "cuSPARSE", dace.float32, marks=pytest.mark.gpu),
    pytest.param(False, 2.0, 1.0, "cuSPARSE", dace.float64, marks=pytest.mark.gpu),
    pytest.param(True, 1.0, 0.0, "cuSPARSE", dace.float32, marks=pytest.mark.gpu),
    pytest.param(True, 1.0, 0.0, "cuSPARSE", dace.float64, marks=pytest.mark.gpu),
    pytest.param(True, 1.0, 1.0, "cuSPARSE", dace.float32, marks=pytest.mark.gpu),
    pytest.param(True, 1.0, 1.0, "cuSPARSE", dace.float64, marks=pytest.mark.gpu),
    pytest.param(True, 2.0, 1.0, "cuSPARSE", dace.float32, marks=pytest.mark.gpu),
    pytest.param(True, 2.0, 1.0, "cuSPARSE", dace.float64, marks=pytest.mark.gpu),
])
def test_csrmm(transB, alpha, beta, implementation, dtype):
    sdfg = make_sdfg(transB, alpha, beta, implementation, dtype)

    n = 16
    m = 8
    k = 4

    A = np.random.random((n, m)).astype(dtype.as_numpy_dtype())
    if transB:
        B = np.random.random((k, m)).astype(dtype.as_numpy_dtype())
    else:
        B = np.random.random((m, k)).astype(dtype.as_numpy_dtype())
    C = np.random.random((n, k)).astype(dtype.as_numpy_dtype())
    C_ = copy.deepcopy(C)

    A_csr = csr_matrix(A)
    nnz = A_csr.nnz
    A_val = np.copy(A_csr.data)
    A_row = np.copy(A_csr.indptr)
    A_col = np.copy(A_csr.indices)

    sdfg.compile()

    sdfg(A_row=A_row, A_val=A_val, A_col=A_col, B=B, C=C, N=n, M=m, K=k, NNZ=nnz)

    if transB:
        ref = alpha * (A_csr @ B.T) + beta * C_
    else:
        ref = alpha * (A_csr @ B) + beta * C_

    assert np.allclose(ref, C)


if __name__ == "__main__":
    test_csrmm(False, 1.0, 0.0, "pure", dace.float32)
    test_csrmm(False, 1.0, 0.0, "pure", dace.float64)
    test_csrmm(False, 1.0, 1.0, "pure", dace.float32)
    test_csrmm(False, 1.0, 1.0, "pure", dace.float64)
    test_csrmm(False, 2.0, 2.0, "pure", dace.float32)
    test_csrmm(False, 2.0, 2.0, "pure", dace.float64)
    test_csrmm(True, 1.0, 0.0, "pure", dace.float32)
    test_csrmm(True, 1.0, 0.0, "pure", dace.float64)
    test_csrmm(True, 1.0, 1.0, "pure", dace.float32)
    test_csrmm(True, 1.0, 1.0, "pure", dace.float64)
    test_csrmm(True, 2.0, 2.0, "pure", dace.float32)
    test_csrmm(True, 2.0, 2.0, "pure", dace.float64)
    test_csrmm(False, 1.0, 0.0, "MKL", dace.float32)
    test_csrmm(False, 1.0, 0.0, "MKL", dace.float64)
    test_csrmm(False, 1.0, 1.0, "MKL", dace.float32)
    test_csrmm(False, 1.0, 1.0, "MKL", dace.float64)
    test_csrmm(False, 2.0, 2.0, "MKL", dace.float32)
    test_csrmm(False, 2.0, 2.0, "MKL", dace.float64)
    test_csrmm(False, 1.0, 0.0, "cuSPARSE", dace.float32)
    test_csrmm(False, 1.0, 0.0, "cuSPARSE", dace.float64)
    test_csrmm(False, 1.0, 1.0, "cuSPARSE", dace.float32)
    test_csrmm(False, 1.0, 1.0, "cuSPARSE", dace.float64)
    test_csrmm(False, 2.0, 2.0, "cuSPARSE", dace.float32)
    test_csrmm(False, 2.0, 2.0, "cuSPARSE", dace.float64)
    test_csrmm(True, 1.0, 0.0, "cuSPARSE", dace.float32)
    test_csrmm(True, 1.0, 0.0, "cuSPARSE", dace.float64)
    test_csrmm(True, 1.0, 1.0, "cuSPARSE", dace.float32)
    test_csrmm(True, 1.0, 1.0, "cuSPARSE", dace.float64)
    test_csrmm(True, 2.0, 2.0, "cuSPARSE", dace.float32)
    test_csrmm(True, 2.0, 2.0, "cuSPARSE", dace.float64)
