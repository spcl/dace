import dace
import copy
import numpy as np
from dace.transformation.layout.collapse_dimension import CollapseDimensions

M = dace.symbol("M")
N = dace.symbol("N")

@dace.program
def matrix_add(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M:1, 0:N:1]:
        C[i, j] = A[i, j] + B[i, j]

def test_matrix_add():
    sdfg1 = matrix_add.to_sdfg()
    sdfg2 = copy.deepcopy(sdfg1)
    CollapseDimensions(collapse_map={"A": (0, 1), "B": (0, 1), "C": (0, 1)}).apply_pass(sdfg2, {})

    # Create test data
    Mv, Nv = 64, 32
    A = np.random.rand(Mv, Nv)
    B = np.random.rand(Mv, Nv)
    C1 = np.zeros((Mv, Nv))
    C2 = np.zeros((Mv, Nv))

    # Run original SDFG
    sdfg1(A=A, B=B, C=C1, M=Mv, N=Nv)

    # Run transformed SDFG
    sdfg2(A=A, B=B, C=C2, M=Mv, N=Nv)

    # Compare
    if not np.allclose(C1, C2):
        print("ERROR: Results differ!")
        max_diff = np.max(np.abs(C1 - C2))
        print("Max difference:", max_diff)
        print("C1:", C1)
        print("C2:", C2)
    assert np.allclose(C1, C2)


def test_matrix_add_partial():
    sdfg1 = matrix_add.to_sdfg()
    sdfg2 = copy.deepcopy(sdfg1)
    CollapseDimensions(collapse_map={"A": (0, 1), }).apply_pass(sdfg2, {})

    # Create test data
    Mv, Nv = 64, 32
    A = np.random.rand(Mv, Nv)
    B = np.random.rand(Mv, Nv)
    C1 = np.zeros((Mv, Nv))
    C2 = np.zeros((Mv, Nv))

    # Run original SDFG
    sdfg1(A=A, B=B, C=C1, M=Mv, N=Nv)

    # Run transformed SDFG
    sdfg2(A=A, B=B, C=C2, M=Mv, N=Nv)

    # Compare
    if not np.allclose(C1, C2):
        print("ERROR: Results differ!")
        max_diff = np.max(np.abs(C1 - C2))
        print("Max difference:", max_diff)
        print("C1:", C1)
        print("C2:", C2)
    assert np.allclose(C1, C2)

if __name__ == "__main__":
    test_matrix_add()
    test_matrix_add_partial()