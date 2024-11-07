import torch


def computeSPMV(
    nx: int, nz: int, ny: int, A: torch.sparse.Tensor, x: torch.Tensor, y: torch.Tensor
) -> int:

    n_rows = nx * ny * nz
    indices = A.indices()
    values = A.values()

    for i in range(n_rows):

        row_ind = indices[0] == i
        nnz_cols = indices[1][row_ind]
        nnz_values = values[row_ind]

        y[i] = torch.dot(nnz_values, x[nnz_cols])

    return 0


# test the SpMV kernel
def test_spmv():
    nx = 2
    ny = 2
    nz = 2
    n_rows = nx * ny * nz
    n_cols = nx * ny * nz

    n = n_rows
    A = torch.rand(n, n)
    A[0, 0] = 0
    A[1, 1] = 0
    A[2, 1] = 0
    A[2, 3] = 0
    A[3, 1] = 0
    A[3, 2] = 0
    A_sparse = A.to_sparse()
    x = torch.rand(n)

    y = torch.zeros(n_rows)
    computeSPMV(nx, nz, ny, A_sparse, x, y)

    y_ref = A_sparse @ x

    if torch.allclose(y, y_ref):
        print("Test passed")
    else:
        print("Test failed")


test_spmv()
