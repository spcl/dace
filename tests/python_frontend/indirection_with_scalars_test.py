# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np


@dc.program
def toplevel_scalar_indirection(A: dc.float32[2, 3, 4, 5], B: dc.float32[4]):
    i = 0
    j = 0
    k = 0
    B[:] = A[:, i, :, j][k, :]


def test_toplevel_scalar_indirection():
    A = np.random.rand(2, 3, 4, 5).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    toplevel_scalar_indirection(A, B)
    ref = A[0, 0, :, 0]
    assert (np.array_equal(B, ref))


@dc.program
def nested_scalar_indirection(A: dc.float32[2, 3, 4, 5], B: dc.float32[2, 4]):
    for l in dc.map[0:2]:
        i = 0
        j = 0
        k = l
        B[k] = A[:, i, :, j][k, :]


def test_nested_scalar_indirection():
    A = np.random.rand(2, 3, 4, 5).astype(np.float32)
    B = np.random.rand(2, 4).astype(np.float32)
    nested_scalar_indirection(A, B)
    ref = A[:, 0, :, 0]
    assert (np.array_equal(B, ref))


def test_array_element_scalar_indirection():

    @dc.program
    def nested_arrayindex_indirection(A: dc.float64[20, 10], indices: dc.int32[2]):
        start = indices[0]
        finish = indices[1]
        A[start:finish] = 0

    A = np.random.rand(20, 10)
    indices = np.array([2, 5], dtype=np.int32)
    expected = np.copy(A)
    nested_arrayindex_indirection.f(expected, indices)
    nested_arrayindex_indirection(A, indices)

    assert np.allclose(expected, A)


def test_array_element_scalar_indirection_in_map():

    @dc.program
    def nested_arrayindex_indirection_map(A: dc.float64[20, 10], indices: dc.int32[4]):
        for i in dc.map[0:2]:
            start = indices[2 * i]
            finish = indices[2 * i + 1]
            A[start:finish] = 0

    A = np.random.rand(20, 10)
    indices = np.array([2, 5, 13, 14], dtype=np.int32)
    expected = np.copy(A)
    nested_arrayindex_indirection_map.f(expected, indices)
    nested_arrayindex_indirection_map(A, indices)

    assert np.allclose(expected, A)


def test_submatrix():
    dtype = dc.float64
    data_index = dc.int32
    M, N, P = (dc.symbol(s) for s in 'MNP')
    x0, x1 = dc.symbol('x0'), dc.symbol('x1')

    @dc.program
    def create_submatrix():
        return np.zeros([x1 - x0, N], dtype=dtype)

    @dc.program
    def zero_submatrix(mat: dtype[M, N], starts: data_index[P], ends: data_index[P]):
        for m in dc.map[0:P]:
            start = starts[m]
            finish = ends[m]
            temp = create_submatrix(x0=start, x1=finish, N=N)
            mat[start:finish, 0:N] = temp

    A = np.random.rand(20, 21)
    starts = np.array([2, 13], dtype=np.int32)
    ends = np.array([5, 14], dtype=np.int32)

    # Regression
    expected = np.copy(A)
    expected[2:5] = 0
    expected[13:14] = 0

    zero_submatrix(A, starts, ends)
    assert np.allclose(A, expected)


def test_promoted_scalar_reaches_a_loop_range():
    """A scalar promoted to ``__sym_<name>`` by a slice, then read back through the slice's shape.

    Promotion adds the symbol straight to ``sdfg.symbols`` without touching the visitor scope, so a
    range over the promoted slice's own extent came back as an undefined variable.
    """
    N = dc.symbol('N', dtype=dc.int64)

    @dc.program
    def promoted(rv: dc.float64[N], off: dc.int64[1], out: dc.float64[N]):
        first_i = int(off[0])
        last_i = first_i + 4
        rv_i = rv[first_i:last_i]
        for k in range(rv_i.shape[0]):
            out[k] = rv_i[k] * 2.0

    rv = np.arange(8.0, dtype=np.float64)
    out = np.zeros(8, dtype=np.float64)
    promoted(rv=rv, off=np.array([2], dtype=np.int64), out=out, N=8)
    want = np.zeros(8)
    want[:4] = rv[2:6] * 2.0
    assert np.allclose(out, want)


if __name__ == "__main__":
    test_toplevel_scalar_indirection()
    test_nested_scalar_indirection()
    test_array_element_scalar_indirection()
    test_array_element_scalar_indirection_in_map()
    test_submatrix()
    test_promoted_scalar_reaches_a_loop_range()
