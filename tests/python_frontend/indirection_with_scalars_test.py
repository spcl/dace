# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dc


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
    assert np.array_equal(B, ref)


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
    assert np.array_equal(B, ref)


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
    M, N, P = (dc.symbol(s) for s in "MNP")
    x0, x1 = dc.symbol("x0"), dc.symbol("x1")

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


def test_scalar_index_in_map():
    """A scalar that a map indexes with is an input of the map body, which assigns it to a symbol: the map body does
    not read it as a symbol of its own."""
    W = dc.symbol("W")

    @dc.program
    def scalar_index_in_map(index: dc.int64, a: dc.float64[4, W], b: dc.float64[4, W]):
        for column in dc.map[0:W]:
            b[index, column] = a[index, column] + 1

    sdfg = scalar_index_in_map.to_sdfg(simplify=False)
    sdfg.validate()
    sdfg.simplify()
    sdfg.validate()
    a = np.random.rand(4, 8)
    b = np.zeros((4, 8))
    sdfg(index=2, a=a, b=b, W=8)
    assert np.allclose(b[2], a[2] + 1)
    assert not b[[0, 1, 3]].any()


def test_scalar_argument_index_in_map_of_callee():
    """A map parameter given to a callee as a scalar argument of another name, which a map of the callee indexes
    with: once the callee is simplified and inlined, its maps still index with the value of the argument."""
    W = dc.symbol("W")

    @dc.program
    def body(index, a, b, c):
        c[index] = 0
        for column in dc.map[0:W]:
            b[index, column] = a[index, column] + 1
        c[index] = c[index] + 1

    @dc.program
    def scalar_argument_index_in_map_of_callee(a: dc.float64[4, W], b: dc.float64[4, W], c: dc.float64[4]):
        for row in dc.map[0:4]:
            body(row, a, b, c)

    sdfg = scalar_argument_index_in_map_of_callee.to_sdfg(simplify=False)
    sdfg.validate()
    sdfg.simplify()
    sdfg.validate()
    a = np.random.rand(4, 8)
    b = np.zeros((4, 8))
    c = np.zeros(4)
    sdfg(a=a, b=b, c=c, W=8)
    assert np.allclose(b, a + 1)
    assert np.array_equal(c, np.ones(4))


if __name__ == "__main__":
    test_toplevel_scalar_indirection()
    test_nested_scalar_indirection()
    test_array_element_scalar_indirection()
    test_array_element_scalar_indirection_in_map()
    test_submatrix()
    test_scalar_index_in_map()
    test_scalar_argument_index_in_map_of_callee()
