# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for numpy advanced indexing syntax. See also:
https://numpy.org/devdocs/reference/arrays.indexing.html
"""
import ast
import dace
from dace.frontend.python.common import DaceSyntaxError
from dace.sdfg import nodes as dace_nodes
import numpy as np
import pytest

N = dace.symbol('N')
M = dace.symbol('M')


def test_flat():

    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A.flat

    A = np.random.rand(20, 30)
    res = indexing_test(A)
    assert np.allclose(A.flat, res)


def test_flat_noncontiguous():
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):

        @dace.program
        def indexing_test(A):
            return A.flat

        A = np.random.rand(20, 30).transpose()
        res = indexing_test(A)
        assert np.allclose(A.flat, res)


def test_ellipsis():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        return A[1:5, ..., 0]

    A = np.random.rand(5, 5, 5, 5, 5)
    res = indexing_test(A)
    assert np.allclose(A[1:5, ..., 0], res)


def test_ellipsis_and_newaxis():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        return A[None, 1:5, ..., 0]

    A = np.random.rand(5, 5, 5, 5, 5)
    res = indexing_test(A)
    assert np.allclose(A[None, 1:5, ..., 0], res)


def test_ellipsis_and_newaxis_2():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        return A[None, 1:5, ..., None, 2]

    A = np.random.rand(5, 5, 5, 5, 5)
    res = indexing_test(A)
    assert np.allclose(A[None, 1:5, ..., None, 2], res)


def test_aug_implicit():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        A[:, 1:5][:, 0:2] += 5

    A = np.random.rand(5, 5, 5, 5, 5)
    regression = np.copy(A)
    regression[:, 1:5][:, 0:2] += 5
    indexing_test(A)
    assert np.allclose(A, regression)


def test_aug_implicit_attribute():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        A.flat[10:15][0:2] += 5

    A = np.random.rand(5, 5, 5, 5, 5)
    regression = np.copy(A)
    # FIXME: NumPy does not support augmented assignment on a sub-iterator of a flat iterator
    regression.flat[10:12] += 5
    indexing_test(A)
    assert np.allclose(A, regression)


def test_ellipsis_aug():

    @dace.program
    def indexing_test(A: dace.float64[5, 5, 5, 5, 5]):
        A[1:5, ..., 0] += 5

    A = np.random.rand(5, 5, 5, 5, 5)
    regression = np.copy(A)
    regression[1:5, ..., 0] += 5
    indexing_test(A)
    assert np.allclose(A, regression)


def test_newaxis():

    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A[:, np.newaxis, None, :]

    A = np.random.rand(20, 30)
    res = indexing_test(A)
    assert res.shape == (20, 1, 1, 30)
    assert np.allclose(A[:, np.newaxis, None, :], res)


def test_multiple_newaxis():

    @dace.program
    def indexing_test(A: dace.float64[10, 20, 30]):
        return A[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis]

    A = np.random.rand(10, 20, 30)
    res = indexing_test(A)
    assert res.shape == (1, 10, 1, 1, 20, 1, 30, 1)
    assert np.allclose(A[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis, :, np.newaxis], res)


def test_index_intarr_1d():

    @dace.program
    def indexing_test(A: dace.float64[N], indices: dace.int32[M]):
        return A[indices]

    A = np.random.rand(20)
    indices = [1, 10, 15]
    res = indexing_test(A, indices, M=3)
    assert np.allclose(A[indices], res)


def test_index_intarr_1d_literal():

    @dace.program
    def indexing_test(A: dace.float64[20]):
        return A[[1, 10, 15]]

    A = np.random.rand(20)
    indices = [1, 10, 15]
    res = indexing_test(A)
    assert np.allclose(A[indices], res)


def test_index_intarr_1d_constant():
    indices = [1, 10, 15]

    @dace.program
    def indexing_test(A: dace.float64[20]):
        return A[indices]

    A = np.random.rand(20)
    res = indexing_test(A)
    assert np.allclose(A[indices], res)


def test_index_intarr_1d_multi():

    @dace.program
    def indexing_test(A: dace.float64[20, 10, 30], indices: dace.int32[3]):
        return A[indices, 2:7:2, [15, 10, 1]]

    A = np.random.rand(20, 10, 30)
    indices = np.array([1, 10, 15], dtype=np.int32)
    res = indexing_test(A, indices)
    assert np.allclose(A[indices, 2:7:2, [15, 10, 1]], res)


def test_index_intarr_nd():

    @dace.program
    def indexing_test(A: dace.float64[4, 3], rows: dace.int64[2, 2], columns: dace.int64[2, 2]):
        return A[rows, columns]

    A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float64)
    rows = np.array([[0, 0], [3, 3]], dtype=np.intp)
    columns = np.array([[0, 2], [0, 2]], dtype=np.intp)
    expected = A[rows, columns]
    res = indexing_test(A, rows, columns)
    assert np.allclose(expected, res)


def indirection_tasklet(sdfg: dace.SDFG):
    """Returns the advanced-indexing tasklet and the memlet feeding its ``__arr`` connector."""
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace_nodes.Tasklet) and node.label.startswith('indirection'):
                arr_edge = next(e for e in state.in_edges(node) if e.dst_conn == '__arr')
                return state, node, arr_edge.data
    raise AssertionError('no indirection tasklet found in SDFG')


def assert_arr_subscript_matches_memlet(sdfg: dace.SDFG) -> int:
    """The ``__arr`` subscript must carry exactly as many indices as its memlet has non-scalar
    dimensions -- the connector rule that conflating "extent 1" with "scalar index" broke.
    """
    _, tasklet, memlet = indirection_tasklet(sdfg)
    code = tasklet.code.as_string.strip()
    value = ast.parse(code).body[0].value
    if isinstance(value, ast.Subscript):
        indices = len(value.slice.elts) if isinstance(value.slice, ast.Tuple) else 1
    else:
        indices = 0
    assert indices == memlet.subset.data_dims(), f'"{code}" does not match memlet "{memlet}"'
    return indices


def test_index_intarr_into_length_one_dim():
    """An advanced index into an extent-1 dimension used to raise KeyError('__i0').

    The index mapping is built only for NON-degenerate dimensions, but the loop that hands a
    dimension over to its index array deleted the entry unconditionally. A ravelled 1x1 lookup
    table -- one species in a pair potential, one channel in a layer -- is exactly that shape,
    and it crashed the parser before any memlet existed to be wrong.
    """

    @dace.program
    def indexing_test(table: dace.float64[1], idx: dace.int64[3, 2]):
        return table[idx]

    table = np.array([7.5], dtype=np.float64)
    idx = np.zeros((3, 2), dtype=np.int64)
    expected = table[idx]

    sdfg = indexing_test.to_sdfg(simplify=False)
    assert expected.shape == (3, 2)

    # The gather degenerates to a broadcast: no index is read, and the map spans the index shape.
    state, tasklet, memlet = indirection_tasklet(sdfg)
    assert assert_arr_subscript_matches_memlet(sdfg) == 0
    assert set(tasklet.in_connectors) == {'__arr'}
    assert str(memlet.subset) == '0'
    map_entry = next(n for n in state.nodes() if isinstance(n, dace_nodes.MapEntry))
    assert map_entry.map.params == ['__ind0', '__ind1']
    assert str(map_entry.map.range) == '0:3, 0:2'
    assert tuple(sdfg.arrays['__return'].shape) == expected.shape

    res = sdfg(table=table, idx=idx)
    assert res.shape == expected.shape, f"gather returned {res.shape}, numpy returns {expected.shape}"
    assert np.allclose(expected, res)


def test_index_intarr_into_length_one_dim_of_many():
    """The same, with the degenerate dimension NEXT TO a normal one, so the surviving basic-index
    dimension still has to reach the output."""

    @dace.program
    def indexing_test(table: dace.float64[1, 4], idx: dace.int64[3]):
        return table[idx, :]

    table = np.arange(4, dtype=np.float64).reshape(1, 4)
    idx = np.zeros((3, ), dtype=np.int64)
    expected = table[idx, :]

    sdfg = indexing_test.to_sdfg(simplify=False)
    state, tasklet, memlet = indirection_tasklet(sdfg)
    assert assert_arr_subscript_matches_memlet(sdfg) == 0
    # The basic dimension stays a map parameter, the degenerate one is pinned by the memlet.
    assert str(memlet.subset) == '0, __i1'
    map_entry = next(n for n in state.nodes() if isinstance(n, dace_nodes.MapEntry))
    assert sorted(map_entry.map.params) == ['__i1', '__ind0']
    assert tuple(sdfg.arrays['__return'].shape) == expected.shape

    res = indexing_test(table, idx)
    assert res.shape == expected.shape, f"gather returned {res.shape}, numpy returns {expected.shape}"
    assert np.allclose(expected, res)


def test_index_intarr_length_one_dim_trailing_basic():
    """An extent-1 dimension gathered while a trailing dimension is implicitly sliced whole."""

    @dace.program
    def indexing_test(table: dace.float64[1, 10], idx: dace.int64[3, 2]):
        return table[idx]

    table = np.arange(10, dtype=np.float64).reshape(1, 10)
    idx = np.zeros((3, 2), dtype=np.int64)
    expected = table[idx]

    sdfg = indexing_test.to_sdfg(simplify=False)
    _, tasklet, memlet = indirection_tasklet(sdfg)
    assert assert_arr_subscript_matches_memlet(sdfg) == 0
    assert str(memlet.subset) == '0, __i1'
    assert tuple(sdfg.arrays['__return'].shape) == expected.shape == (3, 2, 10)

    res = sdfg(table=table, idx=idx)
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_index_intarr_into_trailing_length_one_dim():
    """The degenerate dimension is the LAST one, so the basic dimension in front of it must keep
    its position in the output."""

    @dace.program
    def indexing_test(table: dace.float64[4, 1], idx: dace.int64[3, 2]):
        return table[:, idx]

    table = np.arange(4, dtype=np.float64).reshape(4, 1)
    idx = np.zeros((3, 2), dtype=np.int64)
    expected = table[:, idx]

    sdfg = indexing_test.to_sdfg(simplify=False)
    _, tasklet, memlet = indirection_tasklet(sdfg)
    assert assert_arr_subscript_matches_memlet(sdfg) == 0
    assert str(memlet.subset) == '__i0, 0'
    assert tuple(sdfg.arrays['__return'].shape) == expected.shape == (4, 3, 2)

    res = sdfg(table=table, idx=idx)
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


@pytest.mark.parametrize('degenerate_dim', [0, 1])
def test_index_intarr_length_one_dim_beside_real_gather(degenerate_dim):
    """A degenerate advanced dimension NEXT TO a real one. The surviving index array has to keep
    its position in the tasklet subscript even though the dropped one came before or after it.
    """

    @dace.program
    def gather_first(table: dace.float64[1, 5], i0: dace.int64[3], i1: dace.int64[3]):
        return table[i0, i1]

    @dace.program
    def gather_second(table: dace.float64[5, 1], i0: dace.int64[3], i1: dace.int64[3]):
        return table[i0, i1]

    if degenerate_dim == 0:
        prog, shape, live = gather_first, (1, 5), 'i1'
        i0 = np.zeros(3, dtype=np.int64)
        i1 = np.array([0, 3, 4], dtype=np.int64)
    else:
        prog, shape, live = gather_second, (5, 1), 'i0'
        i0 = np.array([0, 4, 2], dtype=np.int64)
        i1 = np.zeros(3, dtype=np.int64)

    table = np.arange(5, dtype=np.float64).reshape(shape)
    expected = table[i0, i1]

    sdfg = prog.to_sdfg(simplify=False)
    state, tasklet, _ = indirection_tasklet(sdfg)
    # Exactly one index survives, and it is the one belonging to the non-degenerate dimension.
    assert assert_arr_subscript_matches_memlet(sdfg) == 1
    assert set(tasklet.in_connectors) == {'__arr', '__inp0'}
    inp_edge = next(e for e in state.in_edges(tasklet) if e.dst_conn == '__inp0')
    assert inp_edge.data.data == live
    assert tuple(sdfg.arrays['__return'].shape) == expected.shape == (3, )

    res = sdfg(table=table, i0=i0, i1=i1)
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_index_intarr_into_reshaped_length_one_dim():
    """The examinimd gather: a per-pair coefficient table ravelled to one element, then gathered.
    The sliced array is a view, so the degenerate dimension arrives through a reshape.
    """

    @dace.program
    def indexing_test(cutsq: dace.float64[1, 1], pair: dace.int64[4]):
        return cutsq.reshape((1, ))[pair]

    cutsq = np.array([[2.5]], dtype=np.float64)
    pair = np.zeros((4, ), dtype=np.int64)
    expected = cutsq.reshape((1, ))[pair]

    sdfg = indexing_test.to_sdfg(simplify=False)
    assert assert_arr_subscript_matches_memlet(sdfg) == 0
    assert tuple(sdfg.arrays['__return'].shape) == expected.shape == (4, )

    res = sdfg(cutsq=cutsq, pair=pair)
    assert res.shape == expected.shape
    assert np.allclose(expected, res)


def test_index_boolarr_rhs():

    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        return A[A > 15]

    A = np.ndarray((20, 30), dtype=np.float64)
    for i in range(20):
        A[i, :] = np.arange(0, 30)
    regression = A[A > 15]

    # Right-hand side boolean array indexing is unsupported
    with pytest.raises(IndexError):
        res = indexing_test(A)
        assert np.allclose(regression, res)


def test_index_multiboolarr():

    @dace.program
    def indexing_test(A: dace.float64[20, 20], B: dace.bool[20]):
        A[B, B] = 2

    A = np.ndarray((20, 20), dtype=np.float64)
    for i in range(20):
        A[i, :] = np.arange(0, 20)
    B = A[:, 1] > 0

    # Advanced indexing with multiple boolean arrays should be disallowed
    with pytest.raises(DaceSyntaxError):
        indexing_test(A, B)


def test_index_boolarr_fixed():

    @dace.program
    def indexing_test(A: dace.float64[20, 30], barr: dace.bool[20, 30]):
        A[barr] += 5

    A = np.ndarray((20, 30), dtype=np.float64)
    for i in range(20):
        A[i, :] = np.arange(0, 30)
    barr = A > 15
    regression = np.copy(A)
    regression[barr] += 5

    indexing_test(A, barr)

    assert np.allclose(regression, A)


def test_index_boolarr_inline():

    @dace.program
    def indexing_test(A: dace.float64[20, 30]):
        A[A > 15] = 2

    A = np.ndarray((20, 30), dtype=np.float64)
    for i in range(20):
        A[i, :] = np.arange(0, 30)
    regression = np.copy(A)
    regression[A > 15] = 2

    indexing_test(A)

    assert np.allclose(regression, A)


def test_out_index_intarr():

    @dace.program
    def indexing_test(A: dace.float64[N], indices: dace.int32[M]):
        A[indices] = 2

    A = np.random.rand(20)
    indices = [1, 10, 15]
    ref = np.copy(A)
    ref[indices] = 2
    indexing_test(A, indices, M=3)

    assert np.allclose(A, ref)


def test_out_index_intarr_bcast():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N], B: dace.float64[N, N], indices: dace.int32[M]):
        A[indices] = B

    A = np.random.rand(20, 20, 20)
    B = np.random.rand(20, 20)
    indices = [1, 10, 15]
    ref = np.copy(A)
    ref[indices] = B
    indexing_test(A, B, indices, M=3)

    assert np.allclose(A, ref)


def test_out_index_intarr_aug():

    @dace.program
    def indexing_test(A: dace.float64[N], indices: dace.int32[M]):
        A[indices] += 1

    A = np.random.rand(20)
    indices = [1, 10, 15]
    ref = np.copy(A)
    ref[indices] += 1
    indexing_test(A, indices, M=3)

    assert np.allclose(A, ref)


def test_out_index_intarr_aug_bcast():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N], B: dace.float64[N, N], indices: dace.int32[M]):
        A[indices] += B

    A = np.random.rand(20, 20, 20)
    B = np.random.rand(20, 20)
    indices = [1, 10, 15]
    ref = np.copy(A)
    ref[indices] += B
    indexing_test(A, B, indices, M=3)

    assert np.allclose(A, ref)


def test_out_index_intarr_multidim():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N], indices: dace.int32[M]):
        A[1:2, indices, 3:4] = 2

    A = np.random.rand(20, 20, 20)
    indices = [1, 10, 15]
    ref = np.copy(A)
    ref[1:2, indices, 3:4] = 2
    indexing_test(A, indices, M=3)

    assert np.allclose(A, ref)


def test_out_index_intarr_multidim_range():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N], indices: dace.int32[M]):
        A[1:2, indices, 3:10] = 2

    A = np.random.rand(20, 20, 20)
    indices = [1, 10, 15]
    ref = np.copy(A)
    ref[1:2, indices, 3:10] = 2
    indexing_test(A, indices, M=3)

    assert np.allclose(A, ref)


@pytest.mark.parametrize('tuple_index', (False, True))
def test_advanced_indexing_syntax(tuple_index):

    @dace.program
    def indexing_test(A: dace.float64[N, N, N]):
        if tuple_index:
            A[
                (1, 2, 3),
            ] = 2
        else:
            A[[1, 2, 3]] = 2
        A[(1, 2, 3)] = 1

    A = np.random.rand(20, 20, 20)
    ref = np.copy(A)
    ref[
        (1, 2, 3),
    ] = 2
    ref[(1, 2, 3)] = 1
    indexing_test(A)

    assert np.allclose(A, ref)


@pytest.mark.parametrize('contiguous', (False, True))
def test_multidim_tuple_index(contiguous):

    if contiguous:

        @dace.program
        def indexing_test(A: dace.float64[N, M]):
            return A[:, (1, 2, 3)]
    else:

        @dace.program
        def indexing_test(A: dace.float64[N, M]):
            return A[:, (1, 3, 0)]

    sdfg = indexing_test.to_sdfg()
    assert tuple(sdfg.arrays['__return'].shape) == (N, 3)

    A = np.random.rand(20, 10)
    if contiguous:
        ref = A[:, (1, 2, 3)]
    else:
        ref = A[:, (1, 3, 0)]

    res = indexing_test(A)

    assert np.allclose(res, ref)


def test_multidim_tuple_index_longer():

    @dace.program
    def indexing_test(A: dace.float64[N, M]):
        return A[:, (1, 2, 3, 4, 5, 7)]

    sdfg = indexing_test.to_sdfg()
    assert tuple(sdfg.arrays['__return'].shape) == (N, 6)

    A = np.random.rand(20, 10)
    ref = A[:, (1, 2, 3, 4, 5, 7)]

    res = indexing_test(A)

    assert np.allclose(res, ref)


def test_multidim_tuple_multidim_index():
    with pytest.raises(IndexError, match='could not be broadcast together'):

        @dace.program
        def indexing_test(A: dace.float64[N, M, N]):
            return A[:, (1, 2, 3, 4, 5, 7), (0, 1)]

        indexing_test.to_sdfg()


@pytest.mark.skip("Combined basic and advanced indexing with writes is not supported")
def test_multidim_tuple_multidim_index_write():
    with pytest.raises(IndexError, match='could not be broadcast together'):

        @dace.program
        def indexing_test(A: dace.float64[N, M, N]):
            A[:, (1, 2, 3, 4, 5, 7), (0, 1)] = 2

        indexing_test.to_sdfg()


def test_advanced_index_broadcasting():

    @dace.program
    def indexing_test(A: dace.float64[N, M, N], indices: dace.int32[3, 3]):
        return A[indices, (1, 2, 4), :]

    sdfg = indexing_test.to_sdfg()
    assert tuple(sdfg.arrays['__return'].shape) == (3, 3, N)

    A = np.random.rand(20, 10, 20)
    indices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    ref = A[indices, (1, 2, 4), :]

    res = indexing_test(A, indices)

    assert np.allclose(res, ref)


def test_combining_basic_and_advanced_indexing():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N, N, N, N, N], indices: dace.int32[3, 3], indices2: dace.int32[3, 3, 3]):
        return A[:5, indices, indices2, ..., 1:3, 4]

    n = 6
    A = np.random.rand(n, n, n, n, n, n, n)
    indices = np.random.randint(0, n, size=(3, 3)).astype(np.int32)
    indices2 = np.random.randint(0, n, size=(3, 3, 3)).astype(np.int32)
    ref = A[:5, indices, indices2, ..., 1:3, 4]

    # Advanced indexing dimensions should be prepended to the shape
    sdfg = indexing_test.to_sdfg()
    assert tuple(sdfg.arrays['__return'].shape) == (3, 3, 3, 5, N, N, 2)

    res = indexing_test(A, indices, indices2)

    assert np.allclose(res, ref)


@pytest.mark.skip("Combined basic and advanced indexing with writes is not supported")
def test_combining_basic_and_advanced_indexing_write():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N, N, N, N, N], indices: dace.int32[3, 3], indices2: dace.int32[3, 3, 3]):
        A[:5, indices, indices2, ..., 1:3, 4] = 2

    n = 6
    A = np.random.rand(n, n, n, n, n, n, n)
    indices = np.random.randint(0, n, size=(3, 3)).astype(np.int32)
    indices2 = np.random.randint(0, n, size=(3, 3, 3)).astype(np.int32)
    ref = np.copy(A)
    A[:5, indices, indices2, ..., 1:3, 4] = 2

    # Advanced indexing dimensions should be prepended to the shape
    res = indexing_test(A, indices, indices2)

    assert np.allclose(res, ref)


def test_combining_basic_and_advanced_indexing_with_newaxes():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N, N, N, N, N], indices: dace.int32[3, 3], indices2: dace.int32[3, 3, 3]):
        return A[None, :5, indices, indices2, ..., 1:6:3, 4, np.newaxis]

    n = 6
    A = np.random.rand(n, n, n, n, n, n, n)
    indices = np.random.randint(0, n, size=(3, 3)).astype(np.int32)
    indices2 = np.random.randint(0, n, size=(3, 3, 3)).astype(np.int32)
    ref = A[None, :5, indices, indices2, ..., 1:6:3, 4, np.newaxis]

    # Advanced indexing dimensions should be prepended to the shape
    sdfg = indexing_test.to_sdfg()
    assert tuple(sdfg.arrays['__return'].shape) == (3, 3, 3, 1, 5, N, N, 2, 1)

    res = indexing_test(A, indices, indices2)

    assert np.allclose(res, ref)


def test_combining_basic_and_advanced_indexing_with_newaxes_2():

    @dace.program
    def indexing_test(A: dace.float64[N, N, N, N, N, N, N], indices: dace.int32[3, 3], indices2: dace.int32[3, 3, 3]):
        return A[None, :5, indices, indices2, ..., 1:6:3, np.newaxis]

    n = 6
    A = np.random.rand(n, n, n, n, n, n, n)
    indices = np.random.randint(0, n, size=(3, 3)).astype(np.int32)
    indices2 = np.random.randint(0, n, size=(3, 3, 3)).astype(np.int32)
    ref = A[None, :5, indices, indices2, ..., 1:6:3, np.newaxis]

    # Advanced indexing dimensions should be prepended to the shape
    sdfg = indexing_test.to_sdfg()
    assert tuple(sdfg.arrays['__return'].shape) == (1, 5, 3, 3, 3, N, N, N, 2, 1)

    res = indexing_test(A, indices, indices2)

    assert np.allclose(res, ref)


def test_index_array_with_integer_index_write_is_refused():
    """An integer index drops an axis the indirection still counts, so the write is refused."""

    @dace.program
    def scatter_row(A: dace.float64[4, 8], idx: dace.int64[3], out: dace.float64[4, 8]):
        out[0, idx] = A[0, 0:3]

    with pytest.raises(DaceSyntaxError):
        scatter_row.to_sdfg()


def test_index_array_followed_by_integer_index_write_is_allowed():
    """A dropped axis AFTER the index array leaves its numbering alone, so this one stands."""

    @dace.program
    def scatter_column(A: dace.float64[4, 8], idx: dace.int64[3], out: dace.float64[4, 8]):
        out[idx, 0] = A[0:3, 0]

    A = np.random.rand(4, 8)
    idx = np.array([3, 0, 2], dtype=np.int64)
    out = np.zeros((4, 8))
    reference = np.zeros((4, 8))
    reference[idx, 0] = A[0:3, 0]

    scatter_column(A=A, idx=idx, out=out)
    assert np.allclose(out, reference)


if __name__ == '__main__':
    test_flat()
    test_flat_noncontiguous()
    test_ellipsis()
    test_ellipsis_and_newaxis()
    test_ellipsis_and_newaxis_2()
    test_aug_implicit()
    test_aug_implicit_attribute()
    test_ellipsis_aug()
    test_newaxis()
    test_multiple_newaxis()
    test_index_intarr_1d()
    test_index_intarr_1d_literal()
    test_index_intarr_1d_constant()
    test_index_intarr_1d_multi()
    test_index_intarr_nd()
    test_index_intarr_into_length_one_dim()
    test_index_intarr_into_length_one_dim_of_many()
    test_index_intarr_length_one_dim_trailing_basic()
    test_index_intarr_into_trailing_length_one_dim()
    test_index_intarr_length_one_dim_beside_real_gather(0)
    test_index_intarr_length_one_dim_beside_real_gather(1)
    test_index_intarr_into_reshaped_length_one_dim()
    test_index_boolarr_rhs()
    test_index_multiboolarr()
    test_index_boolarr_fixed()
    test_index_boolarr_inline()
    test_out_index_intarr()
    test_out_index_intarr_bcast()
    test_out_index_intarr_aug()
    test_out_index_intarr_aug_bcast()
    test_out_index_intarr_multidim()
    test_out_index_intarr_multidim_range()
    test_advanced_indexing_syntax(False)
    test_advanced_indexing_syntax(True)
    test_multidim_tuple_index(False)
    test_multidim_tuple_index(True)
    test_multidim_tuple_index_longer()
    test_multidim_tuple_multidim_index()
    # test_multidim_tuple_multidim_index_write()
    test_advanced_index_broadcasting()
    test_combining_basic_and_advanced_indexing()
    # test_combining_basic_and_advanced_indexing_write()
    test_combining_basic_and_advanced_indexing_with_newaxes()
    test_combining_basic_and_advanced_indexing_with_newaxes_2()
    test_index_array_with_integer_index_write_is_refused()
    test_index_array_followed_by_integer_index_write_is_allowed()
