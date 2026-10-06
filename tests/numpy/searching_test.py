# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = 100


def test_numpy_where():

    @dace.program
    def numpy_where(A: dace.float64[N]):
        return np.where(A > 0.5, A, 0.0)

    for _ in range(10):
        A = np.random.randn(N)
        assert (np.allclose(numpy_where(A), np.where(A > 0.5, A, 0.0)))


def test_numpy_select():

    @dace.program
    def numpy_where(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        return np.select([A > 0.5, B > 0.5, C > 0.5], [A, B, C], 0.0)

    for _ in range(10):
        A = np.random.randn(N)
        B = np.random.randn(N)
        C = np.random.randn(N)
        assert (np.allclose(numpy_where(A, B, C), np.select([A > 0.5, B > 0.5, C > 0.5], [A, B, C], 0.0)))


def test_numpy_where_uses_the_merge_library_node():
    """ Three real arrays and no cast is exactly what MergeLibraryNode expresses, so the
        frontend must hand that case to the node rather than inline a tasklet -- otherwise
        nothing downstream can pick a different lowering for the select. """
    from dace.libraries.standard.nodes import MergeLibraryNode

    @dace.program
    def where_arrays(A: dace.float64[N], B: dace.float64[N], C: dace.bool_[N]):
        return np.where(C, A, B)

    sdfg = where_arrays.to_sdfg(simplify=False)
    merges = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MergeLibraryNode)]
    assert len(merges) == 1, f'expected one MergeLibraryNode, found {len(merges)}'

    A, B = np.random.randn(N), np.random.randn(N)
    C = A > B
    assert np.allclose(where_arrays(A, B, C), np.where(C, A, B))


def test_numpy_where_partial_broadcast():
    """ A (N, 1) operand against an (N, M) result: the axis of extent 1 must be read at index 0
        for every column, not indexed by the column iterator. """

    @dace.program
    def where_partial(A: dace.float64[N, 1], B: dace.float64[N, 4], C: dace.bool_[N, 4]):
        return np.where(C, A, B)

    A = np.random.randn(N, 1)
    B = np.random.randn(N, 4)
    C = B > 0.0
    assert np.allclose(where_partial(A, B, C), np.where(C, A, B))


def merge_nodes(program):
    from dace.libraries.standard.nodes import MergeLibraryNode
    sdfg = program.to_sdfg(simplify=False)
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MergeLibraryNode)]


def test_numpy_where_cast_stays_a_tasklet():
    """ The library node only expresses ``where`` without a cast, so an operand needing a cast to the result type
        keeps the inlined-tasklet path. """

    @dace.program
    def where_mixed(A: dace.float64[N], B: dace.int32[N], C: dace.bool_[N]):
        return np.where(C, A, B)

    assert not merge_nodes(where_mixed)

    A = np.random.randn(N)
    B = np.random.randint(-8, 8, size=N).astype(np.int32)
    C = np.random.rand(N) > 0.5
    assert np.allclose(where_mixed(A, B, C), np.where(C, A, B))


def test_numpy_where_of_an_unsigned_and_a_signed_array_keeps_the_sign():
    """ numpy promotes uint64 with int64 to float64; a C++ conditional would give uint64 and wrap the negatives. """

    @dace.program
    def where_unsigned(A: dace.uint64[N], B: dace.int64[N], C: dace.bool_[N]):
        return np.where(C, A, B)

    A = np.random.randint(0, 8, size=N).astype(np.uint64)
    B = -np.random.randint(1, 8, size=N).astype(np.int64)
    C = np.random.rand(N) > 0.5
    assert np.array_equal(where_unsigned(A, B, C), np.where(C, A, B))


def test_numpy_where_with_a_condition_wider_than_the_operands():
    """ The condition broadcasts the result past both operands: (N, 1) and (N, 1) against an (N, 4) condition. """

    @dace.program
    def where_wide_condition(A: dace.float64[N, 1], B: dace.float64[N, 1], C: dace.bool_[N, 4]):
        return np.where(C, A, B)

    A = np.random.randn(N, 1)
    B = np.random.randn(N, 1)
    C = np.random.rand(N, 4) > 0.5
    assert np.allclose(where_wide_condition(A, B, C), np.where(C, A, B))


def test_numpy_where_with_a_constant():
    """ A scalar operand stays a tasklet, which autodiff differentiates (``hdiff``'s ``np.where(..., 0, res)``). """

    @dace.program
    def where_constant(A: dace.float64[N]):
        return np.where(A > 0.5, A, 0.0)

    assert not merge_nodes(where_constant)
    A = np.random.rand(N)
    assert np.allclose(where_constant(A), np.where(A > 0.5, A, 0.0))


if __name__ == "__main__":
    test_numpy_where()
    test_numpy_select()
    test_numpy_where_uses_the_merge_library_node()
    test_numpy_where_partial_broadcast()
    test_numpy_where_cast_stays_a_tasklet()
    test_numpy_where_of_an_unsigned_and_a_signed_array_keeps_the_sign()
    test_numpy_where_with_a_condition_wider_than_the_operands()
    test_numpy_where_with_a_constant()
