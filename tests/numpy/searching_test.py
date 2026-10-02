# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
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


def test_numpy_where_cast_stays_a_tasklet():
    """ The library node's tasklet assigns straight across, so an operand needing a cast to the
        result type has to keep the inlined-tasklet path. """
    from dace.libraries.standard.nodes import MergeLibraryNode

    @dace.program
    def where_mixed(A: dace.float64[N], B: dace.int32[N], C: dace.bool_[N]):
        return np.where(C, A, B)

    sdfg = where_mixed.to_sdfg(simplify=False)
    assert not [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MergeLibraryNode)]

    A = np.random.randn(N)
    B = np.random.randint(-8, 8, size=N).astype(np.int32)
    C = A > 0.0
    assert np.allclose(where_mixed(A, B, C), np.where(C, A, B))


if __name__ == "__main__":
    test_numpy_where()
    test_numpy_select()
    test_numpy_where_uses_the_merge_library_node()
    test_numpy_where_partial_broadcast()
    test_numpy_where_cast_stays_a_tasklet()
