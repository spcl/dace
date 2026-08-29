# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

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


def test_numpy_where_scalar_operands():
    """ Both x and y are constants: the result is broadcast to the shape of the condition. """

    @dace.program
    def numpy_where_scalars(A: dace.float64[N]):
        return np.where(A > 0.5, 0.0, 1.0)

    A = np.random.randn(N)
    ref = np.where(A > 0.5, 0.0, 1.0)
    res = numpy_where_scalars(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_int():
    """ Two integer constants must not promote the result to floating point. """

    @dace.program
    def numpy_where_scalars_int(A: dace.float64[N]):
        return np.where(A > 0.5, 1, 2)

    A = np.random.randn(N)
    ref = np.where(A > 0.5, 1, 2)
    res = numpy_where_scalars_int(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_mixed():
    """ Mixed integer/floating-point constants follow NumPy's type promotion. """

    @dace.program
    def numpy_where_scalars_mixed(A: dace.float64[N]):
        return np.where(A > 0.5, 0.0, 1)

    A = np.random.randn(N)
    ref = np.where(A > 0.5, 0.0, 1)
    res = numpy_where_scalars_mixed(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_2d():

    @dace.program
    def numpy_where_scalars_2d(A: dace.float64[N, 5]):
        return np.where(A > 0.5, -1.0, 2.5)

    A = np.random.randn(N, 5)
    ref = np.where(A > 0.5, -1.0, 2.5)
    res = numpy_where_scalars_2d(A)
    assert np.allclose(res, ref)
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype


def test_numpy_where_scalar_operands_scalar_condition():
    """ Scalar condition and scalar operands would give a 0-dimensional result, which DaCe cannot represent. """

    @dace.program
    def numpy_where_scalar_cond(a: dace.float64):
        return np.where(a > 0.5, 0.0, 1.0)

    with pytest.raises(ValueError, match='0-dimensional'):
        numpy_where_scalar_cond(1.0)


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


def test_where_with_a_symbolic_branch():
    """A branch that is a symbolic scalar, not an array or a Python number.

    ``2 * S`` arrives as a sympy expression whose ``type()`` -- ``sympy.Mul`` -- is in no dtype map,
    so the dtype lookup raised ``KeyError`` before the tasklet was ever built.
    """
    S = dace.symbol('S', dtype=dace.int64)

    @dace.program
    def where_sym(a: dace.float64[S], out: dace.float64[S]):
        out[:] = np.where(a > 0.0, a, 2 * S)

    a = np.array([1.0, -1.0, 3.0, -4.0], dtype=np.float64)
    out = np.zeros(4, dtype=np.float64)
    where_sym(a=a, out=out, S=4)
    assert np.allclose(out, np.where(a > 0.0, a, 2.0 * 4))


if __name__ == "__main__":
    test_numpy_where()
    test_where_with_a_symbolic_branch()
    test_numpy_select()
    test_numpy_where_scalar_operands()
    test_numpy_where_scalar_operands_int()
    test_numpy_where_scalar_operands_mixed()
    test_numpy_where_scalar_operands_2d()
    test_numpy_where_scalar_operands_scalar_condition()
    test_numpy_where_uses_the_merge_library_node()
    test_numpy_where_partial_broadcast()
    test_numpy_where_cast_stays_a_tasklet()
