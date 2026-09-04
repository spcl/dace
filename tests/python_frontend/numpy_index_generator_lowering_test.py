# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The numpy sequence and index generators lower to maps, not to a pyobject callback.

Every case asserts the STRUCTURE first: a callback returns the right numbers, so a numeric assertion
on its own passes straight through one and proves nothing about the lowering.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd


def callback_free(sdfg: dace.SDFG) -> bool:
    """A pyobject callback shows up as a ``__pystate`` container plus a ``numpy_<name>`` tasklet."""
    for nested in sdfg.all_sdfgs_recursive():
        if any('__pystate' in name for name in nested.arrays):
            return False
        for state in nested.states():
            for node in state.nodes():
                if isinstance(node, nd.Tasklet) and ('numpy_' in node.code.as_string
                                                     or node.label.startswith('callback')):
                    return False
    return True


def map_count(sdfg: dace.SDFG) -> int:
    return sum(1 for state in sdfg.states() for node in state.nodes() if isinstance(node, nd.MapEntry))


def assert_native(program: dace.frontend.python.parser.DaceProgram) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=False)
    assert callback_free(sdfg)
    assert map_count(sdfg) > 0
    return sdfg


def assert_same(result: np.ndarray, reference: np.ndarray, exact: bool = True) -> None:
    assert result.shape == reference.shape
    assert result.dtype == reference.dtype
    if exact:
        assert np.array_equal(result, reference)
    else:
        np.testing.assert_allclose(result, reference, rtol=1e-13, atol=0.0)


def test_logspace() -> None:

    @dace.program
    def prog():
        return np.logspace(0.0, 3.0, 7)

    assert_native(prog)
    assert_same(prog(), np.logspace(0.0, 3.0, 7), exact=False)


def test_logspace_base_and_axis() -> None:

    @dace.program
    def prog():
        return np.logspace(1.0, 5.0, 5, base=2.0)

    assert_native(prog)
    assert_same(prog(), np.logspace(1.0, 5.0, 5, base=2.0))


def test_geomspace() -> None:

    @dace.program
    def prog():
        return np.geomspace(1.0, 1000.0, 5)

    assert_native(prog)
    assert_same(prog(), np.geomspace(1.0, 1000.0, 5), exact=False)


def test_geomspace_negative_endpoints() -> None:

    @dace.program
    def prog():
        return np.geomspace(-1.0, -8.0, 4)

    assert_native(prog)
    assert_same(prog(), np.geomspace(-1.0, -8.0, 4))


def test_geomspace_refuses_data_endpoint() -> None:

    @dace.program
    def prog(a: dace.float64[1]):
        return np.geomspace(a[0], 1000.0, 5)

    with pytest.raises(ValueError, match='compile-time constant'):
        prog.to_sdfg(simplify=False)


def test_fromfunction() -> None:

    @dace.program
    def prog():
        return np.fromfunction(lambda i, j: i * 10 + j, (4, 5), dtype=np.float64)

    assert_native(prog)
    assert_same(prog(), np.fromfunction(lambda i, j: i * 10 + j, (4, 5), dtype=np.float64))


def test_fromfunction_integer_indices() -> None:

    @dace.program
    def prog():
        return np.fromfunction(lambda i: i * i, (5, ), dtype=np.int64)

    assert_native(prog)
    assert_same(prog(), np.fromfunction(lambda i: i * i, (5, ), dtype=np.int64))


def test_fromfunction_symbolic_shape() -> None:
    N = dace.symbol('N')

    @dace.program
    def prog():
        return np.fromfunction(lambda i, j: i * (j + 2) / N, (N, N), dtype=np.float64)

    assert_native(prog)
    assert_same(prog(N=7), np.fromfunction(lambda i, j: i * (j + 2) / 7, (7, 7), dtype=np.float64), exact=False)


def test_fromfunction_refuses_named_callable() -> None:

    @dace.program
    def prog():
        return np.fromfunction(np.sqrt, (4, ), dtype=np.float64)

    with pytest.raises(ValueError, match='cannot be inlined'):
        prog.to_sdfg(simplify=False)


def test_fromfunction_refuses_array_read() -> None:

    @dace.program
    def prog(a: dace.float64[4]):
        return np.fromfunction(lambda i: a[0] + i, (4, ), dtype=np.float64)

    with pytest.raises(ValueError, match='arithmetic lambda'):
        prog.to_sdfg(simplify=False)


def test_indices() -> None:

    @dace.program
    def prog():
        return np.indices((3, 4))

    assert_native(prog)
    result = prog()
    assert result.dtype == np.int64
    assert_same(result, np.indices((3, 4)))


def test_indices_sparse_keeps_every_axis() -> None:

    @dace.program
    def prog():
        rows, cols = np.indices((3, 4), sparse=True)
        return rows, cols

    assert_native(prog)
    rows, cols = prog()
    ref_rows, ref_cols = np.indices((3, 4), sparse=True)
    assert_same(rows, ref_rows)
    assert_same(cols, ref_cols)


def test_ix_open_mesh_shapes() -> None:

    @dace.program
    def prog(a: dace.int64[3], b: dace.int64[2]):
        rows, cols = np.ix_(a, b)
        return rows, cols

    a = np.array([1, 5, 7], np.int64)
    b = np.array([2, 4], np.int64)
    sdfg = prog.to_sdfg(simplify=False)
    assert callback_free(sdfg)
    rows, cols = prog(a=a, b=b)
    ref_rows, ref_cols = np.ix_(a, b)
    assert_same(rows, ref_rows)
    assert_same(cols, ref_cols)
    # The open mesh is what makes a[np.ix_(a, b)] broadcast into a rectangle.
    assert (rows + cols).shape == (3, 2)


def test_ix_refuses_boolean_mask() -> None:

    @dace.program
    def prog(a: dace.bool[3]):
        rows, = np.ix_(a)
        return rows

    with pytest.raises(ValueError, match='data-dependent'):
        prog.to_sdfg(simplify=False)


def test_ravel_multi_index() -> None:

    @dace.program
    def prog(rows: dace.int64[4], cols: dace.int64[4]):
        return np.ravel_multi_index((rows, cols), (3, 5))

    assert_native(prog)
    rows = np.array([0, 1, 2, 1], np.int64)
    cols = np.array([4, 0, 3, 1], np.int64)
    assert_same(prog(rows=rows, cols=cols), np.ravel_multi_index((rows, cols), (3, 5)))


def test_ravel_multi_index_modes_and_order() -> None:
    rows = np.array([5, -1, 2, 9], np.int64)
    cols = np.array([7, 0, -3, 1], np.int64)

    @dace.program
    def wrapped(r: dace.int64[4], c: dace.int64[4]):
        return np.ravel_multi_index((r, c), (3, 5), mode='wrap')

    @dace.program
    def clipped(r: dace.int64[4], c: dace.int64[4]):
        return np.ravel_multi_index((r, c), (3, 5), mode='clip')

    @dace.program
    def fortran(r: dace.int64[4], c: dace.int64[4]):
        return np.ravel_multi_index((r, c), (3, 5), order='F')

    assert_native(wrapped)
    assert_native(clipped)
    assert_native(fortran)
    assert_same(wrapped(r=rows, c=cols), np.ravel_multi_index((rows, cols), (3, 5), mode='wrap'))
    assert_same(clipped(r=rows, c=cols), np.ravel_multi_index((rows, cols), (3, 5), mode='clip'))
    inrange_r = np.array([0, 1, 2, 1], np.int64)
    inrange_c = np.array([4, 0, 3, 1], np.int64)
    assert_same(fortran(r=inrange_r, c=inrange_c), np.ravel_multi_index((inrange_r, inrange_c), (3, 5), order='F'))


def test_ravel_multi_index_broadcasts_operands() -> None:

    @dace.program
    def prog(rows: dace.int64[2, 3], cols: dace.int64[2, 3]):
        return np.ravel_multi_index((rows, cols), (4, 5))

    assert_native(prog)
    rows = np.array([[0, 1, 2], [3, 0, 1]], np.int64)
    cols = np.array([[1, 2, 3], [4, 0, 1]], np.int64)
    assert_same(prog(rows=rows, cols=cols), np.ravel_multi_index((rows, cols), (4, 5)))


def test_ravel_multi_index_refuses_dimension_mismatch() -> None:

    @dace.program
    def prog(rows: dace.int64[4], cols: dace.int64[4]):
        return np.ravel_multi_index((rows, cols), (3, 5, 7))

    with pytest.raises(ValueError, match='2 indices given for 3 dimensions'):
        prog.to_sdfg(simplify=False)


@pytest.mark.parametrize('n, k, m', [(4, 1, None), (3, -1, None), (4, -1, 3), (5, 0, None), (4, 0, 6)])
def test_triu_indices(n: int, k: int, m: int | None) -> None:

    @dace.program
    def prog():
        rows, cols = np.triu_indices(n, k, m)
        return rows, cols

    assert_native(prog)
    rows, cols = prog()
    ref_rows, ref_cols = np.triu_indices(n, k, m)
    assert_same(rows, ref_rows)
    assert_same(cols, ref_cols)


def test_triu_indices_refuses_symbolic_extent() -> None:
    N = dace.symbol('N')

    @dace.program
    def prog():
        rows, cols = np.triu_indices(N)
        return rows, cols

    with pytest.raises(ValueError, match='static extent'):
        prog.to_sdfg(simplify=False)


if __name__ == '__main__':
    pytest.main([__file__])
