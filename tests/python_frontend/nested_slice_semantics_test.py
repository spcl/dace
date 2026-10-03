# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""NumPy subset semantics for subscripts parsed inside a nested scope (a ``dace.map`` body).

Outside a map the frontend builds a real View for a slice; inside one it builds a nested SDFG
connector instead, and that second path used to lose the write-through, the length-1 axis, the
new axis and the source/destination aliasing.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python.common import DaceSyntaxError
from dace.sdfg import nodes


def nested_sdfg_node(sdfg: dace.SDFG) -> nodes.NestedSDFG:
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            return node
    raise AssertionError('no nested SDFG in the parsed program')


@dace.program
def view_write_in_map(A: dace.float64[8, 8], B: dace.float64[8]):
    for i in dace.map[0:8]:
        v = A[i, :]
        v[0] = B[i]


def test_view_write_in_map():
    sdfg = view_write_in_map.to_sdfg(simplify=False)
    sdfg.validate()

    nsdfg = nested_sdfg_node(sdfg)
    written = nsdfg.in_connectors.keys() & nsdfg.out_connectors.keys()
    assert written, 'the sliced row must reach the map body as an IN/OUT connector'

    A = np.arange(64, dtype=np.float64).reshape(8, 8).copy()
    B = np.arange(1, 9, dtype=np.float64).copy()
    expected = A.copy()
    expected[:, 0] = B
    view_write_in_map(A, B)
    assert np.allclose(A, expected)


@dace.program
def view_of_view_write_in_map(A: dace.float64[4, 6, 8], B: dace.float64[4]):
    for i in dace.map[0:4]:
        p = A[i]
        q = p[2]
        q[3] = B[i]


def test_view_of_view_write_in_map():
    A = np.arange(192, dtype=np.float64).reshape(4, 6, 8).copy()
    B = np.arange(1, 5, dtype=np.float64).copy()
    expected = A.copy()
    expected[:, 2, 3] = B
    view_of_view_write_in_map(A, B)
    assert np.allclose(A, expected)


@dace.program
def augmented_view_write_in_map(A: dace.float64[8, 8], B: dace.float64[8]):
    for i in dace.map[0:8]:
        v = A[i, :]
        v[2:4] += B[i]


def test_augmented_view_write_in_map():
    A = np.arange(64, dtype=np.float64).reshape(8, 8).copy()
    B = np.arange(1, 9, dtype=np.float64).copy()
    expected = A.copy()
    expected[:, 2:4] += B[:, None]
    augmented_view_write_in_map(A, B)
    assert np.allclose(A, expected)


@dace.program
def len1_slice_in_map(A: dace.float64[4, 4, 4], B: dace.float64[4, 4, 4], out: dace.float64[4, 4, 4]):
    for i in dace.map[0:4]:
        out[i, :, :] = A[i, :, 0:1] * B[i, :, :]


@dace.program
def index_in_map(A: dace.float64[4, 4, 4], B: dace.float64[4, 4, 4], out: dace.float64[4, 4, 4]):
    for i in dace.map[0:4]:
        out[i, :, :] = A[i, :, 0] * B[i, :, :]


def test_len1_slice_keeps_its_axis_in_map():
    sdfg = len1_slice_in_map.to_sdfg(simplify=False)
    inner = [sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg][0]
    assert any(tuple(desc.shape) == (4, 1) for desc in inner.arrays.values()), \
        f'the length-1 axis was squeezed away: {[tuple(d.shape) for d in inner.arrays.values()]}'

    A = np.arange(64, dtype=np.float64).reshape(4, 4, 4).copy()
    B = (np.arange(64, dtype=np.float64).reshape(4, 4, 4) % 7 + 1).copy()
    out = np.zeros((4, 4, 4))
    len1_slice_in_map(A, B, out)
    assert np.allclose(out, A[:, :, 0:1] * B)


def test_integer_index_drops_its_axis_in_map():
    A = np.arange(64, dtype=np.float64).reshape(4, 4, 4).copy()
    B = (np.arange(64, dtype=np.float64).reshape(4, 4, 4) % 7 + 1).copy()
    out = np.zeros((4, 4, 4))
    index_in_map(A, B, out)
    expected = np.zeros((4, 4, 4))
    for i in range(4):
        expected[i] = A[i, :, 0] * B[i]
    assert np.allclose(out, expected)


@dace.program
def newaxis_in_map(A: dace.float64[4, 8], B: dace.float64[4, 8], out: dace.float64[4, 8, 8]):
    for i in dace.map[0:4]:
        out[i] = A[i, :, None] * B[i, None, :]


def test_newaxis_in_map():
    sdfg = newaxis_in_map.to_sdfg(simplify=False)
    inner = [sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg][0]
    shapes = [tuple(desc.shape) for desc in inner.arrays.values()]
    assert (8, 1) in shapes and (1, 8) in shapes, f'np.newaxis was dropped: {shapes}'

    A = np.arange(32, dtype=np.float64).reshape(4, 8).copy()
    B = np.arange(1, 33, dtype=np.float64).reshape(4, 8).copy()
    out = np.zeros((4, 8, 8))
    newaxis_in_map(A, B, out)
    assert np.allclose(out, A[:, :, None] * B[:, None, :])


@dace.program
def overlapping_row_copy_in_map(A: dace.float64[8, 8]):
    for i in dace.map[0:8]:
        A[i, 1:] = A[i, :-1]


def test_overlapping_slice_copy_in_map():
    # Both sides become connectors that POINT INTO the same row, so the copy needs a temporary.
    sdfg = overlapping_row_copy_in_map.to_sdfg(simplify=False)
    inner = [sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg][0]
    assert any(desc.transient for desc in inner.arrays.values()), \
        'the overlapping self-copy was left aliased, with no temporary in between'

    A = np.arange(64, dtype=np.float64).reshape(8, 8).copy()
    expected = A.copy()
    expected[:, 1:] = expected[:, :-1]
    sdfg(A=A)
    assert np.allclose(A, expected)


@dace.program
def negative_step_in_map(A: dace.float64[8, 8], out: dace.float64[8, 8]):
    for i in dace.map[0:8]:
        out[i, :] = A[i, ::-1]


def test_negative_step_in_map_is_refused():
    with pytest.raises(DaceSyntaxError, match='Negative strides'):
        negative_step_in_map.to_sdfg(simplify=False)


@dace.program
def fancy_index_in_map(A: dace.float64[4, 8], idx: dace.int64[3], out: dace.float64[4, 3]):
    for i in dace.map[0:4]:
        out[i, :] = A[i, idx]


def test_fancy_index_in_map():
    sdfg = fancy_index_in_map.to_sdfg(simplify=False)
    sdfg.validate()

    found = [(n, p) for n, p in sdfg.all_nodes_recursive()
             if isinstance(n, nodes.Tasklet) and n.label.startswith('indirection')]
    assert found, 'advanced indexing in a map body must lower to an indirection tasklet'
    for tasklet, state in found:
        for edge in state.in_edges(tasklet):
            if edge.dst_conn.startswith('__inp'):
                assert edge.data.subset.num_elements() == 1, \
                    'the index fed to the tasklet must be one element, not the whole index array'

    A = np.arange(32, dtype=np.float64).reshape(4, 8).copy()
    idx = np.array([5, 0, 2], dtype=np.int64)
    out = np.zeros((4, 3))
    fancy_index_in_map(A, idx, out)
    assert np.allclose(out, A[:, idx])


@dace.program
def fancy_index_leading_in_map(A: dace.float64[4, 8], idx: dace.int64[3], out: dace.float64[2, 3]):
    for i in dace.map[0:2]:
        out[i] = A[idx, i]


def test_fancy_index_leading_in_map():
    A = np.arange(32, dtype=np.float64).reshape(4, 8).copy()
    idx = np.array([3, 0, 2], dtype=np.int64)
    out = np.zeros((2, 3))
    expected = np.stack([A[idx, i] for i in range(2)])
    fancy_index_leading_in_map(A, idx, out)
    assert np.allclose(out, expected)


@dace.program
def boolean_mask_read_in_map(A: dace.float64[4, 8], m: dace.bool[4, 8], out: dace.float64[4]):
    for i in dace.map[0:4]:
        out[i] = np.sum(A[m])


def test_boolean_mask_read_in_map_is_refused():
    with pytest.raises(DaceSyntaxError, match='Boolean array indexing'):
        boolean_mask_read_in_map.to_sdfg(simplify=False)


@dace.program
def subscripted_binop_in_map(A: dace.float64[4, 8], B: dace.float64[4, 8], out: dace.float64[4]):
    for i in dace.map[0:4]:
        out[i] = (A[i] + B[i])[2]


def test_subscripted_binop_in_map():
    A = np.arange(32, dtype=np.float64).reshape(4, 8).copy()
    B = np.ones((4, 8))
    out = np.zeros(4)
    subscripted_binop_in_map(A, B, out)
    assert np.allclose(out, (A + B)[:, 2])


if __name__ == '__main__':
    test_view_write_in_map()
    test_view_of_view_write_in_map()
    test_augmented_view_write_in_map()
    test_len1_slice_keeps_its_axis_in_map()
    test_integer_index_drops_its_axis_in_map()
    test_newaxis_in_map()
    test_overlapping_slice_copy_in_map()
    test_negative_step_in_map_is_refused()
    test_fancy_index_in_map()
    test_fancy_index_leading_in_map()
    test_boolean_mask_read_in_map_is_refused()
    test_subscripted_binop_in_map()
