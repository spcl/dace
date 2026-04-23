# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import warnings
import numpy as np

import dace
from dace import data, nodes, Memlet

from dace.transformation.passes.remove_views import RemoveViews


def _count_views(sdfg: dace.SDFG) -> int:
    num = 0
    for n, _ in sdfg.all_nodes_recursive():
        if (isinstance(n, nodes.AccessNode) and isinstance(sdfg.arrays[n.data], data.View)):
            num += 1
    return num


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_view_array_array():
    """Reshape view (2x10 -> flat 20)"""
    sdfg = dace.SDFG('redarrtest')
    sdfg.add_view('v', [2, 10], dace.float64)
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_transient('tmp', [20], dace.float64)

    state = sdfg.add_state()
    t = state.add_tasklet('something', {}, {'out'}, 'out[1, 1] = 6')
    v = state.add_access('v')
    tmp = state.add_access('tmp')
    w = state.add_write('A')
    state.add_edge(t, 'out', v, None, Memlet('v[0:2, 0:10]'))
    state.add_nedge(v, tmp, Memlet('tmp[0:20]'))
    state.add_nedge(tmp, w, Memlet('A[0:20]'))

    sdfg.validate()
    num_before = _count_views(sdfg)
    assert num_before == 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None

    num_after = _count_views(sdfg)
    assert num_after == 0
    sdfg.validate()


def test_view_slice_detect_simple():
    """Squeeze view: A[1,1] -> V[1], map writes through V to A mapping [0] to [0, 0]."""
    sdfg = dace.SDFG('view_squeeze_test')
    sdfg.add_array('A', [1, 1], dace.float64)
    sdfg.add_view('V', [1], dace.float64)

    state = sdfg.add_state()
    a = state.add_write('A')
    v = state.add_access('V')

    state.add_edge(v, 'views', a, None, Memlet(data='A', subset='0, 0:1', other_subset='0:1'))

    state.add_mapped_tasklet(
        'produce',
        {'i': '0:1'},
        {},
        'out = 42.0',
        {'out': Memlet('V[i]')},
        output_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A_ref = np.zeros((1, 1), dtype=np.float64)
    sdfg(A=A_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})

    num_after = _count_views(sdfg)
    assert num_after == 0
    assert result is not None
    sdfg.validate()

    A_new = np.zeros((1, 1), dtype=np.float64)
    sdfg(A=A_new)
    np.testing.assert_allclose(A_new, A_ref)
    assert A_new[0, 0] == 42.0


@dace.program
def jacobi1d_half(TMAX: dace.int32, A: dace.float32[12], B: dace.float32[12]):
    for _ in range(TMAX):
        B[1:-1] = 0.3333 * (A[:-2] + A[1:-1] + A[2:])


def test_read_slice():
    """Three read-slice views from jacobi1d (A[:-2], A[1:-1], A[2:]) should be removed."""
    sdfg = jacobi1d_half.to_sdfg(simplify=False)

    num_before = _count_views(sdfg)
    if num_before != 3:
        warnings.warn("Unexpected number of Views; test may need updating "
                      "for this DaCe version.")

    A = np.arange(12, dtype=np.float32)
    B_ref = np.zeros(12, dtype=np.float32)
    sdfg(TMAX=1, A=A.copy(), B=B_ref)

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})

    num_after = _count_views(sdfg)
    assert num_after == 0
    assert result is not None
    sdfg.validate()

    B_new = np.zeros(12, dtype=np.float32)
    sdfg(TMAX=1, A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref, rtol=1e-5)


def test_simple_slice_view():
    """1D contiguous slice: A[10] -> V[6] via A[2:8]."""
    sdfg = dace.SDFG('test_simple_slice')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    sdfg.add_view('V', [6], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')

    state.add_edge(a, None, v, 'views', Memlet(data='A', subset='2:8', other_subset='0:6'))

    state.add_mapped_tasklet(
        'copy',
        {'i': '0:6'},
        {'inp': Memlet('V[i]')},
        'out = inp * 2.0',
        {'out': Memlet('B[i]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(10, dtype=np.float64)
    B_ref = np.zeros(6, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None
    assert _count_views(sdfg) == 0

    sdfg.validate()

    B_new = np.zeros(6, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(B_new, A[2:8] * 2.0)


def test_reshape_view():
    """Dense reshape via numpy frontend: A[9] -> tmp[3,3]."""

    @dace.program
    def reshape_prog(A: dace.float64[9], B: dace.float64[3, 3]):
        tmp = np.reshape(A, (3, 3))
        B[:] = tmp + 1.0

    sdfg = reshape_prog.to_sdfg(simplify=False)
    sdfg.validate()

    A = np.arange(9, dtype=np.float64)
    B_ref = np.zeros((3, 3), dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})

    num_after = _count_views(sdfg)
    assert num_after == 0
    assert result is not None

    sdfg.validate()

    B_new = np.zeros((3, 3), dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)


def test_squeeze_view():
    """Squeeze: A[1, N] -> V[N], map writes through V."""
    N = 8
    sdfg = dace.SDFG('test_squeeze')
    sdfg.add_array('A', [1, N], dace.float64)
    sdfg.add_view('V', [N], dace.float64)

    state = sdfg.add_state()
    v = state.add_access('V')
    a = state.add_write('A')

    state.add_edge(v, 'views', a, None, Memlet(data='A', subset='0, 0:{}'.format(N), other_subset='0:{}'.format(N)))

    state.add_mapped_tasklet(
        'produce',
        {'i': '0:{}'.format(N)},
        {},
        'out = double(i)',
        {'out': Memlet('V[i]')},
        output_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A_ref = np.zeros((1, N), dtype=np.float64)
    sdfg(A=A_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None
    assert _count_views(sdfg) == 0

    sdfg.validate()

    A_new = np.zeros((1, N), dtype=np.float64)
    sdfg(A=A_new)
    np.testing.assert_allclose(A_new, A_ref)


def test_view_chain():
    """Chained views: A[4:12] -> V1[8], V1[1:7] -> V2[6]; fixpoint collapses both."""
    sdfg = dace.SDFG('test_chain')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [6], dace.float64)
    sdfg.add_view('V1', [8], dace.float64)
    sdfg.add_view('V2', [6], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    v1 = state.add_access('V1')
    v2 = state.add_access('V2')

    state.add_edge(a, None, v1, 'views', Memlet(data='A', subset='4:12', other_subset='0:8'))
    state.add_edge(v1, None, v2, 'views', Memlet(data='V1', subset='1:7', other_subset='0:6'))

    state.add_mapped_tasklet(
        'copy',
        {'i': '0:6'},
        {'inp': Memlet('V2[i]')},
        'out = inp',
        {'out': Memlet('B[i]')},
        input_nodes={'V2': v2},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(20, dtype=np.float64)
    B_ref = np.zeros(6, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)
    assert num_before == 2

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None
    assert _count_views(sdfg) == 0

    sdfg.validate()

    B_new = np.zeros(6, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(B_new, A[5:11])


def test_noop_no_views():
    """No views present; pass returns None."""
    sdfg = dace.SDFG('test_noop')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    a = state.add_read('A')
    state.add_mapped_tasklet(
        'copy',
        {'i': '0:10'},
        {'inp': Memlet('A[i]')},
        'out = inp',
        {'out': Memlet('B[i]')},
        input_nodes={'A': a},
        external_edges=True,
    )

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is None


def test_unsqueeze_view():
    """Unsqueeze: A[N] -> V[1, N, 1], map writes through V."""
    N = 8
    sdfg = dace.SDFG('test_unsqueeze')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_view('V', [1, N, 1], dace.float64)

    state = sdfg.add_state()
    v = state.add_access('V')
    a = state.add_write('A')

    state.add_edge(v, 'views', a, None, Memlet(data='A', subset='0:{}'.format(N), other_subset='0, 0:{}, 0'.format(N)))

    state.add_mapped_tasklet(
        'produce',
        {'i': '0:{}'.format(N)},
        {},
        'out = double(i) + 1.0',
        {'out': Memlet('V[0, i, 0]')},
        output_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A_ref = np.zeros(N, dtype=np.float64)
    sdfg(A=A_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})

    num_after = _count_views(sdfg)
    assert num_after == 0
    assert result is not None
    sdfg.validate()

    A_new = np.zeros(N, dtype=np.float64)
    sdfg(A=A_new)
    np.testing.assert_allclose(A_new, A_ref)


def test_multiple_views_same_state():
    """Two independent views of the same array in one state."""
    sdfg = dace.SDFG('test_multi')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    sdfg.add_array('C', [5], dace.float64)
    sdfg.add_view('V1', [5], dace.float64)
    sdfg.add_view('V2', [5], dace.float64)

    state = sdfg.add_state()
    a1 = state.add_read('A')
    a2 = state.add_read('A')
    v1 = state.add_access('V1')
    v2 = state.add_access('V2')

    state.add_edge(a1, None, v1, 'views', Memlet(data='A', subset='0:5', other_subset='0:5'))
    state.add_edge(a2, None, v2, 'views', Memlet(data='A', subset='10:15', other_subset='0:5'))

    state.add_mapped_tasklet(
        'map1',
        {'i': '0:5'},
        {'inp': Memlet('V1[i]')},
        'out = inp + 1.0',
        {'out': Memlet('B[i]')},
        input_nodes={'V1': v1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        'map2',
        {'i': '0:5'},
        {'inp': Memlet('V2[i]')},
        'out = inp + 2.0',
        {'out': Memlet('C[i]')},
        input_nodes={'V2': v2},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(20, dtype=np.float64)
    B_ref = np.zeros(5, dtype=np.float64)
    C_ref = np.zeros(5, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref, C=C_ref)

    num_before = _count_views(sdfg)
    assert num_before == 2

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None
    assert _count_views(sdfg) == 0

    sdfg.validate()

    B_new = np.zeros(5, dtype=np.float64)
    C_new = np.zeros(5, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new, C=C_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(C_new, C_ref)


def test_write_view():
    """Write-side view: map -> V[6] -> A[3:9]."""
    sdfg = dace.SDFG('test_write_view')
    sdfg.add_array('A', [12], dace.float64)
    sdfg.add_view('V', [6], dace.float64)

    state = sdfg.add_state()
    v = state.add_access('V')
    a = state.add_write('A')

    state.add_edge(v, 'views', a, None, Memlet(data='A', subset='3:9', other_subset='0:6'))

    state.add_mapped_tasklet(
        'produce',
        {'i': '0:6'},
        {},
        'out = double(i) * 3.0',
        {'out': Memlet('V[i]')},
        output_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A_ref = np.zeros(12, dtype=np.float64)
    sdfg(A=A_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None
    assert _count_views(sdfg) == 0

    sdfg.validate()

    A_new = np.zeros(12, dtype=np.float64)
    sdfg(A=A_new)
    np.testing.assert_allclose(A_new, A_ref)


# ---------------------------------------------------------------------------
# Column views, strided views, flatten
# ---------------------------------------------------------------------------


def test_column_view():
    """Column extraction: A[M,N] row-major -> V[M] via A[:,COL], stride N."""
    M, N, COL = 6, 8, 2
    sdfg = dace.SDFG('test_column_view')
    sdfg.add_array('A', [M, N], dace.float64)
    sdfg.add_array('B', [M], dace.float64)
    sdfg.add_view('V', [M], dace.float64, strides=[N])

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')

    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{M}, {COL}', other_subset=f'0:{M}'))

    state.add_mapped_tasklet(
        'add_one',
        {'i': f'0:{M}'},
        {'inp': Memlet('V[i]')},
        'out = inp + 1.0',
        {'out': Memlet('B[i]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(M * N, dtype=np.float64).reshape(M, N)
    B_ref = np.zeros(M, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    sdfg.validate()

    assert _count_views(sdfg) == 0

    B_new = np.zeros(M, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(B_new, A[:, COL] + 1.0)


def test_column_view_w_offset():
    """Column extraction with row offset: A[2:M, COL] -> V[M-2], stride N."""
    M, N, COL = 6, 8, 2
    sdfg = dace.SDFG('test_column_view_w_offset')
    sdfg.add_array('A', [M, N], dace.float64)
    sdfg.add_array('B', [M], dace.float64)
    sdfg.add_view('V', [M - 2], dace.float64, strides=[N])

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')

    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'2:{M}, {COL}', other_subset=f'0:{M - 2}'))

    state.add_mapped_tasklet(
        'add_one',
        {'i': f'0:{M - 2}'},
        {'inp': Memlet('V[i]')},
        'out = inp + 1.0',
        {'out': Memlet('B[i]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(M * N, dtype=np.float64).reshape(M, N)
    B_ref = np.zeros(M, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    sdfg.validate()

    assert _count_views(sdfg) == 0

    B_new = np.zeros(M, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(B_new[0:4], A[2:6, COL] + 1.0)


def test_strided_column_view():
    """Strided column: A[0:M:2, COL] -> V[M//2], stride 2*N."""
    M, N, COL = 8, 6, 3
    HALF = M // 2
    sdfg = dace.SDFG('test_strided_column_view')
    sdfg.add_array('A', [M, N], dace.float64)
    sdfg.add_array('B', [HALF], dace.float64)
    sdfg.add_view('V', [HALF], dace.float64, strides=[N * 2])

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')

    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{M}:2, {COL}', other_subset=f'0:{HALF}'))

    state.add_mapped_tasklet(
        'add_one',
        {'i': f'0:{HALF}'},
        {'inp': Memlet('V[i]')},
        'out = inp + 1.0',
        {'out': Memlet('B[i]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(M * N, dtype=np.float64).reshape(M, N)
    B_ref = np.zeros(HALF, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None
    assert _count_views(sdfg) == 0

    sdfg.validate()

    B_new = np.zeros(HALF, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(B_new, A[0:M:2, COL] + 1.0)


def test_flatten_view():
    """Dense flatten: A[M,N] row-major -> V[M*N], linearize/delinearize path."""
    M, N = 4, 5
    MN = M * N
    sdfg = dace.SDFG('test_flatten_view')
    sdfg.add_array('A', [M, N], dace.float64)
    sdfg.add_array('B', [MN], dace.float64)
    sdfg.add_view('V', [MN], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')

    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{M}, 0:{N}', other_subset=f'0:{MN}'))

    state.add_mapped_tasklet(
        'copy',
        {'i': f'0:{MN}'},
        {'inp': Memlet('V[i]')},
        'out = inp',
        {'out': Memlet('B[i]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    A = np.arange(MN, dtype=np.float64).reshape(M, N)
    B_ref = np.zeros(MN, dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)

    num_before = _count_views(sdfg)
    assert num_before >= 1

    p = RemoveViews()
    result = p.apply_pass(sdfg, {})

    num_after = _count_views(sdfg)
    assert num_after == 0
    sdfg.validate()

    B_new = np.zeros(MN, dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)
    np.testing.assert_allclose(B_new, A.ravel())

    if result is not None:
        assert num_after < num_before


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_view_array_array()
    test_view_slice_detect_simple()
    test_read_slice()
    test_simple_slice_view()
    test_reshape_view()
    test_squeeze_view()
    test_view_chain()
    test_noop_no_views()
    test_unsqueeze_view()
    test_multiple_views_same_state()
    test_write_view()
    test_column_view()
    test_column_view_w_offset()
    test_strided_column_view()
    test_flatten_view()
