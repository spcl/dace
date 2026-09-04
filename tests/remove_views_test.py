# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import warnings
import numpy as np

import dace
from dace import data, nodes, Memlet

from dace.transformation.passes.remove_views import RemoveViews


def dangling_connectors(sdfg: dace.SDFG):
    """Every connector in the SDFG that has no edge attached to it, as ``(state, node, side, conn)``.

    ``validate`` reports the first one it meets, but only after ``scope_dict`` has succeeded -- a
    graph whose scopes are already broken never reaches that check. Listing them directly keeps the
    assertion about the connectors themselves.
    """
    rows = []
    for parent in sdfg.all_sdfgs_recursive():
        for state in parent.states():
            for node in state.nodes():
                wired_in = {e.dst_conn for e in state.in_edges(node)}
                wired_out = {e.src_conn for e in state.out_edges(node)}
                rows += [(state.label, str(node), 'in', c) for c in node.in_connectors if c not in wired_in]
                rows += [(state.label, str(node), 'out', c) for c in node.out_connectors if c not in wired_out]
    return rows


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


def test_view_in_interstate_edge():
    """View referenced from an interstate-edge assignment, consumed later via a symbol.

    Flow: state1 holds ``A -> V`` (flat reshape of A[M,N]). The interstate edge
    assigns ``s = V[7]``. state2 writes ``B[0] = s``. After RemoveViews runs,
    V must be folded into A and the interstate edge's RHS must be rewritten
    to ``A[1, 2]`` (for M=4, N=5: 7 // 5 = 1, 7 % 5 = 2).
    """
    M, N = 4, 5
    MN = M * N

    sdfg = dace.SDFG('view_in_interstate_edge')
    sdfg.add_array('A', [M, N], dace.int64)
    sdfg.add_array('B', [1], dace.int64)
    sdfg.add_view('V', [MN], dace.int64)

    state1 = sdfg.add_state('entry')
    a1 = state1.add_read('A')
    v1 = state1.add_access('V')
    state1.add_edge(a1, None, v1, 'views', Memlet(data='A', subset=f'0:{M}, 0:{N}', other_subset=f'0:{MN}'))

    state2 = sdfg.add_state('consume')
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments={'s': 'V[7]'}))

    t = state2.add_tasklet('write', {}, {'out'}, 'out = s')
    b = state2.add_write('B')
    state2.add_edge(t, 'out', b, None, Memlet('B[0]'))

    sdfg.validate()

    A = np.arange(MN, dtype=np.int64).reshape(M, N)
    B_ref = np.zeros(1, dtype=np.int64)
    sdfg(A=A.copy(), B=B_ref)
    assert B_ref[0] == A.ravel()[7]

    assert _count_views(sdfg) == 1
    p = RemoveViews()
    result = p.apply_pass(sdfg, {})
    assert result is not None

    assert _count_views(sdfg) == 0
    sdfg.validate()

    # Every interstate edge must now reference A, not V.
    for e in sdfg.all_interstate_edges():
        for rhs in e.data.assignments.values():
            assert 'V[' not in rhs

    B_new = np.zeros(1, dtype=np.int64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, B_ref)


def test_view_edge_from_map_entry():
    """Read view staged in through a Map: ``A -> MapEntry -> V -> tasklet``.

    The view's defining edge starts at the MapEntry, not at the viewed AccessNode (which sits at
    the far end of the memlet path, outside the scope). Splicing the view out must leave the
    MapEntry on the path -- reattaching the consumer to the outer AccessNode instead drops the
    scope node's outgoing edge and leaves its ``IN_1``/``OUT_1`` pair dangling.
    """
    sdfg = dace.SDFG('view_edge_from_map_entry')
    sdfg.add_array('A', [4, 3], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    sdfg.add_view('vrow', [3], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_write('B')
    entry, exit_node = state.add_map('rows', {'i': '0:4'})
    v = state.add_access('vrow')
    t = state.add_tasklet('pick', {'inp'}, {'out'}, 'out = inp')

    state.add_memlet_path(a, entry, v, memlet=Memlet(data='A', subset='i, 0:3'))
    state.add_edge(v, None, t, 'inp', Memlet(data='vrow', subset='1'))
    state.add_memlet_path(t, exit_node, b, src_conn='out', memlet=Memlet(data='B', subset='i'))

    sdfg.validate()
    assert _count_views(sdfg) == 1

    assert RemoveViews().apply_pass(sdfg, {}) is not None
    assert _count_views(sdfg) == 0
    assert dangling_connectors(sdfg) == []
    sdfg.validate()
    # The consumer must still hang off the MapEntry, not off the outer AccessNode.
    assert state.scope_dict()[t] is entry

    a_arr = np.arange(12, dtype=np.float64).reshape(4, 3).copy()
    b_arr = np.zeros(4, dtype=np.float64)
    sdfg(A=a_arr, B=b_arr)
    np.testing.assert_allclose(b_arr, a_arr[:, 1])


def test_view_edge_into_map_exit():
    """Write view staged out through a Map: ``tasklet -> V -> MapExit -> B``.

    Mirror of :func:`test_view_edge_from_map_entry`: the defining edge ends at the MapExit, so
    reattaching the producer to the outer AccessNode leaves the MapExit's ``IN_1`` dangling.
    """
    sdfg = dace.SDFG('view_edge_into_map_exit')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4, 3], dace.float64)
    sdfg.add_view('vrow', [3], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_write('B')
    entry, exit_node = state.add_map('rows', {'i': '0:4'})
    v = state.add_access('vrow')
    t = state.add_tasklet('spread', {'inp'}, {'out'}, 'out = inp + 1.0')

    state.add_memlet_path(a, entry, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
    state.add_edge(t, 'out', v, None, Memlet(data='vrow', subset='1'))
    state.add_memlet_path(v, exit_node, b, memlet=Memlet(data='B', subset='i, 0:3'))

    sdfg.validate()
    assert _count_views(sdfg) == 1

    assert RemoveViews().apply_pass(sdfg, {}) is not None
    assert _count_views(sdfg) == 0
    assert dangling_connectors(sdfg) == []
    sdfg.validate()
    assert state.scope_dict()[t] is entry

    a_arr = np.arange(4, dtype=np.float64)
    b_arr = np.zeros((4, 3), dtype=np.float64)
    sdfg(A=a_arr, B=b_arr)
    np.testing.assert_allclose(b_arr[:, 1], a_arr + 1.0)


def test_plane_view_copied_by_dst_keyed_memlet():
    """A plane view read by a COPY whose memlet names the destination must keep its plane.

    ``data_col = np.array(dcol[:, :, K-1])`` (npbench vadv) lands as ``V -> B`` with the memlet
    keyed to ``B``, so the view side is unspelled -- and splicing the view out used to rewire the
    edge to the 3-D ``A`` with that side STILL unspelled, dropping which plane is copied. Nothing
    rejected the result: ``B`` is 2-D and the memlet names ``B``, so validation checked the only
    side that was spelled, and the plane was lost silently until a later pass materialised the copy.
    """
    M, N, K, PLANE = 4, 5, 3, 2
    sdfg = dace.SDFG('test_plane_view_dst_keyed')
    sdfg.add_array('A', [M, N, K], dace.float64)
    sdfg.add_array('B', [M, N], dace.float64)
    sdfg.add_view('V', [M, N], dace.float64, strides=[N * K, K])

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')
    b = state.add_write('B')

    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{M}, 0:{N}, {PLANE}', other_subset=f'0:{M}, 0:{N}'))
    # Keyed to the DESTINATION, with no other_subset: the shape the numpy frontend emits for a copy.
    state.add_edge(v, None, b, None, Memlet(data='B', subset=f'0:{M}, 0:{N}'))

    sdfg.validate()

    A = np.arange(M * N * K, dtype=np.float64).reshape(M, N, K)
    B_ref = np.zeros((M, N), dtype=np.float64)
    sdfg(A=A.copy(), B=B_ref)
    np.testing.assert_allclose(B_ref, A[:, :, PLANE])

    assert _count_views(sdfg) == 1
    assert RemoveViews().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert _count_views(sdfg) == 0

    # STRUCTURAL: the surviving A -> B edge must still say WHICH plane it reads. An unspelled
    # source side here is the silent miscompile this test exists for, so assert the subset text.
    copy_edges = [
        e for s in sdfg.states() for e in s.edges() if isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'B'
    ]
    assert len(copy_edges) == 1
    src_subset = copy_edges[0].data.get_src_subset(copy_edges[0], sdfg.states()[0])
    assert src_subset is not None, 'RemoveViews dropped the view plane from the copy memlet'
    assert src_subset.dims() == 3
    assert str(src_subset) == f'0:{M}, 0:{N}, {PLANE}'

    B_new = np.zeros((M, N), dtype=np.float64)
    sdfg(A=A.copy(), B=B_new)
    np.testing.assert_allclose(B_new, A[:, :, PLANE])


def _empty_edges(state):
    """``(src.data, dst.data)`` for every ordering (empty-memlet) edge in ``state``."""
    return [(e.src.data, e.dst.data) for e in state.edges() if e.data.is_empty()]


def test_read_view_keeps_ordering_edge():
    """An ordering edge into a read view must survive the view's removal.

    ``X --(empty)--> V`` says X runs before whoever reads V. ``_reconnect_edges`` only re-homes
    the view's OUT-edges for a read view, so the in-edge was left on the node and destroyed by
    ``remove_node`` -- and nothing rejected the result, because X keeps its other edges and the
    graph stays valid. The constraint just disappeared, so the writer of X could be scheduled
    after the read of A.
    """
    N = 8
    sdfg = dace.SDFG('read_view_ordering')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('X', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')
    b = state.add_write('B')
    x = state.add_access('X')
    # X is produced here, so dropping the ordering edge leaves a VALID graph: silent misordering.
    t = state.add_tasklet('produce', {}, {'o'}, 'o = 1.0')
    state.add_edge(t, 'o', x, None, Memlet(data='X', subset='0'))
    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{N}'))
    state.add_nedge(x, v, Memlet())
    state.add_nedge(v, b, Memlet(data='V', subset=f'0:{N}'))
    sdfg.validate()

    assert _count_views(sdfg) == 1
    assert RemoveViews().apply_pass(sdfg, {}) is not None
    assert _count_views(sdfg) == 0
    sdfg.validate()

    state = sdfg.states()[0]
    # STRUCTURAL: the constraint now lands on the view's READER, which is what X preceded.
    assert _empty_edges(state) == [('X', 'B')]
    # ... and NOT on the viewed array, which would be a strictly stronger claim.
    assert not state.edges_between(x, a)
    assert dangling_connectors(sdfg) == []


def test_write_view_keeps_ordering_edge():
    """Mirror of :func:`test_read_view_keeps_ordering_edge` for a write view.

    ``V --(empty)--> Y`` says whoever writes V runs before Y. For a write view
    ``_reconnect_edges`` only re-homes IN-edges, so this one was dropped with the node.
    """
    N = 8
    sdfg = dace.SDFG('write_view_ordering')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('Y', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64)

    state = sdfg.add_state()
    b = state.add_read('B')
    v = state.add_access('V')
    a = state.add_write('A')
    y = state.add_access('Y')
    # Y is consumed here, so dropping the ordering edge leaves a VALID graph.
    t = state.add_tasklet('consume', {'i'}, {}, 'pass')
    state.add_edge(y, None, t, 'i', Memlet(data='Y', subset='0'))
    state.add_nedge(b, v, Memlet(data='B', subset=f'0:{N}'))
    state.add_edge(v, 'views', a, None, Memlet(data='A', subset=f'0:{N}'))
    state.add_nedge(v, y, Memlet())
    sdfg.validate()

    assert _count_views(sdfg) == 1
    assert RemoveViews().apply_pass(sdfg, {}) is not None
    assert _count_views(sdfg) == 0
    sdfg.validate()

    state = sdfg.states()[0]
    # STRUCTURAL: the constraint lands on the view's WRITER, not on the viewed array A.
    assert _empty_edges(state) == [('B', 'Y')]
    assert not state.edges_between(a, y)
    assert dangling_connectors(sdfg) == []


def test_ordering_edge_never_rehomed_onto_the_viewed_array():
    """The ordering edge goes to the view's readers, because the viewed array can CYCLE.

    Here ``A -> X`` already exists, so re-homing ``X --(empty)--> V`` onto the view edge's
    neighbour ``A`` would mint ``X -> A`` and close ``A -> X -> A``. Re-homing onto the reader
    ``B`` cannot: ``X -> V -> B`` already made B reachable from X, so ``X -> B`` adds no
    reachability and no cycle the graph did not already have.
    """
    N = 8
    sdfg = dace.SDFG('ordering_cycle_risk')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('X', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')
    b = state.add_write('B')
    x = state.add_access('X')
    state.add_nedge(a, x, Memlet(data='A', subset=f'0:{N}'))
    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{N}'))
    state.add_nedge(x, v, Memlet())
    state.add_nedge(v, b, Memlet(data='V', subset=f'0:{N}'))
    sdfg.validate()

    assert RemoveViews().apply_pass(sdfg, {}) is not None
    assert _count_views(sdfg) == 0

    state = sdfg.states()[0]
    assert not state.has_cycles()
    assert not state.edges_between(x, a)
    assert _empty_edges(state) == [('X', 'B')]
    # ``validate`` rejects a cyclic state, so this is the end-to-end check on the choice.
    sdfg.validate()


def test_view_with_ordering_edge_and_no_reader_is_kept():
    """With no reader to carry the constraint, the view is KEPT rather than removed.

    ``A -> V`` plus ``X --(empty)--> V`` and nothing else: there is no successor of V to re-home
    ``X -> V`` onto, and ``X -> A`` is the strictly stronger claim this pass must not invent.
    Refusing the removal is the only choice that neither drops nor strengthens the constraint.
    """
    N = 8
    sdfg = dace.SDFG('ordering_no_reader')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('X', [N], dace.float64)
    sdfg.add_view('V', [N], dace.float64)

    state = sdfg.add_state()
    a = state.add_read('A')
    v = state.add_access('V')
    x = state.add_access('X')
    t = state.add_tasklet('produce', {}, {'o'}, 'o = 1.0')
    state.add_edge(t, 'o', x, None, Memlet(data='X', subset='0'))
    state.add_edge(a, None, v, 'views', Memlet(data='A', subset=f'0:{N}'))
    state.add_nedge(x, v, Memlet())

    assert _count_views(sdfg) == 1
    RemoveViews().apply_pass(sdfg, {})
    assert _count_views(sdfg) == 1, 'RemoveViews must refuse a view whose ordering edge has no home'

    state = sdfg.states()[0]
    assert _empty_edges(state) == [('X', 'V')]
    assert len(state.edges_between(a, v)) == 1


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
    test_view_in_interstate_edge()
    test_view_edge_from_map_entry()
    test_view_edge_into_map_exit()
    test_plane_view_copied_by_dst_keyed_memlet()
    test_read_view_keeps_ordering_edge()
    test_write_view_keeps_ordering_edge()
    test_ordering_edge_never_rehomed_onto_the_viewed_array()
    test_view_with_ordering_edge_and_no_reader_is_kept()
