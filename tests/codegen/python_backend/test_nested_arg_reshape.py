# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the Python backend's nested-SDFG connector shape reconciliation.

A nested connector whose shape differs from the outer memlet subset is
reconciled by (in order):

1. a genuine STRIDED VIEW of the outer array, when the connector's declared
   shape/strides provably match the outer subset view (column views, singleton
   squeezes, transposes) -- valid for inputs AND outputs;
2. a flat C-order reshape, when it is provably order-preserving (same element
   count and C-contiguous connector strides);
3. otherwise a loud NotImplementedError -- never silently wrong data.
"""
import numpy as np
import pytest

import dace
from dace import data, dtypes, subsets
from dace.codegen.py.framecode import DaCePythonCodeGenerator
from dace.codegen.py.python_target import PythonCodeGen
from dace.memlet import Memlet


def _build_nested_scale_sdfg(name: str,
                             outer_in,
                             outer_out,
                             subset_in: str,
                             subset_out: str,
                             conn_in,
                             conn_out,
                             symbols=()) -> dace.SDFG:
    """Outer arrays ``X`` (shape ``outer_in``) / ``Y`` (shape ``outer_out``);
    a nested SDFG computes ``_y = 2 * _x`` with explicit connector layouts and
    explicit outer memlet subsets.

    :param name: SDFG name.
    :param outer_in: Shape of outer input ``X``.
    :param outer_out: Shape of outer output ``Y``.
    :param subset_in: Memlet subset string for the input edge.
    :param subset_out: Memlet subset string for the output edge.
    :param conn_in: (shape, strides) of the nested input connector.
    :param conn_out: (shape, strides) of the nested output connector.
    :param symbols: Symbol names to declare on both SDFGs.
    :returns: The configured (Python-backend) SDFG.
    """
    sdfg = dace.SDFG(name)
    for sym in symbols:
        sdfg.add_symbol(sym, dace.int64)
    state = sdfg.add_state()
    sdfg.add_array('X', outer_in, dace.float64)
    sdfg.add_array('Y', outer_out, dace.float64)

    nsdfg = dace.SDFG(name + '_inner')
    for sym in symbols:
        nsdfg.add_symbol(sym, dace.int64)
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', conn_in[0], dace.float64, strides=conn_in[1])
    nsdfg.add_array('_y', conn_out[0], dace.float64, strides=conn_out[1])
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet(f'X[{subset_in}]'))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet(f'Y[{subset_out}]'))
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _build_scale_sdfg(name: str,
                      conn_in_shape,
                      conn_in_strides,
                      conn_out_shape,
                      conn_out_strides,
                      outer_shape=(2, 3, 4)) -> dace.SDFG:
    """Outer arrays ``X``/``Y`` of ``outer_shape``; a nested SDFG computes
    ``_y = 2 * _x`` with the given connector shapes/strides.

    :param name: SDFG name.
    :param conn_in_shape: Nested input connector shape.
    :param conn_in_strides: Nested input connector strides.
    :param conn_out_shape: Nested output connector shape.
    :param conn_out_strides: Nested output connector strides.
    :param outer_shape: Shape of the outer arrays.
    :returns: The configured (Python-backend) SDFG.
    """
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array('X', outer_shape, dace.float64)
    sdfg.add_array('Y', outer_shape, dace.float64)

    nsdfg = dace.SDFG(name + '_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', conn_in_shape, dace.float64, strides=conn_in_strides)
    nsdfg.add_array('_y', conn_out_shape, dace.float64, strides=conn_out_strides)
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet.from_array('Y', sdfg.arrays['Y']))
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def test_output_side_flat_reshape():
    """A flat-order-preserving OUTPUT connector mismatch writes back correctly.

    Outer ``(2, 3, 4)`` vs C-contiguous ``(6, 4)`` connectors on both sides:
    writes through the reshaped output must land in the outer array (bug 08b:
    they were previously unhandled).
    """
    sdfg = _build_scale_sdfg('nested_out_reshape', (6, 4), (4, 1), (6, 4), (4, 1))
    X = np.random.rand(2, 3, 4)
    Y = np.zeros((2, 3, 4))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X), f"max diff = {np.max(np.abs(Y - 2 * X))}"


def test_input_permutation_mismatch_raises():
    """A permutation-type INPUT mismatch raises instead of transposing data.

    Connector shape ``(4, 6)`` with transposed strides ``(1, 4)`` is not a flat
    C-order reinterpretation of the outer ``(2, 3, 4)`` slice; a flat reshape
    would silently produce transposed data (bug 08b).
    """
    sdfg = _build_scale_sdfg('nested_perm_mismatch', (4, 6), (1, 4), (6, 4), (4, 1))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


def test_output_permutation_mismatch_raises():
    """A permutation-type OUTPUT mismatch raises loudly too."""
    sdfg = _build_scale_sdfg('nested_perm_out_mismatch', (6, 4), (4, 1), (4, 6), (1, 4))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


def test_element_count_mismatch_raises():
    """A same-rank different-size mismatch raises (no silent wrong indexing)."""
    sdfg = _build_scale_sdfg('nested_count_mismatch', (5, 4), (4, 1), (6, 4), (4, 1))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


def test_matching_shapes_unchanged():
    """Identical connector/outer shapes need no reconciliation."""
    sdfg = _build_scale_sdfg('nested_matching', (2, 3, 4), (12, 4, 1), (2, 3, 4), (12, 4, 1))
    X = np.random.rand(2, 3, 4)
    Y = np.zeros((2, 3, 4))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X)


def test_inout_connector_flat_reshape():
    """An in/out connector with a flat reshape mismatch reads AND writes back.

    ``_y`` is both input and output of the nested SDFG (``_y = 2*_x + _y``),
    bound through the same contiguous bridge with a copy-back.
    """
    sdfg = dace.SDFG('nested_inout_reshape')
    state = sdfg.add_state()
    sdfg.add_array('X', (2, 3, 4), dace.float64)
    sdfg.add_array('Y', (2, 3, 4), dace.float64)

    nsdfg = dace.SDFG('nested_inout_reshape_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (6, 4), dace.float64, strides=(4, 1))
    nsdfg.add_array('_y', (6, 4), dace.float64, strides=(4, 1))
    t = nstate.add_tasklet('axpy', {'__i', '__c'}, {'__o'}, '__o = 2 * __i + __c')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(nstate.add_read('_y'), None, t, '__c', Memlet.from_array('_y', nsdfg.arrays['_y']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x', '_y'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(state.add_read('Y'), None, node, '_y', Memlet.from_array('Y', sdfg.arrays['Y']))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet.from_array('Y', sdfg.arrays['Y']))
    sdfg.backend = dtypes.BackendLanguage.Python

    X = np.random.rand(2, 3, 4)
    Y0 = np.random.rand(2, 3, 4)
    Y = Y0.copy()
    sdfg(X=X, Y=Y)
    ref = 2 * X + Y0
    assert np.allclose(Y, ref), f"max diff = {np.max(np.abs(Y - ref))}"


def test_column_view_input_output():
    """Column views of a row-major matrix bind as strided views (gramschmidt).

    Connector shape ``(6,)`` with stride ``(4,)`` against the outer subset
    ``[0:6, 1]`` of a ``(6, 4)`` array: reads AND writes go through the view.
    """
    sdfg = _build_nested_scale_sdfg('nested_column_view', (6, 4), (6, 4), '0:6, 1', '0:6, 1', ((6, ), (4, )),
                                    ((6, ), (4, )))
    X = np.random.rand(6, 4)
    Y = np.zeros((6, 4))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y[:, 1], 2 * X[:, 1])
    Y[:, 1] = 0
    assert np.all(Y == 0), 'writes leaked outside the column view'


def test_column_view_symbolic_length():
    """A column view with SYMBOLIC length/stride binds correctly (symm)."""
    M, N = dace.symbol('M'), dace.symbol('N')
    sdfg = _build_nested_scale_sdfg('nested_column_view_sym', (M, N), (M, N),
                                    '0:M, 1',
                                    '0:M, 1', ((M, ), (N, )), ((M, ), (N, )),
                                    symbols=('M', 'N'))
    # Reference M outside data shapes/subsets so the top-level frame keeps it
    # as an argument (``used_symbols(all_symbols=False)`` drops shape-only
    # symbols at the top level).
    sdfg.add_state_after(sdfg.start_state, assignments={'__use_m': 'M'})
    X = np.random.rand(6, 4)
    Y = np.zeros((6, 4))
    sdfg(X=X, Y=Y, M=6, N=4)
    assert np.allclose(Y[:, 1], 2 * X[:, 1])
    Y[:, 1] = 0
    assert np.all(Y == 0)


def test_strided_output_writeback():
    """An OUTPUT connector with non-contiguous outer-dim strides writes back
    through the strided view (conv2d_bias class).

    Connector ``(2, 5)`` with strides ``(60, 1)`` against subset
    ``[0:2, 1, 2, 0:5]`` of a ``(2, 3, 4, 5)`` C-order array.
    """
    sdfg = _build_nested_scale_sdfg('nested_strided_out', (2, 5), (2, 3, 4, 5), '0:2, 0:5', '0:2, 1, 2, 0:5',
                                    ((2, 5), (5, 1)), ((2, 5), (60, 1)))
    X = np.random.rand(2, 5)
    Y = np.zeros((2, 3, 4, 5))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y[:, 1, 2, :], 2 * X)
    Y[:, 1, 2, :] = 0
    assert np.all(Y == 0), 'writes leaked outside the strided view'


def test_transpose_view_supported():
    """A connector layout that is a genuine TRANSPOSE view binds via
    ``.transpose`` instead of raising (extension over the flat-reshape rule).
    """
    sdfg = _build_nested_scale_sdfg('nested_transpose_view', (2, 4), (2, 4), '0:2, 0:4', '0:2, 0:4', ((4, 2), (1, 4)),
                                    ((4, 2), (1, 4)))
    X = np.random.rand(2, 4)
    Y = np.zeros((2, 4))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X)


def test_inconsistent_stride_raises():
    """A NON-squeeze shape mismatch whose strides match no view of the outer
    subset (and are not C-contiguous) still raises loudly.

    Connector ``(4, 6)`` with strides ``(3, 7)`` against the full ``(6, 4)``
    subset: not a squeeze (no size-1 dims; rendered shapes differ), not the
    transpose view (that would be strides ``(1, 4)``), not C-contiguous
    (that would be ``(6, 1)``).

    Note: a pure squeeze/unsqueeze mismatch (equal non-singleton dims in
    order) is accepted REGARDLESS of declared strides — in the Python
    backend the bound runtime view's strides govern indexing, not the
    descriptor metadata (correlation regression,
    ``test_rendered_column_slice_needs_no_reshape``).
    """
    sdfg = _build_nested_scale_sdfg('nested_bad_stride', (6, 4), (6, 4), '0:6, 0:4', '0:6, 0:4', ((4, 6), (3, 7)),
                                    ((6, 4), (4, 1)))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


@pytest.mark.gpu
def test_column_view_gpu_global():
    """The same column-view binding works on GPU_Global (cupy) arrays."""
    import cupy
    sdfg = _build_nested_scale_sdfg('nested_column_view_gpu', (6, 4), (6, 4), '0:6, 1', '0:6, 1', ((6, ), (4, )),
                                    ((6, ), (4, )))
    for arr in sdfg.arrays.values():
        arr.storage = dtypes.StorageType.GPU_Global
    X = np.random.rand(6, 4)
    X_gpu = cupy.asarray(X)
    Y_gpu = cupy.zeros((6, 4))
    sdfg(X=X_gpu, Y=Y_gpu)
    Y = cupy.asnumpy(Y_gpu)
    assert np.allclose(Y[:, 1], 2 * X[:, 1])
    Y[:, 1] = 0
    assert np.all(Y == 0)


def test_rendered_column_slice_needs_no_reshape():
    """A strided column slice whose RENDERED shape matches the connector
    (correlation regression).

    The outer memlet ``X[0:5, 2]`` has subset size ``(5, 1)`` but renders to
    a shape-``(5,)`` view (the size-1 dim collapses to an index), exactly
    matching the 1-D connector ``_x`` with column stride ``(7,)``.  Before
    the fix this raised ``NotImplementedError`` (non-contiguous strides
    rejected the flat reshape that is not actually needed).
    """
    sdfg = dace.SDFG('nested_column_slice')
    state = sdfg.add_state()
    sdfg.add_array('X', (5, 7), dace.float64)
    sdfg.add_array('Y', (5, ), dace.float64)

    nsdfg = dace.SDFG('nested_column_slice_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (5, ), dace.float64, strides=(7, ))
    nsdfg.add_array('_y', (5, ), dace.float64)
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet('X[0:5, 2]'))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet.from_array('Y', sdfg.arrays['Y']))
    sdfg.backend = dtypes.BackendLanguage.Python

    X = np.random.rand(5, 7)
    Y = np.zeros(5)
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X[:, 2])


def test_writeback_through_collapsed_dim():
    """A reshape-bridge writeback into a target with a collapsed size-1 dim
    (mlp regression).

    The outer memlet ``Y[0:6, 0]`` renders to a shape-``(6,)`` target, so
    the bridge writeback must reshape to ``(6,)``, not the raw subset size
    ``(6, 1)`` (which raised a cupy/numpy shape mismatch).
    """
    sdfg = dace.SDFG('nested_collapsed_writeback')
    state = sdfg.add_state()
    sdfg.add_array('X', (2, 3), dace.float64)
    sdfg.add_array('Y', (6, 2), dace.float64)

    nsdfg = dace.SDFG('nested_collapsed_writeback_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (2, 3), dace.float64, strides=(3, 1))
    nsdfg.add_array('_y', (2, 3), dace.float64, strides=(3, 1))
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet('Y[0:6, 0]'))
    sdfg.backend = dtypes.BackendLanguage.Python

    X = np.random.rand(2, 3)
    Y = np.zeros((6, 2))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y[:, 0], (2 * X).reshape(6))
    assert np.allclose(Y[:, 1], 0.0)


def _build_inout_full_in_window_out_sdfg(name: str, use_wcr: bool) -> dace.SDFG:
    """An in/out connector ``_y`` with an explicit-FULL IN subset and a
    zero-based WINDOW OUT subset (the ``ExpandNestedSDFGInputs`` WCR-skip
    shape: widened IN edge, deliberately narrow OUT edge).

    The inner map covers rows ``0:3`` of the ``(4, 6)`` arrays; inner
    subscripts are outer-origin coordinates, so both edges are semantically
    the whole array.

    :param name: SDFG name.
    :param use_wcr: Put a sum WCR on the inner write and the OUT boundary edge
        (accumulate) instead of a plain assignment.
    :returns: The configured (Python-backend) SDFG.
    """
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array('X', (4, 6), dace.float64)
    sdfg.add_array('Y', (4, 6), dace.float64)

    nsdfg = dace.SDFG(name + '_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (4, 6), dace.float64)
    nsdfg.add_array('_y', (4, 6), dace.float64)
    me, mx = nstate.add_map('rows', dict(i='0:3', j='0:6'))
    if use_wcr:
        t = nstate.add_tasklet('acc', {'__i'}, {'__o'}, '__o = 2 * __i')
        nstate.add_memlet_path(nstate.add_read('_x'), me, t, dst_conn='__i', memlet=Memlet('_x[i, j]'))
        nstate.add_memlet_path(t,
                               mx,
                               nstate.add_write('_y'),
                               src_conn='__o',
                               memlet=Memlet('_y[i, j]', wcr='lambda a, b: a + b'))
        # The IN edge exists because the connector is in/out (widened boundary).
        read_y = nstate.add_read('_y')
        t2 = nstate.add_tasklet('touch', {'__c'}, {'__u'}, '__u = __c')
        nsdfg.add_scalar('_unused', dace.float64, transient=True)
        nstate.add_edge(read_y, None, t2, '__c', Memlet('_y[3, 0]'))
        nstate.add_edge(t2, '__u', nstate.add_write('_unused'), None, Memlet('_unused[0]'))
    else:
        t = nstate.add_tasklet('axpy', {'__i', '__c'}, {'__o'}, '__o = 2 * __i + __c')
        nstate.add_memlet_path(nstate.add_read('_x'), me, t, dst_conn='__i', memlet=Memlet('_x[i, j]'))
        nstate.add_memlet_path(nstate.add_read('_y'), me, t, dst_conn='__c', memlet=Memlet('_y[i, j]'))
        nstate.add_memlet_path(t, mx, nstate.add_write('_y'), src_conn='__o', memlet=Memlet('_y[i, j]'))

    node = state.add_nested_sdfg(nsdfg, {'_x', '_y'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet('X[0:4, 0:6]'))
    # Explicit-full IN subset: shapes match, so pre-fix the predicate was never
    # consulted and the edge rendered as a full view expression ...
    state.add_edge(state.add_read('Y'), None, node, '_y', Memlet('Y[0:4, 0:6]'))
    # ... while the zero-based WINDOW OUT subset rendered the bare name ->
    # "bound inconsistently".
    out_memlet = Memlet('Y[0:3, 0:6]', wcr='lambda a, b: a + b') if use_wcr else Memlet('Y[0:3, 0:6]')
    state.add_edge(node, '_y', state.add_write('Y'), None, out_memlet)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def test_inout_full_in_window_out_binds_consistently():
    """Explicit-full IN + zero-based-window OUT on an in/out connector.

    Both edges are semantically the whole array; before consulting the
    whole-array predicate also on shape matches this raised
    ``NotImplementedError: ... bound inconsistently``.
    """
    sdfg = _build_inout_full_in_window_out_sdfg('nested_inout_full_window', use_wcr=False)
    X = np.random.rand(4, 6)
    Y0 = np.random.rand(4, 6)
    Y = Y0.copy()
    sdfg(X=X, Y=Y)
    ref = Y0.copy()
    ref[0:3] = 2 * X[0:3] + Y0[0:3]
    assert np.allclose(Y, ref), f"max diff = {np.max(np.abs(Y - ref))}"


def test_inout_full_in_window_out_wcr():
    """The same mixed-edge shape with a sum WCR OUT edge (the realistic
    ``ExpandNestedSDFGInputs`` producer: widened IN, WCR-skipped narrow OUT).
    Accumulation is realized at the inner write site; boundary WCR is
    metadata."""
    sdfg = _build_inout_full_in_window_out_sdfg('nested_inout_full_window_wcr', use_wcr=True)
    X = np.random.rand(4, 6)
    Y0 = np.random.rand(4, 6)
    Y = Y0.copy()
    sdfg(X=X, Y=Y)
    ref = Y0.copy()
    ref[0:3] += 2 * X[0:3]
    assert np.allclose(Y, ref), f"max diff = {np.max(np.abs(Y - ref))}"


class TestNestedDescSpansFullOuter:
    """Unit tests for ``PythonCodeGen._nested_desc_spans_full_outer``.

    The whole-outer-array shortcut must only fire for a connector declared
    over the entire outer array with a zero-based, step-1 window subset and
    no inner-side (``other_subset``) remapping.
    """

    @staticmethod
    def _codegen() -> PythonCodeGen:
        sdfg = dace.SDFG('spans_full_outer_unit')
        sdfg.backend = dtypes.BackendLanguage.Python
        return PythonCodeGen(DaCePythonCodeGenerator(sdfg), sdfg)

    @staticmethod
    def _descs(inner_dtype=dace.float64, outer_dtype=dace.float64, inner_shape=(4, 5), outer_shape=(4, 5)):
        return data.Array(inner_dtype, inner_shape), data.Array(outer_dtype, outer_shape)

    def test_accepts_zero_based_step1_window(self):
        inner, outer = self._descs()
        assert self._codegen()._nested_desc_spans_full_outer(Memlet('A[0:3, 0:5]'), inner, outer)

    def test_rejects_dtype_mismatch(self):
        inner, outer = self._descs(outer_dtype=dace.float32)
        assert not self._codegen()._nested_desc_spans_full_outer(Memlet('A[0:3, 0:5]'), inner, outer)

    def test_rejects_ndim_mismatch(self):
        inner, outer = self._descs(outer_shape=(4, 5, 6))
        assert not self._codegen()._nested_desc_spans_full_outer(Memlet('A[0:3, 0:5, 0:6]'), inner, outer)

    def test_rejects_nonzero_start(self):
        inner, outer = self._descs()
        assert not self._codegen()._nested_desc_spans_full_outer(Memlet('A[1:4, 0:5]'), inner, outer)

    def test_rejects_non_unit_step(self):
        inner, outer = self._descs()
        assert not self._codegen()._nested_desc_spans_full_outer(Memlet('A[0:4:2, 0:5]'), inner, outer)

    def test_rejects_nonzero_single_element_subset(self):
        inner, outer = self._descs()
        memlet = Memlet(data='A', subset=subsets.Range([(2, 2, 1), (3, 3, 1)]))
        assert not self._codegen()._nested_desc_spans_full_outer(memlet, inner, outer)

    def test_rejects_other_subset(self):
        inner, outer = self._descs()
        memlet = Memlet('A[0:3, 0:5]')
        memlet.other_subset = subsets.Range.from_string('0:3, 0:5')
        assert not self._codegen()._nested_desc_spans_full_outer(memlet, inner, outer)

    def test_rejects_view_outer_desc(self):
        """A View outer descriptor is not the array itself; the shortcut must
        stay sound even if ``preprocess``'s ``RemoveViews`` did not run."""
        inner, _ = self._descs()
        outer_view = data.ArrayView(dace.float64, (4, 5))
        assert not self._codegen()._nested_desc_spans_full_outer(Memlet('A[0:3, 0:5]'), inner, outer_view)


if __name__ == '__main__':
    pytest.main(['-q', __file__])
