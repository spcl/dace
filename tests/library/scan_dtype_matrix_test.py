# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the three dtype gaps closed in ``dace/runtime/include/dace/scan.hpp``'s
CPU (blocked parallel) lowering. ``tests/library/reduce_scan_dtype_matrix_test.py`` covers the
working matrix; the cells here were the ones missing from it because they used to fail:

  S1  complex64 / complex128 inclusive sum / product, ``CPU`` implementation -> used to be a
      CompilationError ("user defined reduction not found"): the complex ``+`` / ``*`` OpenMP
      user-defined reductions were declared in ``namespace dace::reduce`` (reduction.h), a
      namespace ``dace::scan::detail`` cannot see; they are now declared in
      ``dace::scan::detail`` itself.
  S2  float16 / bfloat16 inclusive max, ``CPU`` implementation -> used to be a CompilationError:
      ``max_identity``'s ternary arms had types ``float`` (unary minus on the struct converts
      through ``operator float()``) and ``T`` (``lowest()``), which have no common type.
  S3  complex min / max, any dtype -> now a named compile-time rejection
      (``dace::scan::detail::is_ordered``) instead of raw ``std::min`` / ``numeric_limits``
      operator noise. Only the ``CPU`` implementation routes through ``scan.hpp`` -- ``pure`` is
      a hand-written loop with no call into the header (see the header's own top-of-file note),
      so it is not part of this rejection's contract and is not exercised here.

Both S1 and S2 are run at a small and a large N. The entry point no longer branches on the size
-- whether a scan earns a team is settled at compile time by the specialization band -- so both
sizes reach the SAME blocked code, which is exactly why the pair is still worth running: the small
one covers a team whose blocks are shorter than the tile, the large one covers several tiles, and
both need the same UDR / identity to compile at all.
"""
import numpy as np
import pytest

import dace
from dace import memlet as mm
from dace.codegen.exceptions import CompilationError
from dace.libraries.standard.nodes.scan import Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, INIT_CONNECTOR_NAME

_LOWP = (dace.float16, dace.bfloat16)
_COMPLEX = (dace.complex64, dace.complex128)

#: One N shorter than a single tile, one spanning several.
_N_BELOW = 64
_N_ABOVE = 40000

_RTOL = {'float16': 1e-2, 'bfloat16': 1e-1, 'complex64': 1e-6, 'complex128': 1e-12}


def _tol(dtype):
    return _RTOL[dtype.to_string()]


def _scan_sdfg(dtype, op, n, implementation='CPU', with_init=False):
    sdfg = dace.SDFG('scangap_%s_%s_%d_%d' % (op.value, dtype.to_string().replace(':', '_'), n, with_init))
    sdfg.add_array('arr_in', [n], dtype)
    sdfg.add_array('arr_out', [n], dtype)
    state = sdfg.add_state()
    node = Scan('Scan', op=op, exclusive=False)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('arr_in'), None, node, INPUT_CONNECTOR_NAME, mm.Memlet('arr_in[0:%d]' % n))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('arr_out'), None, mm.Memlet('arr_out[0:%d]' % n))
    if with_init:
        sdfg.add_array('seed', [1], dtype)
        node.add_in_connector(INIT_CONNECTOR_NAME)
        state.add_edge(state.add_read('seed'), None, node, INIT_CONNECTOR_NAME, mm.Memlet('seed[0]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('n', [_N_BELOW, _N_ABOVE], ids=['below_threshold', 'above_threshold'])
@pytest.mark.parametrize('dtype', _COMPLEX, ids=lambda d: d.to_string())
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT])
def test_scan_complex_parallel_shape(op, dtype, n):
    """GAP S1. A unit-modulus factor keeps the product bounded over 40000 elements."""
    np_t = dtype.as_numpy_dtype()
    if op is ScanOp.SUM:
        arr = np.full(n, 1.0 + 0.25j, np_t)
    else:
        arr = np.ones(n, np_t)
        arr[2] = 1j
    out = np.zeros(n, np_t)
    _scan_sdfg(dtype, op, n)(arr_in=arr.copy(), arr_out=out)
    want = (np.cumsum if op is ScanOp.SUM else np.cumprod)(arr.astype(np.complex128))
    assert np.allclose(out.astype(np.complex128), want, rtol=_tol(dtype))


@pytest.mark.parametrize('dtype', _COMPLEX, ids=lambda d: d.to_string())
def test_scan_complex_sum_seeded_parallel_shape(dtype):
    """GAP S1, seeded overload: ``_scan_init`` wired in, above the parallel threshold."""
    np_t = dtype.as_numpy_dtype()
    n = _N_ABOVE
    arr = np.full(n, 1.0 + 0.25j, np_t)
    seed = np.array([0.5 - 0.5j], np_t)
    out = np.zeros(n, np_t)
    _scan_sdfg(dtype, ScanOp.SUM, n, with_init=True)(arr_in=arr.copy(), arr_out=out, seed=seed)
    want = seed[0].astype(np.complex128) + np.cumsum(arr.astype(np.complex128))
    assert np.allclose(out.astype(np.complex128), want, rtol=_tol(dtype))


@pytest.mark.parametrize('n', [_N_BELOW, _N_ABOVE], ids=['below_threshold', 'above_threshold'])
@pytest.mark.parametrize('dtype', _LOWP, ids=lambda d: d.to_string())
def test_scan_low_precision_max_parallel_shape(dtype, n):
    """GAP S2. Values stay dyadic and small, so the running maximum is exact in fp16 and bf16."""
    np_t = dtype.as_numpy_dtype()
    arr = (1.0 + 0.5 * (np.arange(n) % 5)).astype(np_t)
    out = np.zeros(n, np_t)
    _scan_sdfg(dtype, ScanOp.MAX, n)(arr_in=arr.copy(), arr_out=out)
    want = np.maximum.accumulate(arr.astype(np.float64))
    assert np.allclose(out.astype(np.float64), want, rtol=_tol(dtype))


@pytest.mark.parametrize('dtype', _COMPLEX, ids=lambda d: d.to_string())
@pytest.mark.parametrize('op', [ScanOp.MIN, ScanOp.MAX])
def test_scan_names_the_complex_ordering_rejection(op, dtype):
    """GAP S3. Complex has no ordering; the refusal must be the runtime's own diagnostic.

    ``CPU`` only: ``pure`` is a hand-written loop that never calls into ``scan.hpp``, so it is
    not this rejection's contract (see the module docstring).
    """
    with pytest.raises(CompilationError, match='needs an ordered element type'):
        _scan_sdfg(dtype, op, _N_BELOW, implementation='CPU').compile()
