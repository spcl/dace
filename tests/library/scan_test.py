# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.libraries.standard.nodes.scan.Scan`.

Covers inclusive / exclusive scans, all supported ops (``SUM``, ``PRODUCT``,
``MIN``, ``MAX``), several integer and floating dtypes, and the ``CPU`` /
``pure`` implementations. ``CUDA`` is not exercised in this CPU-only file; it
would need a GPU test environment.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME)


def _safe_label(s: str) -> str:
    return s.replace(':', '_').replace(' ', '_')


def _build_scan_sdfg(dace_dtype: dace.dtypes.typeclass, n: int, op: ScanOp, exclusive: bool,
                     implementation: str, identity=None) -> dace.SDFG:
    """Build a single-state SDFG that scans ``arr_in[0:N]`` into ``arr_out[0:N]``."""
    name = f"scan_{op.value}_{int(exclusive)}_{implementation}_{_safe_label(dace_dtype.to_string())}_{n}"
    sdfg = dace.SDFG(name)
    sdfg.add_array('arr_in', [n], dace_dtype)
    sdfg.add_array('arr_out', [n], dace_dtype)
    state = sdfg.add_state('scan')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=op, exclusive=exclusive, identity=identity)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'arr_in[0:{n}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet(f'arr_out[0:{n}]'))
    sdfg.validate()
    return sdfg


def _numpy_inclusive(arr, op: ScanOp):
    if op is ScanOp.SUM:
        return np.cumsum(arr)
    if op is ScanOp.PRODUCT:
        return np.cumprod(arr)
    if op is ScanOp.MIN:
        return np.minimum.accumulate(arr)
    if op is ScanOp.MAX:
        return np.maximum.accumulate(arr)
    raise AssertionError(f'Unknown op: {op}')


def _numpy_exclusive(arr, op: ScanOp, identity):
    """Exclusive scan: out[0] = identity, out[k] = identity OP in[0] OP ... OP in[k-1]."""
    out = np.empty_like(arr)
    out[0] = identity
    if len(arr) == 1:
        return out
    incl = _numpy_inclusive(arr, op)
    out[1:] = incl[:-1] if op is ScanOp.SUM else incl[:-1]
    if op is ScanOp.SUM:
        out[1:] = identity + incl[:-1]
    elif op is ScanOp.PRODUCT:
        out[1:] = identity * incl[:-1]
    elif op is ScanOp.MIN:
        out[1:] = np.minimum(identity, incl[:-1])
    elif op is ScanOp.MAX:
        out[1:] = np.maximum(identity, incl[:-1])
    return out


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
def test_scan_inclusive_matches_numpy(op: ScanOp, implementation: str):
    """Inclusive scan over float64 matches numpy's cum* / accumulate."""
    n = 64
    rng = np.random.default_rng(int(op.value.encode().hex(), 16) & 0xFFFF)
    if op is ScanOp.PRODUCT:
        arr_in = rng.uniform(0.95, 1.05, size=n)  # keep magnitudes finite
    else:
        arr_in = rng.uniform(-1.0, 1.0, size=n)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_scan_sdfg(dace.float64, n, op, exclusive=False, implementation=implementation)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _numpy_inclusive(arr_in, op).astype(np.float64)
    assert np.allclose(arr_out, expected), (
        f'{op.value} inclusive scan mismatch on {implementation}; '
        f'max diff {np.max(np.abs(arr_out - expected))}.')


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_inclusive_sum_int32(implementation: str):
    """Integer dtype inclusive sum scan -- exact equality, no floating tolerance."""
    n = 50
    arr_in = np.arange(1, n + 1, dtype=np.int32)
    arr_out = np.zeros(n, dtype=np.int32)
    sdfg = _build_scan_sdfg(dace.int32, n, ScanOp.SUM, exclusive=False, implementation=implementation)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.array_equal(arr_out, np.cumsum(arr_in))


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_exclusive_sum_with_seed(implementation: str):
    """Exclusive sum scan with a non-zero seed identity -- prefix-then-shift semantics."""
    n = 32
    arr_in = np.arange(1, n + 1, dtype=np.float64)
    arr_out = np.zeros(n, dtype=np.float64)
    seed = 10.0
    sdfg = _build_scan_sdfg(dace.float64, n, ScanOp.SUM, exclusive=True, implementation=implementation,
                            identity=seed)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _numpy_exclusive(arr_in, ScanOp.SUM, seed)
    assert np.allclose(arr_out, expected)


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_single_element(implementation: str):
    """A length-1 inclusive scan returns the single element; exclusive returns the seed."""
    arr_in = np.array([3.5], dtype=np.float64)
    arr_out = np.zeros(1, dtype=np.float64)
    inc_sdfg = _build_scan_sdfg(dace.float64, 1, ScanOp.SUM, exclusive=False, implementation=implementation)
    inc_sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert arr_out[0] == arr_in[0]

    arr_out = np.zeros(1, dtype=np.float64)
    exc_sdfg = _build_scan_sdfg(dace.float64, 1, ScanOp.SUM, exclusive=True, implementation=implementation,
                                identity=7.0)
    exc_sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert arr_out[0] == 7.0


def test_scan_refuses_dtype_mismatch():
    """Validation rejects an output array whose dtype differs from the input's."""
    sdfg = dace.SDFG('scan_refuses_mismatch')
    sdfg.add_array('arr_in', [8], dace.float64)
    sdfg.add_array('arr_out', [8], dace.float32)
    state = sdfg.add_state('scan')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=ScanOp.SUM)
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet('arr_in[0:8]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet('arr_out[0:8]'))
    with pytest.raises(Exception):
        sdfg.compile()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
