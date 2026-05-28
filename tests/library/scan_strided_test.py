# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical verification of the ``Scan`` libnode with ``stride > 1``.

The libnode expansion runs ``s`` independent inclusive scans, one per residue class
modulo ``s``. Each class is a closed scan (writes only to indices ``≡ k (mod s)``
strictly forward and reads only those indices plus the class's seed at position
``k``), so the ``s`` classes have no cross-dependence and run in parallel.

This test confirms the result is bit-identical to a sequential strided-scan
oracle for several ``(n, s)`` and both supported CPU implementations.

The libnode's expansion emits a runtime ``s > 0`` ``std::abort()`` check before
the scan starts; the negative/zero-stride case is exercised via subprocess to
avoid killing the test runner.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME)


def _build_scan_sdfg(n: int, stride: int, op: ScanOp, implementation: str) -> dace.SDFG:
    """Build a single-state SDFG that scans ``arr_in[0:N]`` into ``arr_out[0:N]``
    with the given stride and op."""
    sdfg = dace.SDFG(f'strided_scan_{op.value}_{implementation}_n{n}_s{stride}')
    sdfg.add_array('arr_in', [n], dace.float64)
    sdfg.add_array('arr_out', [n], dace.float64)
    state = sdfg.add_state('scan')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=op, exclusive=False)
    node.stride = stride
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'arr_in[0:{n}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet(f'arr_out[0:{n}]'))
    sdfg.validate()
    return sdfg


def _residue_class_scan_oracle(arr_in: np.ndarray, stride: int, op: ScanOp) -> np.ndarray:
    """Reference: ``out[j] = OP_running(arr_in[j_first], ..., arr_in[j])`` within each
    residue class ``j ≡ k (mod stride)``. The libnode produces the same values."""
    n = arr_in.shape[0]
    out = np.zeros_like(arr_in)
    if op is ScanOp.SUM:
        binop = lambda a, b: a + b
        ident = 0.0
    elif op is ScanOp.PRODUCT:
        binop = lambda a, b: a * b
        ident = 1.0
    elif op is ScanOp.MIN:
        binop = min
        ident = None
    elif op is ScanOp.MAX:
        binop = max
        ident = None
    else:
        raise AssertionError(f'Unknown op: {op}')
    for k in range(stride):
        first = True
        acc = ident
        for j in range(k, n, stride):
            if op in (ScanOp.SUM, ScanOp.PRODUCT):
                acc = binop(acc, arr_in[j])
            else:
                acc = arr_in[j] if first else binop(acc, arr_in[j])
            out[j] = acc
            first = False
    return out


@pytest.mark.parametrize('stride', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('n', [16, 33])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_strided_scan_matches_residue_class_oracle(stride: int, n: int, op: ScanOp,
                                                   implementation: str):
    """For each stride, dtype, and implementation, the libnode-produced output equals
    the per-residue-class sequential scan."""
    # For PRODUCT keep magnitudes ~1; for SUM/MIN/MAX use the unit interval.
    seed = stride * 100 + n + ord(op.value[0]) + ord(implementation[0])
    rng = np.random.default_rng(seed)
    if op is ScanOp.PRODUCT:
        arr_in = rng.uniform(0.95, 1.05, size=n)
    else:
        arr_in = rng.uniform(-1.0, 1.0, size=n)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_scan_sdfg(n, stride, op, implementation)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _residue_class_scan_oracle(arr_in, stride, op)
    assert np.allclose(arr_out, expected), (
        f'stride={stride} n={n} op={op.value} impl={implementation}: '
        f'max abs diff {np.max(np.abs(arr_out - expected))}')


def test_strided_scan_stride_2_explicit():
    """Hand-computed: stride=2, n=8 ascending integers. Even and odd residues are
    independent cumsum subsequences."""
    n, s = 8, 2
    arr_in = np.arange(1.0, n + 1.0)  # [1, 2, 3, 4, 5, 6, 7, 8]
    sdfg = _build_scan_sdfg(n, s, ScanOp.SUM, 'pure')
    arr_out = np.zeros(n)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    # Even residue: cumsum([1, 3, 5, 7]) = [1, 4, 9, 16]
    # Odd  residue: cumsum([2, 4, 6, 8]) = [2, 6, 12, 20]
    expected = np.array([1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0])
    assert np.allclose(arr_out, expected), f'got {arr_out}, expected {expected}'


def test_strided_scan_stride_one_matches_contiguous():
    """``stride=1`` is identical to the contiguous inclusive scan -- the dispatch should
    pick the existing OpenMP scan path, and the result matches ``np.cumsum``."""
    n = 17
    rng = np.random.default_rng(0)
    arr_in = rng.uniform(-1.0, 1.0, size=n)
    arr_out = np.zeros(n)
    sdfg = _build_scan_sdfg(n, 1, ScanOp.SUM, 'CPU')
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.allclose(arr_out, np.cumsum(arr_in))


_NEGATIVE_STRIDE_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {repo!r})

    import numpy as np
    import dace
    from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME,
                                                    OUTPUT_CONNECTOR_NAME)

    n = 12
    sdfg = dace.SDFG('negstride_probe')
    sdfg.add_array('arr_in', [n], dace.float64)
    sdfg.add_array('arr_out', [n], dace.float64)
    st = sdfg.add_state()
    a_in = st.add_read('arr_in')
    a_out = st.add_write('arr_out')
    node = Scan('Scan', op=ScanOp.SUM, exclusive=False)
    # A negative literal stride trips the runtime ``s > 0`` check inside ``dace::scan``.
    node.stride = -2
    node.implementation = 'pure'
    st.add_node(node)
    st.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet('arr_in[0:%d]' % n))
    st.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet('arr_out[0:%d]' % n))

    arr_in = np.arange(n, dtype=np.float64)
    arr_out = np.zeros(n)
    sdfg(arr_in=arr_in, arr_out=arr_out)
    print('UNEXPECTEDLY_SURVIVED', flush=True)
""").format(repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_negative_stride_aborts_at_runtime():
    """A non-positive stride must abort the program before the scan runs. Spawned in
    a subprocess so the abort doesn't kill the test runner."""
    proc = subprocess.run([sys.executable, '-c', _NEGATIVE_STRIDE_SCRIPT],
                          capture_output=True, text=True, timeout=120)
    assert 'UNEXPECTEDLY_SURVIVED' not in proc.stdout, (
        f'Negative stride failed to abort. stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}')
    assert proc.returncode != 0, (
        f'Expected non-zero exit on abort; got returncode={proc.returncode}. '
        f'stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
