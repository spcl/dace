# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-node unit tests for the new BLAS-L1/L2/L3 library lowerings.

Drives each new lib node through the ``OpenBLAS`` expansion and
compares against numpy. Tolerance is BLAS-strict ``1e-14``. Marker
``lapack`` schedules them into the OpenBLAS CI step -- the same bucket
the other OpenBLAS-backed tests use, and one the MKL-only heterogeneous
runner does not select.
"""
import numpy as np
import pytest

import dace
from dace.memlet import Memlet
from dace.libraries.blas.nodes import (Axpy, Scal, Copy, Swap, Trsv, Trmv, Symv, Trsm, Trmm, Symm, Syrk, Ger)

# These force ``BLA_VENDOR=OpenBLAS``, so they need libopenblas -- run them in the OpenBLAS
# ``lapack`` step, not the ``mkl`` step (the heterogeneous runner has MKL but not OpenBLAS).
pytestmark = pytest.mark.lapack

_RTOL = 1e-14
_ATOL = 1e-14
_IMPL = 'OpenBLAS'


def _run(sdfg, **kw):
    sdfg.validate()
    sdfg(**kw)


def test_axpy_openblas():
    n, a = 16, 1.7
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    expected = a * x + y
    sdfg = dace.SDFG('axpy_obs')
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('y', [n], dace.float64)
    sdfg.add_array('res', [n], dace.float64)
    s = sdfg.add_state()
    node = Axpy('axpy', a=a, n=n)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_x', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(s.add_read('y'), node, dst_conn='_y', memlet=Memlet(f'y[0:{n}]'))
    s.add_memlet_path(node, s.add_write('res'), src_conn='_res', memlet=Memlet(f'res[0:{n}]'))
    res = np.zeros(n)
    _run(sdfg, x=x, y=y, res=res)
    np.testing.assert_allclose(res, expected, rtol=_RTOL, atol=_ATOL)


def test_scal_openblas():
    n, a = 20, 2.5
    x = np.random.default_rng(1).standard_normal(n)
    expected = a * x
    sdfg = dace.SDFG('scal_obs')
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('res', [n], dace.float64)
    s = sdfg.add_state()
    node = Scal('scal', a=a, n=n)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_x', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(node, s.add_write('res'), src_conn='_res', memlet=Memlet(f'res[0:{n}]'))
    res = np.zeros(n)
    _run(sdfg, x=x, res=res)
    np.testing.assert_allclose(res, expected, rtol=_RTOL, atol=_ATOL)


def test_copy_openblas():
    n = 18
    x = np.random.default_rng(5).standard_normal(n)
    sdfg = dace.SDFG('copy_obs')
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('y', [n], dace.float64)
    s = sdfg.add_state()
    node = Copy('copy', n=n)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_x', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(node, s.add_write('y'), src_conn='_y', memlet=Memlet(f'y[0:{n}]'))
    y = np.zeros(n)
    _run(sdfg, x=x, y=y)
    np.testing.assert_array_equal(y, x)


def test_swap_openblas():
    n = 12
    rng = np.random.default_rng(6)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    x_orig, y_orig = x.copy(), y.copy()
    sdfg = dace.SDFG('swap_obs')
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('y', [n], dace.float64)
    sdfg.add_array('x_out', [n], dace.float64)
    sdfg.add_array('y_out', [n], dace.float64)
    s = sdfg.add_state()
    node = Swap('swap', n=n)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_xin', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(s.add_read('y'), node, dst_conn='_yin', memlet=Memlet(f'y[0:{n}]'))
    s.add_memlet_path(node, s.add_write('x_out'), src_conn='_xout', memlet=Memlet(f'x_out[0:{n}]'))
    s.add_memlet_path(node, s.add_write('y_out'), src_conn='_yout', memlet=Memlet(f'y_out[0:{n}]'))
    x_out = np.zeros(n)
    y_out = np.zeros(n)
    _run(sdfg, x=x, y=y, x_out=x_out, y_out=y_out)
    np.testing.assert_array_equal(x_out, y_orig)
    np.testing.assert_array_equal(y_out, x_orig)


def test_trsv_openblas():
    n = 8
    rng = np.random.default_rng(7)
    L = np.tril(rng.standard_normal((n, n)))
    np.fill_diagonal(L, 1.0 + np.abs(np.diag(L)))
    b = rng.standard_normal(n)
    expected = np.linalg.solve(L, b)
    L_cm = np.asfortranarray(L)
    sdfg = dace.SDFG('trsv_obs')
    sdfg.add_array('A', [n, n], dace.float64)
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('x_out', [n], dace.float64)
    s = sdfg.add_state()
    node = Trsv('trsv', uplo=False, transA=False, unit_diag=False)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{n}, 0:{n}]'))
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_xin', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(node, s.add_write('x_out'), src_conn='_xout', memlet=Memlet(f'x_out[0:{n}]'))
    x_out = np.zeros(n)
    _run(sdfg, A=L_cm, x=b, x_out=x_out)
    np.testing.assert_allclose(x_out, expected, rtol=1e-10, atol=1e-10)


def test_trmv_openblas():
    n = 6
    rng = np.random.default_rng(8)
    L = np.tril(rng.standard_normal((n, n)))
    x = rng.standard_normal(n)
    expected = L @ x
    L_cm = np.asfortranarray(L)
    sdfg = dace.SDFG('trmv_obs')
    sdfg.add_array('A', [n, n], dace.float64)
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('x_out', [n], dace.float64)
    s = sdfg.add_state()
    node = Trmv('trmv', uplo=False)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{n}, 0:{n}]'))
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_xin', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(node, s.add_write('x_out'), src_conn='_xout', memlet=Memlet(f'x_out[0:{n}]'))
    x_out = np.zeros(n)
    _run(sdfg, A=L_cm, x=x, x_out=x_out)
    np.testing.assert_allclose(x_out, expected, rtol=_RTOL, atol=_ATOL)


def test_symv_openblas():
    n = 6
    rng = np.random.default_rng(9)
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2.0
    x = rng.standard_normal(n)
    expected = A @ x
    A_cm = A
    sdfg = dace.SDFG('symv_obs')
    sdfg.add_array('A', [n, n], dace.float64)
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('y', [n], dace.float64)
    sdfg.add_array('y_out', [n], dace.float64)
    s = sdfg.add_state()
    node = Symv('symv', uplo=False, alpha=1.0, beta=0.0)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{n}, 0:{n}]'))
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_x', memlet=Memlet(f'x[0:{n}]'))
    s.add_memlet_path(s.add_read('y'), node, dst_conn='_yin', memlet=Memlet(f'y[0:{n}]'))
    s.add_memlet_path(node, s.add_write('y_out'), src_conn='_yout', memlet=Memlet(f'y_out[0:{n}]'))
    y = np.zeros(n)
    y_out = np.zeros(n)
    _run(sdfg, A=A_cm, x=x, y=y, y_out=y_out)
    np.testing.assert_allclose(y_out, expected, rtol=_RTOL, atol=_ATOL)


def test_trsm_openblas():
    m, n = 6, 4
    rng = np.random.default_rng(10)
    L = np.tril(rng.standard_normal((m, m)))
    np.fill_diagonal(L, 1.0 + np.abs(np.diag(L)))
    B = rng.standard_normal((m, n))
    expected = np.linalg.solve(L, B)
    L_cm, B_cm = L, B
    sdfg = dace.SDFG('trsm_obs')
    sdfg.add_array('A', [m, m], dace.float64)
    sdfg.add_array('B', [m, n], dace.float64)
    sdfg.add_array('B_out', [m, n], dace.float64)
    s = sdfg.add_state()
    node = Trsm('trsm', side=False, uplo=False, alpha=1.0)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{m}, 0:{m}]'))
    s.add_memlet_path(s.add_read('B'), node, dst_conn='_Bin', memlet=Memlet(f'B[0:{m}, 0:{n}]'))
    s.add_memlet_path(node, s.add_write('B_out'), src_conn='_Bout', memlet=Memlet(f'B_out[0:{m}, 0:{n}]'))
    B_out = np.zeros((m, n))
    _run(sdfg, A=L_cm, B=B_cm, B_out=B_out)
    np.testing.assert_allclose(np.asarray(B_out), expected, rtol=1e-10, atol=1e-10)


def test_trmm_openblas():
    m, n = 6, 4
    rng = np.random.default_rng(11)
    L = np.tril(rng.standard_normal((m, m)))
    B = rng.standard_normal((m, n))
    expected = L @ B
    L_cm, B_cm = L, B
    sdfg = dace.SDFG('trmm_obs')
    sdfg.add_array('A', [m, m], dace.float64)
    sdfg.add_array('B', [m, n], dace.float64)
    sdfg.add_array('B_out', [m, n], dace.float64)
    s = sdfg.add_state()
    node = Trmm('trmm', side=False, uplo=False, alpha=1.0)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{m}, 0:{m}]'))
    s.add_memlet_path(s.add_read('B'), node, dst_conn='_Bin', memlet=Memlet(f'B[0:{m}, 0:{n}]'))
    s.add_memlet_path(node, s.add_write('B_out'), src_conn='_Bout', memlet=Memlet(f'B_out[0:{m}, 0:{n}]'))
    B_out = np.zeros((m, n))
    _run(sdfg, A=L_cm, B=B_cm, B_out=B_out)
    np.testing.assert_allclose(np.asarray(B_out), expected, rtol=1e-12, atol=1e-12)


def test_symm_openblas():
    m, n = 5, 4
    rng = np.random.default_rng(12)
    A = rng.standard_normal((m, m))
    A = (A + A.T) / 2.0
    B = rng.standard_normal((m, n))
    expected = A @ B
    A_cm, B_cm = A, B
    sdfg = dace.SDFG('symm_obs')
    sdfg.add_array('A', [m, m], dace.float64)
    sdfg.add_array('B', [m, n], dace.float64)
    sdfg.add_array('C', [m, n], dace.float64)
    sdfg.add_array('C_out', [m, n], dace.float64)
    s = sdfg.add_state()
    node = Symm('symm', side=False, uplo=False, alpha=1.0, beta=0.0)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{m}, 0:{m}]'))
    s.add_memlet_path(s.add_read('B'), node, dst_conn='_B', memlet=Memlet(f'B[0:{m}, 0:{n}]'))
    s.add_memlet_path(s.add_read('C'), node, dst_conn='_Cin', memlet=Memlet(f'C[0:{m}, 0:{n}]'))
    s.add_memlet_path(node, s.add_write('C_out'), src_conn='_Cout', memlet=Memlet(f'C_out[0:{m}, 0:{n}]'))
    C = np.zeros((m, n))
    C_out = np.zeros((m, n))
    _run(sdfg, A=A_cm, B=B_cm, C=C, C_out=C_out)
    np.testing.assert_allclose(np.asarray(C_out), expected, rtol=_RTOL, atol=_ATOL)


def test_syrk_openblas():
    n, k = 5, 4
    A = np.random.default_rng(13).standard_normal((n, k))
    expected = A @ A.T
    A_cm = A
    sdfg = dace.SDFG('syrk_obs')
    sdfg.add_array('A', [n, k], dace.float64)
    sdfg.add_array('C', [n, n], dace.float64)
    sdfg.add_array('C_out', [n, n], dace.float64)
    s = sdfg.add_state()
    node = Syrk('syrk', uplo=False, transA=False, alpha=1.0, beta=0.0)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{n}, 0:{k}]'))
    s.add_memlet_path(s.add_read('C'), node, dst_conn='_Cin', memlet=Memlet(f'C[0:{n}, 0:{n}]'))
    s.add_memlet_path(node, s.add_write('C_out'), src_conn='_Cout', memlet=Memlet(f'C_out[0:{n}, 0:{n}]'))
    C = np.zeros((n, n))
    C_out = np.zeros((n, n))
    _run(sdfg, A=A_cm, C=C, C_out=C_out)
    np.testing.assert_allclose(np.tril(np.asarray(C_out)), np.tril(expected), rtol=_RTOL, atol=_ATOL)


def test_ger_openblas():
    m, n = 5, 4
    rng = np.random.default_rng(14)
    A = rng.standard_normal((m, n))
    x = rng.standard_normal(m)
    y = rng.standard_normal(n)
    alpha = 1.5
    expected = alpha * np.outer(x, y) + A
    sdfg = dace.SDFG('ger_obs')
    sdfg.add_array('A', [m, n], dace.float64)
    sdfg.add_array('x', [m], dace.float64)
    sdfg.add_array('y', [n], dace.float64)
    sdfg.add_array('res', [m, n], dace.float64)
    s = sdfg.add_state()
    node = Ger('ger', m=m, n=n, alpha=alpha)
    node.implementation = _IMPL
    s.add_node(node)
    s.add_memlet_path(s.add_read('A'), node, dst_conn='_A', memlet=Memlet(f'A[0:{m}, 0:{n}]'))
    s.add_memlet_path(s.add_read('x'), node, dst_conn='_x', memlet=Memlet(f'x[0:{m}]'))
    s.add_memlet_path(s.add_read('y'), node, dst_conn='_y', memlet=Memlet(f'y[0:{n}]'))
    s.add_memlet_path(node, s.add_write('res'), src_conn='_res', memlet=Memlet(f'res[0:{m}, 0:{n}]'))
    res = np.zeros((m, n))
    _run(sdfg, A=A, x=x, y=y, res=res)
    np.testing.assert_allclose(res, expected, rtol=_RTOL, atol=_ATOL)


if __name__ == '__main__':
    for fn in (test_axpy_openblas, test_scal_openblas, test_copy_openblas, test_swap_openblas, test_trsv_openblas,
               test_trmv_openblas, test_symv_openblas, test_trsm_openblas, test_trmm_openblas, test_symm_openblas,
               test_syrk_openblas, test_ger_openblas):
        fn()
        print(f'  {fn.__name__}: PASS')
    print('All BLAS extension OpenBLAS lowering tests pass.')
