# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-node unit tests for the new LAPACK library lowerings.

Drives :class:`Potrs`, :class:`Geqrf`, :class:`Orgqr` through the
``OpenBLAS`` (LAPACKE) expansion and compares against numpy.
"""
import numpy as np
import pytest

import dace
from dace.memlet import Memlet
from dace.libraries.lapack.nodes import Potrf, Potrs, Geqrf, Orgqr

_RTOL = 1e-10
_ATOL = 1e-10
_IMPL = 'OpenBLAS'


@pytest.mark.lapack
def test_potrs_openblas():
    """POTRF + POTRS round-trip: solve ``A X = B`` where ``A = L L^T``."""
    n, nrhs = 6, 3
    rng = np.random.default_rng(15)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)  # SPD, well-conditioned
    B = rng.standard_normal((n, nrhs))
    expected = np.linalg.solve(A, B)

    sdfg = dace.SDFG('potrs_obs')
    sdfg.add_array('A', [n, n], dace.float64)
    sdfg.add_array('B', [n, nrhs], dace.float64)
    sdfg.add_array('X', [n, nrhs], dace.float64)
    sdfg.add_array('info_f', [1], dace.int32)
    sdfg.add_array('info_s', [1], dace.int32)
    s = sdfg.add_state()
    factor = Potrf('potrf', lower=True, n=n)
    factor.implementation = _IMPL
    solver = Potrs('potrs', lower=True)
    solver.implementation = _IMPL
    s.add_node(factor)
    s.add_node(solver)
    A_in = s.add_read('A')
    s.add_memlet_path(A_in, factor, dst_conn='_xin', memlet=Memlet(f'A[0:{n}, 0:{n}]'))
    s.add_memlet_path(factor, s.add_write('info_f'), src_conn='_res', memlet=Memlet('info_f[0]'))
    factor_A = s.add_write('A')
    s.add_memlet_path(factor, factor_A, src_conn='_xout', memlet=Memlet(f'A[0:{n}, 0:{n}]'))
    A_for_solve = s.add_read('A')
    s.add_edge(factor_A, None, A_for_solve, None, dace.Memlet())
    s.add_memlet_path(A_for_solve, solver, dst_conn='_a', memlet=Memlet(f'A[0:{n}, 0:{n}]'))
    s.add_memlet_path(s.add_read('B'), solver, dst_conn='_bin', memlet=Memlet(f'B[0:{n}, 0:{nrhs}]'))
    s.add_memlet_path(solver, s.add_write('X'), src_conn='_bout', memlet=Memlet(f'X[0:{n}, 0:{nrhs}]'))
    s.add_memlet_path(solver, s.add_write('info_s'), src_conn='_res', memlet=Memlet('info_s[0]'))

    A_run, B_run = A.copy(), B.copy()
    X = np.zeros((n, nrhs))
    info_f = np.zeros(1, dtype=np.int32)
    info_s = np.zeros(1, dtype=np.int32)
    sdfg.validate()
    sdfg(A=A_run, B=B_run, X=X, info_f=info_f, info_s=info_s)
    assert int(info_f[0]) == 0
    assert int(info_s[0]) == 0
    np.testing.assert_allclose(X, expected, rtol=_RTOL, atol=_ATOL)


@pytest.mark.lapack
def test_geqrf_orgqr_openblas():
    """GEQRF + ORGQR: factor ``A = QR`` then reconstruct an explicit ``Q``."""
    m, n = 8, 5
    rng = np.random.default_rng(16)
    A = rng.standard_normal((m, n))

    sdfg = dace.SDFG('qr_obs')
    sdfg.add_array('A', [m, n], dace.float64)
    sdfg.add_array('R', [m, n], dace.float64)
    sdfg.add_array('Q', [m, n], dace.float64)
    sdfg.add_array('tau', [min(m, n)], dace.float64)
    sdfg.add_array('info_q', [1], dace.int32)
    sdfg.add_array('info_o', [1], dace.int32)
    s = sdfg.add_state()
    qr = Geqrf('geqrf')
    qr.implementation = _IMPL
    org = Orgqr('orgqr')
    org.implementation = _IMPL
    s.add_node(qr)
    s.add_node(org)
    s.add_memlet_path(s.add_read('A'), qr, dst_conn='_ain', memlet=Memlet(f'A[0:{m}, 0:{n}]'))
    qr_R = s.add_write('R')
    s.add_memlet_path(qr, qr_R, src_conn='_aout', memlet=Memlet(f'R[0:{m}, 0:{n}]'))
    s.add_memlet_path(qr, s.add_write('tau'), src_conn='_tau', memlet=Memlet(f'tau[0:{min(m,n)}]'))
    s.add_memlet_path(qr, s.add_write('info_q'), src_conn='_res', memlet=Memlet('info_q[0]'))

    R_for_org = s.add_read('R')
    s.add_edge(qr_R, None, R_for_org, None, dace.Memlet())
    s.add_memlet_path(R_for_org, org, dst_conn='_ain', memlet=Memlet(f'R[0:{m}, 0:{n}]'))
    s.add_memlet_path(s.add_read('tau'), org, dst_conn='_tau', memlet=Memlet(f'tau[0:{min(m,n)}]'))
    s.add_memlet_path(org, s.add_write('Q'), src_conn='_aout', memlet=Memlet(f'Q[0:{m}, 0:{n}]'))
    s.add_memlet_path(org, s.add_write('info_o'), src_conn='_res', memlet=Memlet('info_o[0]'))

    A_run = A.copy()
    R = np.zeros((m, n))
    Q = np.zeros((m, n))
    tau = np.zeros(min(m, n))
    info_q = np.zeros(1, dtype=np.int32)
    info_o = np.zeros(1, dtype=np.int32)
    sdfg.validate()
    sdfg(A=A_run, R=R, Q=Q, tau=tau, info_q=info_q, info_o=info_o)
    assert int(info_q[0]) == 0
    assert int(info_o[0]) == 0
    # Q should be orthonormal: Q^T Q == I_n.
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=_RTOL, atol=_ATOL)


if __name__ == '__main__':
    test_potrs_openblas()
    print('  test_potrs_openblas: PASS')
    test_geqrf_orgqr_openblas()
    print('  test_geqrf_orgqr_openblas: PASS')
    print('All LAPACK extension OpenBLAS lowering tests pass.')
