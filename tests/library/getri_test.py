import dace
from dace.data import Array
from dace.libraries.blas.nodes import Getrf, Getri
import numpy as np
from scipy.linalg.lapack import dgetrf, dgetri
from typing import Dict


def relative_error(val, ref):
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


def test_getri_node():
    # Construct graph
    sdfg = dace.SDFG('GETRI')
    sdfg.add_array('A', [128, 128], dace.float64)
    sdfg.add_array('IPIV', [128], dace.int32)
    sdfg.add_array('INFO', [1], dace.int32)
    state = sdfg.add_state()

    node1 = Getrf('getrf_node', dtype=dace.float64)
    node1.implementation = 'MKL'
    node2 = Getri('getri_node', dtype=dace.float64)
    node2.implementation = 'MKL'
    a = state.add_read('A')
    b = state.add_access('A')
    c = state.add_access('A')
    ipiv = state.add_access('IPIV')
    info1 = state.add_write('INFO')
    info2 = state.add_write('INFO')

    state.add_edge(a, None, node1, '_a_in',
                   dace.Memlet.simple('A', '0:128, 0:128'))
    state.add_edge(node1, '_a_out', b, None,
                   dace.Memlet.simple('A', '0:128, 0:128'))
    state.add_edge(node1, '_ipiv', ipiv, None,
                   dace.Memlet.simple('IPIV', '0:128'))
    state.add_edge(node1, '_info', info1, None, dace.Memlet.simple('INFO', '0'))

    state.add_edge(b, None, node2, '_a_in',
                   dace.Memlet.simple('A', '0:128, 0:128'))
    state.add_edge(node2, '_a_out', c, None,
                   dace.Memlet.simple('A', '0:128, 0:128'))
    state.add_edge(ipiv, None, node2, '_ipiv',
                   dace.Memlet.simple('IPIV', '0:128'))
    state.add_edge(node2, '_info', info2, None, dace.Memlet.simple('INFO', '0'))

    # Run graph
    A = np.random.rand(128, 128).astype(np.float64)
    A_orig = np.zeros((128, 128), dtype=np.float64)
    A_orig[:] = A[:]
    IPIV = np.zeros((128,), dtype=np.int32)
    INFO = np.zeros((1, ), dtype=np.int32)
    sdfg(A=A, IPIV=IPIV, INFO=INFO)

    # Validate
    assert(INFO[0] == 0)
    A_ref, ipiv_ref, info_ref = dgetrf(A_orig)
    assert(info_ref == 0)
    # dgetrf of scipy returns 0-indexed ipiv
    assert(relative_error(IPIV, ipiv_ref + 1) == 0)
    A_ref, info_ref = dgetri(A_ref, ipiv_ref)
    assert(relative_error(A, A_ref) < 1e-12)


if __name__ == '__main__':
    test_getri_node()