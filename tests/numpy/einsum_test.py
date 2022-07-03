# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')


def test_general_einsum():
    @dace.program
    def einsumtest(A: dace.float64[M, N], B: dace.float64[N, M], C: dace.float64[M]):
        return np.einsum('ij,ji,i->', A, B, C)

    A = np.random.rand(10, 20)
    B = np.random.rand(20, 10)
    C = np.random.rand(10)
    out = einsumtest(A, B, C)
    assert np.allclose(out, np.einsum('ij,ji,i->', A, B, C))


def test_matmul():
    @dace.program
    def einsumtest(A: dace.float64[M, N], B: dace.float64[N, M]):
        return np.einsum('ik,kj', A, B)

    A = np.random.rand(10, 20)
    B = np.random.rand(20, 10)
    assert np.allclose(einsumtest(A, B), A @ B)


def test_batch_matmul():
    @dace.program
    def einsumtest(A: dace.float64[4, M, N], B: dace.float64[4, N, M]):
        return np.einsum('bik,bkj->bij', A, B)

    A = np.random.rand(4, 10, 20)
    B = np.random.rand(4, 20, 10)
    assert np.allclose(einsumtest(A, B), A @ B)


def test_opteinsum_sym():
    @dace.program
    def einsumtest(A: dace.float64[N, N, N, N], B: dace.float64[N, N, N, N], C: dace.float64[N, N, N, N],
                   D: dace.float64[N, N, N, N], E: dace.float64[N, N, N, N]):
        return np.einsum('bdik,acaj,ikab,ajac,ikbd->', A, B, C, D, E, optimize=True)

    A, B, C, D, E = tuple(np.random.rand(10, 10, 10, 10) for _ in range(5))
    try:
        einsumtest(A, B, C, D, E)
        raise AssertionError('Exception should have been raised')
    except ValueError:
        print('Exception successfully caught')


def test_opteinsum():
    N = 10

    @dace.program
    def einsumtest(A: dace.float64[N, N, N, N], B: dace.float64[N, N, N, N], C: dace.float64[N, N, N, N],
                   D: dace.float64[N, N, N, N], E: dace.float64[N, N, N, N]):
        return np.einsum('bdik,acaj,ikab,ajac,ikbd->', A, B, C, D, E, optimize=True)

    A, B, C, D, E = tuple(np.random.rand(10, 10, 10, 10) for _ in range(5))

    assert np.allclose(einsumtest(A, B, C, D, E), np.einsum('bdik,acaj,ikab,ajac,ikbd->', A, B, C, D, E))


def test_einsum_libnode():
    from dace.libraries.blas.nodes.einsum import Einsum

    sdfg = dace.SDFG('tester')
    sdfg.arg_names = ['A', 'B']
    sdfg.add_array('A', (20, 21), dace.float64)
    sdfg.add_array('B', (21, 22), dace.float64)
    sdfg.add_array('__return', (20, 22), dace.float64)

    state = sdfg.add_state()
    r1 = state.add_read('A')
    r2 = state.add_read('B')
    w = state.add_write('__return')
    enode = Einsum('einsum')
    enode.einsum_str = 'ik,kj->ij'
    enode.in_connectors = {'a': None, 'b': None}
    enode.out_connectors = {'out': None}
    state.add_node(enode)

    state.add_edge(r1, None, enode, 'a', dace.Memlet('A'))
    state.add_edge(r2, None, enode, 'b', dace.Memlet('B'))
    state.add_edge(enode, 'out', w, None, dace.Memlet('__return'))

    A = np.random.rand(20, 21)
    B = np.random.rand(21, 22)
    assert np.allclose(sdfg(A, B), A @ B)


def test_lift_einsum():
    from dace.transformation.dataflow import LiftEinsum
    N = dace.symbol('N')
    M = dace.symbol('M')
    K = dace.symbol('K')

    @dace.program
    def tester(A: dace.float64[M, K], B: dace.float64[K, N]):
        C = np.zeros([M, N], A.dtype)
        for i, j, k in dace.map[0:M, 0:N, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]
                c = a * b
        return C

    sdfg = tester.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(LiftEinsum) == 1

    A = np.random.rand(20, 21)
    B = np.random.rand(21, 22)
    assert np.allclose(sdfg(A, B, M=20, K=21, N=22), A @ B)


def test_lift_einsum_mttkrp():
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.transformation.dataflow import LiftEinsum

    @dace.program
    def tester(A, B, C, D):
        for i, j, k, a in dace.map[0:A.shape[0], 0:A.shape[1], 0:A.shape[2], 0:B.shape[1]]:
            with dace.tasklet:
                x << A[i, j, k]
                y << B[j, a]
                z << C[k, a]
                w >> D(1, lambda a, b: a + b)[i, a]
                w = x * y * z

    A = np.random.rand(10, 11, 9)
    B = np.random.rand(11, 8)
    C = np.random.rand(9, 8)
    D = np.zeros((10, 8))

    sdfg = tester.to_sdfg(A, B, C, D, simplify=True)
    assert sdfg.apply_transformations(LiftEinsum) == 1
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Einsum):
            assert node.einsum_str == 'ijk,jl,kl->il'

    sdfg(A, B, C, D)
    assert np.allclose(D, np.einsum('ijk,jl,kl->il', A, B, C))


def test_lift_einsum_reduce():
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.transformation.dataflow import LiftEinsum

    @dace.program
    def tester(A, B):
        B[:] = np.sum(A)

    A = np.random.rand(10, 11, 9)
    B = np.random.rand(1)

    sdfg = tester.to_sdfg(A, B, simplify=True)
    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(LiftEinsum) == 1
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Einsum):
            assert node.einsum_str == 'ijk->'

    sdfg(A, B)
    assert np.allclose(B, np.einsum('ijk->', A))


def test_lift_einsum_outerproduct():
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.transformation.dataflow import LiftEinsum

    @dace.program
    def tester(A, B):
        C = np.ndarray([B.shape[0], A.shape[0]], A.dtype)
        for i, j in dace.map[0:A.shape[0], 0:B.shape[0]]:
            with dace.tasklet:
                a << A[i]
                b << B[j]
                c >> C[j, i]
                c = a * b
        return C

    A = np.random.rand(10)
    B = np.random.rand(11)

    sdfg = tester.to_sdfg(A, B, simplify=True)
    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(LiftEinsum) == 1
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Einsum):
            assert node.einsum_str == 'i,j->ji'

    assert np.allclose(sdfg(A, B), np.einsum('i,j->ji', A, B))


def test_lift_einsum_beta():
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.transformation.dataflow import LiftEinsum

    @dace.program
    def tester(A, B):
        C = np.ones_like(A)
        for i, j, k in dace.map[0:A.shape[0], 0:A.shape[0], 0:A.shape[0]]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]
                c = a * b
        return C

    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    C = 1 + A @ B

    sdfg = tester.to_sdfg(A, B, simplify=True)
    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(LiftEinsum) == 1
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Einsum):
            assert node.einsum_str == 'ij,jk->ik'
            assert node.alpha == 1.0
            assert node.beta == 1.0

    assert np.allclose(sdfg(A, B), C)


@pytest.mark.parametrize('symbolic', (False, True))
def test_lift_einsum_alpha_beta(symbolic):
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.transformation.dataflow import LiftEinsum

    alph = dace.symbol('alph') if symbolic else 2

    @dace.program
    def tester(A, B):
        C = np.ones_like(A)
        for i, j, k in dace.map[0:A.shape[0], 0:A.shape[0], 0:A.shape[0]]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]
                c = alph * a * b
        return C

    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)

    sdfg = tester.to_sdfg(A, B, simplify=True)
    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(LiftEinsum) == 1
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, Einsum):
            assert node.einsum_str == 'ij,jk->ik'
            assert node.alpha == alph
            assert node.beta == 1.0

    if not symbolic:
        C = 1 + 2 * A @ B
        assert np.allclose(sdfg(A, B), C)


if __name__ == '__main__':
    test_general_einsum()
    test_matmul()
    test_batch_matmul()
    test_opteinsum_sym()
    test_opteinsum()
    test_einsum_libnode()
    test_lift_einsum()
    test_lift_einsum_mttkrp()
    test_lift_einsum_reduce()
    test_lift_einsum_outerproduct()
    test_lift_einsum_beta()
    test_lift_einsum_alpha_beta(False)
    test_lift_einsum_alpha_beta(True)
