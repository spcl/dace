# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
from dace import symbolic
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
    from dace.libraries.standard.nodes.reduce import Reduce
    from dace.transformation.dataflow import LiftEinsum

    @dace.program
    def tester(A, B):
        B[:] = np.sum(A)

    A = np.random.rand(10, 11, 9)
    B = np.random.rand(1)

    sdfg = tester.to_sdfg(A, B, simplify=True)
    # Capture the original Reduce node before lowering; afterwards it lives in a
    # nested SDFG that LiftEinsum intentionally does not traverse.
    rnode = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, Reduce))
    assert rnode.axes is None

    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(LiftEinsum) == 0

    sdfg(A, B)
    assert np.allclose(B, np.einsum('ijk->', A))


def test_lift_einsum_reduce_partial():
    from dace.libraries.standard.nodes.reduce import Reduce
    from dace.transformation.dataflow import LiftEinsum

    @dace.program
    def tester(A, B):
        B[:] = np.sum(A, axis=1)

    A = np.random.rand(10, 11, 9)
    B = np.random.rand(10, 9)

    sdfg = tester.to_sdfg(A, B, simplify=True)
    # Capture the original Reduce node before lowering; afterwards it lives in a
    # nested SDFG that LiftEinsum intentionally does not traverse.
    rnode = next(node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, Reduce))
    assert tuple(rnode.axes) == (1, )

    sdfg.expand_library_nodes()
    assert sdfg.apply_transformations(LiftEinsum) == 0

    sdfg(A, B)
    assert np.allclose(B, np.einsum('ijk->ik', A))


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
            assert symbolic.equal_valued(1, node.alpha)
            assert symbolic.equal_valued(1, node.beta)

    assert np.allclose(sdfg(A, B), C)


@pytest.mark.parametrize('symbolic_alpha', (False, True))
def test_lift_einsum_alpha_beta(symbolic_alpha):
    from dace.libraries.blas.nodes.einsum import Einsum
    from dace.transformation.dataflow import LiftEinsum

    alph = dace.symbol('alph') if symbolic_alpha else 2

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
            assert symbolic.equal_valued(1, node.beta)

    if not symbolic_alpha:
        C = 1 + 2 * A @ B
        assert np.allclose(sdfg(A, B), C)


def test_rowwise_dot():
    """``ik,ik->i`` is a batched dot: the batch index survives into the output and neither operand
    has a private index, so there is no GEMM/GEMV form -- it must lower to the pure einsum map."""

    @dace.program
    def einsumtest(A: dace.float64[M, N]):
        return np.einsum('ik,ik->i', A, A)

    A = np.random.rand(10, 20)
    assert np.allclose(einsumtest(A), np.einsum('ik,ik->i', A, A), rtol=1e-12)


def test_batched_dot_4d():
    """Same batched-dot class over a 3-D batch. Simplify collapses the box into a flat view, which
    the degenerate M=N=1 MatMul the contraction path used to mint could no longer dispatch. The
    contracted extent is the symbol ``k``, the contraction index letter as well: the map parameter
    must not shadow it."""
    L = dace.symbol('L')
    k = dace.symbol('k')

    @dace.program
    def einsumtest(A: dace.float64[L, L, L, k]):
        return np.einsum('xyzk,xyzk->xyz', A, A)

    A = np.random.rand(4, 4, 4, 7)
    assert np.allclose(einsumtest(A), np.einsum('xyzk,xyzk->xyz', A, A), rtol=1e-12)


def test_batched_dot_in_loop():
    """The pure einsum path prepends a state that initializes the accumulator. Inside a loop body
    that state belongs to the loop's region, not to the SDFG, and it must run on every iteration."""
    F = dace.symbol('F')
    L = dace.symbol('L')
    k = dace.symbol('k')

    @dace.program
    def einsumtest(psi: dace.float64[F, L, L, L, k], out: dace.float64[F, L, L, L]):
        for f in range(F):
            dens = np.einsum('xyzk,xyzk->xyz', psi[f], psi[f])
            out[f] = dens

    psi = np.random.rand(3, 2, 2, 2, 5)
    out = np.zeros((3, 2, 2, 2))
    einsumtest(psi, out)
    assert np.allclose(out, np.einsum('fxyzk,fxyzk->fxyz', psi, psi), rtol=1e-12)


def test_c_transposed():
    N, F_in, F_out = 2, 3, 3

    @dace.program
    def fn(a, b, c):
        c[:] = np.einsum('nm,nf->fm', a, b)

    a = np.random.rand(N, F_in)
    b = np.random.rand(N, F_out)
    c_expected = np.zeros((F_out, F_in))
    c = np.zeros((F_out, F_in))

    fn.f(a, b, c_expected)
    fn(a, b, c)

    assert np.allclose(c, c_expected)


@pytest.mark.parametrize('beta', [0.0, 1.0])
def test_einsum_dot_node(beta):
    """A directly-constructed scalar-output ``i,i->`` Einsum node expands to a stride-aware DDOT
    (a ``Dot`` node), NOT the degenerate 1x1 GEMM the contraction path emits (illegal ``lda=1``
    for a strided operand -> silently dropped contraction). Covers the ``SpecializeEinsum`` dot
    path that ``LiftEinsum`` deliberately no longer produces: ``out = alpha*dot(x,y) +
    beta*out_prior``."""
    from dace.libraries.blas.nodes.einsum import Einsum
    n = 40
    sdfg = dace.SDFG('einsum_dot')
    sdfg.add_array('x', [n], dace.float64)
    sdfg.add_array('y', [n], dace.float64)
    sdfg.add_array('r', [1], dace.float64)
    state = sdfg.add_state()
    enode = Einsum('einsum')
    enode.einsum_str = 'i,i->'
    enode.alpha = 2.0
    enode.beta = beta
    enode.in_connectors = {'a': None, 'b': None}
    enode.out_connectors = {'out': None}
    state.add_node(enode)
    state.add_edge(state.add_read('x'), None, enode, 'a', dace.Memlet(f'x[0:{n}]'))
    state.add_edge(state.add_read('y'), None, enode, 'b', dace.Memlet(f'y[0:{n}]'))
    state.add_edge(enode, 'out', state.add_write('r'), None, dace.Memlet('r[0]'))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(0)
    x = rng.random(n)
    y = rng.random(n)
    r = np.array([7.0])
    prior = r[0]
    sdfg(x=x, y=y, r=r)
    assert np.allclose(r[0], 2.0 * np.dot(x, y) + beta * prior), f'got {r[0]}'


def test_einsum_shape_check_equalizes_symbols():
    """One symbol name can reach the shape check as several sympy instances -- a descriptor a layout
    pass rebuilt against one parsed from a string -- which compare unequal by identity. The einsum
    dimension check must go through the name, or it rejects a contraction whose shapes agree."""
    from dace.libraries.blas.nodes.einsum import Einsum
    rows, cols = 4, 6
    wide = dace.symbol('ESN', dace.int32)
    narrow = dace.symbol('ESN', dace.int64)
    # the premise: identity says these differ, the name says they do not
    assert wide is not narrow and wide != narrow
    assert not symbolic.inequal_symbols(wide, narrow)

    sdfg = dace.SDFG('einsum_symbol_instances')
    sdfg.add_array('A', [rows, wide], dace.float64)
    sdfg.add_array('v', [narrow], dace.float64)
    sdfg.add_array('out', [rows], dace.float64)
    state = sdfg.add_state()
    enode = Einsum('einsum')
    enode.einsum_str = 'ij,j->i'
    enode.in_connectors = {'_ein00': None, '_ein01': None}
    enode.out_connectors = {'_out': None}
    state.add_node(enode)
    # from_array, not a parsed string: parsing both bounds from 'ESN' would mint ONE instance and
    # the mismatch under test could not arise.
    state.add_edge(state.add_read('A'), None, enode, '_ein00', dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(state.add_read('v'), None, enode, '_ein01', dace.Memlet.from_array('v', sdfg.arrays['v']))
    state.add_edge(enode, '_out', state.add_write('out'), None, dace.Memlet.from_array('out', sdfg.arrays['out']))

    # the two instances survive as far as the check; without equalization this raises
    assert sdfg.arrays['A'].shape[1] is not sdfg.arrays['v'].shape[0]
    sdfg.expand_library_nodes()
    assert not any(isinstance(n, Einsum) for n in state.nodes())

    rng = np.random.default_rng(3)
    a = rng.random((rows, cols))
    v = rng.random(cols)
    out = np.zeros(rows)
    sdfg(A=a, v=v, out=out, ESN=cols)
    assert np.allclose(out, a @ v), f'got {out}, want {a @ v}'


def test_einsum_libnode_ordering_edge():
    """An ordering memlet into the Einsum node names no container; expansion must skip it."""
    from dace.libraries.blas.nodes.einsum import Einsum

    sdfg = dace.SDFG('tester_ordering')
    sdfg.arg_names = ['A', 'B']
    sdfg.add_array('A', (20, 21), dace.float64)
    sdfg.add_array('B', (21, 22), dace.float64)
    sdfg.add_array('__return', (20, 22), dace.float64)
    sdfg.add_transient('anchor', (1, ), dace.float64)

    state = sdfg.add_state()
    enode = Einsum('einsum')
    enode.einsum_str = 'ik,kj->ij'
    enode.in_connectors = {'a': None, 'b': None}
    enode.out_connectors = {'out': None}
    state.add_node(enode)
    state.add_edge(state.add_read('A'), None, enode, 'a', dace.Memlet('A'))
    state.add_edge(state.add_read('B'), None, enode, 'b', dace.Memlet('B'))
    state.add_edge(enode, 'out', state.add_write('__return'), None, dace.Memlet('__return'))

    anchor = state.add_access('anchor')
    state.add_edge(state.add_tasklet('t', {}, {'o': None}, 'o = 1.0'), 'o', anchor, None, dace.Memlet('anchor[0]'))
    state.add_nedge(anchor, enode, dace.Memlet())
    sdfg.validate()

    A = np.random.rand(20, 21)
    B = np.random.rand(21, 22)
    assert np.allclose(sdfg(A, B), A @ B)


if __name__ == '__main__':
    test_general_einsum()
    test_matmul()
    test_batch_matmul()
    test_opteinsum_sym()
    test_opteinsum()
    test_einsum_libnode()
    test_einsum_libnode_ordering_edge()
    test_lift_einsum()
    test_lift_einsum_mttkrp()
    test_lift_einsum_reduce()
    test_lift_einsum_reduce_partial()
    test_lift_einsum_outerproduct()
    test_lift_einsum_beta()
    test_lift_einsum_alpha_beta(False)
    test_lift_einsum_alpha_beta(True)
    test_rowwise_dot()
    test_batched_dot_4d()
    test_batched_dot_in_loop()
    test_c_transposed()
    test_einsum_shape_check_equalizes_symbols()
