# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Self-contained equivalence tests for the experimental (readable) CPU code
generator: no external corpus needed. Each kernel is generated + run with both
``compiler.cpu.implementation = legacy`` and ``experimental`` on identical
inputs; the outputs must be bit-exact.
"""
import copy
import numpy as np
import pytest
import dace
from dace.config import Config

N, M, K = (dace.symbol(s) for s in ('N', 'M', 'K'))


@dace.program
def ew(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    C[:] = A + B


@dace.program
def red(A: dace.float64[N], s: dace.float64[1]):
    s[0] = np.sum(A)


@dace.program
def mm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    C[:] = A @ B


@dace.program
def jac(A: dace.float64[N], B: dace.float64[N]):
    B[1:N - 1] = 0.33 * (A[0:N - 2] + A[1:N - 1] + A[2:N])


@dace.program
def trans(A: dace.float64[N], B: dace.float64[N]):
    tmp = A + 1.0
    B[:] = tmp * 2.0


def _build(prog, impl, name):
    Config.set('compiler', 'cpu', 'implementation', value=impl)
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg.compile()


def _equivalence(prog, args):
    base = args()
    a_leg = copy.deepcopy(base)
    _build(prog, 'legacy', '%s_legacy' % prog.name)(**a_leg)
    a_exp = copy.deepcopy(base)
    _build(prog, 'experimental_readable', '%s_experimental' % prog.name)(**a_exp)
    for key in a_leg:
        v1, v2 = a_leg[key], a_exp[key]
        if isinstance(v1, np.ndarray):
            assert np.array_equal(v1, v2), f'{prog.name}: output {key} differs'


def test_elementwise():
    _equivalence(ew, lambda: dict(A=np.random.rand(6, 8), B=np.random.rand(6, 8), C=np.zeros((6, 8)), M=6, N=8))


def test_reduction_wcr():
    _equivalence(red, lambda: dict(A=np.random.rand(64), s=np.zeros(1), N=64))


def test_matmul_library():
    _equivalence(mm, lambda: dict(A=np.random.rand(6, 5), B=np.random.rand(5, 7), C=np.zeros((6, 7)), M=6, K=5, N=7))


def test_jacobi_stencil():
    _equivalence(jac, lambda: dict(A=np.random.rand(32), B=np.zeros(32), N=32))


def test_transient():
    _equivalence(trans, lambda: dict(A=np.random.rand(20), B=np.zeros(20), N=20))


def test_const_init_constexpr():
    # Constant-shape write-once array is promoted to a constexpr initializer.
    sdfg_e = _const_sdfg('experimental_readable')
    code = sdfg_e.generate_code()[0].clean_code
    assert any('constexpr' in l and 'tbl[' in l and '= {' in l for l in code.splitlines())
    # No runtime allocation of the promoted array.
    assert not any('tbl = new' in l for l in code.splitlines())
    # Correct result.
    A = np.arange(4, dtype=np.float64)
    B = np.zeros(4)
    sdfg_e.compile()(A=A.copy(), B=B)
    assert np.array_equal(B, A + 2.0)


def _const_sdfg(impl):
    Config.set('compiler', 'cpu', 'implementation', value=impl)
    sdfg = dace.SDFG('cinit_%s' % impl)
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    sdfg.add_transient('tbl', [4], dace.float64)
    s1 = sdfg.add_state('init')
    me, mx = s1.add_map('initmap', dict(i='0:4'))
    t1 = s1.add_tasklet('setc', {}, {'o'}, 'o = 2.0')
    w1 = s1.add_access('tbl')
    s1.add_edge(me, None, t1, None, dace.Memlet())
    s1.add_edge(t1, 'o', mx, 'IN_tbl', dace.Memlet('tbl[i]'))
    s1.add_edge(mx, 'OUT_tbl', w1, None, dace.Memlet('tbl[0:4]'))
    mx.add_in_connector('IN_tbl')
    mx.add_out_connector('OUT_tbl')
    s2 = sdfg.add_state_after(s1, 'compute')
    ra, rt, wb = s2.add_access('A'), s2.add_access('tbl'), s2.add_access('B')
    me2, mx2 = s2.add_map('cmap', dict(i='0:4'))
    t2 = s2.add_tasklet('add', {'a', 't'}, {'o'}, 'o = a + t')
    s2.add_memlet_path(ra, me2, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    s2.add_memlet_path(rt, me2, t2, dst_conn='t', memlet=dace.Memlet('tbl[i]'))
    s2.add_memlet_path(t2, mx2, wb, src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


if __name__ == '__main__':
    test_elementwise()
    test_reduction_wcr()
    test_matmul_library()
    test_jacobi_stencil()
    test_transient()
    test_const_init_constexpr()
    print('ok')
