# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Codegen-integration tests for const / constexpr initialization in the
experimental (readable) code generator. Verify that write-once data is emitted
as a ``const``/``constexpr`` initializer and that NO redundant runtime
initialization (``memset`` / ``new`` / ``= {0}`` allocation) is emitted for it,
and that results stay bit-exact vs legacy.
"""
import numpy as np
import dace
from dace.config import Config


def _gen(sdfg_factory, impl, name):
    Config.set('compiler', 'cpu', 'implementation', value=impl)
    sdfg = sdfg_factory(name)
    return sdfg, sdfg.generate_code()[0].clean_code


def _array_const_sdfg(name, shape, values, read=True):
    """ Element-wise constant init of a transient, then B = A + arr. """
    sdfg = dace.SDFG(name)
    n = int(np.prod(shape))
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n], dace.float64)
    sdfg.add_transient('arr', [n], dace.float64)
    s1 = sdfg.add_state('init')
    acc = s1.add_access('arr')
    for i, v in values.items():
        t = s1.add_tasklet('set_%d' % i, {}, {'o'}, 'o = %s' % v)
        s1.add_edge(t, 'o', acc, None, dace.Memlet('arr[%d]' % i))
    s2 = sdfg.add_state_after(s1, 'compute')
    ra, rr, wb = s2.add_access('A'), s2.add_access('arr'), s2.add_access('B')
    me, mx = s2.add_map('m', dict(i='0:%d' % n))
    t2 = s2.add_tasklet('add', {'a', 'r'}, {'o'}, 'o = a + r')
    s2.add_memlet_path(ra, me, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    s2.add_memlet_path(rr, me, t2, dst_conn='r', memlet=dace.Memlet('arr[i]'))
    s2.add_memlet_path(t2, mx, wb, src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def _scalar_const_sdfg(name):
    """ s = 3.0 (write-once constant scalar), then B[i] = A[i] * s. """
    N = dace.symbol('N')
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('s', [1], dace.float64)
    s1 = sdfg.add_state('init')
    ts = s1.add_tasklet('setc', {}, {'o'}, 'o = 3.0')
    s1.add_edge(ts, 'o', s1.add_access('s'), None, dace.Memlet('s[0]'))
    s2 = sdfg.add_state_after(s1, 'compute')
    ra, rs, wb = s2.add_access('A'), s2.add_access('s'), s2.add_access('B')
    me, mx = s2.add_map('m', dict(i='0:N'))
    t2 = s2.add_tasklet('mul', {'a', 'sc'}, {'o'}, 'o = a * sc')
    s2.add_memlet_path(ra, me, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    s2.add_memlet_path(rs, me, t2, dst_conn='sc', memlet=dace.Memlet('s[0]'))
    s2.add_memlet_path(t2, mx, wb, src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def _no_redundant_init(code, arrname):
    """ Assert the promoted const array has no runtime init/allocation. """
    for line in code.splitlines():
        if arrname in line:
            assert 'new' not in line or 'constexpr' in line, f'runtime alloc of const {arrname}: {line}'
            assert 'memset' not in line, f'redundant memset of const {arrname}: {line}'


def test_scalar_constexpr_no_memset():
    sdfg, code = _gen(_scalar_const_sdfg, 'experimental_readable', 'sc_exp')
    lines = code.splitlines()
    # A write-once scalar is promoted to a length-1 constexpr array: constexpr double s[1] = {3.0};
    assert any('constexpr' in l and 's[1]' in l and '3' in l for l in lines), 'scalar not emitted const/constexpr'
    _no_redundant_init(code, ' s[')
    # correctness
    sdfg_l = _scalar_const_sdfg('sc_leg')
    Config.set('compiler', 'cpu', 'implementation', value='legacy')
    A = np.random.rand(8)
    bl, be = np.zeros(8), np.zeros(8)
    sdfg_l.compile()(A=A.copy(), B=bl, N=8)
    Config.set('compiler', 'cpu', 'implementation', value='experimental_readable')
    sdfg.compile()(A=A.copy(), B=be, N=8)
    assert np.array_equal(bl, be) and np.allclose(be, A * 3.0)


def test_array_constexpr_full_no_memset():
    vals = {0: '0.0', 1: '1.0', 2: '2.0', 3: '3.0'}
    fac = lambda name: _array_const_sdfg(name, (4, ), vals)
    sdfg, code = _gen(fac, 'experimental_readable', 'ac_exp')
    assert any('constexpr' in l and 'arr[' in l and '= {' in l for l in code.splitlines()), \
        'array not emitted constexpr initializer'
    _no_redundant_init(code, 'arr[')
    # correctness
    Config.set('compiler', 'cpu', 'implementation', value='legacy')
    A = np.random.rand(4)
    bl, be = np.zeros(4), np.zeros(4)
    fac('ac_leg').compile()(A=A.copy(), B=bl)
    Config.set('compiler', 'cpu', 'implementation', value='experimental_readable')
    sdfg.compile()(A=A.copy(), B=be)
    assert np.array_equal(bl, be) and np.allclose(be, A + np.array([0., 1., 2., 3.]))


def test_array_constexpr_partial_zerofill():
    # Only indices 1,2 written -> unwritten 0,3 must be zero-filled in the constexpr.
    vals = {1: '5.0', 2: '6.0'}
    fac = lambda name: _array_const_sdfg(name, (4, ), vals)
    sdfg, code = _gen(fac, 'experimental_readable', 'ap_exp')
    lines = [l for l in code.splitlines() if 'constexpr' in l and 'arr[' in l and '= {' in l]
    assert lines, 'partial-init array not constexpr'
    _no_redundant_init(code, 'arr[')
    # Partial init: the readable generator zero-fills the unwritten elements (a
    # deliberate, defined-behavior choice). Legacy leaves them uninitialized
    # (garbage), so we assert the experimental result is the zero-filled value
    # and only compare the WRITTEN region against legacy.
    A = np.zeros(4)
    be = np.zeros(4)
    Config.set('compiler', 'cpu', 'implementation', value='experimental_readable')
    sdfg.compile()(A=A.copy(), B=be)
    assert np.allclose(be, np.array([0., 5., 6., 0.])), f'zero-fill wrong: {be}'
    bl = np.zeros(4)
    Config.set('compiler', 'cpu', 'implementation', value='legacy')
    fac('ap_leg').compile()(A=A.copy(), B=bl)
    assert np.allclose(bl[[1, 2]], be[[1, 2]]), 'written region differs from legacy'


def _scalar_constant_subscript_sdfg(name):
    """A 0-d scalar SDFG constant read via a subscript ``C[0]`` in a tasklet body. MarkConstInit
    promotes a write-once scalar to exactly such a constant (emitted as bare ``constexpr T C = v;``),
    so a subscript on it must lower to the bare name -- the classic path trips on the scalar's empty
    stride list (``Missing dimensions in expression (expected one, got 0)``)."""
    from dace import data as dt
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', [4], dace.float64)
    sdfg.add_constant('C', np.float64(3.0), dt.Scalar(dace.float64))
    st = sdfg.add_state('main')
    me, mx = st.add_map('m', dict(i='0:4'))
    t = st.add_tasklet('r', {}, {'o'}, 'o = C[0]', language=dace.Language.Python)
    oacc = st.add_access('out')
    st.add_edge(me, None, t, None, dace.Memlet())
    st.add_memlet_path(t, mx, oacc, src_conn='o', memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def test_scalar_constant_subscript_lowered_to_bare_name():
    # Before the fix, generate_code raises SyntaxError('Missing dimensions ... got 0').
    _sdfg, code = _gen(_scalar_constant_subscript_sdfg, 'experimental_readable', 'sc_exp')
    assert 'constexpr double C = 3.0;' in code, 'scalar constant not emitted as bare constexpr'
    assert 'C[' not in code, 'scalar-constant subscript survived (should lower to bare name)'
    assert '= C;' in code, 'read did not lower to the bare name'


if __name__ == '__main__':
    test_scalar_constexpr_no_memset()
    test_array_constexpr_full_no_memset()
    test_array_constexpr_partial_zerofill()
    test_scalar_constant_subscript_lowered_to_bare_name()
    print('ok')
