# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Codegen-integration tests for const / constexpr initialization in the
experimental (readable) code generator. Verify that write-once data is emitted
as a ``const``/``constexpr`` initializer and that NO redundant runtime
initialization (``memset`` / ``new`` / ``= {0}`` allocation) is emitted for it,
and that results stay bit-exact vs legacy.
"""
import re

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


def test_const_init_flag_off_keeps_the_runtime_write():
    """``codegen_params.const_init = off`` takes MarkConstInit out of the readable pipeline.

    Unlike most of this group, the default is ``on``: the pass already runs today, so ``on`` is what
    reproduces today's output byte-for-byte and ``off`` is the deviation. With it off the write-once
    scalar stays a mutable buffer written at runtime instead of a ``constexpr`` initializer, and the
    result must be unchanged either way.
    """
    Config.set('compiler', 'cpu', 'codegen_params', 'const_init', value='off')
    try:
        sdfg_off, code_off = _gen(_scalar_const_sdfg, 'experimental_readable', 'sc_flag_off')
    finally:
        Config.set('compiler', 'cpu', 'codegen_params', 'const_init', value='on')
    sdfg_on, code_on = _gen(_scalar_const_sdfg, 'experimental_readable', 'sc_flag_on')

    assert not any('constexpr' in l and 's[1]' in l for l in code_off.splitlines()), \
        'const_init=off still promoted the scalar to a constexpr'
    assert any('constexpr' in l and 's[1]' in l for l in code_on.splitlines()), \
        'const_init=on (the default) must still promote the scalar'

    # Semantics are identical -- this key changes what is emitted, never what is computed.
    A = np.random.rand(8)
    b_off, b_on = np.zeros(8), np.zeros(8)
    sdfg_off.compile()(A=A.copy(), B=b_off, N=8)
    sdfg_on.compile()(A=A.copy(), B=b_on, N=8)
    assert np.array_equal(b_off, b_on) and np.allclose(b_on, A * 3.0)


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
    # Before the fix, generate_code raises SyntaxError('Missing dimensions ... got 0'). The fix lowers
    # the `C[0]` subscript on the 0-d scalar constant to the bare name; assert that behavior (the exact
    # `constexpr T C = v;` declaration is framecode's job and its formatting is not under test here).
    _sdfg, code = _gen(_scalar_constant_subscript_sdfg, 'experimental_readable', 'sc_exp')
    assert 'C[' not in code, 'scalar-constant subscript survived (should lower to bare name)'
    assert '= C;' in code, 'read did not lower to the bare name'


def const_chain_sdfg(name, n=4, m=3):
    """A partial-sum chain of write-once scalars, one link of which keeps a connector.

    ``res[j] = ((((0 + X[0, j]) + X[1, j]) + ...))``, each partial sum its own scope-local scalar, as a
    reduction split into per-element tasklets produces. The seed ``p0`` is a compile-time constant, so
    ``MarkConstInit`` promotes it to an SDFG constant -- and a read of an SDFG constant is exactly what
    ``InlineTaskletConnectors`` refuses to inline, so the tasklet consuming it keeps its ``a`` connector
    and is emitted in its own ``{ }`` block while the rest of the chain is brace-free.

    ``p0`` is declared LAST on purpose: ``MarkConstInit`` classifies descriptors in declaration order,
    so this is the order in which ``p1`` is classified (and its writer predicted brace-free) BEFORE
    ``p0`` becomes the constant that keeps that writer's connector alive.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('X', [n, m], dace.float64)
    sdfg.add_array('res', [m], dace.float64)
    for i in list(range(1, n)) + [0]:
        sdfg.add_scalar('p%d' % i,
                        dace.float64,
                        transient=True,
                        storage=dace.StorageType.Register,
                        lifetime=dace.AllocationLifetime.Scope)
    st = sdfg.add_state('main')
    me, mx = st.add_map('outer', dict(j='0:%d' % m))
    rx = st.add_read('X')
    partials = {i: st.add_access('p%d' % i) for i in range(n)}
    seed = st.add_tasklet('seed', {}, {'o'}, 'o = 0.0')
    st.add_edge(me, None, seed, None, dace.Memlet())
    st.add_edge(seed, 'o', partials[0], None, dace.Memlet('p0[0]'))
    for i in range(n):
        t = st.add_tasklet('acc%d' % i, {'a', 'x'}, {'o'}, 'o = a + x')
        st.add_edge(partials[i], None, t, 'a', dace.Memlet('p%d[0]' % i))
        st.add_memlet_path(rx, me, t, dst_conn='x', memlet=dace.Memlet('X[%d, j]' % i))
        if i + 1 < n:
            st.add_edge(t, 'o', partials[i + 1], None, dace.Memlet('p%d[0]' % (i + 1)))
        else:
            mx.add_in_connector('IN_res')
            mx.add_out_connector('OUT_res')
            st.add_edge(t, 'o', mx, 'IN_res', dace.Memlet('res[j]'))
            st.add_edge(mx, 'OUT_res', st.add_write('res'), None, dace.Memlet('res[0:%d]' % m))
    sdfg.validate()
    return sdfg


def scope_end_line(lines, decl_line):
    """Index of the line that closes the C++ block ``lines[decl_line]`` was declared in."""
    depth = 0
    for index in range(decl_line + 1, len(lines)):
        depth += lines[index].count('{') - lines[index].count('}')
        if depth < 0:
            return index
    return len(lines)


def assert_declared_in_readers_scope(code, names):
    """Assert each of ``names`` is still in scope at its last use (the block it was declared in has
    not closed yet). Catches a binding emitted inside a tasklet's own ``{ }`` block while its readers
    sit in the enclosing one -- code that references an out-of-scope name."""
    lines = code.splitlines()
    for name in names:
        decl = re.compile(r'^\s*(const\s+)?double\s+%s\s*(=|;)' % name)
        use = re.compile(r'\b%s\b' % name)
        decl_lines = [i for i, l in enumerate(lines) if decl.match(l)]
        assert len(decl_lines) == 1, f'{name}: expected exactly one declaration, got {decl_lines}'
        end = scope_end_line(lines, decl_lines[0])
        uses = [i for i, l in enumerate(lines) if i != decl_lines[0] and use.search(l)]
        assert uses, f'{name}: no use found -- the chain was folded away, test is vacuous'
        assert max(uses) < end, (f'{name} is declared at line {decl_lines[0] + 1} in a block that closes '
                                 f'at line {end + 1}, but is used at line {max(uses) + 1}:\n' +
                                 '\n'.join(lines[decl_lines[0]:max(uses) + 1]))


def test_const_binding_stays_in_scope_of_its_readers():
    """A write-once scalar whose producing tasklet needs its own ``{ }`` block must NOT get its
    ``const T x = expr;`` binding folded into that block -- the readers live in the enclosing scope.
    Before the fix this emitted C++ that does not compile ("'p1' was not declared in this scope")."""
    sdfg, code = _gen(const_chain_sdfg, 'experimental_readable', 'const_chain_exp')
    assert '{  // acc0' in code, 'the connector-keeping link no longer needs its own block; test is vacuous'
    assert_declared_in_readers_scope(code, ['p1', 'p2', 'p3'])

    # Same numbers as legacy (this also proves the emitted C++ compiles).
    X = np.arange(12.0).reshape(4, 3).copy()
    res_exp, res_legacy = np.zeros(3), np.zeros(3)
    sdfg.compile()(X=X.copy(), res=res_exp)
    Config.set('compiler', 'cpu', 'implementation', value='legacy')
    const_chain_sdfg('const_chain_leg').compile()(X=X.copy(), res=res_legacy)
    Config.set('compiler', 'cpu', 'implementation', value='experimental_readable')
    assert np.array_equal(res_exp, res_legacy) and np.array_equal(res_exp, X.sum(axis=0))


if __name__ == '__main__':
    test_scalar_constexpr_no_memset()
    test_array_constexpr_full_no_memset()
    test_array_constexpr_partial_zerofill()
    test_scalar_constant_subscript_lowered_to_bare_name()
    test_const_binding_stays_in_scope_of_its_readers()
    print('ok')
