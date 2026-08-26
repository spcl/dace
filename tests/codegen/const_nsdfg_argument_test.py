# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Const-qualifier detection for nested-SDFG (emitted function) parameters.

A nested SDFG's parameters must be ``const``-qualified exactly when the connector is read-only --
for array references as well as scalars. Codegen only; nothing is compiled or run.
"""
import re

import dace
from dace.codegen.targets.cpu import CPUCodeGen


def _signature(inner: dace.SDFG, in_conns, out_conns, wirings) -> str:
    """Build ``map -> inner`` and return the emitted parameter list of ``inner``.

    ``wirings``: list of ``(connector, outer_name, subset, is_input)``.
    """
    sdfg = dace.SDFG('outer_' + inner.name)
    for _conn, oname, _sub, _isin in wirings:
        if oname not in sdfg.arrays:
            sdfg.add_array(oname, (16, 8), dace.float64)
    state = sdfg.add_state('main')
    entry, exit_ = state.add_map('m', dict(i='0:16'))
    nsdfg = state.add_nested_sdfg(inner, dict.fromkeys(in_conns), dict.fromkeys(out_conns))
    for conn, oname, sub, is_input in wirings:
        access = state.add_access(oname)
        if is_input:
            state.add_memlet_path(access, entry, nsdfg, dst_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))
        else:
            state.add_memlet_path(nsdfg, exit_, access, src_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))

    code = '\n'.join(o.code for o in sdfg.generate_code())
    match = re.search(r'void %s\w*\(([^)]*)\)' % re.escape(inner.name), code)
    assert match is not None, f'no function emitted for {inner.name}'
    return match.group(1)


def _param(signature: str, connector: str) -> str:
    for part in signature.split(','):
        if re.search(r'\b%s\b' % re.escape(connector), part):
            return part.strip()
    raise AssertionError(f'connector {connector!r} not in signature: {signature!r}')


def _inner_copy(name: str) -> dace.SDFG:
    """Inner SDFG ``b[j] = a[j]``: ``a`` read-only, ``b`` written."""
    g = dace.SDFG(name)
    g.add_array('a', (8, ), dace.float64)
    g.add_array('b', (8, ), dace.float64)
    s = g.add_state('s')
    s.add_mapped_tasklet('cp',
                         dict(j='0:8'), {'x': dace.Memlet('a[j]')},
                         'y = x', {'y': dace.Memlet('b[j]')},
                         external_edges=True)
    return g


def _inner_with_view(name: str, write_through_view: bool) -> dace.SDFG:
    """Inner SDFG with a ``View`` ``av`` of ``a``, read-direction or write-direction."""
    g = dace.SDFG(name)
    g.add_array('a', (8, ), dace.float64)
    g.add_array('b', (8, ), dace.float64)
    g.add_view('av', (8, ), dace.float64)
    s = g.add_state('s')
    a, b, av = s.add_access('a'), s.add_access('b'), s.add_access('av')
    t = s.add_tasklet('cp', {'x': None}, {'y': None}, 'y = x')
    if write_through_view:
        s.add_edge(b, None, t, 'x', dace.Memlet('b[0]'))
        s.add_edge(t, 'y', av, None, dace.Memlet('av[0]'))
        s.add_edge(av, None, a, None, dace.Memlet('a[0:8]'))
    else:
        s.add_edge(a, None, av, None, dace.Memlet('a[0:8]'))
        s.add_edge(av, None, t, 'x', dace.Memlet('av[0]'))
        s.add_edge(t, 'y', b, None, dace.Memlet('b[0]'))
    return g


def test_readonly_input_is_const_and_written_output_is_not():
    sig = _signature(_inner_copy('roarr'), {'a'}, {'b'}, [('a', 'A', 'i,0:8', True), ('b', 'B', 'i,0:8', False)])
    assert _param(sig, 'a').startswith('const '), _param(sig, 'a')
    assert not _param(sig, 'b').startswith('const '), _param(sig, 'b')


def test_inout_array_is_not_const():
    g = dace.SDFG('inout')
    g.add_array('d', (8, ), dace.float64)
    s = g.add_state('s')
    s.add_mapped_tasklet('inc',
                         dict(j='0:8'), {'v': dace.Memlet('d[j]')},
                         'w = v + 1.0', {'w': dace.Memlet('d[j]')},
                         external_edges=True)
    sig = _signature(g, {'d'}, {'d'}, [('d', 'D', 'i,0:8', True), ('d', 'D', 'i,0:8', False)])
    assert not _param(sig, 'd').startswith('const '), _param(sig, 'd')


def test_a_read_view_of_a_const_input_is_emitted_const():
    """A non-const view of const data is an illegal ``const T* -> T*`` conversion."""
    inner = _inner_with_view('rov', write_through_view=False)
    sdfg = dace.SDFG('outer_rov')
    for n in ('A', 'B'):
        sdfg.add_array(n, (16, 8), dace.float64)
    state = sdfg.add_state('main')
    entry, exit_ = state.add_map('m', dict(i='0:16'))
    nsdfg = state.add_nested_sdfg(inner, {'a': None}, {'b': None})
    state.add_memlet_path(state.add_access('A'), entry, nsdfg, dst_conn='a', memlet=dace.Memlet('A[i, 0:8]'))
    state.add_memlet_path(nsdfg, exit_, state.add_access('B'), src_conn='b', memlet=dace.Memlet('B[i, 0:8]'))
    code = '\n'.join(o.code for o in sdfg.generate_code())

    assert re.search(r'const\s+double\s*\*\s*av\s*;', code), 'view "av" is not pointer-to-const'


def test_mutated_descriptors_follows_view_direction():
    """A read view leaves its parent const-qualifiable; a written view taints it."""
    assert 'a' not in CPUCodeGen._mutated_descriptors(_inner_with_view('rov', write_through_view=False))
    assert 'a' in CPUCodeGen._mutated_descriptors(_inner_with_view('wov', write_through_view=True))


if __name__ == '__main__':
    test_readonly_input_is_const_and_written_output_is_not()
    test_inout_array_is_not_const()
    test_a_read_view_of_a_const_input_is_emitted_const()
    test_mutated_descriptors_follows_view_direction()
