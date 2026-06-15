# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Const-qualifier detection for nested-SDFG (device-function) parameters.

A nested SDFG inside a ``GPU_Device`` map is emitted as a ``DACE_DFI`` device
function. Its parameters must be ``const``-qualified exactly when the connector
is read-only (an input that is never written), for both scalars *and* array
references -- see :func:`dace.codegen.targets.cpp.emit_memlet_reference`.

These tests build a minimal kernel (``GPU_Device`` map containing one nested
SDFG) and assert the ``const`` qualifier of each device-function parameter.
Codegen only -- no GPU / nvcc required.
"""
import re

import pytest

import dace
from dace import dtypes

GPU_GLOBAL = dtypes.StorageType.GPU_Global
GPU_DEVICE = dtypes.ScheduleType.GPU_Device


def _device_function_signature(inner: dace.SDFG, in_conns, out_conns, wirings, impl='experimental') -> str:
    """Build ``GPU_Device map -> inner`` and return the emitted ``DACE_DFI`` parameter list.

    ``wirings``: list of ``(connector, outer_name, outer_shape, subset, is_input)``.
    """
    sdfg = dace.SDFG('outer_' + inner.name)
    declared = {}
    for _conn, oname, oshape, _sub, _isin in wirings:
        if oname not in declared:
            sdfg.add_array(oname, oshape, dace.float64, storage=GPU_GLOBAL)
            declared[oname] = oshape
    state = sdfg.add_state('main')
    entry, exit_ = state.add_map('kmap', dict(i='0:16'), schedule=GPU_DEVICE)
    nsdfg = state.add_nested_sdfg(inner, set(in_conns), set(out_conns))
    for conn, oname, _oshape, sub, is_input in wirings:
        access = state.add_access(oname)
        if is_input:
            state.add_memlet_path(access, entry, nsdfg, dst_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))
        else:
            state.add_memlet_path(nsdfg, exit_, access, src_conn=conn, memlet=dace.Memlet(data=oname, subset=sub))

    dace.Config.set('compiler', 'cuda', 'implementation', value=impl)
    code = '\n'.join(o.code for o in sdfg.generate_code())
    match = re.search(r'DACE_DFI void %s\w*\(([^)]*)\)' % re.escape(inner.name), code)
    assert match is not None, f'no DACE_DFI device function emitted for {inner.name}\n{code[:2000]}'
    return match.group(1)


def _param(signature: str, connector: str) -> str:
    """The parameter substring for ``connector`` from a comma-separated signature."""
    for part in signature.split(','):
        if re.search(r'\b%s\b' % re.escape(connector), part):
            return part.strip()
    raise AssertionError(f'connector {connector!r} not found in signature: {signature!r}')


def _inner_copy(name: str, in_name: str, out_name: str) -> dace.SDFG:
    """Inner SDFG: elementwise ``out[j] = in[j]`` over an 8-vector (in read-only, out written)."""
    g = dace.SDFG(name)
    g.add_array(in_name, (8, ), dace.float64)
    g.add_array(out_name, (8, ), dace.float64)
    s = g.add_state('s')
    s.add_mapped_tasklet('cp',
                         dict(j='0:8'), {'x': dace.Memlet(f'{in_name}[j]')},
                         'y = x', {'y': dace.Memlet(f'{out_name}[j]')},
                         external_edges=True)
    return g


@pytest.mark.parametrize('impl', ['experimental', 'legacy'])
def test_readonly_array_input_is_const(impl):
    """A read-only array input becomes ``const T*`` (the previously-missing case)."""
    inner = _inner_copy('roarr', 'a', 'b')
    sig = _device_function_signature(inner, {'a'}, {'b'}, [('a', 'A', (16, 8), 'i,0:8', True),
                                                           ('b', 'B', (16, 8), 'i,0:8', False)],
                                     impl=impl)
    assert _param(sig, 'a').startswith('const '), _param(sig, 'a')


@pytest.mark.parametrize('impl', ['experimental', 'legacy'])
def test_written_array_output_is_not_const(impl):
    """A written array output must NOT be const (no over-qualification)."""
    inner = _inner_copy('woarr', 'a', 'b')
    sig = _device_function_signature(inner, {'a'}, {'b'}, [('a', 'A', (16, 8), 'i,0:8', True),
                                                           ('b', 'B', (16, 8), 'i,0:8', False)],
                                     impl=impl)
    assert not _param(sig, 'b').startswith('const '), _param(sig, 'b')


@pytest.mark.parametrize('impl', ['experimental', 'legacy'])
def test_readonly_scalar_input_is_const(impl):
    """A read-only scalar input is ``const`` (scalar reference or by-value)."""
    g = dace.SDFG('roscalar')
    g.add_array('a', (8, ), dace.float64)
    g.add_array('b', (8, ), dace.float64)
    g.add_scalar('c', dace.float64)
    s = g.add_state('s')
    s.add_mapped_tasklet('cp',
                         dict(j='0:8'), {
                             'x': dace.Memlet('a[j]'),
                             'sc': dace.Memlet('c[0]')
                         },
                         'y = x + sc', {'y': dace.Memlet('b[j]')},
                         external_edges=True)
    sig = _device_function_signature(g, {'a', 'c'}, {'b'}, [('a', 'A', (16, 8), 'i,0:8', True),
                                                            ('c', 'C', (16, ), 'i', True),
                                                            ('b', 'B', (16, 8), 'i,0:8', False)])
    assert 'const' in _param(sig, 'c'), _param(sig, 'c')


@pytest.mark.parametrize('impl', ['experimental', 'legacy'])
def test_inout_array_is_not_const(impl):
    """An in/out (read-modify-write) array connector must NOT be const."""
    g = dace.SDFG('inout')
    g.add_array('d', (8, ), dace.float64)
    s = g.add_state('s')
    s.add_mapped_tasklet('inc',
                         dict(j='0:8'), {'v': dace.Memlet('d[j]')},
                         'w = v + 1.0', {'w': dace.Memlet('d[j]')},
                         external_edges=True)
    sig = _device_function_signature(g, {'d'}, {'d'}, [('d', 'D', (16, 8), 'i,0:8', True),
                                                       ('d', 'D', (16, 8), 'i,0:8', False)],
                                     impl=impl)
    assert not _param(sig, 'd').startswith('const '), _param(sig, 'd')


if __name__ == '__main__':
    for impl in ('experimental', 'legacy'):
        test_readonly_array_input_is_const(impl)
        test_written_array_output_is_not_const(impl)
        test_readonly_scalar_input_is_const(impl)
        test_inout_array_is_not_const(impl)
    print('ok')
