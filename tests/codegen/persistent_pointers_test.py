# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests loading the pointers to persistent data into local constants of the program function. """
import re

import numpy as np

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers as xfh

N = dace.symbol('N')


def _loops_sdfg(name: str, lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Persistent) -> dace.SDFG:
    """ Two loops writing and reading a transient, then a state that copies it to the output. """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('tmp', [N], dace.float64, lifetime=lifetime)
    loops = []
    for i, (src, dst, expr) in enumerate([('A', 'tmp', 'a + 1'), ('tmp', 'tmp', 'a * 3')]):
        loop = LoopRegion(f'loop_{i}', 'i < N', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(loop, is_start_block=(i == 0))
        body = loop.add_state(f'body_{i}', is_start_block=True)
        t = body.add_tasklet(f'compute_{i}', {'a'}, {'b'}, f'b = {expr}')
        body.add_edge(body.add_read(src), None, t, 'a', dace.Memlet(f'{src}[i]'))
        body.add_edge(t, 'b', body.add_write(dst), None, dace.Memlet(f'{dst}[i]'))
        if loops:
            sdfg.add_edge(loops[-1], loop, dace.InterstateEdge())
        loops.append(loop)
    final = sdfg.add_state_after(loops[-1], 'copy_out')
    final.add_nedge(final.add_read('tmp'), final.add_write('B'), dace.Memlet('tmp[0:N]'))
    return sdfg


def _program_function(code: str, name: str) -> str:
    """ The body of the program function of the SDFG called ``name`` in the frame code. """
    start = code.index(f'void __program_{name}_internal(')
    return code[start:code.index('\n}\n', start)]


def _sources(sdfg: dace.SDFG):
    return [o.clean_code for o in sdfg.generate_code() if o.language == 'cpp' and o.linkable]


def _run(sdfg: dace.SDFG):
    A = np.random.rand(16)
    B = np.zeros(16)
    sdfg(A=A, B=B, N=16)
    assert np.allclose(B, (A + 1) * 3)


def test_persistent_pointer_loaded_once():
    sdfg = _loops_sdfg('persistent_pointer_loaded_once')
    with dace.config.set_temporary('compiler', 'cpu', 'hoist_persistent_pointers', value=True):
        body = _program_function(_sources(sdfg)[0], sdfg.name)
        assert re.search(r'double\s*\* const __p__\d+_tmp = __state->__\d+_tmp;', body)
        # Every access goes through the local constant
        assert len(re.findall(r'__state->__\d+_tmp', body)) == 1
        assert re.search(r'__p__\d+_tmp\[', body)
        _run(sdfg)


def test_persistent_pointer_disabled():
    sdfg = _loops_sdfg('persistent_pointer_disabled')
    with dace.config.set_temporary('compiler', 'cpu', 'hoist_persistent_pointers', value=False):
        body = _program_function(_sources(sdfg)[0], sdfg.name)
        assert '__p__' not in body
        assert re.search(r'__state->__\d+_tmp\[', body)
        _run(sdfg)


def test_non_persistent_data_unchanged():
    """ Only persistent data lives in the state struct; other transients keep their names. """
    sdfg = _loops_sdfg('persistent_pointer_sdfg_lifetime', dtypes.AllocationLifetime.SDFG)
    with dace.config.set_temporary('compiler', 'cpu', 'hoist_persistent_pointers', value=True):
        body = _program_function(_sources(sdfg)[0], sdfg.name)
        assert '__p__' not in body
        _run(sdfg)


def test_region_receives_local_pointer():
    """ A function region receives the local constant as its argument and names the parameter after the member. """
    sdfg = _loops_sdfg('persistent_pointer_region')
    xfh.wrap_in_function_region([b for b in sdfg.nodes() if isinstance(b, LoopRegion)], 'loops',
                                dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    with dace.config.set_temporary('compiler', 'cpu', 'hoist_persistent_pointers', value=True):
        frame, unit = _sources(sdfg)
        assert re.search(r'loops\w*\(__state, [^;]*__p__\d+_tmp', _program_function(frame, sdfg.name))
        params = re.search(r'void loops\w*\(([^)]*)\)', unit).group(1)
        assert re.search(r'__restrict__ __\d+_tmp\b', params)
        assert '__p__' not in unit
        _run(sdfg)


def test_region_reaching_state_struct_uses_member():
    """ A region that keeps persistent data in the state struct (it contains a nested SDFG) does not see the locals. """
    sdfg = _loops_sdfg('persistent_pointer_region_nested')
    inner = dace.SDFG('noop')
    inner.add_array('X', [1], dace.float64)
    state = inner.add_state()
    state.add_edge(state.add_tasklet('noop', {}, {'o'}, 'o = 0'), 'o', state.add_write('X'), None, dace.Memlet('X[0]'))
    loops = [b for b in sdfg.nodes() if isinstance(b, LoopRegion)]
    body = loops[0].start_block
    nested = body.add_nested_sdfg(inner, {}, {'X'})
    body.add_edge(nested, 'X', body.add_write('B'), None, dace.Memlet('B[0]'))
    xfh.wrap_in_function_region(loops, 'loops', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    with dace.config.set_temporary('compiler', 'cpu', 'hoist_persistent_pointers', value=True):
        frame, unit = _sources(sdfg)
        assert '__p__' not in unit
        assert re.search(r'__state->__\d+_tmp\[', unit)
        # The copy after the region still uses the local constant
        assert re.search(r'__p__\d+_tmp', _program_function(frame, sdfg.name))
        _run(sdfg)


def test_external_pointer_loaded_per_call():
    """ External memory may change between calls: the pointer is loaded on every call, not cached. """

    @dace.program
    def external_workspace(a: dace.float64[20]):
        workspace = dace.ndarray([20], dace.float64, lifetime=dace.AllocationLifetime.External)
        workspace[:] = a
        workspace += 1
        a[:] = workspace

    sdfg = external_workspace.to_sdfg()
    with dace.config.set_temporary('compiler', 'cpu', 'hoist_persistent_pointers', value=True):
        assert re.search(r'const __p__\d+_workspace = __state->__\d+_workspace;',
                         _program_function(_sources(sdfg)[0], sdfg.name))
        csdfg = sdfg.compile()
    a = np.random.rand(20)
    csdfg.initialize(a)
    for _ in range(2):
        workspace = np.zeros(20)
        csdfg.set_workspace(dace.StorageType.CPU_Heap, workspace)
        ref = a + 1
        csdfg(a)
        assert np.allclose(a, ref)
        assert np.allclose(workspace, ref)


if __name__ == '__main__':
    test_persistent_pointer_loaded_once()
    test_persistent_pointer_disabled()
    test_non_persistent_data_unchanged()
    test_region_receives_local_pointer()
    test_region_reaching_state_struct_uses_member()
    test_external_pointer_loaded_per_call()
