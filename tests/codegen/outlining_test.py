# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests generating code generator function regions as separate functions and translation units. """
import copy
import re
from typing import List

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.exceptions import CodegenError
from dace.sdfg.state import CodeGeneratorFunctionRegion, LoopRegion
from dace.transformation import helpers as xfh

N = dace.symbol('N')


@dace.program
def three_loops(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    tmp = np.empty_like(A)
    for i in range(N):
        tmp[i] = A[i] * 2
    for i in range(1, N):
        B[i] = tmp[i] + tmp[i - 1]
    s = 0.0
    for i in range(N):
        s += B[i]
    C[:] = s


def _top_level_loops(sdfg: dace.SDFG) -> List[LoopRegion]:
    return [b for b in sdfg.bfs_nodes(sdfg.start_block) if isinstance(b, LoopRegion)]


def _linkable_sources(sdfg: dace.SDFG) -> List[str]:
    return [o.clean_code for o in sdfg.generate_code() if o.language == 'cpp' and o.linkable]


def _signature(code: str, name: str) -> str:
    """ The parameter list of the first definition or declaration of a function whose name starts with ``name``. """
    return re.search(rf'void {name}\w*\(([^)]*)\)', code).group(1)


def _run_three_loops(sdfg: dace.SDFG):
    A = np.random.rand(20)
    B = np.zeros(20)
    C = np.zeros(20)
    sdfg(A=A, B=B, C=C, N=20)
    tmp = A * 2
    B_ref = np.zeros(20)
    B_ref[1:] = tmp[1:] + tmp[:-1]
    assert np.allclose(B, B_ref)
    assert np.allclose(C, B_ref.sum())


def _three_loop_regions(name: str,
                        placement: dtypes.FunctionPlacement,
                        unit: str = '',
                        inlining: dtypes.FunctionInlining = dtypes.FunctionInlining.Default) -> dace.SDFG:
    sdfg = three_loops.to_sdfg(simplify=True)
    sdfg.name = name
    loops = _top_level_loops(sdfg)
    xfh.wrap_in_function_region(loops[:2], 'first_loops', placement, unit, inlining)
    xfh.wrap_in_function_region(loops[2:], 'last_loop', placement, unit, inlining)
    sdfg.validate()
    return sdfg


def test_region_function_interface():
    sdfg = _three_loop_regions('fnregion_interface', dtypes.FunctionPlacement.CallerUnit)
    regions = [b for b in sdfg.nodes() if isinstance(b, CodeGeneratorFunctionRegion)]
    assert len(regions) == 2

    sources = _linkable_sources(sdfg)
    assert len(sources) == 1
    params = _signature(sources[0], 'first_loops')
    # Both arrays are restrict and the symbol is passed by value. The loop variable ``i`` is assigned inside and only
    # read by loops that assign it themselves, so it is local to the function
    assert re.search(r'__restrict__ A\b', params) and re.search(r'__restrict__ B\b', params)
    assert re.search(r'\bN\b', params)
    assert not re.search(r'\bi\b', params)
    _run_three_loops(sdfg)


@pytest.mark.parametrize('inlining, specifier', [
    (dtypes.FunctionInlining.Default, 'static void'),
    (dtypes.FunctionInlining.Inline, 'static inline void'),
    (dtypes.FunctionInlining.NoInline, 'static DACE_NOINLINE void'),
    (dtypes.FunctionInlining.ForceInline, 'static DACE_FORCEINLINE void'),
])
def test_region_inlining_hints(inlining: dtypes.FunctionInlining, specifier: str):
    sdfg = _three_loop_regions(f'fnregion_inlining_{inlining.name}', dtypes.FunctionPlacement.CallerUnit, '', inlining)
    sources = _linkable_sources(sdfg)
    assert len(sources) == 1
    assert len(re.findall(rf'{specifier} (first_loops|last_loop)', sources[0])) == 2
    _run_three_loops(sdfg)


@pytest.mark.parametrize('inlining', [dtypes.FunctionInlining.Inline, dtypes.FunctionInlining.ForceInline])
def test_region_separate_unit_cannot_inline(inlining: dtypes.FunctionInlining):
    sdfg = _three_loop_regions(f'fnregion_bad_inlining_{inlining.name}', dtypes.FunctionPlacement.SeparateUnit, '',
                               inlining)
    with pytest.raises(CodegenError, match='cannot be inlined'):
        sdfg.generate_code()


def test_region_separate_unit_noinline():
    sdfg = _three_loop_regions('fnregion_separate_noinline', dtypes.FunctionPlacement.SeparateUnit, '',
                               dtypes.FunctionInlining.NoInline)
    frame, *units = _linkable_sources(sdfg)
    assert len(re.findall(r'DACE_HIDDEN DACE_NOINLINE void (first_loops|last_loop)\w*\(.*;', frame)) == 2
    _run_three_loops(sdfg)


def test_region_name_attributes_restrict():
    sdfg = three_loops.to_sdfg(simplify=True)
    sdfg.name = 'fnregion_name_attributes'
    region = xfh.wrap_in_function_region(_top_level_loops(sdfg)[:2], 'first_loops')
    region.function_name = 'my_kernel'
    region.attributes = '__attribute__((cold))'
    region.restrict_arguments = False
    sources = _linkable_sources(sdfg)
    params = _signature(sources[0], 'my_kernel')
    assert re.search(r'static __attribute__\(\(cold\)\) void my_kernel\(', sources[0])
    assert '__restrict__' not in params and re.search(r'\bA\b', params)
    _run_three_loops(sdfg)


def test_region_separate_units():
    sdfg = _three_loop_regions('fnregion_separate', dtypes.FunctionPlacement.SeparateUnit)
    frame, *units = _linkable_sources(sdfg)
    assert len(units) == 2
    declarations = [line for line in frame.splitlines() if re.search(r'DACE_HIDDEN void (first_loops|last_loop)', line)]
    assert len(declarations) == 2 and all(line.rstrip().endswith(';') for line in declarations)
    for unit in units:
        assert len(re.findall(r'DACE_HIDDEN void (first_loops|last_loop)\w*\(.*\{', unit)) == 1
        assert 'struct fnregion_separate_state_t' in unit
    _run_three_loops(sdfg)


def test_region_shared_unit():
    sdfg = _three_loop_regions('fnregion_shared', dtypes.FunctionPlacement.SeparateUnit, 'unit0')
    frame, *units = _linkable_sources(sdfg)
    assert len(units) == 1
    assert len(re.findall(r'DACE_HIDDEN void (first_loops|last_loop)\w*\(.*\{', units[0])) == 2
    _run_three_loops(sdfg)


def _transient_loops_sdfg(name: str, lifetime: dtypes.AllocationLifetime) -> dace.SDFG:
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


@pytest.mark.parametrize('lifetime', [dtypes.AllocationLifetime.SDFG, dtypes.AllocationLifetime.Persistent])
def test_region_transient_used_after(lifetime: dtypes.AllocationLifetime):
    sdfg = _transient_loops_sdfg(f'fnregion_transient_{lifetime.name}', lifetime)
    xfh.wrap_in_function_region(_top_level_loops(sdfg), 'loops', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, unit = _linkable_sources(sdfg)
    params = _signature(unit, 'loops')
    if lifetime == dtypes.AllocationLifetime.Persistent:
        # Persistent data is passed as an argument instead of through the state struct, which compilers do not treat
        # as unaliased
        assert re.search(r'__restrict__ __\d+_tmp\b', params)
        assert not re.search(r'__state->__\d+_tmp', unit)
    else:
        assert re.search(r'__restrict__ tmp\b', params)

    A = np.random.rand(16)
    B = np.zeros(16)
    sdfg(A=A, B=B, N=16)
    assert np.allclose(B, (A + 1) * 3)


def test_region_persistent_data_with_nested_sdfg():
    """ Persistent data stays in the state struct if a nested SDFG in the region could reach it there. """
    sdfg = _transient_loops_sdfg('fnregion_persistent_nested', dtypes.AllocationLifetime.Persistent)
    inner = dace.SDFG('noop')
    inner.add_array('X', [1], dace.float64)
    state = inner.add_state()
    state.add_edge(state.add_tasklet('noop', {}, {'o'}, 'o = 0'), 'o', state.add_write('X'), None, dace.Memlet('X[0]'))
    loops = _top_level_loops(sdfg)
    body = loops[0].start_block
    nested = body.add_nested_sdfg(inner, {}, {'X'})
    body.add_edge(nested, 'X', body.add_write('B'), None, dace.Memlet('B[0]'))
    xfh.wrap_in_function_region(loops, 'loops', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, unit = _linkable_sources(sdfg)
    assert not re.search(r'__\d+_tmp', _signature(unit, 'loops'))
    assert re.search(r'__state->__\d+_tmp', unit)


def test_region_state_local_scalar():
    """ A scalar every state writes before reading it is a local of each function, even if several regions use it. """
    sdfg = dace.SDFG('fnregion_state_local')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_scalar('t', dace.float64, transient=True)
    loops = []
    for k in range(2):
        loop = LoopRegion(f'loop_{k}', 'i < N', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(loop, is_start_block=(k == 0))
        body = loop.add_state(f'body_{k}', is_start_block=True)
        first = body.add_tasklet(f'first_{k}', {'a'}, {'b'}, 'b = a * 2')
        second = body.add_tasklet(f'second_{k}', {'a'}, {'b'}, 'b = a + 1')
        t = body.add_access('t')
        body.add_edge(body.add_read('A'), None, first, 'a', dace.Memlet('A[i]'))
        body.add_edge(first, 'b', t, None, dace.Memlet('t'))
        body.add_edge(t, None, second, 'a', dace.Memlet('t'))
        body.add_edge(second, 'b', body.add_write('A'), None, dace.Memlet('A[i]'))
        if loops:
            sdfg.add_edge(loops[-1], loop, dace.InterstateEdge())
        loops.append(loop)
    for k, loop in enumerate(loops):
        xfh.wrap_in_function_region([loop], f'loop_fn_{k}', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, *units = _linkable_sources(sdfg)
    for k, unit in enumerate(units):
        assert not re.search(r'\bt\b', _signature(unit, f'loop_fn_{k}'))
        assert re.search(r'double t;', unit)
    A = np.random.rand(10)
    ref = (A * 2 + 1) * 2 + 1
    sdfg(A=A, N=10)
    assert np.allclose(A, ref)


def test_region_live_scalar_copied_in_and_out():
    """ A scalar the region updates and code around it reads is copied into a local and back. """
    sdfg = three_loops.to_sdfg(simplify=True)
    sdfg.name = 'fnregion_live_scalar'
    xfh.wrap_in_function_region([_top_level_loops(sdfg)[-1]], 'reduction', dtypes.FunctionPlacement.SeparateUnit)
    frame, unit = _linkable_sources(sdfg)
    assert re.search(r'double &__ref_s\b', _signature(unit, 'reduction'))
    assert re.search(r'double s = __ref_s;', unit) and re.search(r'__ref_s = s;', unit)
    _run_three_loops(sdfg)


def test_region_allocates_local_scalar():
    """ A scalar only the region uses is a local variable of the function, not an argument passed by reference. """
    sdfg = three_loops.to_sdfg(simplify=True)
    sdfg.name = 'fnregion_local_scalar'
    last_loop = _top_level_loops(sdfg)[-1]
    init = sdfg.in_edges(last_loop)[0].src
    final = sdfg.out_edges(last_loop)[0].dst
    xfh.wrap_in_function_region([init, last_loop, final], 'reduction', dtypes.FunctionPlacement.SeparateUnit)
    frame, unit = _linkable_sources(sdfg)
    assert not re.search(r'\bs\b', _signature(unit, 'reduction'))
    assert re.search(r'double s;', unit)
    assert not re.search(r'double s;', frame)
    _run_three_loops(sdfg)


@dace.program
def loop_with_inner_loops(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i in range(N):
        for j in range(N):
            B[i, j] = A[i, j] + i
        for j in range(N):
            B[i, j] = B[i, j] * 2


def test_region_in_loop_body():
    sdfg = loop_with_inner_loops.to_sdfg(simplify=True)
    outer = _top_level_loops(sdfg)[0]
    body = list(outer.bfs_nodes(outer.start_block))
    xfh.wrap_in_function_region(body, 'body', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, unit = _linkable_sources(sdfg)
    params = _signature(unit, 'body')
    # The enclosing loop's variable is read by value, the inner loop variable is local
    assert re.search(r'\bi\b', params) and '&i' not in params.replace(' ', '')
    assert not re.search(r'\bj\b', params)

    A = np.random.rand(8, 8)
    B = np.zeros((8, 8))
    sdfg(A=A, B=B, N=8)
    assert np.allclose(B, (A + np.arange(8)[:, None]) * 2)


def test_region_symbol_assigned_inside_used_after():
    """ A symbol the region assigns and the code after it reads is copied in and out through a reference. """
    sdfg = dace.SDFG('fnregion_live_out_symbol')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.add_symbol('k', dace.int64)
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    last = sdfg.add_state('last')
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'k': '5'}))
    sdfg.add_edge(second, last, dace.InterstateEdge())
    t = last.add_tasklet('use', {}, {'o'}, 'o = k')
    last.add_edge(t, 'o', last.add_write('A'), None, dace.Memlet('A[0]'))
    sdfg.remove_symbol('k')
    xfh.wrap_in_function_region([first, second], 'assigns', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, unit = _linkable_sources(sdfg)
    assert re.search(r'&\s*__ref_k\b', _signature(unit, 'assigns'))
    assert re.search(r'\bk = __ref_k;', unit) and re.search(r'__ref_k = k;', unit)
    A = np.zeros(1, dtype=np.int64)
    sdfg(A=A)
    assert A[0] == 5


@dace.program
def loop_with_break(A: dace.float64[N]):
    for i in range(N):
        if A[i] > 0.5:
            break
        A[i] = 1.0


def test_region_rejects_escaping_break():
    sdfg = loop_with_break.to_sdfg(simplify=True)
    outer = _top_level_loops(sdfg)[0]
    body = list(outer.bfs_nodes(outer.start_block))
    with pytest.raises(ValueError, match='leaves'):
        xfh.wrap_in_function_region(body, 'body')


def test_region_rejects_non_chain():
    sdfg = three_loops.to_sdfg(simplify=True)
    loops = _top_level_loops(sdfg)
    with pytest.raises(ValueError, match='chain'):
        xfh.wrap_in_function_region([loops[0], loops[2]], 'loops')


def test_region_units_define_their_nested_functions():
    """ Equal nested SDFGs in regions of different units get a function in each unit, as one cannot call another's. """
    inner = dace.SDFG('double_in_place')
    inner.add_array('X', [N], dace.float64)
    inner.add_state().add_mapped_tasklet('double', {'i': '0:N'}, {'x': dace.Memlet('X[i]')},
                                         'y = x * 2', {'y': dace.Memlet('X[i]')},
                                         external_edges=True)

    sdfg = dace.SDFG('fnregion_nested_functions')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    states = []
    for name in ('A', 'B'):
        state = sdfg.add_state(f'call_{name}', is_start_block=not states)
        node = state.add_nested_sdfg(copy.deepcopy(inner), {'X'}, {'X'})
        state.add_edge(state.add_read(name), None, node, 'X', dace.Memlet(f'{name}[0:N]'))
        state.add_edge(node, 'X', state.add_write(name), None, dace.Memlet(f'{name}[0:N]'))
        if states:
            sdfg.add_edge(states[-1], state, dace.InterstateEdge())
        states.append(state)
    for state in states:
        xfh.wrap_in_function_region([state], f'region_{state.label}', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, *units = _linkable_sources(sdfg)
    assert len(units) == 2
    for unit in units:
        assert len(re.findall(r'inline void double_in_place\w*\(.*\{', unit)) == 1

    A = np.random.rand(10)
    B = np.random.rand(10)
    A_ref, B_ref = A * 2, B * 2
    sdfg(A=A, B=B, N=10)
    assert np.allclose(A, A_ref) and np.allclose(B, B_ref)


_callback_values = []


def _record(x):
    _callback_values.append(x.copy())


@dace.program
def loops_with_callback(A: dace.float64[N]):
    for i in range(N):
        A[i] = A[i] + 1
    _record(A)
    for i in range(N):
        A[i] = A[i] * 2


def test_region_with_callback():
    """ A callback called in the region is passed as a function pointer, and its data through the state struct. """
    sdfg = loops_with_callback.to_sdfg(simplify=True)
    sdfg.name = 'fnregion_callback'
    blocks = list(sdfg.bfs_nodes(sdfg.start_block))
    xfh.wrap_in_function_region(blocks, 'with_callback', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, unit = _linkable_sources(sdfg)
    assert 'dace.callback' not in _signature(unit, 'with_callback')
    _callback_values.clear()
    A = np.random.rand(8)
    ref = (A + 1) * 2
    # The frontend parses ``_record`` and calls back into ``_callback_values.append``
    sdfg(A=A, N=8, _callback_values_append=lambda x: _callback_values.append(x.copy()))
    assert np.allclose(A, ref)
    assert len(_callback_values) == 1 and np.allclose(_callback_values[0], ref / 2)


def test_region_equal_functions_shared():
    """ Equal regions in separate units share one function; a region differing in a constant does not. """
    sdfg = dace.SDFG('fnregion_equal_functions')
    sdfg.add_array('A', [N], dace.float64)
    loops = []
    for k, expr in enumerate(['a + 1', 'a + 1', 'a + 2']):
        loop = LoopRegion(f'loop_{k}', 'i < N', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(loop, is_start_block=(k == 0))
        body = loop.add_state(f'body_{k}', is_start_block=True)
        t = body.add_tasklet('compute', {'a'}, {'b'}, f'b = {expr}')
        body.add_edge(body.add_read('A'), None, t, 'a', dace.Memlet('A[i]'))
        body.add_edge(t, 'b', body.add_write('A'), None, dace.Memlet('A[i]'))
        if loops:
            sdfg.add_edge(loops[-1], loop, dace.InterstateEdge())
        loops.append(loop)
    for k, loop in enumerate(loops):
        xfh.wrap_in_function_region([loop], f'step_{k}', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    frame, *units = _linkable_sources(sdfg)
    assert len(units) == 2
    calls = re.findall(r'\b(step_\d+_\d+)\(__state', frame)
    assert len(calls) == 3 and calls[0] == calls[1] != calls[2]
    A = np.random.rand(10)
    ref = A + 4
    sdfg(A=A, N=10)
    assert np.allclose(A, ref)


def _literal_scalar_sdfg(name: str, second_value: str, second_reads_data: bool) -> dace.SDFG:
    """
    Sets scalars ``c`` to a literal and ``d`` to ``second_value`` (which reads ``A[0]`` as ``a`` if
    ``second_reads_data``), then scales ``A`` by both in a loop.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_scalar('c', dace.float64, transient=True)
    sdfg.add_scalar('d', dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    for scalar, code, inputs in [('c', 'o = dace.float64(-2.0)', {}),
                                 ('d', f'o = {second_value}', {'a'} if second_reads_data else {})]:
        t = init.add_tasklet(f'set_{scalar}', inputs, {'o'}, code)
        if inputs:
            init.add_edge(init.add_read('A'), None, t, 'a', dace.Memlet('A[0]'))
        init.add_edge(t, 'o', init.add_write(scalar), None, dace.Memlet(scalar))
    loop = LoopRegion('scale', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('scale', {'a', 'f', 'g'}, {'b'}, 'b = a * f + g')
    body.add_edge(body.add_read('A'), None, t, 'a', dace.Memlet('A[i]'))
    body.add_edge(body.add_read('c'), None, t, 'f', dace.Memlet('c'))
    body.add_edge(body.add_read('d'), None, t, 'g', dace.Memlet('d'))
    body.add_edge(t, 'b', body.add_write('A'), None, dace.Memlet('A[i]'))
    xfh.wrap_in_function_region([loop], 'scale_fn', dtypes.FunctionPlacement.SeparateUnit)
    sdfg.validate()
    return sdfg


def test_region_literal_scalars_folded():
    """ A scalar only ever assigned one literal is a constant of the function; one computed from data is not. """
    sdfg = _literal_scalar_sdfg('fnregion_literals', 'a * 3', True)
    frame, unit = _linkable_sources(sdfg)
    params = _signature(unit, 'scale_fn')
    assert not re.search(r'\bc\b', params) and re.search(r'const double c = -2\.0;', unit)
    assert re.search(r'\bd\b', params)
    A = np.random.rand(10)
    ref = A * -2.0 + A[0] * 3
    sdfg(A=A, N=10)
    assert np.allclose(A, ref)


def test_region_literal_scalar_not_folded_with_call():
    """ A function call that is not a cast (here ``abs``) does not count as a literal. """
    sdfg = _literal_scalar_sdfg('fnregion_literal_call', 'abs(-3.0)', False)
    frame, unit = _linkable_sources(sdfg)
    assert re.search(r'\bd\b', _signature(unit, 'scale_fn'))


def test_region_serialization():
    sdfg = _three_loop_regions('fnregion_serialization', dtypes.FunctionPlacement.SeparateUnit, 'unit0')
    loaded = dace.SDFG.from_json(sdfg.to_json())
    regions = [b for b in loaded.nodes() if isinstance(b, CodeGeneratorFunctionRegion)]
    assert len(regions) == 2
    assert all(r.function_placement == dtypes.FunctionPlacement.SeparateUnit for r in regions)
    assert all(r.translation_unit == 'unit0' for r in regions)


if __name__ == '__main__':
    test_region_function_interface()
    for inlining, specifier in [(dtypes.FunctionInlining.Default, 'static void'),
                                (dtypes.FunctionInlining.Inline, 'static inline void'),
                                (dtypes.FunctionInlining.NoInline, 'static DACE_NOINLINE void'),
                                (dtypes.FunctionInlining.ForceInline, 'static DACE_FORCEINLINE void')]:
        test_region_inlining_hints(inlining, specifier)
    test_region_separate_unit_cannot_inline(dtypes.FunctionInlining.Inline)
    test_region_separate_unit_cannot_inline(dtypes.FunctionInlining.ForceInline)
    test_region_separate_unit_noinline()
    test_region_name_attributes_restrict()
    test_region_separate_units()
    test_region_shared_unit()
    test_region_transient_used_after(dtypes.AllocationLifetime.SDFG)
    test_region_transient_used_after(dtypes.AllocationLifetime.Persistent)
    test_region_persistent_data_with_nested_sdfg()
    test_region_state_local_scalar()
    test_region_live_scalar_copied_in_and_out()
    test_region_allocates_local_scalar()
    test_region_in_loop_body()
    test_region_symbol_assigned_inside_used_after()
    test_region_rejects_escaping_break()
    test_region_rejects_non_chain()
    test_region_units_define_their_nested_functions()
    test_region_with_callback()
    test_region_equal_functions_shared()
    test_region_literal_scalars_folded()
    test_region_literal_scalar_not_folded_with_call()
    test_region_serialization()
