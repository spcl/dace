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
        # Persistent data is reached through the state struct
        assert not re.search(r'\btmp\b', params)
    else:
        assert re.search(r'__restrict__ tmp\b', params)

    A = np.random.rand(16)
    B = np.zeros(16)
    sdfg(A=A, B=B, N=16)
    assert np.allclose(B, (A + 1) * 3)


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
    """ A symbol the region assigns and the code after it reads is passed by reference. """
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
    assert re.search(r'&\s*k\b', _signature(unit, 'assigns'))
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
    test_region_allocates_local_scalar()
    test_region_in_loop_body()
    test_region_symbol_assigned_inside_used_after()
    test_region_rejects_escaping_break()
    test_region_rejects_non_chain()
    test_region_units_define_their_nested_functions()
    test_region_serialization()
