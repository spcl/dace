# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`NestedGPUDeviceMapLowering`: nested ``GPU_Device`` maps become one bound-checked kernel."""
import re
from typing import Literal

import dace
import numpy as np
import pytest
from dace.sdfg.state import ConditionalBlock, StateSubgraphView
from dace.transformation import helpers
from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering

GPU_DEVICE = dace.dtypes.ScheduleType.GPU_Device
GPU_GLOBAL = dace.dtypes.StorageType.GPU_Global
K = dace.symbol('K', dtype=dace.int32)


def build_outer_with_two_sibling_inner_gpu_kernels(j_ranges: tuple[str, str] = ('0:J + 1', '0:J')) -> dace.SDFG:
    """``vertical_loop (0:K)`` wrapping a NestedSDFG with sibling GPU maps over ``(j_range, 0:I)``."""
    J = dace.symbol('J', dtype=dace.int32)
    I_SIZE = dace.symbol('I', dtype=dace.int32)

    sdfg = dace.SDFG('lower_nested_gpu_maps')
    sdfg.add_array('A', [K, J + 1, I_SIZE], dace.float64, storage=GPU_GLOBAL)
    sdfg.add_array('B', [K, J, I_SIZE], dace.float64, storage=GPU_GLOBAL)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical_loop', dict(__k='0:K'), schedule=GPU_DEVICE)

    inner = dace.SDFG('nested_sdfg')
    inner.add_symbol('__k', dace.int32)
    inner.add_array('a_in', [K, J + 1, I_SIZE], dace.float64, storage=GPU_GLOBAL)
    inner.add_array('b_out', [K, J, I_SIZE], dace.float64, storage=GPU_GLOBAL)
    inner_state = inner.add_state('nested_root', is_start_block=True)

    for label, j_range, out, value in (('a', j_ranges[0], 'a_in', '1.0'), ('b', j_ranges[1], 'b_out', '2.0')):
        me, mx = inner_state.add_map(f'horizontal_loop_{label}', dict(__j=j_range, __i='0:I'), schedule=GPU_DEVICE)
        tasklet = inner_state.add_tasklet(f'write_{label}', {}, {'_o': dace.float64}, f'_o = {value}')
        inner_state.add_memlet_path(me, tasklet, memlet=dace.Memlet())
        inner_state.add_memlet_path(tasklet,
                                    mx,
                                    inner_state.add_write(out),
                                    src_conn='_o',
                                    memlet=dace.Memlet(f'{out}[__k, __j, __i]'))

    nsdfg = state.add_nested_sdfg(inner, {}, {'a_in': None, 'b_out': None})
    state.add_memlet_path(outer_me, nsdfg, memlet=dace.Memlet())
    state.add_memlet_path(nsdfg,
                          outer_mx,
                          state.add_write('A'),
                          src_conn='a_in',
                          memlet=dace.Memlet('A[__k, 0:J + 1, 0:I]'))
    state.add_memlet_path(nsdfg,
                          outer_mx,
                          state.add_write('B'),
                          src_conn='b_out',
                          memlet=dace.Memlet('B[__k, 0:J, 0:I]'))
    return sdfg


def gpu_device_maps(sdfg: dace.SDFG) -> list[tuple[bool, dace.nodes.MapEntry]]:
    """Every ``GPU_Device`` map of the hierarchy, flagged whether it sits in a NestedSDFG."""
    return [(sub is not sdfg, node) for sub in sdfg.all_sdfgs_recursive() for state in sub.states()
            for node in state.nodes() if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == GPU_DEVICE]


def count_gpu_device_maps(sdfg: dace.SDFG) -> tuple[int, int]:
    """``(top_level, inside_nsdfgs)`` counts of ``GPU_Device`` ``MapEntry``s."""
    nested = [in_nsdfg for in_nsdfg, _ in gpu_device_maps(sdfg)]
    return nested.count(False), nested.count(True)


def kernel_entry(sdfg: dace.SDFG) -> dace.nodes.MapEntry:
    return next(node for in_nsdfg, node in gpu_device_maps(sdfg) if not in_nsdfg)


def nested_sdfg_nodes(sdfg: dace.SDFG) -> list[dace.nodes.NestedSDFG]:
    return [
        node for sub in sdfg.all_sdfgs_recursive() for state in sub.states() for node in state.nodes()
        if isinstance(node, dace.nodes.NestedSDFG)
    ]


def test_pass_flattens_nested_gpu_kernels_validates_clean():
    """The kernel gains the inner params, the inner maps disappear, and each guarded body binds ``__k``."""
    sdfg = build_outer_with_two_sibling_inner_gpu_kernels()
    assert count_gpu_device_maps(sdfg) == (1, 2)

    assert NestedGPUDeviceMapLowering().apply_pass(sdfg, {}) == 2

    assert count_gpu_device_maps(sdfg) == (1, 0)
    assert set(kernel_entry(sdfg).map.params) >= {'__j', '__i'}, kernel_entry(sdfg).map.params
    guarded = [n for n in nested_sdfg_nodes(sdfg) if n.label.startswith('if_of_nested_')]
    assert len(guarded) == 2, [n.label for n in guarded]
    for node in guarded:
        assert '__k' in node.symbol_mapping, (node.label, sorted(node.symbol_mapping))
    sdfg.validate()


def build_nested_kernel_with_internal_inout_node() -> dace.SDFG:
    """Kernel wrapping a NestedSDFG whose inner GPU map reads, bumps and rewrites ``c_io`` through an
    internal non-transient access node: ``c_read -> map -> t1 -> c_mid -> t2 -> map -> c_write``."""
    J = dace.symbol('J', dtype=dace.int32)

    sdfg = dace.SDFG('lower_nested_inout')
    sdfg.add_array('C', [K, J], dace.float64, storage=GPU_GLOBAL)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical_loop', dict(__k='0:K'), schedule=GPU_DEVICE)

    inner = dace.SDFG('nested_sdfg')
    inner.add_symbol('__k', dace.int32)
    inner.add_array('c_io', [K, J], dace.float64, storage=GPU_GLOBAL)
    inner_state = inner.add_state('nested_root', is_start_block=True)

    me, mx = inner_state.add_map('horizontal_loop', dict(__j='0:J'), schedule=GPU_DEVICE)
    c_read = inner_state.add_read('c_io')
    c_mid = inner_state.add_access('c_io')
    c_write = inner_state.add_write('c_io')
    t1 = inner_state.add_tasklet('bump', {'i': None}, {'o': None}, 'o = i + 1.0')
    t2 = inner_state.add_tasklet('bump2', {'i': None}, {'o': None}, 'o = i + 2.0')
    inner_state.add_memlet_path(c_read, me, t1, dst_conn='i', memlet=dace.Memlet('c_io[__k, __j]'))
    inner_state.add_edge(t1, 'o', c_mid, None, dace.Memlet('c_io[__k, __j]'))
    inner_state.add_edge(c_mid, None, t2, 'i', dace.Memlet('c_io[__k, __j]'))
    inner_state.add_memlet_path(t2, mx, c_write, src_conn='o', memlet=dace.Memlet('c_io[__k, __j]'))

    nsdfg = state.add_nested_sdfg(inner, {'c_io': None}, {'c_io': None})
    state.add_memlet_path(state.add_read('C'), outer_me, nsdfg, dst_conn='c_io', memlet=dace.Memlet('C[__k, 0:J]'))
    state.add_memlet_path(nsdfg, outer_mx, state.add_write('C'), src_conn='c_io', memlet=dace.Memlet('C[__k, 0:J]'))
    return sdfg


def test_inner_kernel_with_internal_inout_node_lowers_clean():
    """An internal inout node of a non-transient array is nested by data name, not as an ``AccessNode`` leaking
    into the NestedSDFG connectors."""
    sdfg = build_nested_kernel_with_internal_inout_node()

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    assert count_gpu_device_maps(sdfg) == (1, 0)
    for node in nested_sdfg_nodes(sdfg):
        for conn in [*node.in_connectors, *node.out_connectors]:
            assert isinstance(conn, str), conn


def build_inner_kernel_with_range(inner_range: str,
                                  symbol_mapping: dict[str, str] | None = None,
                                  param: str = '__j',
                                  loop_var: str | None = None,
                                  sequential_scope: Literal['inside', 'around'] | None = None) -> dace.SDFG:
    """Kernel ``__k`` over ``0:K`` wrapping a NestedSDFG with one ``GPU_Device`` map ``param`` over ``inner_range``.

    With ``loop_var``, the map sits in a loop ``loop_var = 0..3`` inside the NestedSDFG. With ``sequential_scope``,
    it sits in a sequential map ``__t`` over ``0:2``, inside the NestedSDFG or around the NestedSDFG in the kernel.
    """
    symbol_mapping = {'K': 'K', **(symbol_mapping or {'__k': '__k'})}
    sdfg = dace.SDFG('inner_range_kernel')
    sdfg.add_symbol('K', dace.int32)
    for value in symbol_mapping.values():
        for name in map(str, dace.symbolic.pystr_to_symbolic(value).free_symbols):
            if name not in sdfg.symbols and name != '__k':
                sdfg.add_symbol(name, dace.int32)
    sdfg.add_array('A', [K, 32], dace.float64, storage=GPU_GLOBAL)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical', dict(__k='0:K'), schedule=GPU_DEVICE)

    inner = dace.SDFG('nested')
    for name in symbol_mapping:
        inner.add_symbol(name, dace.int32)
    inner.add_array('a_out', [K, 32], dace.float64, storage=GPU_GLOBAL)
    region = inner
    if loop_var is not None:
        region = dace.sdfg.state.LoopRegion('time', f'{loop_var} < 4', loop_var, f'{loop_var} = 0',
                                            f'{loop_var} = {loop_var} + 1')
        inner.add_node(region, is_start_block=True)
    inner_state = region.add_state('root', is_start_block=True)
    scopes = [inner_state.add_map('horizontal', {param: inner_range}, schedule=GPU_DEVICE)]
    if sequential_scope == 'inside':
        scopes.insert(0, inner_state.add_map('seq', dict(__t='0:2'), schedule=dace.dtypes.ScheduleType.Sequential))
    tasklet = inner_state.add_tasklet('w', {}, {'_a': dace.float64}, '_a = 1.0')
    inner_state.add_memlet_path(*(entry for entry, _ in scopes), tasklet, memlet=dace.Memlet())
    inner_state.add_memlet_path(tasklet,
                                *(exit_node for _, exit_node in reversed(scopes)),
                                inner_state.add_write('a_out'),
                                src_conn='_a',
                                memlet=dace.Memlet(f'a_out[__k, {param}]'))

    nsdfg = state.add_nested_sdfg(inner, {}, {'a_out': None}, symbol_mapping=symbol_mapping)
    kernel_scopes = [(outer_me, outer_mx)]
    if sequential_scope == 'around':
        kernel_scopes.append(state.add_map('seq', dict(__t='0:2'), schedule=dace.dtypes.ScheduleType.Sequential))
    state.add_memlet_path(*(entry for entry, _ in kernel_scopes), nsdfg, memlet=dace.Memlet())
    state.add_memlet_path(nsdfg,
                          *(exit_node for _, exit_node in reversed(kernel_scopes)),
                          state.add_write('A'),
                          src_conn='a_out',
                          memlet=dace.Memlet('A[__k, 0:32]'))
    return sdfg


def absorbed_range(sdfg: dace.SDFG, param: str) -> tuple:
    kernel = kernel_entry(sdfg)
    return kernel.map.range[kernel.map.params.index(param)]


def guard_conditions(sdfg: dace.SDFG) -> dict[str, str]:
    """Condition string of every ``ConditionalBlock`` in the hierarchy, by block label."""
    return {
        block.label: block.branches[0][0].as_string
        for sub in sdfg.all_sdfgs_recursive()
        for block in sub.all_control_flow_blocks() if isinstance(block, ConditionalBlock)
    }


@pytest.mark.parametrize('inner_range, symbol_mapping, expected', [
    pytest.param('5:10', None, ('5', '9'), id='lower_bound_is_not_widened_to_the_origin'),
    pytest.param('0:M', {
        '__k': '__k',
        'M': 'N',
        'N': 'M'
    }, ('0', 'N - 1'),
                 id='swapping_symbol_mapping_is_applied_simultaneously'),
])
def test_the_absorbed_range_is_the_inner_range_as_the_kernel_sees_it(inner_range, symbol_mapping, expected):
    """A kernel must not launch iterations no inner map owns, nor read a bound through a mapping applied twice."""
    sdfg = build_inner_kernel_with_range(inner_range, symbol_mapping)
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, _ = absorbed_range(sdfg, '__j')
    assert (str(begin), str(end)) == expected


@pytest.mark.parametrize('nest_twice', [False, True])
def test_a_bound_naming_a_nested_scope_symbol_is_hoisted_in_the_kernel_symbols(nest_twice):
    """``M`` exists only inside the NestedSDFG(s); the hoisted bound must say the outer ``K`` it maps to."""
    sdfg = build_inner_kernel_with_range('0:M', {'__k': '__k', 'M': 'K'})
    if nest_twice:
        inner = nested_sdfg_nodes(sdfg)[0].sdfg
        body = inner.start_block
        helpers.nest_state_subgraph(inner, body, StateSubgraphView(body, body.nodes()))
        assert len(nested_sdfg_nodes(sdfg)) == 2
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, _ = absorbed_range(sdfg, '__j')
    assert (str(begin), str(end)) == ('0', 'K - 1')
    assert 'M' not in sdfg.free_symbols


@pytest.mark.parametrize('inner_range, has_step_term', [('0:10:2', True), ('0:10', False)])
def test_bound_check_checks_the_step_only_of_a_strided_range(inner_range, has_step_term):
    """A strided map must not let the iterations it skips into its body; a unit step needs no check."""
    sdfg = build_inner_kernel_with_range(inner_range)
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    (condition, ) = guard_conditions(sdfg).values()
    assert ('% 2' in condition) == has_step_term, condition


@pytest.mark.parametrize('j_ranges', [('0:10:2', '1:10:3'), ('3:12:3', '0:6'), ('4:8', '0:10:4')])
def test_each_guard_admits_exactly_the_iterations_of_its_own_map(j_ranges):
    """Siblings share one bounding-box kernel dimension, so the guard alone keeps a body from the iterations
    only its sibling owns."""
    sdfg = build_outer_with_two_sibling_inner_gpu_kernels(j_ranges)
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, step = (int(x) for x in absorbed_range(sdfg, '__j'))
    launched = range(begin, end + 1, step)
    conditions = guard_conditions(sdfg)
    for label, j_range in zip('ab', j_ranges, strict=True):
        owned_begin, owned_end, owned_step = (int(x) for x in dace.subsets.Range.from_string(j_range)[0])
        condition = conditions[f'bound_check_horizontal_loop_{label}']
        admitted = [j for j in launched if eval(condition, {'__j': j, '__i': 0, 'I': 1})]
        assert admitted == list(range(owned_begin, owned_end + 1, owned_step)), (label, condition, admitted)


def test_a_descending_inner_map_is_refused_not_silently_lowered():
    """``Map.validate`` rejects negative steps, so an inner map walking downward never reaches a guard that would
    admit none of its iterations."""
    sdfg = build_inner_kernel_with_range('9:-1:-1')

    with pytest.raises(dace.sdfg.InvalidSDFGError, match='negative step'):
        NestedGPUDeviceMapLowering().apply_pass(sdfg, {})


def kernel_with_directly_nested_gpu_maps(ranges: tuple[str, ...], in_nested_sdfg: bool = False) -> dace.SDFG:
    """``GPU_Device`` maps over ``ranges`` nested straight in each other's scope, writing ``A[k, j, i] = 1``.

    The maps are joined only by empty memlets, which carry ordering rather than data. With ``in_nested_sdfg``, every
    map below the outermost one sits in a NestedSDFG.
    """
    params = ['__k', '__j', '__i'][:len(ranges)]
    sdfg = dace.SDFG(f'direct_{len(ranges)}_{"nested" if in_nested_sdfg else "flat"}')
    sdfg.add_array('A', [16, 32, 32][:len(ranges)], dace.float64, storage=GPU_GLOBAL)
    state = sdfg.add_state('s')
    scopes = [state.add_map(f'map{p}', {p: r}, schedule=GPU_DEVICE) for p, r in zip(params, ranges, strict=True)]
    tasklet = state.add_tasklet('w', {}, {'_v': dace.float64}, '_v = 1.0')
    state.add_memlet_path(*(entry for entry, _ in scopes), tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet,
                          *(exit_node for _, exit_node in reversed(scopes)),
                          state.add_write('A'),
                          src_conn='_v',
                          memlet=dace.Memlet(f'A[{", ".join(params)}]'))
    if in_nested_sdfg:
        kernel, kernel_exit = scopes[0]
        body = StateSubgraphView(state, list(state.all_nodes_between(kernel, kernel_exit)))
        helpers.nest_state_subgraph(sdfg, state, body)
    sdfg.validate()
    return sdfg


def run_on_cpu(sdfg: dace.SDFG, **arrays: np.ndarray) -> None:
    """Run a lowered kernel on the host: every ``GPU_Device`` map becomes multicore and every array host memory."""
    for sub in sdfg.all_sdfgs_recursive():
        for desc in sub.arrays.values():
            if desc.storage == GPU_GLOBAL:
                desc.storage = dace.dtypes.StorageType.CPU_Heap
        for state in sub.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == GPU_DEVICE:
                    node.map.schedule = dace.dtypes.ScheduleType.CPU_Multicore
    sdfg(**arrays)


def test_directly_nested_gpu_map_lowers_without_detaching_its_body():
    """The ordering edge into the inner scope must be re-anchored, not dropped with the map."""
    sdfg = kernel_with_directly_nested_gpu_maps(('0:16', '0:32'))
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    assert count_gpu_device_maps(sdfg) == (1, 0)
    assert set(kernel_entry(sdfg).map.params) == {'__k', '__j'}, kernel_entry(sdfg).map.params
    state = sdfg.states()[0]
    body = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert state.in_degree(body) > 0, 'the guarded body was detached from the kernel scope'


@pytest.mark.parametrize('in_nested_sdfg', [False, True])
def test_gpu_maps_nested_three_deep_become_one_kernel_computing_each_iteration_once(in_nested_sdfg):
    """Every layer is absorbed into the outermost kernel, whether the inner maps are direct or behind a NestedSDFG."""
    sdfg = kernel_with_directly_nested_gpu_maps(('0:16', '2:30:4', '1:7'), in_nested_sdfg)

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    assert count_gpu_device_maps(sdfg) == (1, 0)
    assert kernel_entry(sdfg).map.params == ['__k', '__j', '__i']
    A = np.zeros((16, 32, 32))
    run_on_cpu(sdfg, A=A)
    expected = np.zeros((16, 32, 32))
    expected[:, 2:30:4, 1:7] = 1.0
    assert np.array_equal(A, expected)


@pytest.mark.parametrize('in_nested_sdfg', [False, True])
def test_a_deeper_bound_naming_an_enclosing_map_param_is_refused(in_nested_sdfg):
    """A triangular nest has no rectangular grid: ``__i in 0:__j`` cannot be sized where the kernel is launched."""
    sdfg = kernel_with_directly_nested_gpu_maps(('0:16', '0:32', '0:__j'), in_nested_sdfg)

    with pytest.raises(NotImplementedError, match='__j'):
        NestedGPUDeviceMapLowering().apply_pass(sdfg, {})


@pytest.mark.parametrize('inner_range, loop_var, unavailable', [('__k:__k + 4', None, '__k'), ('0:t + 1', 't', "'t'")])
def test_a_bound_the_host_cannot_evaluate_is_refused(inner_range, loop_var, unavailable):
    """The grid is sized on the host, where neither a kernel parameter nor an in-kernel loop variable exists."""
    sdfg = build_inner_kernel_with_range(inner_range, loop_var=loop_var)
    sdfg.validate()

    with pytest.raises(NotImplementedError, match=unavailable):
        NestedGPUDeviceMapLowering().apply_pass(sdfg, {})


def build_inner_kernel_reusing_the_outer_param(name: str) -> dace.SDFG:
    """Kernel ``i`` writes ``B[i]``; a same-state inner kernel reuses ``i`` and writes ``A[i, j] = 10 * i + j``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [3, 4], dace.float64, storage=GPU_GLOBAL)
    sdfg.add_array('B', [1], dace.float64, storage=GPU_GLOBAL)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('outer', dict(i='0:1'), schedule=GPU_DEVICE)
    outer_me.map.gpu_block_size = [32, 1, 1]
    inner_me, inner_mx = state.add_map('inner', dict(i='0:3', j='0:4'), schedule=GPU_DEVICE)

    seven = state.add_tasklet('seven', {}, {'b': dace.float64}, 'b = 7.0')
    state.add_memlet_path(outer_me, seven, memlet=dace.Memlet())
    state.add_memlet_path(seven, outer_mx, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))

    index = state.add_tasklet('index', {}, {'a': dace.float64}, 'a = 10 * i + j')
    state.add_memlet_path(outer_me, inner_me, index, memlet=dace.Memlet())
    state.add_memlet_path(index, inner_mx, outer_mx, state.add_write('A'), src_conn='a', memlet=dace.Memlet('A[i, j]'))
    return sdfg


def test_an_inner_param_reusing_a_kernel_param_gets_its_own_dimension():
    """The reused name appears once in the kernel params; the outer body keeps the outer index."""
    sdfg = build_inner_kernel_reusing_the_outer_param('inner_reuses_outer_param')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    state = sdfg.states()[0]
    params = kernel_entry(sdfg).map.params
    assert len(params) == 3 and len(set(params)) == 3, params
    assert params[0] == 'i' and 'j' in params, params
    b_write = next(e for e in state.edges() if e.data.data == 'B' and isinstance(e.src, dace.nodes.Tasklet))
    assert str(b_write.data.subset) == 'i', b_write.data.subset


def test_the_flattened_kernel_declares_each_index_once():
    """nvcc rejects a second declaration of one index name in a kernel scope."""
    sdfg = build_inner_kernel_reusing_the_outer_param('inner_reuses_outer_param_codegen')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})
    code = ''.join(obj.clean_code for obj in sdfg.generate_code() if obj.title == 'CUDA')

    kernel = code.split('__global__ void')[1].split('DACE_EXPORTED')[0]
    declared = re.findall(r'\bint\s+(\w+)\s*=', kernel)
    assert declared.count('i') == 1 and declared.count('j') == 1, declared


@pytest.mark.gpu
def test_the_flattened_kernel_computes_both_indices():
    """Outer and inner writes each land at their own index."""
    import cupy  # Only present on GPU runners.
    sdfg = build_inner_kernel_reusing_the_outer_param('inner_reuses_outer_param_run')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})
    A = cupy.zeros((3, 4), dtype=cupy.float64)
    B = cupy.zeros((1, ), dtype=cupy.float64)

    sdfg(A=A, B=B)

    expected = np.array([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
    assert np.array_equal(cupy.asnumpy(A), expected)
    assert cupy.asnumpy(B).tolist() == [7.0]


def test_an_inner_param_shadowing_a_nested_symbol_is_renamed():
    """The NestedSDFG binds ``j`` to the outer ``N``; the absorbed param must not inherit that binding."""
    sdfg = build_inner_kernel_with_range('0:4', {'__k': '__k', 'j': 'N'}, param='j')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    absorbed = kernel_entry(sdfg).map.params[1]
    assert absorbed != 'j', kernel_entry(sdfg).map.params
    nested = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert str(nested.symbol_mapping[absorbed]) == absorbed, nested.symbol_mapping
    assert str(nested.symbol_mapping['j']) == 'N', nested.symbol_mapping


def test_a_transient_local_to_the_inner_body_moves_with_it():
    """A scalar the inner body allocates for itself crosses no map edge but must still be declared."""
    sdfg = build_inner_kernel_with_range('0:32')
    inner = nested_sdfg_nodes(sdfg)[0].sdfg
    inner.add_scalar('tmp', dace.float64, transient=True, storage=dace.dtypes.StorageType.Register)
    body = inner.start_block
    write = next(n for n in body.nodes() if isinstance(n, dace.nodes.Tasklet))
    exit_edge = body.out_edges(write)[0]
    copy = body.add_tasklet('copy', {'_in': dace.float64}, {'_out': dace.float64}, '_out = _in')
    tmp = body.add_access('tmp')
    body.add_edge(write, '_a', tmp, None, dace.Memlet('tmp[0]'))
    body.add_edge(tmp, None, copy, '_in', dace.Memlet('tmp[0]'))
    body.add_edge(copy, '_out', exit_edge.dst, exit_edge.dst_conn, exit_edge.data)
    body.remove_edge(exit_edge)
    sdfg.validate()

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    holder = next(sub for sub in sdfg.all_sdfgs_recursive() for state in sub.states() for n in state.nodes()
                  if isinstance(n, dace.nodes.AccessNode) and n.data == 'tmp')
    assert 'tmp' in holder.arrays and holder.arrays['tmp'].transient


def test_a_sequential_map_below_the_kernel_is_not_absorbed():
    """Only ``GPU_Device`` maps are flattened; absorbing a sequential one would parallelize it."""
    sdfg = build_inner_kernel_with_range('0:32')
    inner_map = next(node for sub in sdfg.all_sdfgs_recursive() if sub is not sdfg for state in sub.states()
                     for node in state.nodes() if isinstance(node, dace.nodes.MapEntry))
    inner_map.map.schedule = dace.dtypes.ScheduleType.Sequential

    assert NestedGPUDeviceMapLowering().apply_pass(sdfg, {}) is None
    assert kernel_entry(sdfg).map.params == ['__k']
    assert inner_map.map.schedule == dace.dtypes.ScheduleType.Sequential


@pytest.mark.parametrize('sequential_scope', ['inside', 'around'])
def test_a_gpu_map_behind_a_sequential_map_is_absorbed(sequential_scope):
    """A sequential scope between the kernel and an inner GPU map, above or below the NestedSDFG, does not hide it."""
    sdfg = build_inner_kernel_with_range('0:32', sequential_scope=sequential_scope)

    assert NestedGPUDeviceMapLowering().apply_pass(sdfg, {}) == 1

    assert count_gpu_device_maps(sdfg) == (1, 0)
    assert kernel_entry(sdfg).map.params == ['__k', '__j']


def test_sibling_bounds_carrying_an_overapproximation_are_unioned():
    """A ``SymExpr`` bound (main plus overapproximation) must not break the sibling union."""
    sdfg = build_outer_with_two_sibling_inner_gpu_kernels()
    first = next(node for in_nsdfg, node in gpu_device_maps(sdfg) if in_nsdfg)
    rng = list(first.map.range)
    rng[0] = (0, dace.symbolic.SymExpr('J', 'J + 1'), 1)
    first.map.range = dace.subsets.Range(rng)

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, _ = absorbed_range(sdfg, '__j')
    assert begin == 0
    widest = end.approx if isinstance(end, dace.symbolic.SymExpr) else end
    assert widest == dace.symbolic.pystr_to_symbolic('J + 1'), end


if __name__ == '__main__':
    test_pass_flattens_nested_gpu_kernels_validates_clean()
    test_inner_kernel_with_internal_inout_node_lowers_clean()
    for inner_range, symbol_mapping, expected in [('5:10', None, ('5', '9')),
                                                  ('0:M', {
                                                      '__k': '__k',
                                                      'M': 'N',
                                                      'N': 'M'
                                                  }, ('0', 'N - 1'))]:
        test_the_absorbed_range_is_the_inner_range_as_the_kernel_sees_it(inner_range, symbol_mapping, expected)
    for nest_twice in [False, True]:
        test_a_bound_naming_a_nested_scope_symbol_is_hoisted_in_the_kernel_symbols(nest_twice)
    for inner_range, has_step_term in [('0:10:2', True), ('0:10', False)]:
        test_bound_check_checks_the_step_only_of_a_strided_range(inner_range, has_step_term)
    for j_ranges in [('0:10:2', '1:10:3'), ('3:12:3', '0:6'), ('4:8', '0:10:4')]:
        test_each_guard_admits_exactly_the_iterations_of_its_own_map(j_ranges)
    test_a_descending_inner_map_is_refused_not_silently_lowered()
    test_directly_nested_gpu_map_lowers_without_detaching_its_body()
    for in_nested_sdfg in [False, True]:
        test_gpu_maps_nested_three_deep_become_one_kernel_computing_each_iteration_once(in_nested_sdfg)
    for in_nested_sdfg in [False, True]:
        test_a_deeper_bound_naming_an_enclosing_map_param_is_refused(in_nested_sdfg)
    for inner_range, loop_var, unavailable in [('__k:__k + 4', None, '__k'), ('0:t + 1', 't', "'t'")]:
        test_a_bound_the_host_cannot_evaluate_is_refused(inner_range, loop_var, unavailable)
    test_an_inner_param_reusing_a_kernel_param_gets_its_own_dimension()
    test_the_flattened_kernel_declares_each_index_once()
    test_an_inner_param_shadowing_a_nested_symbol_is_renamed()
    test_a_transient_local_to_the_inner_body_moves_with_it()
    test_a_sequential_map_below_the_kernel_is_not_absorbed()
    for sequential_scope in ['inside', 'around']:
        test_a_gpu_map_behind_a_sequential_map_is_absorbed(sequential_scope)
    test_sibling_bounds_carrying_an_overapproximation_are_unioned()
    test_the_flattened_kernel_computes_both_indices()
