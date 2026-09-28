# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`NestedGPUDeviceMapLowering`.

The pass rewrites the ``GPU_Device`` nested inside ``GPU_Device`` pattern that the
experimental CUDA codegen explicitly refuses (``Dynamic parallelism ... not supported``):
the outer kernel's iteration range is union-expanded with the inner kernels' params, each
inner kernel's body is moved into a ``NestedSDFG`` guarded by an if-bound-check, and the
inner ``GPU_Device`` map itself is removed. The result is a single flat ``GPU_Device``
kernel whose body uses if-guards to fan out to each original inner kernel's range.
"""
import re

import dace
import numpy as np
import pytest

K = dace.symbol('K', dtype=dace.int32)

from dace.transformation.passes.lower_nested_gpu_device_maps import NestedGPUDeviceMapLowering


def build_outer_with_two_sibling_inner_gpu_kernels() -> dace.SDFG:
    """``vertical_loop (0:K)`` (GPU_Device) wrapping a NestedSDFG that holds two sibling
    ``horizontal_loop`` ``GPU_Device`` maps of ranges ``(0:J+1, 0:I)`` and ``(0:J, 0:I)``.

    Mirrors the ICON ``native_functions_main`` reproducer in miniature.
    """
    K = dace.symbol('K', dtype=dace.int32)
    J = dace.symbol('J', dtype=dace.int32)
    I = dace.symbol('I', dtype=dace.int32)

    sdfg = dace.SDFG('lower_nested_gpu_maps')
    sdfg.add_array('A', [K, J + 1, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [K, J, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical_loop', dict(__k='0:K'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    inner = dace.SDFG('nested_sdfg')
    inner.add_symbol('__k', dace.int32)
    inner.add_array('a_in', [J + 1, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner.add_array('b_out', [J, I], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('nested_root', is_start_block=True)

    # Inner kernel 1: writes a_in slice via WCR-free overwrite (range J+1, I).
    me_a, mx_a = inner_state.add_map('horizontal_loop_a',
                                     dict(__j='0:J + 1', __i='0:I'),
                                     schedule=dace.dtypes.ScheduleType.GPU_Device)
    t_a = inner_state.add_tasklet('write_a', {}, {'_a': dace.float64}, '_a = 1.0')
    inner_state.add_memlet_path(me_a, t_a, memlet=dace.Memlet())
    inner_state.add_memlet_path(t_a,
                                mx_a,
                                inner_state.add_write('a_in'),
                                src_conn='_a',
                                memlet=dace.Memlet('a_in[__j, __i]'))

    # Inner kernel 2: writes b_out (range J, I).
    me_b, mx_b = inner_state.add_map('horizontal_loop_b',
                                     dict(__j='0:J', __i='0:I'),
                                     schedule=dace.dtypes.ScheduleType.GPU_Device)
    t_b = inner_state.add_tasklet('write_b', {}, {'_b': dace.float64}, '_b = 2.0')
    inner_state.add_memlet_path(me_b, t_b, memlet=dace.Memlet())
    inner_state.add_memlet_path(t_b,
                                mx_b,
                                inner_state.add_write('b_out'),
                                src_conn='_b',
                                memlet=dace.Memlet('b_out[__j, __i]'))

    nsdfg = state.add_nested_sdfg(inner, {}, {"a_in": None, "b_out": None}, symbol_mapping={"__k": "__k"})
    a_write = state.add_write('A')
    b_write = state.add_write('B')
    state.add_memlet_path(outer_me, nsdfg, memlet=dace.Memlet())
    state.add_memlet_path(nsdfg, outer_mx, a_write, src_conn='a_in', memlet=dace.Memlet('A[__k, 0:J + 1, 0:I]'))
    state.add_memlet_path(nsdfg, outer_mx, b_write, src_conn='b_out', memlet=dace.Memlet('B[__k, 0:J, 0:I]'))
    return sdfg


def count_gpu_device_maps(sdfg: dace.SDFG) -> tuple[int, int]:
    """Return ``(top_level, inside_nsdfgs)`` counts of ``GPU_Device`` ``MapEntry``s.

    Top-level counts the outer-state maps. Inside-NSDFG counts maps within any
    ``NestedSDFG`` in the SDFG hierarchy.
    """
    top = sum(1 for state in sdfg.states() for n in state.nodes()
              if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    inner = 0
    for s in sdfg.all_sdfgs_recursive():
        if s is sdfg:
            continue
        inner += sum(1 for state in s.states() for n in state.nodes()
                     if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    return top, inner


def test_pass_flattens_nested_gpu_kernels_validates_clean():
    """After the pass: outer ``GPU_Device`` map gains the inner kernels' params; inner
    ``GPU_Device`` maps disappear (their bodies live in if-bound-checked NSDFGs); the SDFG
    validates."""
    sdfg = build_outer_with_two_sibling_inner_gpu_kernels()

    top_before, inner_before = count_gpu_device_maps(sdfg)
    assert (top_before, inner_before) == (1, 2), (top_before, inner_before)

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    top_after, inner_after = count_gpu_device_maps(sdfg)
    assert (top_after, inner_after) == (1, 0), (top_after, inner_after)

    # The outer map now carries the inner kernels' iteration params.
    outer = next(n for state in sdfg.states() for n in state.nodes()
                 if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    assert set(outer.map.params) >= {'__j', '__i'}, outer.map.params

    sdfg.validate()


def build_nested_kernel_with_internal_inout_node() -> dace.SDFG:
    """Outer ``GPU_Device`` kernel wrapping a NestedSDFG whose inner ``GPU_Device`` map
    has a *non-transient* array as an internal inout ``AccessNode`` (written then read
    inside the kernel body: ``c_read -> map -> t1 -> c_mid -> t2 -> map -> c_write``).

    This is the shape that drives the inout-collection branch of ``move_map_to_if``.
    """
    K = dace.symbol('K', dtype=dace.int32)
    J = dace.symbol('J', dtype=dace.int32)

    sdfg = dace.SDFG('lower_nested_inout')
    sdfg.add_array('C', [K, J], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical_loop', dict(__k='0:K'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    inner = dace.SDFG('nested_sdfg')
    inner.add_symbol('__k', dace.int32)
    inner.add_array('c_io', [J], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    inner_state = inner.add_state('nested_root', is_start_block=True)

    me, mx = inner_state.add_map('horizontal_loop', dict(__j='0:J'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    c_read = inner_state.add_read('c_io')
    c_mid = inner_state.add_access('c_io')  # internal inout node (non-transient)
    c_write = inner_state.add_write('c_io')
    t1 = inner_state.add_tasklet('bump', {'i': None}, {'o': None}, 'o = i + 1.0')
    t2 = inner_state.add_tasklet('bump2', {'i': None}, {'o': None}, 'o = i + 2.0')
    inner_state.add_memlet_path(c_read, me, t1, dst_conn='i', memlet=dace.Memlet('c_io[__j]'))
    inner_state.add_edge(t1, 'o', c_mid, None, dace.Memlet('c_io[__j]'))
    inner_state.add_edge(c_mid, None, t2, 'i', dace.Memlet('c_io[__j]'))
    inner_state.add_memlet_path(t2, mx, c_write, src_conn='o', memlet=dace.Memlet('c_io[__j]'))

    nsdfg = state.add_nested_sdfg(inner, {'c_io': None}, {'c_io': None}, symbol_mapping={'__k': '__k'})
    c_outer_read = state.add_read('C')
    c_outer_write = state.add_write('C')
    state.add_memlet_path(c_outer_read, outer_me, nsdfg, dst_conn='c_io', memlet=dace.Memlet('C[__k, 0:J]'))
    state.add_memlet_path(nsdfg, outer_mx, c_outer_write, src_conn='c_io', memlet=dace.Memlet('C[__k, 0:J]'))
    return sdfg


def test_inner_kernel_with_internal_inout_node_lowers_clean():
    """A non-transient array that is an internal inout ``AccessNode`` is collected by
    data name (a ``str``), not by the ``AccessNode`` object -- otherwise the pass leaks a
    node into the NestedSDFG connector set and raises ``KeyError`` on ``sdfg.arrays[node]``."""
    sdfg = build_nested_kernel_with_internal_inout_node()

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    top, inner = count_gpu_device_maps(sdfg)
    assert (top, inner) == (1, 0), (top, inner)
    # Every NestedSDFG connector must be a real (str) array name, never an AccessNode.
    for s in sdfg.all_sdfgs_recursive():
        for cf_state in s.states():
            for n in cf_state.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    for conn in list(n.in_connectors) + list(n.out_connectors):
                        assert isinstance(conn, str), conn
    sdfg.validate()


def build_inner_kernel_with_range(inner_range: str,
                                  symbol_mapping: dict[str, str] | None = None,
                                  param: str = '__j',
                                  loop_var: str | None = None) -> dace.SDFG:
    """Kernel ``__k`` over ``0:K`` wrapping a NestedSDFG with one ``GPU_Device`` map ``param`` over ``inner_range``.

    With ``loop_var``, the map sits in a loop ``loop_var = 0..3`` inside the NestedSDFG.
    """
    symbol_mapping = symbol_mapping or {'__k': '__k'}
    sdfg = dace.SDFG('inner_range_kernel')
    sdfg.add_symbol('K', dace.int32)
    for value in symbol_mapping.values():
        for name in map(str, dace.symbolic.pystr_to_symbolic(value).free_symbols):
            if name not in sdfg.symbols and name != '__k':
                sdfg.add_symbol(name, dace.int32)
    sdfg.add_array('A', [K, 32], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('vertical', dict(__k='0:K'), schedule=dace.dtypes.ScheduleType.GPU_Device)

    inner = dace.SDFG('nested')
    for name in symbol_mapping:
        inner.add_symbol(name, dace.int32)
    inner.add_array('a_out', [32], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    region = inner
    if loop_var is not None:
        region = dace.sdfg.state.LoopRegion('time', f'{loop_var} < 4', loop_var, f'{loop_var} = 0',
                                            f'{loop_var} = {loop_var} + 1')
        inner.add_node(region, is_start_block=True)
    inner_state = region.add_state('root', is_start_block=True)
    me, mx = inner_state.add_map('horizontal', {param: inner_range}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    tasklet = inner_state.add_tasklet('w', {}, {'_a': dace.float64}, '_a = 1.0')
    inner_state.add_memlet_path(me, tasklet, memlet=dace.Memlet())
    inner_state.add_memlet_path(tasklet,
                                mx,
                                inner_state.add_write('a_out'),
                                src_conn='_a',
                                memlet=dace.Memlet(f'a_out[{param}]'))

    nsdfg = state.add_nested_sdfg(inner, {}, {'a_out': None}, symbol_mapping=symbol_mapping)
    state.add_memlet_path(outer_me, nsdfg, memlet=dace.Memlet())
    state.add_memlet_path(nsdfg, outer_mx, state.add_write('A'), src_conn='a_out', memlet=dace.Memlet('A[__k, 0:32]'))
    return sdfg


def absorbed_range(sdfg: dace.SDFG, param: str) -> tuple:
    """The kernel map's range for ``param`` after lowering."""
    kernel = next(n for state in sdfg.states() for n in state.nodes()
                  if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    return kernel.map.range[kernel.map.params.index(param)]


def guard_conditions(sdfg: dace.SDFG) -> list:
    """Condition strings of every ``ConditionalBlock`` branch in the hierarchy."""
    return [
        branch[0].as_string for s in sdfg.all_sdfgs_recursive() for block in s.all_control_flow_blocks()
        if isinstance(block, dace.sdfg.state.ConditionalBlock) for branch in block.branches if branch[0] is not None
    ]


def test_absorbed_range_keeps_the_inner_lower_bound():
    """A kernel starting above zero must not be widened down to an origin no map asked for."""
    sdfg = build_inner_kernel_with_range('5:10')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, _ = absorbed_range(sdfg, '__j')
    assert begin == 5, f'lower bound widened to {begin}, launching iterations no inner map owned'
    assert end == 9, end
    sdfg.validate()


def test_bound_check_reproduces_a_strided_range():
    """A strided inner map must not let the iterations it skips into its body."""
    sdfg = build_inner_kernel_with_range('0:10:2')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    conditions = guard_conditions(sdfg)
    assert len(conditions) == 1, conditions
    # The step is what distinguishes the owned iterations from the absorbed unit-step range.
    assert '% 2' in conditions[0], f'guard {conditions[0]!r} admits the iterations the step skips'
    sdfg.validate()


def test_unit_step_bound_check_stays_a_plain_interval():
    """The step term is only emitted when there is a step to check."""
    sdfg = build_inner_kernel_with_range('0:10')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    conditions = guard_conditions(sdfg)
    assert len(conditions) == 1, conditions
    assert '%' not in conditions[0], conditions[0]


def kernel_with_a_directly_nested_gpu_map() -> dace.SDFG:
    """Inner ``GPU_Device`` map sitting straight in the kernel's scope, with no NestedSDFG between.

    The two maps are joined only by empty memlets, which carry ordering rather than data.
    """
    sdfg = dace.SDFG('direct')
    sdfg.add_array('A', [16, 32], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('outer', dict(__k='0:16'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    inner_me, inner_mx = state.add_map('inner', dict(__j='0:32'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('w', {}, {'_v': dace.float64}, '_v = 1.0')
    state.add_memlet_path(outer_me, inner_me, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet,
                          inner_mx,
                          outer_mx,
                          state.add_write('A'),
                          src_conn='_v',
                          memlet=dace.Memlet('A[__k, __j]'))
    sdfg.validate()
    return sdfg


def test_directly_nested_gpu_map_lowers_without_detaching_its_body():
    """The ordering edge into the inner scope must be re-anchored, not dropped with the map."""
    sdfg = kernel_with_a_directly_nested_gpu_map()
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    top, inner = count_gpu_device_maps(sdfg)
    assert (top, inner) == (1, 0), (top, inner)
    kernel = next(n for state in sdfg.states() for n in state.nodes()
                  if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)
    assert set(kernel.map.params) == {'__k', '__j'}, kernel.map.params
    # The body's nested SDFG stays attached to the kernel, or the scope traversal cannot place it.
    body = next(n for state in sdfg.states() for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    state = sdfg.states()[0]
    assert state.in_degree(body) > 0, 'the guarded body was detached from the kernel scope'
    sdfg.validate()


def test_enclosing_kernel_symbol_is_bound_on_the_guarded_body():
    """A body may name the outer kernel's parameter, so the nested SDFG node must bind it."""
    sdfg = build_outer_with_two_sibling_inner_gpu_kernels()
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    guarded = [
        node for sub in sdfg.all_sdfgs_recursive() for state in sub.states() for node in state.nodes()
        if isinstance(node, dace.nodes.NestedSDFG) and node.label.startswith('if_of_nested_')
    ]
    assert guarded, 'no guarded body was produced'
    for node in guarded:
        assert '__k' in node.symbol_mapping, (node.label, sorted(node.symbol_mapping))
    sdfg.validate()


def kernel_entry(sdfg: dace.SDFG) -> dace.nodes.MapEntry:
    return next(n for state in sdfg.states() for n in state.nodes()
                if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device)


def test_a_bound_naming_a_nested_scope_symbol_is_hoisted_in_the_kernel_symbols():
    """``M`` exists only inside the NestedSDFG; the hoisted bound must say the outer ``K`` it maps to."""
    sdfg = build_inner_kernel_with_range('0:M', {'__k': '__k', 'M': 'K'})
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    range_symbols = {str(s) for s in kernel_entry(sdfg).map.range.free_symbols}
    assert 'M' not in range_symbols and 'K' in range_symbols, kernel_entry(sdfg).map.range
    assert 'M' not in sdfg.free_symbols
    sdfg.validate()


def test_a_swapping_symbol_mapping_is_applied_simultaneously():
    """Inner ``M`` means outer ``N`` and inner ``N`` means outer ``M``; a sequential rewrite maps ``M`` back to itself."""
    sdfg = build_inner_kernel_with_range('0:M', {'__k': '__k', 'M': 'N', 'N': 'M'})
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, _ = absorbed_range(sdfg, '__j')
    assert (begin, end) == (0, dace.symbolic.pystr_to_symbolic('N - 1')), (begin, end)


def test_a_bound_naming_a_kernel_parameter_is_refused():
    """The kernel's range sizes the grid on the host, where the per-block ``__k`` does not exist."""
    sdfg = build_inner_kernel_with_range('__k:__k + 4')
    with pytest.raises(NotImplementedError, match='__k'):
        NestedGPUDeviceMapLowering().apply_pass(sdfg, {})


def test_a_bound_naming_a_loop_variable_inside_the_kernel_is_refused():
    """A loop inside the kernel defines ``t`` per thread; the host cannot size the grid from it."""
    sdfg = build_inner_kernel_with_range('0:t + 1', loop_var='t')
    sdfg.validate()

    with pytest.raises(NotImplementedError, match="'t'"):
        NestedGPUDeviceMapLowering().apply_pass(sdfg, {})


def build_inner_kernel_reusing_the_outer_param(name: str) -> dace.SDFG:
    """Kernel ``i`` writes ``B[i]``; a same-state inner kernel reuses ``i`` and writes ``A[i, j] = 10 * i + j``."""
    gpu = dace.dtypes.StorageType.GPU_Global
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [3, 4], dace.float64, storage=gpu)
    sdfg.add_array('B', [1], dace.float64, storage=gpu)

    state = sdfg.add_state('s')
    outer_me, outer_mx = state.add_map('outer', dict(i='0:1'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    outer_me.map.gpu_block_size = [32, 1, 1]
    inner_me, inner_mx = state.add_map('inner', dict(i='0:3', j='0:4'), schedule=dace.dtypes.ScheduleType.GPU_Device)

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
    sdfg.validate()


def test_the_flattened_kernel_declares_each_index_once():
    """nvcc rejects a second declaration of one index name in a kernel scope."""
    sdfg = build_inner_kernel_reusing_the_outer_param('inner_reuses_outer_param_codegen')
    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})
    code = ''.join(obj.clean_code for obj in sdfg.generate_code() if obj.language == 'cu')

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
    sdfg.validate()


def test_a_transient_local_to_the_inner_body_moves_with_it():
    """A scalar the inner body allocates for itself crosses no map edge but must still be declared."""
    sdfg = build_inner_kernel_with_range('0:32')
    inner = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
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
    sdfg.validate()


def test_a_sequential_map_below_the_kernel_is_not_absorbed():
    """Only ``GPU_Device`` maps are flattened; absorbing a sequential one would parallelize it."""
    sdfg = build_inner_kernel_with_range('0:32')
    inner_map = next(n for sub in sdfg.all_sdfgs_recursive() if sub is not sdfg for state in sub.states()
                     for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    inner_map.map.schedule = dace.dtypes.ScheduleType.Sequential

    assert NestedGPUDeviceMapLowering().apply_pass(sdfg, {}) is None
    assert kernel_entry(sdfg).map.params == ['__k']
    assert inner_map.map.schedule == dace.dtypes.ScheduleType.Sequential


def test_sibling_bounds_carrying_an_overapproximation_are_unioned():
    """A ``SymExpr`` bound (main plus overapproximation) must not break the sibling union."""
    sdfg = build_outer_with_two_sibling_inner_gpu_kernels()
    first = next(n for sub in sdfg.all_sdfgs_recursive() if sub is not sdfg for state in sub.states()
                 for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    rng = list(first.map.range)
    rng[0] = (0, dace.symbolic.SymExpr('J', 'J + 1'), 1)
    first.map.range = dace.subsets.Range(rng)

    NestedGPUDeviceMapLowering().apply_pass(sdfg, {})

    begin, end, _ = absorbed_range(sdfg, '__j')
    assert begin == 0
    widest = end.approx if isinstance(end, dace.symbolic.SymExpr) else end
    assert widest == dace.symbolic.pystr_to_symbolic('J + 1'), end
    sdfg.validate()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
