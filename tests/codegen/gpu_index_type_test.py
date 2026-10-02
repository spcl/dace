# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests the types that GPU map indices are declared with.

Indices are declared with the configured ``compiler.cuda.thread_id_type`` (``int32`` by default), unless the type
inferred from the map range is a wider integer, e.g., when the range spans a 64-bit symbol. A wider index is computed
from ``blockIdx``/``gridDim`` cast to its type, since those registers are 32-bit and their products would wrap around
before they are widened.

The tests that only inspect the generated code run without a GPU; the ones marked ``gpu`` run the programs.
"""
import re

import numpy as np
import pytest

import dace
from dace import nodes
from dace.codegen import common

N = dace.symbol('N', dace.int64)
M = dace.symbol('M', dace.int32)
O = dace.symbol('O', dace.int64)
H = dace.symbol('H', dace.int64)
W = dace.symbol('W', dace.int64)
nnz = dace.symbol('nnz', dace.int64)


def _cuda_code(sdfg: dace.SDFG, backend: str = 'cuda', **config) -> str:
    """Generates the GPU code of ``sdfg`` for the given backend, with the given ``compiler.cuda`` entries set."""
    with dace.config.temporary_config():
        dace.config.Config.set('compiler', 'cuda', 'backend', value=backend)
        # The expected code below assumes the default block size, whatever a local configuration file sets
        dace.config.Config.set('compiler', 'cuda', 'default_block_size', value='32,1,1')
        for key, value in config.items():
            dace.config.Config.set('compiler', 'cuda', key, value=value)
        # The backend and chiplet count are cached for the whole process; clear them before and after, so that the
        # backend set above reaches the code generator and does not leak into other tests
        common.get_gpu_backend.cache_clear()
        common.get_gpu_chiplet_count.cache_clear()
        try:
            # The GPU code object is named the same for every backend, while its language is the file extension
            return next(c for c in sdfg.generate_code() if c.name == f'{sdfg.name}_cuda').clean_code
        finally:
            common.get_gpu_backend.cache_clear()
            common.get_gpu_chiplet_count.cache_clear()


def _kernels(code: str) -> str:
    """Returns the kernels and device functions in ``code``, without the host code around them."""
    # Every top-level definition starts at the first column, and everything within it is indented
    definitions = re.split(r'\n(?=[^\s}])', code)
    return '\n'.join(d for d in definitions if d.startswith(('__global__', 'DACE_DFI')))


def _declaration(code: str, name: str) -> str:
    """Returns the declaration of the index variable ``name`` in the kernels of ``code``."""
    kernels = _kernels(code)
    match = re.search(rf'((?:for \()?[\w:]+ {name} = [^;]*;)', kernels)
    assert match, f'No declaration of {name} in:\n{kernels}'
    return match.group(1)


def _gpu(sdfg: dace.SDFG) -> dace.SDFG:
    for desc in sdfg.arrays.values():
        if not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
    return sdfg


def _one_dimensional(size: dace.symbol) -> dace.SDFG:
    """A device map over ``0:size``, which code generation splits into a device map (``b_i``) and a thread-block map
    (``i``) of the default block size."""

    @dace.program
    def one_dimensional(A: dace.float64[size], B: dace.float64[size]):
        for i in dace.map[0:size] @ dace.ScheduleType.GPU_Device:
            B[i] = A[i] + 1

    return _gpu(one_dimensional.to_sdfg(simplify=True))


def _spmv(step: int = 1) -> dace.SDFG:
    """A CSR sparse matrix-vector product with 64-bit row pointers, whose inner map is a dynamic thread-block map."""

    @dace.program
    def spmv(A_row: dace.int64[H + 1], A_col: dace.int64[nnz], A_val: dace.float32[nnz], x: dace.float32[W],
             b: dace.float32[H]):
        for i in dace.map[0:H]:
            for j in dace.map[A_row[i]:A_row[i + 1]:step]:
                b[i] += A_val[j] * x[A_col[j]]

    sdfg = spmv.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dace.ScheduleType.Sequential:
            node.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic
    return sdfg


def _persistent(sdfg: dace.SDFG) -> dace.SDFG:
    """Makes the outermost maps of ``sdfg`` persistent kernels, and the maps they contain device maps."""
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.MapEntry):
                continue
            if state.entry_node(node) is None:
                node.map.schedule = dace.ScheduleType.GPU_Persistent
            elif state.entry_node(state.entry_node(node)) is None:
                node.map.schedule = dace.ScheduleType.GPU_Device
    return sdfg


@dace.program
def offset_indices(out: dace.int64[N]):
    for i in dace.map[O:O + N]:
        out[i - O] = i


@dace.program
def persistent_offset_indices(out: dace.int64[N]):
    for _ in dace.map[0:1]:
        for i in dace.map[O:O + N]:
            out[i - O] = i


# Code generation ######################################################################################################


def test_default_indices_are_unchanged():
    code = _cuda_code(_one_dimensional(dace.symbol('N32', dace.int32)))
    assert _declaration(code, 'b_i') == 'int b_i = (32 * blockIdx.x);'
    assert _declaration(code, 'i') == 'int i = (threadIdx.x + b_i);'
    assert 'static_cast' not in _kernels(code)


def test_64bit_range_widens_indices():
    code = _cuda_code(_one_dimensional(N))
    assert _declaration(code, 'b_i') == 'int64_t b_i = (32 * static_cast<int64_t>(blockIdx.x));'
    assert _declaration(code, 'i') == 'int64_t i = (threadIdx.x + b_i);'


def test_each_dimension_takes_its_own_type():

    @dace.program
    def mixed(A: dace.float64[N, M, 4]):
        for i, j, k in dace.map[0:N, 0:M, 0:4] @ dace.ScheduleType.GPU_Device:
            A[i, j, k] = 1.0

    code = _cuda_code(_gpu(mixed.to_sdfg(simplify=True)))
    assert _declaration(code, 'b_i').startswith('int64_t b_i = static_cast<int64_t>(blockIdx.z)')
    assert _declaration(code, 'b_j') == 'int b_j = blockIdx.y;'
    assert _declaration(code, 'b_k').startswith('int b_k = ')


def test_threadblock_map_keeps_its_own_type():

    @dace.program
    def tiled(A: dace.float64[N]):
        for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ dace.ScheduleType.GPU_ThreadBlock:
                A[i + j] = 1.0

    code = _cuda_code(_gpu(tiled.to_sdfg(simplify=True)))
    assert _declaration(code, 'i') == 'int64_t i = (32 * static_cast<int64_t>(blockIdx.x));'
    assert _declaration(code, 'j') == 'int j = threadIdx.x;'


def nested_device_map_code() -> str:
    sdfg = dace.SDFG('nested_device_index_type')
    sdfg.add_array('A', [N, N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map('outer', {'i': '0:N'}, schedule=dace.ScheduleType.GPU_Device)
    inner_entry, inner_exit = state.add_map('inner', {'j': '0:N'}, schedule=dace.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_nedge(outer_entry, inner_entry, dace.Memlet())
    state.add_edge(inner_entry, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet,
                          inner_exit,
                          outer_exit,
                          state.add_write('A'),
                          src_conn='o',
                          memlet=dace.Memlet('A[i, j]'))

    return _cuda_code(sdfg)


@pytest.mark.old_gpu_codegen_only
def test_nested_device_map():
    code = nested_device_map_code()
    assert _declaration(code, 'i') == 'int64_t i = (static_cast<int64_t>(blockIdx.x) * 32 + threadIdx.x);'
    assert _declaration(code, 'j') == 'int64_t j = static_cast<int64_t>(blockIdx.y);'


@pytest.mark.new_gpu_codegen_only
def test_nested_device_map_is_lowered_to_one_two_dimensional_kernel_map():
    code = nested_device_map_code()
    assert _declaration(code, 'b_i') == 'int64_t b_i = static_cast<int64_t>(blockIdx.y);'
    assert _declaration(code, 'i') == 'int64_t i = (threadIdx.y + b_i);'
    assert _declaration(code, 'j') == 'int64_t j = (threadIdx.x + b_j);'


def test_nested_sdfg_receives_wide_index():
    """A map body with control flow is a nested SDFG; its device function must take the index as wide as the map."""

    @dace.program
    def conditional(out: dace.int64[64]):
        for i in dace.map[0:N]:
            if i >= N - 64:
                out[i - N + 64] = i

    sdfg = conditional.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    code = _cuda_code(sdfg)
    assert re.search(r'DACE_DFI void \w+\([^)]*\bint64_t i\)', code), _kernels(code)
    assert _declaration(code, 'i') == 'int64_t i = (threadIdx.x + b_i);'


@pytest.mark.old_gpu_codegen_only
def test_persistent_map():
    code = _cuda_code(_persistent(persistent_offset_indices.to_sdfg(simplify=False)))
    header = _declaration(code, 'i')
    assert header == 'for (int64_t i = (O + (static_cast<int64_t>(blockIdx.x) * 32 + threadIdx.x));', header
    assert 'i += static_cast<int64_t>(gridDim.x) * 32' in code


@pytest.mark.old_gpu_codegen_only
@pytest.mark.parametrize('step', [1, 2])
def test_dynamic_map(step: int):
    code = _cuda_code(_spmv(step), dynamic_map_block_size='64,1,1')
    # The scheduling state (two arrays of 32 * 32 indices per warp) fits in static shared memory
    assert '__shared__ int64_t __dace_dynmap_state[4096];' in code
    assert 'int64_t __dace_dynmap_begin = 0, __dace_dynmap_end = 0;' in code
    assert 'dace::DynamicMap<true, 64, 32, int64_t>::schedule(' in code
    if step != 1:
        assert re.search(r'int64_t j = \w+ \+ 2 \* j_idx;', code)


@pytest.mark.old_gpu_codegen_only
def test_dynamic_map_default_types():
    # A name of its own: SymPy's cache shares a symbol with every module that declares ``M`` with another type
    M32 = dace.symbol('M32', dace.int32)

    @dace.program
    def spmv32(A_row: dace.uint32[M32 + 1], A_col: dace.uint32[M32], A_val: dace.float32[M32], x: dace.float32[M32],
               b: dace.float32[M32]):
        for i in dace.map[0:M32]:
            for j in dace.map[A_row[i]:A_row[i + 1]]:
                b[i] += A_val[j] * x[A_col[j]]

    sdfg = spmv32.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dace.ScheduleType.Sequential:
            node.map.schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic
    code = _cuda_code(sdfg)
    assert '__shared__ int __dace_dynmap_state[8192];' in code
    assert 'unsigned int __dace_dynmap_begin = 0, __dace_dynmap_end = 0;' in code
    assert 'dace::DynamicMap<true, 128>::schedule(' in code


def test_chiplet_distribution():
    code = _cuda_code(_one_dimensional(N), backend='hip', chiplet_number=6)
    assert '((static_cast<int64_t>(blockIdx.x) % 6) * ' in code
    assert ' + static_cast<int64_t>(blockIdx.x) / 6)' in code


def test_configured_64bit_index_type_widens_registers():
    """A 64-bit ``thread_id_type`` also has to compute indices from widened registers, not only store them."""
    code = _cuda_code(_one_dimensional(dace.symbol('N32', dace.int32)), thread_id_type='int64')
    assert _declaration(code, 'b_i') == 'int64_t b_i = (32 * static_cast<int64_t>(blockIdx.x));'


# End-to-end ###########################################################################################################


@pytest.mark.gpu
def test_indices_beyond_32_bits():
    size, offset = 1000, 2**33 + 5
    sdfg = offset_indices.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    out = np.zeros(size, dtype=np.int64)
    sdfg(out=out, N=size, O=offset)
    assert np.array_equal(out, np.arange(size, dtype=np.int64) + offset)


@pytest.mark.gpu
def test_block_index_products_beyond_32_bits():
    """With 2**32 + 64 threads, ``blockIdx.x * blockDim`` exceeds 32 bits for the last blocks."""
    tail = 64

    @dace.program
    def last_indices(out: dace.int64[tail]):
        for i in dace.map[0:N]:
            if i >= N - tail:
                out[i - N + tail] = i

    sdfg = last_indices.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    size = 2**32 + tail
    out = np.zeros(tail, dtype=np.int64)
    sdfg(out=out, N=size)
    assert np.array_equal(out, np.arange(size - tail, size, dtype=np.int64))


@pytest.mark.old_gpu_codegen_only
@pytest.mark.gpu
def test_persistent_indices_beyond_32_bits():
    size, offset = 1000, 2**33 + 5
    sdfg = persistent_offset_indices.to_sdfg(simplify=False)
    sdfg.apply_gpu_transformations()
    _persistent(sdfg)
    out = np.zeros(size, dtype=np.int64)
    sdfg(out=out, N=size, O=offset)
    assert np.array_equal(out, np.arange(size, dtype=np.int64) + offset)


@pytest.mark.old_gpu_codegen_only
@pytest.mark.gpu
@pytest.mark.parametrize('fine_grained,block_size', [(True, 64), (True, 128), (False, 128)])
def test_dynamic_map_with_64bit_row_pointers(fine_grained: bool, block_size: int):
    """With a block size of 128, the fine-grained scheduling state is placed in dynamic shared memory."""
    rng = np.random.default_rng(42)
    height, width = 256, 256
    row_lengths = rng.integers(0, 257, size=height)
    A_row = np.concatenate([[0], np.cumsum(row_lengths)]).astype(np.int64)
    A_col = np.concatenate([np.sort(rng.choice(width, length, replace=False))
                            for length in row_lengths]).astype(np.int64)
    A_val = rng.random(A_row[-1], dtype=np.float32)
    x = rng.random(width, dtype=np.float32)
    b = np.zeros(height, dtype=np.float32)

    with dace.config.temporary_config():
        dace.config.Config.set('compiler', 'cuda', 'dynamic_map_fine_grained', value=fine_grained)
        dace.config.Config.set('compiler', 'cuda', 'dynamic_map_block_size', value=f'{block_size},1,1')
        sdfg = _spmv()
        sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=height, W=width, nnz=A_row[-1])

    expected = np.array([A_val[A_row[r]:A_row[r + 1]] @ x[A_col[A_row[r]:A_row[r + 1]]] for r in range(height)])
    assert np.allclose(b, expected, rtol=1e-5)


if __name__ == '__main__':
    test_default_indices_are_unchanged()
    test_64bit_range_widens_indices()
    test_each_dimension_takes_its_own_type()
    test_threadblock_map_keeps_its_own_type()
    test_nested_device_map()
    test_nested_device_map_is_lowered_to_one_two_dimensional_kernel_map()
    test_nested_sdfg_receives_wide_index()
    test_persistent_map()
    for step in [1, 2]:
        test_dynamic_map(step)
    test_dynamic_map_default_types()
    test_chiplet_distribution()
    test_configured_64bit_index_type_widens_registers()
    test_indices_beyond_32_bits()
    test_block_index_products_beyond_32_bits()
    test_persistent_indices_beyond_32_bits()
    for fine_grained, block_size in [(True, 64), (True, 128), (False, 128)]:
        test_dynamic_map_with_64bit_row_pointers(fine_grained, block_size)
