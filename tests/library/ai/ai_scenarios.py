# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Runnable scenarios for exercising AI expansion by hand.

Each scenario places a library node in a specific slot, expands it with the configured provider,
and -- where the machine can run the result -- checks it numerically. Together with the ``manual``
provider this is the loop for evaluating the prompt without an API key::

    # Writes the prompt and stops, because there is no answer yet
    DACE_ai_provider=manual DACE_ai_manual_dir=/tmp/p python -m tests.library.ai.ai_scenarios gemm_cpu

    # ... paste the prompt into a chat, save the reply next to it, then run again
    DACE_ai_provider=manual DACE_ai_manual_dir=/tmp/p python -m tests.library.ai.ai_scenarios gemm_cpu

Run without arguments to list the scenarios.
"""

import sys
from typing import Callable, Dict, Tuple

import numpy as np

import dace
from dace import dtypes, nodes
from dace.libraries.ai.nodes import AINode

N = 64
TILE = 8

TILE_GEMM = f"""
Compute one {TILE}x{TILE} output tile of a single-precision matrix multiplication:

    for i in 0..{TILE}, j in 0..{TILE}:
        _c[i][j] = sum over p in 0..{N} of _a[i][p] * _b[p][j]

Write it as a register-blocked microkernel using AVX2 intrinsics (immintrin.h), accumulating the
tile in YMM registers across the reduction dimension and storing it once at the end. Use FMA
instructions rather than separate multiplies and adds.
""".strip()

TILE_TRANSPOSE = f"""
Transpose one {TILE}x{TILE} single-precision tile: _out[j][i] = _in[i][j] for i, j in 0..{TILE}.

Write it with AVX2 intrinsics (immintrin.h) as a register transpose: load eight rows into YMM
registers, transpose them in registers with unpack and permute instructions, and store the eight
result rows. Do not write a scalar element-by-element loop.
""".strip()


def _matmul_reference(a, b):
    """
    Returns the expected result of the matrix multiplication scenarios.

    :param a: Left operand.
    :param b: Right operand.
    :return: ``a @ b``.
    """
    return a @ b


def gemm_cpu() -> Tuple[dace.SDFG, Callable]:
    """
    A ``Gemm`` library node on the host over CPU arrays.

    The node carries no description, so its class documentation is the only specification.

    :return: The SDFG and a callable that runs and checks it.
    """
    from dace.libraries.blas.nodes.gemm import Gemm

    sdfg = dace.SDFG('ai_scenario_gemm_cpu')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [N, N], dace.float32, storage=dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state()
    gemm = Gemm('gemm')
    state.add_node(gemm)
    state.add_edge(state.add_read('A'), None, gemm, '_a', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(state.add_read('B'), None, gemm, '_b', dace.Memlet(f'B[0:{N}, 0:{N}]'))
    state.add_edge(gemm, '_c', state.add_write('C'), None, dace.Memlet(f'C[0:{N}, 0:{N}]'))

    def check(compiled_sdfg):
        rng = np.random.default_rng(0)
        a, b = rng.random((N, N), dtype=np.float32), rng.random((N, N), dtype=np.float32)
        c = np.zeros((N, N), dtype=np.float32)
        compiled_sdfg(A=a, B=b, C=c)
        return np.allclose(c, _matmul_reference(a, b), rtol=1e-4, atol=1e-4)

    return sdfg, check


def gemm_tile_avx() -> Tuple[dace.SDFG, Callable]:
    """
    An :class:`AINode` microkernel computing one output tile, inside a parallel map.

    :return: The SDFG and a callable that runs and checks it.
    """
    sdfg = dace.SDFG('ai_scenario_gemm_tile')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [N, N], dace.float32)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('tiles', {
        'ti': f'0:{N}:{TILE}',
        'tj': f'0:{N}:{TILE}'
    },
                                     schedule=dtypes.ScheduleType.CPU_Multicore)
    node = AINode('gemm_tile', TILE_GEMM, inputs={'_a', '_b'}, outputs={'_c'})
    state.add_node(node)
    state.add_memlet_path(state.add_read('A'),
                          entry,
                          node,
                          dst_conn='_a',
                          memlet=dace.Memlet(f'A[ti:ti+{TILE}, 0:{N}]'))
    state.add_memlet_path(state.add_read('B'),
                          entry,
                          node,
                          dst_conn='_b',
                          memlet=dace.Memlet(f'B[0:{N}, tj:tj+{TILE}]'))
    state.add_memlet_path(node,
                          exit_node,
                          state.add_write('C'),
                          src_conn='_c',
                          memlet=dace.Memlet(f'C[ti:ti+{TILE}, tj:tj+{TILE}]'))

    def check(compiled_sdfg):
        rng = np.random.default_rng(0)
        a, b = rng.random((N, N), dtype=np.float32), rng.random((N, N), dtype=np.float32)
        c = np.zeros((N, N), dtype=np.float32)
        compiled_sdfg(A=a, B=b, C=c)
        return np.allclose(c, _matmul_reference(a, b), rtol=1e-4, atol=1e-4)

    return sdfg, check


def transpose_avx() -> Tuple[dace.SDFG, Callable]:
    """
    An :class:`AINode` register transpose of one tile, inside a parallel map.

    :return: The SDFG and a callable that runs and checks it.
    """
    sdfg = dace.SDFG('ai_scenario_transpose')
    for name in ('A', 'B'):
        sdfg.add_array(name, [N, N], dace.float32)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('tiles', {
        'ti': f'0:{N}:{TILE}',
        'tj': f'0:{N}:{TILE}'
    },
                                     schedule=dtypes.ScheduleType.CPU_Multicore)
    node = AINode('transpose_tile', TILE_TRANSPOSE, inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_memlet_path(state.add_read('A'),
                          entry,
                          node,
                          dst_conn='_in',
                          memlet=dace.Memlet(f'A[ti:ti+{TILE}, tj:tj+{TILE}]'))
    state.add_memlet_path(node,
                          exit_node,
                          state.add_write('B'),
                          src_conn='_out',
                          memlet=dace.Memlet(f'B[tj:tj+{TILE}, ti:ti+{TILE}]'))

    def check(compiled_sdfg):
        rng = np.random.default_rng(0)
        a = rng.random((N, N), dtype=np.float32)
        b = np.zeros((N, N), dtype=np.float32)
        compiled_sdfg(A=a, B=b)
        return np.allclose(b, a.T)

    return sdfg, check


def _gemm_gpu(name: str, inside_kernel: bool) -> Tuple[dace.SDFG, Callable]:
    """
    A ``Gemm`` over GPU arrays, either inside a kernel or on the host.

    :param name: Name of the SDFG.
    :param inside_kernel: If True, the node is placed inside a ``GPU_Device`` map.
    :return: The SDFG and a callable that runs and checks it on the GPU.
    """
    from dace.libraries.blas.nodes.gemm import Gemm

    sdfg = dace.SDFG(name)
    for array in ('A', 'B', 'C'):
        sdfg.add_array(array, [N, N], dace.float32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    gemm = Gemm('gemm')
    state.add_node(gemm)
    subset = f'0:{N}, 0:{N}'
    if inside_kernel:
        entry, exit_node = state.add_map('grid', {'b': '0:1'}, schedule=dtypes.ScheduleType.GPU_Device)
        state.add_memlet_path(state.add_read('A'), entry, gemm, dst_conn='_a', memlet=dace.Memlet(f'A[{subset}]'))
        state.add_memlet_path(state.add_read('B'), entry, gemm, dst_conn='_b', memlet=dace.Memlet(f'B[{subset}]'))
        state.add_memlet_path(gemm, exit_node, state.add_write('C'), src_conn='_c', memlet=dace.Memlet(f'C[{subset}]'))
    else:
        state.add_edge(state.add_read('A'), None, gemm, '_a', dace.Memlet(f'A[{subset}]'))
        state.add_edge(state.add_read('B'), None, gemm, '_b', dace.Memlet(f'B[{subset}]'))
        state.add_edge(gemm, '_c', state.add_write('C'), None, dace.Memlet(f'C[{subset}]'))

    def check(compiled_sdfg):
        import cupy

        rng = np.random.default_rng(0)
        a, b = rng.random((N, N), dtype=np.float32), rng.random((N, N), dtype=np.float32)
        ga, gb = cupy.asarray(a), cupy.asarray(b)
        gc = cupy.zeros((N, N), dtype=np.float32)
        compiled_sdfg(A=ga, B=gb, C=gc)
        return np.allclose(cupy.asnumpy(gc), _matmul_reference(a, b), rtol=1e-4, atol=1e-4)

    return sdfg, check


def gemm_device() -> Tuple[dace.SDFG, Callable]:
    """
    A ``Gemm`` inside a GPU kernel.

    :return: The SDFG and a callable that runs and checks it on the GPU.
    """
    return _gemm_gpu('ai_scenario_gemm_device', inside_kernel=True)


def gemm_host_gpu() -> Tuple[dace.SDFG, Callable]:
    """
    A ``Gemm`` on the host over GPU arrays.

    :return: The SDFG and a callable that runs and checks it on the GPU.
    """
    return _gemm_gpu('ai_scenario_gemm_host_gpu', inside_kernel=False)


SCENARIOS: Dict[str, Callable] = {
    'gemm_cpu': gemm_cpu,
    'gemm_tile_avx': gemm_tile_avx,
    'transpose_avx': transpose_avx,
    'gemm_device': gemm_device,
    'gemm_host_gpu': gemm_host_gpu,
}


def run(name: str) -> int:
    """
    Expands one scenario and, where possible, runs and checks it.

    :param name: Key in :data:`SCENARIOS`.
    :return: A process exit code.
    """
    sdfg, check = SCENARIOS[name]()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, nodes.LibraryNode))

    node.expand(state, 'ai')
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))

    print(f'=== generated tasklet for {name} ===')
    print(tasklet.code.as_string)
    for label, block in (('code_global', tasklet.code_global), ('code_init', tasklet.code_init), ('code_exit',
                                                                                                  tasklet.code_exit)):
        text = block.as_string if block is not None else ''
        if text.strip():
            print(f'--- {label} ---\n{text}')
    if tasklet.state_fields:
        print(f'--- state_fields ---\n{tasklet.state_fields}')
    if tasklet.environments:
        print(f'--- environments ---\n{sorted(tasklet.environments)}')

    if check is None:
        print(f'\n{name}: generated only (no runnable check for this scenario).')
        return 0

    print(f'\n{name}: compiling and running...')
    ok = check(sdfg)
    print(f'{name}: {"CORRECT" if ok else "WRONG RESULT"}')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        print(f'usage: {sys.argv[0]} <{" | ".join(SCENARIOS)}>')
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
