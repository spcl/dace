# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live tests of AI-generated CPU code.

These call a model provider and need an API key, so they are marked ``ai`` and excluded from CI.
Run them with::

    ANTHROPIC_API_KEY=... pytest tests/library/ai/ai_cpu_test.py -m ai

They assert behavior -- that the compiled program computes the right thing -- rather than the text
of the generated code, since that varies between runs. The few text assertions that do appear
check that a specific piece of context reached the model (the ISA extensions, the reserved
implementation name), not how it chose to use it.
"""

import platform

import numpy as np
import pytest

import dace
from dace import dtypes, nodes
from dace.libraries.ai.nodes import AINode
from dace.libraries.ai.sysinfo import cpu_has_feature

N = 64
TILE = 8

TILE_GEMM_DESCRIPTION = f"""
Compute one {TILE}x{TILE} output tile of a single-precision matrix multiplication:

    for i in 0..{TILE}, j in 0..{TILE}:
        _c[i][j] = sum over p in 0..N of _a[i][p] * _b[p][j]

Write it as a register-blocked microkernel using AVX2 intrinsics (immintrin.h), accumulating the
tile in YMM registers across the reduction dimension and storing it once at the end. Use FMA
instructions. Do not fall back to a plain scalar triple loop.
""".strip()

ASM_DESCRIPTION = """
Compute _out = _a + _b for two double-precision scalars, using an inline assembly statement
(GCC extended asm with the addsd instruction) rather than the C++ '+' operator. Keep it correct
for x86-64 System V.
""".strip()


def _tile_gemm_sdfg(description: str) -> dace.SDFG:
    """
    Builds a tiled matrix multiplication whose innermost kernel is an :class:`AINode`.

    The AI node sits inside a ``CPU_Multicore`` map over output tiles, and receives pointers that
    are already offset to its own tile.

    :param description: What the node must compute.
    :return: The SDFG.
    """
    sdfg = dace.SDFG('ai_tile_gemm')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [N, N], dace.float32)

    state = sdfg.add_state()
    entry, exit_node = state.add_map('tiles', {
        'ti': f'0:{N}:{TILE}',
        'tj': f'0:{N}:{TILE}'
    },
                                     schedule=dtypes.ScheduleType.CPU_Multicore)
    node = AINode('gemm_tile', description, inputs={'_a', '_b'}, outputs={'_c'})
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
    return sdfg


@pytest.mark.ai
@pytest.mark.skipif(platform.machine() != 'x86_64', reason='needs an x86-64 host')
@pytest.mark.skipif(not cpu_has_feature('avx2') or not cpu_has_feature('fma'),
                    reason='needs a host CPU with AVX2 and FMA')
def test_vectorized_tile_in_a_multicore_map():
    sdfg = _tile_gemm_sdfg(TILE_GEMM_DESCRIPTION)
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, AINode))
    node.expand(state, 'ai')

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    # The include belongs at file scope, not in the body
    assert 'immintrin' in tasklet.code_global.as_string
    assert 'immintrin' not in tasklet.code.as_string
    # The enclosing map is already parallel, so the body must not open its own parallel region
    assert '#pragma omp parallel' not in tasklet.code.as_string

    rng = np.random.default_rng(0)
    a = rng.random((N, N), dtype=np.float32)
    b = rng.random((N, N), dtype=np.float32)
    c = np.zeros((N, N), dtype=np.float32)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.ai
@pytest.mark.skipif(platform.machine() != 'x86_64', reason='needs an x86-64 host')
def test_inline_assembly():
    sdfg = dace.SDFG('ai_inline_asm')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [1], dace.float64)

    state = sdfg.add_state()
    node = AINode('asm_add', ASM_DESCRIPTION, inputs={'_a', '_b'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[0]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet('B[0]'))
    state.add_edge(node, '_out', state.add_write('C'), None, dace.Memlet('C[0]'))

    node.expand(state, 'ai')
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert 'asm' in tasklet.code.as_string, 'the model did not use inline assembly'

    a = np.array([1.5], dtype=np.float64)
    b = np.array([2.25], dtype=np.float64)
    c = np.zeros([1], dtype=np.float64)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, 3.75)


if __name__ == '__main__':
    test_vectorized_tile_in_a_multicore_map()
    test_inline_assembly()
