# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live tests of AI-generated GPU code.

Needs both a GPU and an API key, so these are marked ``gpu`` and ``ai`` and excluded from CI.
"""

import numpy as np
import pytest

import dace
from dace import dtypes, nodes
from dace.libraries.ai.nodes import AINode

N = 1024

DEVICE_DESCRIPTION = """
Compute _out = _a * _b + _c for single-precision scalars, using the GPU's fused multiply-add
device intrinsic (__fmaf_rn on CUDA, or the equivalent on HIP) rather than separate multiply and
add operations. This runs once per thread.
""".strip()


def _device_sdfg() -> dace.SDFG:
    """
    Builds a GPU kernel whose body is an :class:`AINode`.

    :return: The SDFG.
    """
    sdfg = dace.SDFG('ai_gpu_fma')
    for name in ('A', 'B', 'C', 'D'):
        sdfg.add_array(name, [N], dace.float32)

    state = sdfg.add_state()
    entry, exit_node = state.add_map('kernel', {'i': f'0:{N}'}, schedule=dtypes.ScheduleType.GPU_Device)
    node = AINode('fma', DEVICE_DESCRIPTION, inputs={'_a', '_b', '_c'}, outputs={'_out'})
    state.add_node(node)
    for conn, array in (('_a', 'A'), ('_b', 'B'), ('_c', 'C')):
        state.add_memlet_path(state.add_read(array), entry, node, dst_conn=conn, memlet=dace.Memlet(f'{array}[i]'))
    state.add_memlet_path(node, exit_node, state.add_write('D'), src_conn='_out', memlet=dace.Memlet('D[i]'))
    sdfg.apply_gpu_transformations()
    return sdfg


@pytest.mark.gpu
@pytest.mark.ai
def test_gpu_microkernel():
    sdfg = _device_sdfg()
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, AINode))
    node.expand(state, 'ai')

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    body = tasklet.code.as_string
    # __state is not a kernel parameter, so device code must not reach for it
    assert '__state' not in body
    assert not tasklet.state_fields

    rng = np.random.default_rng(0)
    a = rng.random(N, dtype=np.float32)
    b = rng.random(N, dtype=np.float32)
    c = rng.random(N, dtype=np.float32)
    d = np.zeros(N, dtype=np.float32)
    sdfg(A=a, B=b, C=c, D=d)
    assert np.allclose(d, a * b + c, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'ai and gpu'])
