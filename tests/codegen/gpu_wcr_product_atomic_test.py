# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A product that many GPU threads accumulate into one element keeps every factor.

Under ``DACE_USE_GPU_ATOMICS`` the ``Product`` resolution's ``reduce_atomic`` called the NON-atomic
``wcr_custom::reduce``, so blocks racing on one accumulator lost multiplications: TSVC s312's product came back
0.87x, 0.84x and -0.81x of the reference over three calls.
"""
import numpy as np
import pytest

import dace

N = dace.symbol('N')


@dace.program
def product(a: dace.float64[N], out: dace.float64[1]):
    for i in dace.map[0:N]:
        out[0] *= a[i]


@pytest.mark.gpu
def test_a_gpu_product_reduction_keeps_every_factor():
    sdfg = product.to_sdfg()
    sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device:
            node.map.gpu_block_size = [128, 1, 1]
    a = np.ones(1 << 16)
    a[::1 << 11] = 2.0
    out = np.ones(1)
    sdfg(a=a, out=out, N=a.size)
    assert out[0] == 2.0**32, out[0] / 2.0**32
