# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest
import dace
from dace import nodes
from dace.dtypes import ScheduleType
from dace.transformation.dataflow import StripMining

N = dace.symbol("N")


@dace.program
def dot(A: dace.float32[N], B: dace.float32[N], out: dace.float64[1]):
    @dace.map
    def product(i: _[0:N]):
        a << A[i]
        b << B[i]
        o >> out(1, lambda x, y: x + y)
        o = a * b


@pytest.mark.gpu
def test_persistent_thread_block():

    sdfg = dot.to_sdfg()

    sdfg.apply_gpu_transformations()
    sdfg.apply_transformations(StripMining, options={'tile_size': '256'})

    for state in sdfg:
        for scope in state.nodes():
            if not isinstance(scope, nodes.EntryNode):
                continue
            if state.entry_node(scope) is None:
                scope.map.schedule = ScheduleType.GPU_Device
            else:
                scope.map.schedule = ScheduleType.GPU_ThreadBlock

    N = 1050

    print('Dot product (N = {})'.format(N))

    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)
    out_AB = np.zeros(1, dtype=np.float64)
    out_AA = np.zeros(1, dtype=np.float64)

    sdfg(A=A, B=B, out=out_AB, N=N)

    sdfg(A=A, B=A, out=out_AA, N=N)

    assert (np.allclose(out_AB, np.dot(A, B)) and np.allclose(out_AA, np.dot(A, A))), "Result doesn't match!"
    print("Complete.")


if __name__ == "__main__":
    test_persistent_thread_block()
