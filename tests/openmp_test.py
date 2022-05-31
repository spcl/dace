# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace import dtypes, nodes

N = dace.symbol("N")


@dace.program
def arrayop(inp: dace.float32[N], out: dace.float32[N]):
    for i in dace.map[0:N]:
        out[i] = 2 * inp[i]


def test_openmp():
    N = 1000
    rng = np.random.default_rng(42)
    A = rng.random((N,), dtype=np.float32)
    B = np.zeros((N,), dtype=np.float32)

    sdfg = arrayop.to_sdfg(simplify=True)
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.EntryNode):
            assert(isinstance(node, nodes.MapEntry))
            node.map.schedule = dtypes.ScheduleType.CPU_Multicore
            node.map.schedule.omp_num_threads = 3
            node.map.schedule.omp_schedule = dtypes.OMPScheduleType.Guided
            node.map.schedule.omp_chunk_size = 5
            break
    
    sdfg(inp=A, out=B, N=N)


if __name__ == "__main__":
    test_openmp()
