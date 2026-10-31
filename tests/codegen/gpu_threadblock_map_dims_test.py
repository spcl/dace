# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Thread-block maps with fewer dimensions than the thread block run on the first threads of the others."""

import numpy as np
import pytest

import dace


@dace.program
def single_thread_after_block(counter: dace.int32[1], rows: dace.int32[4, 32]):
    for _ in dace.map[0:1]:
        for w, lane in dace.map[0:4, 0:32]:  # A 32 x 4 thread block
            with dace.tasklet:
                r >> rows[w, lane]
                r = 1
        for _one in dace.map[0:1]:  # One thread: x == 0, and y == 0 too
            with dace.tasklet:
                c >> counter(1, lambda a, b: a + b)[0]
                c = 1


@pytest.mark.gpu
def test_one_thread_map_in_two_dimensional_block():
    sdfg = single_thread_after_block.to_sdfg(simplify=True)
    for desc in sdfg.arrays.values():
        desc.storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.schedule = (
                dace.ScheduleType.GPU_Device if node.map.params == ["_"] else dace.ScheduleType.GPU_ThreadBlock
            )
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert "threadIdx.y == 0" in code

    import cupy as cp  # Only GPU tests use CuPy

    counter = cp.zeros(1, dtype=np.int32)
    rows = cp.zeros((4, 32), dtype=np.int32)
    sdfg(counter=counter, rows=rows)
    assert int(counter[0]) == 1
    assert int(rows.sum()) == 128


if __name__ == "__main__":
    test_one_thread_map_in_two_dimensional_block()
