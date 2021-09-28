# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import warnings

import dace
from dace import data, nodes
from dace.transformation.dataflow import RedundantReadSlice, RedundantWriteSlice


def _count_views(sdfg: dace.SDFG) -> int:
    num = 0
    for n, _ in sdfg.all_nodes_recursive():
        if (isinstance(n, nodes.AccessNode) and
                isinstance(sdfg.arrays[n.data], data.View)):
            num += 1
    return num


@dace.program
def jacobi1d_half(TMAX: dace.int32, A: dace.float32[12], B: dace.float32[12]):
    for _ in range(TMAX):
        B[1:-1] = 0.3333 * (A[:-2] + A[1:-1] + A[2:])


def test_read_slice():
    sdfg = jacobi1d_half.to_sdfg(strict=False)
    num_views_before = _count_views(sdfg)
    if num_views_before != 3:
        warnings.Warn("Incorrect number of Views detected. Please ensure that "
                      "the test is compatible with this DaCe version.")
    sdfg.apply_transformations_repeated(RedundantReadSlice)
    num_views_after = _count_views(sdfg)
    assert (num_views_after == 0)


@dace.program
def write_slice(A: dace.float32[10]):
    B = A[2:8]
    B[:] = np.pi


def test_write_slice():
    sdfg = write_slice.to_sdfg(strict=False)
    sdfg.save('test_before.sdfg')
    num_views_before = _count_views(sdfg)
    if num_views_before == 0:
        warnings.Warn("Incorrect number of Views detected. Please ensure that "
                      "the test is compatible with this DaCe version.")
    print(num_views_before)
    sdfg.apply_transformations_repeated(RedundantWriteSlice)
    sdfg.save('test_after.sdfg')
    num_views_after = _count_views(sdfg)
    assert (num_views_after == 0)


if __name__ == '__main__':
    test_read_slice()
    test_write_slice()
