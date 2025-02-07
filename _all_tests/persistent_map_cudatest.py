# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import scipy
import pytest

import dace
from dace import nodes
from dace.dtypes import ScheduleType

W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')


@dace.program(dace.uint32[H + 1], dace.uint32[nnz], dace.float32[nnz], dace.float32[W], dace.float32[H])
def spmv(A_row, A_col, A_val, x, b):
    for ignore in dace.map[0]:
        for i in dace.map[0:H]:

            @dace.map(_[A_row[i]:A_row[i + 1]])
            def compute(j):
                a << A_val[j]
                in_x << x[A_col[j]]
                out >> b(1, lambda x, y: x + y)[i]

                out = a * in_x


@pytest.mark.gpu
def test_persistent_dynamic_map():
    sdfg = spmv.to_sdfg()
    sdfg.apply_gpu_transformations()

    for state in sdfg:
        for scope in state.nodes():
            if not isinstance(scope, nodes.EntryNode):
                continue
            if state.entry_node(scope) is None:
                scope.map.schedule = ScheduleType.GPU_Persistent
            elif state.entry_node(state.entry_node(scope)) is None:
                scope.map.schedule = ScheduleType.GPU_Device
            else:
                scope.map.schedule = ScheduleType.GPU_ThreadBlock_Dynamic

    verify(sdfg)


@pytest.mark.gpu
def test_persistent_default():
    sdfg = spmv.to_sdfg()
    sdfg.apply_gpu_transformations()

    for state in sdfg:
        for scope in state.nodes():
            if not isinstance(scope, nodes.EntryNode):
                continue
            if state.entry_node(scope) is None:
                scope.map.schedule = ScheduleType.GPU_Persistent
            else:
                scope.map.schedule = ScheduleType.Default

    verify(sdfg)


def verify(sdfg):
    height = 1024
    width = 1024

    # Fill input data
    # each row has up (including) 256 elements
    A_row = np.random.randint(257, size=height + 1, dtype=dace.uint32.type)
    A_row[0] = 0
    A_row = np.cumsum(A_row, dtype=dace.uint32.type)

    # Column data
    A_col = dace.ndarray([A_row[height]], dtype=dace.uint32)
    for i in range(height):
        A_col[A_row[i]:A_row[i + 1]] = np.sort(np.random.choice(width, A_row[i + 1] - A_row[i], replace=False))

    # values
    A_val = np.random.rand(A_row[height]).astype(dace.float32.type)

    A_sparse = scipy.sparse.csr_matrix((A_val, A_col, A_row), dtype=dace.float32.type, shape=(1024, 1024))

    x = np.random.rand(width).astype(dace.float32.type)
    b = np.zeros(height, dtype=dace.float32.type)

    sdfg(A_row=A_row, A_col=A_col, A_val=A_val, x=x, b=b, H=A_sparse.shape[0], W=A_sparse.shape[1], nnz=A_sparse.nnz)

    assert np.allclose(b, A_sparse.dot(x)), "Result doesn't match!"


if __name__ == '__main__':
    test_persistent_dynamic_map()
    test_persistent_default()
