# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from dace.libraries import standard


def _make_sdfg(name, storage=dace.dtypes.StorageType.CPU_Heap):

    N = dace.symbol('N', dtype=dace.int32, integer=True, positive=True)
    i = dace.symbol('i', dtype=dace.int32, integer=True)

    sdfg = dace.SDFG(name)
    _, A = sdfg.add_array('A', [N, N, N], dtype=dace.float64)
    _, B = sdfg.add_array('B', [N], dtype=dace.float64)
    _, tmp1 = sdfg.add_transient('tmp1', [N - 4, N - 4, N - i],
                                 dtype=dace.float64,
                                 storage=storage)
    _, tmp2 = sdfg.add_transient('tmp2', [1],
                                 dtype=dace.float64,
                                 storage=storage)

    begin_state = sdfg.add_state("begin", is_start_state=True)
    guard_state = sdfg.add_state("guard")
    body1_state = sdfg.add_state("body1")
    body2_state = sdfg.add_state("body2")
    body3_state = sdfg.add_state("body3")
    end_state = sdfg.add_state("end")

    sdfg.add_edge(begin_state, guard_state,
                  dace.InterstateEdge(assignments=dict(i='0')))
    sdfg.add_edge(guard_state, body1_state,
                  dace.InterstateEdge(condition=f'i<{N}'))
    sdfg.add_edge(guard_state, end_state,
                  dace.InterstateEdge(condition=f'i>={N}'))
    sdfg.add_edge(body1_state, body2_state, dace.InterstateEdge())
    sdfg.add_edge(body2_state, body3_state, dace.InterstateEdge())
    sdfg.add_edge(body3_state, guard_state,
                  dace.InterstateEdge(assignments=dict(i='i+1')))

    read_a = body1_state.add_read('A')
    write_tmp1 = body1_state.add_write('tmp1')
    body1_state.add_nedge(read_a, write_tmp1,
                          dace.Memlet(f'A[2:{N}-2, 2:{N}-2, i:{N}]'))

    read_tmp1 = body2_state.add_read('tmp1')
    rednode = standard.Reduce(wcr='lambda a, b : a + b', identity=0)
    if storage == dace.dtypes.StorageType.GPU_Global:
        rednode.implementation = 'CUDA (device)'
    elif storage == dace.dtypes.StorageType.FPGA_Global:
        rednode.implementation = 'FPGAPartialReduction'
    body2_state.add_node(rednode)
    write_tmp2 = body2_state.add_write('tmp2')
    body2_state.add_nedge(read_tmp1, rednode,
                          dace.Memlet.from_array('tmp1', tmp1))
    body2_state.add_nedge(rednode, write_tmp2, dace.Memlet('tmp2[0]'))

    read_tmp2 = body3_state.add_read('tmp2')
    write_b = body3_state.add_write('B')
    body3_state.add_nedge(read_tmp2, write_b, dace.Memlet('B[i]'))

    return sdfg


def test_symbol_dependent_heap_array():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    sdfg = _make_sdfg("symbol_dependent_heap_array")
    # Compile manually to avoid strict transformations
    sdfg_exec = sdfg.compile()
    sdfg_exec(A=A, B=B, N=10)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert (np.allclose(B, B_ref))


def test_symbol_dependent_register_array():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    sdfg = _make_sdfg("symbol_dependent_register_array",
                      storage=dace.dtypes.StorageType.Register)
    # Compile manually to avoid strict transformations
    sdfg_exec = sdfg.compile()
    sdfg_exec(A=A, B=B, N=10)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert (np.allclose(B, B_ref))


def test_symbol_dependent_threadlocal_array():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    sdfg = _make_sdfg("symbol_dependent_threadlocal_array",
                      storage=dace.dtypes.StorageType.CPU_ThreadLocal)
    # Compile manually to avoid strict transformations
    sdfg_exec = sdfg.compile()
    sdfg_exec(A=A, B=B, N=10)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert (np.allclose(B, B_ref))


@pytest.mark.gpu
def test_symbol_dependent_gpu_global_array():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    sdfg = _make_sdfg("symbol_dependent_gpu_global_array",
                      storage=dace.dtypes.StorageType.GPU_Global)
    # Compile manually to avoid strict transformations
    sdfg_exec = sdfg.compile()
    sdfg_exec(A=A, B=B, N=10)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert (np.allclose(B, B_ref))


@pytest.mark.gpu
def test_symbol_dependent_pinned_array():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    sdfg = _make_sdfg("symbol_dependent_pinned_array",
                      storage=dace.dtypes.StorageType.CPU_Pinned)
    # Compile manually to avoid strict transformations
    sdfg_exec = sdfg.compile()
    sdfg_exec(A=A, B=B, N=10)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert (np.allclose(B, B_ref))


@pytest.mark.skip
def test_symbol_dependent_fpga_global_array():
    A = np.random.randn(10, 10, 10)
    B = np.ndarray(10, dtype=np.float64)
    sdfg = _make_sdfg("symbol_dependent_fpga_global_array",
                      storage=dace.dtypes.StorageType.FPGA_Global)
    # Compile manually to avoid strict transformations
    sdfg_exec = sdfg.compile()
    sdfg_exec(A=A, B=B, N=10)
    B_ref = np.ndarray(10, dtype=np.float64)
    for i in range(10):
        tmp = A[2:-2, 2:-2, i:]
        B_ref[i] = np.sum(tmp)
    assert (np.allclose(B, B_ref))


if __name__ == '__main__':
    test_symbol_dependent_heap_array()
    test_symbol_dependent_register_array()
    test_symbol_dependent_threadlocal_array()
    test_symbol_dependent_gpu_global_array()
    test_symbol_dependent_pinned_array()
    # test_symbol_dependent_fpga_global_array()
