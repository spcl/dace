# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA Tests for reshaping and reinterpretation of existing arrays.
    Part of the following tests are based on the ones in numpy/reshape_test.py"""
import dace
import numpy as np
import pytest
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG, GPUTransformSDFG, NestSDFG
from dace.fpga_testing import fpga_test

N = dace.symbol('N')


@fpga_test()
def test_view_fpga_sdfg():
    """
    Manually built FPGA-SDFG with a view: Array -> view -> Array
    """

    sdfg = dace.SDFG("view_fpga")

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array('A', [2, 3, 4], dace.float32)

    in_host_A = copy_in_state.add_read('A')

    sdfg.add_array("device_A",
                   shape=[2, 3, 4],
                   dtype=dace.float32,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    in_device_A = copy_in_state.add_write("device_A")

    copy_in_state.add_memlet_path(in_host_A, in_device_A, memlet=dace.Memlet("A[0:2,0:3,0:4]"))
    ###########################################################################
    # Copy data from FPGA

    copy_out_state = sdfg.add_state("copy_from_device")
    sdfg.add_array('B', [8, 3], dace.float32)
    sdfg.add_array("device_B",
                   shape=[8, 3],
                   dtype=dace.float32,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    out_device = copy_out_state.add_read("device_B")
    out_host = copy_out_state.add_write("B")

    copy_out_state.add_memlet_path(out_device, out_host, memlet=dace.Memlet("B[0:8,0:3]"))

    ########################################################################
    # FPGA State

    fpga_state = sdfg.add_state("fpga_state")

    sdfg.add_view('Av', [8, 3], dace.float32, storage=dace.dtypes.StorageType.FPGA_Global)
    r = fpga_state.add_read('device_A')
    v = fpga_state.add_access('Av')
    w = fpga_state.add_write('device_B')
    fpga_state.add_edge(r, None, v, 'views', dace.Memlet(data='device_A'))
    fpga_state.add_nedge(v, w, dace.Memlet(data='device_B'))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state, dace.sdfg.sdfg.InterstateEdge())

    sdfg.validate()

    ###########################################################################################
    # Execute

    A = np.random.rand(2, 3, 4).astype(np.float32)
    B = np.random.rand(8, 3).astype(np.float32)
    sdfg(A=A, B=B)
    assert np.allclose(A, np.reshape(B, [2, 3, 4]))

    return sdfg


@fpga_test()
def test_reshape_np():
    """
    Dace program with numpy reshape, transformed for FPGA
    """
    @dace.program
    def reshp_np(A: dace.float32[3, 4], B: dace.float32[2, 6]):
        B[:] = np.reshape(A, [2, 6])

    A = np.random.rand(3, 4).astype(np.float32)
    B = np.random.rand(2, 6).astype(np.float32)

    sdfg = reshp_np.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg(A=A, B=B)
    assert np.allclose(np.reshape(A, [2, 6]), B)

    return sdfg


@fpga_test()
def test_reshape_dst_explicit():
    """ Tasklet->View->Array """
    sdfg = dace.SDFG('reshapedst')
    sdfg.add_array('A', [2, 3, 4], dace.float64)
    sdfg.add_view('Bv', [2, 3, 4], dace.float64)
    sdfg.add_array('B', [8, 3], dace.float64)
    state = sdfg.add_state()

    me, mx = state.add_map('compute', dict(i='0:2', j='0:3', k='0:4'))
    t = state.add_tasklet('add', {'a'}, {'b'}, 'b = a + 1')
    state.add_memlet_path(state.add_read('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i,j,k]'))
    v = state.add_access('Bv')
    state.add_memlet_path(t, mx, v, src_conn='b', memlet=dace.Memlet('Bv[i,j,k]'))
    state.add_nedge(v, state.add_write('B'), dace.Memlet('B'))
    sdfg.validate()

    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg(A=A, B=B)
    assert np.allclose(A + 1, np.reshape(B, [2, 3, 4]))

    return sdfg


@fpga_test(assert_ii_1=False)
def test_view_slice():
    """
        In this test we use slice. In this case a view is used to access
        the desired portion of the original array
        (this is part of symm polybench kernel)
    """
    N = dace.symbol('N', dace.int32)
    M = dace.symbol('M', dace.int32)

    @dace.program
    def view_slice(alpha: dace.float32, beta: dace.float32, C: dace.float32[M, N], A: dace.float32[M, M],
                   B: dace.float32[M, N]):

        C *= beta
        for i in range(M):
            for j in range(N):
                C[:i, j] += alpha * B[i, j] * A[i, :i]

    def kernel_numpy(M, N, alpha, beta, C, A, B):
        C *= beta
        for i in range(M):
            for j in range(N):
                C[:i, j] += alpha * B[i, j] * A[i, :i]

    M, N = 16, 32
    alpha = 2
    beta = 3
    A = np.random.rand(M, M).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    C = np.random.rand(M, N).astype(np.float32)
    np_C = np.copy(C)
    kernel_numpy(M, N, alpha, beta, np_C, A, B)
    sdfg = view_slice.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg(A=A, B=B, C=C, alpha=alpha, beta=beta, M=M, N=N)
    assert np.allclose(C, np_C, atol=1e-06)

    return sdfg


if __name__ == "__main__":
    test_reshape_np(None)
    test_view_fpga_sdfg(None)
    test_reshape_dst_explicit(None)
    test_view_slice(None)
