# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Tests GEMM expansions for FPGA
import dace
import numpy as np
import dace.libraries.blas as blas
from dace.transformation.interstate import InlineSDFG, FPGATransformSDFG
from multiprocessing import Process, Queue
from dace.config import Config


def create_gemm_sdfg(sdfg_name,
                     alpha,
                     beta,
                     A,
                     B,
                     C,
                     dtype,
                     transA=False,
                     transB=False,
                     vec_width=1,
                     expansion_args=None):
    '''
    Build an SDFG that perform the given GEMM operation along the given axis
    Input data A, B, and C is not vectorized
    '''
    sdfg = dace.SDFG(sdfg_name)

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")
    A_shape = A.shape
    B_shape = B.shape
    C_shape = C.shape
    N = A_shape[0]
    K = A_shape[1]
    M = B_shape[1]
    vec_type = dace.vector(dtype, vec_width)

    # Create data containers
    sdfg.add_array('A', A_shape, dtype)
    sdfg.add_array("A_device",
                   shape=A_shape,
                   dtype=dtype,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("B", [K, M / vec_width], dtype=vec_type)
    sdfg.add_array("B_device", [K, M / vec_width],
                   dtype=vec_type,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)

    sdfg.add_array("C", [N, M / vec_width], dtype=vec_type)
    sdfg.add_array("C_device", [N, M / vec_width],
                   dtype=vec_type,
                   transient=True,
                   storage=dace.dtypes.StorageType.FPGA_Global)

    # Copy A
    in_host_A = copy_in_state.add_read("A")
    in_device_A = copy_in_state.add_write("A_device")
    copy_in_state.add_memlet_path(in_host_A,
                                  in_device_A,
                                  memlet=dace.Memlet(f"A[0:{N}, 0:{K}]"))

    # Copy B
    in_host_B = copy_in_state.add_read("B")
    in_device_B = copy_in_state.add_write("B_device")
    copy_in_state.add_memlet_path(
        in_host_B,
        in_device_B,
        memlet=dace.Memlet(f"B[0:{K}, 0:{M}/{vec_width}]"))

    # Copy C
    in_host_C = copy_in_state.add_read("C")
    in_device_C = copy_in_state.add_write("C_device")
    copy_in_state.add_memlet_path(
        in_host_C,
        in_device_C,
        memlet=dace.Memlet(f"C[0:{N}, 0:{M}/{vec_width}]"))

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_from_device")

    out_device = copy_out_state.add_read("C_device")
    out_host = copy_out_state.add_write("C")
    copy_out_state.add_memlet_path(
        out_device,
        out_host,
        memlet=dace.Memlet(f"C[0:{N}, 0:{M}//{vec_width}]"))

    ########################################################################
    # FPGA State

    fpga_state = sdfg.add_state("fpga_state")
    in_A = fpga_state.add_read("A_device")
    in_B = fpga_state.add_read("B_device")
    in_C = fpga_state.add_read("C_device")
    out_C = fpga_state.add_read("C_device")

    gemm_node = blas.Gemm("gemm",
                          transA=transA,
                          transB=transB,
                          alpha=alpha,
                          beta=beta)
    gemm_node.implementation = "FPGA1DSystolic"

    fpga_state.add_memlet_path(in_A,
                               gemm_node,
                               dst_conn="_a",
                               memlet=dace.Memlet(f"A_device[0:{N}, 0:{K}]"))
    fpga_state.add_memlet_path(
        in_B,
        gemm_node,
        dst_conn="_b",
        memlet=dace.Memlet(f"B_device[0:{K}, 0:{M}/{vec_width}]"))
    fpga_state.add_memlet_path(
        in_C,
        gemm_node,
        dst_conn="_cin",
        memlet=dace.Memlet(f"C_device[0:{N}, 0:{M}/{vec_width}]"))
    fpga_state.add_memlet_path(
        gemm_node,
        out_C,
        src_conn="_c",
        memlet=dace.Memlet(f"C_device[0:{N}, 0:{M}/{vec_width}]"))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.validate()

    if expansion_args is not None:
        gemm_node.expand(sdfg, fpga_state, **expansion_args)

    return sdfg


def evaluate(sdfg, A, B, C, C_regression):
    '''
    Utility function to evaluate the SDFG in a separate process.
    This is needed to avoid Intel FPGA emulation runtime to segfault, or to leave
    threads running that will slow down evaluation
     (which seems to be the case since in these tests we generate autorun kernels).
    '''
    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, C_regression, atol=1e-6)


def test_gemm_vectorized():
    # Test with vectorization
    A = np.random.rand(128, 128).astype(np.float32)
    B = np.random.rand(128, 128).astype(np.float32)
    C = np.random.rand(128, 128).astype(np.float32)
    alpha = 2.1
    beta = 1.5
    vec_width = 4
    sdfg = create_gemm_sdfg("gemm_vectorized",
                            alpha,
                            beta,
                            A,
                            B,
                            C,
                            dace.float32,
                            vec_width=vec_width)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    # compute ground truth
    C_regression = alpha * (A @ B) + beta * C
    p = Process(target=evaluate, args=(sdfg, A, B, C, C_regression))
    p.start()
    p.join()
    del sdfg


def test_gemm_size_not_multiples_of():

    # Test with matrix sizes that are not a multiple of #PEs and Tile sizes
    A = np.random.rand(120, 128).astype(np.float32)
    B = np.random.rand(128, 128).astype(np.float32)
    C = np.random.rand(120, 128).astype(np.float32)
    expansion_args = {"tile_size_m": 50, "num_pes": 7}
    sdfg = create_gemm_sdfg("gemm_not_multiple_of",
                            1,
                            1,
                            A,
                            B,
                            C,
                            dace.float32,
                            expansion_args=expansion_args)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    # compute ground truth
    C_regression = A @ B + C
    p = Process(target=evaluate, args=(sdfg, A, B, C, C_regression))
    p.start()
    p.join()
    del sdfg


def test_matmul_np():
    # Test with numpy matmul, and double precision
    @dace.program
    def matmul_np(A: dace.float64[128, 64], B: dace.float64[64, 32],
                  C: dace.float64[128, 32]):
        C[:] = A @ B

    A = np.random.rand(128, 64).astype(np.float64)
    B = np.random.rand(64, 32).astype(np.float64)
    C = np.random.rand(128, 32).astype(np.float64)

    sdfg = matmul_np.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])
    from dace.libraries.blas import Gemm
    Gemm.default_implementation = "FPGA1DSystolic"
    # We have to Inline
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    C_regression = A @ B
    p = Process(target=evaluate, args=(sdfg, A, B, C, C_regression))
    p.start()
    p.join()
    del sdfg


if __name__ == "__main__":
    # These tests will be executed on seperate process. The tranfform_on_call DACE configuration must be set to false
    Config.set('optimizer', 'transform_on_call', value=False)

    test_gemm_vectorized()
    test_gemm_size_not_multiples_of()
    test_matmul_np()
