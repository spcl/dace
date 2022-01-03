# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.sdfg.nodes as nodes
from dace.fpga_testing import xilinx_test
import importlib.util
import numpy as np
from pathlib import Path


@xilinx_test()
def test_map_unroll_processing_elements():

    # Grab the systolic GEMM implementation the samples directory
    spec = importlib.util.spec_from_file_location(
        "gemm",
        Path(__file__).parent.parent.parent / "samples" / "fpga" /
        "gemm_systolic_vectorized.py")
    gemm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm)

    N = 128
    K = 256
    M = 512
    P = 8
    W = 4
    TN = 32
    TM = 128

    # Create an SDFG with multiple processing elements
    sdfg = gemm.make_sdfg("map_unroll_processing_elements",
                          dace.vector(dace.float32, W))
    sdfg.specialize({"P": P, "W": W, "TN": TN, "TM": TM})
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and node.params == ["p"]:
                node.unroll = False
                node.schedule = dace.ScheduleType.Unrolled

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N, K], dtype=dace.float32.type)
    B = np.ndarray([K, M], dtype=dace.float32.type)
    C = np.ndarray([N, M], dtype=dace.float32.type)
    A[:] = np.random.rand(N, K).astype(dace.float32.type)
    B[:] = np.random.rand(K, M).astype(dace.float32.type)
    C[:] = np.random.rand(N, M).astype(dace.float32.type)

    C_regression = A @ B + C

    sdfg(A=A, B=B, C=C, N=N, M=M, K=K)
    diff = np.linalg.norm(C_regression - C) / float(N * M)
    if not np.allclose(C_regression, C):
        raise ValueError("Verification failed.")

    return sdfg


if __name__ == "__main__":
    test_map_unroll_processing_elements(None)
