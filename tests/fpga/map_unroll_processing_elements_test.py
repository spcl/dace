# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.sdfg.nodes as nodes
from dace.fpga_testing import fpga_test
import importlib.util
import numpy as np
from pathlib import Path


@fpga_test()
def test_map_unroll_processing_elements():

    # Grab the systolic GEMM implementation the samples directory
    spec = importlib.util.spec_from_file_location(
        "gemm",
        Path(__file__).parent.parent.parent / "samples" / "fpga" /
        "gemm_systolic_vectorized.py")
    gemm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm)

    # Create an SDFG with multiple processing elements
    sdfg = gemm.make_sdfg("map_unroll_processing_elements", 4)
    sdfg.specialize({"P": 4, "M": 32})
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and node.params == ["p"]:
                node.unroll = False
                node.schedule = dace.ScheduleType.Unrolled

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([32, 32], dtype=dace.float32.type)
    B = np.ndarray([32, 32], dtype=dace.float32.type)
    C = np.ndarray([32, 32], dtype=dace.float32.type)
    A[:] = np.random.rand(32, 32).astype(dace.float32.type)
    B[:] = np.random.rand(32, 32).astype(dace.float32.type)
    C[:] = np.random.rand(32, 32).astype(dace.float32.type)

    C_regression = np.ndarray([32, 32], dtype=np.float32)
    C_regression = A @ B + C

    sdfg(A=A, B=B, C=C, N=32, K=32)
    diff = np.linalg.norm(C_regression - C) / float(32 * 32)
    assert diff < 1e-6

    return sdfg


if __name__ == "__main__":
    test_map_unroll_processing_elements(None)
