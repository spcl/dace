# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.config import set_temporary

C_in, N, S0, S1, S2, N1, N2 = (dc.symbol(s, dtype=dc.int64) for s in ('C_in', 'N', 'S0', 'S1', 'S2', 'N1', 'N2'))


@dc.program
def relu(x: dc.float32[N1, N2]):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@dc.program
def softmax(x: dc.float32[N1, N2]):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
@dc.program
def mlp_kernel(input: dc.float32[N, C_in], w1: dc.float32[C_in, S0], b1: dc.float32[S0], w2: dc.float32[S0, S1],
               b2: dc.float32[S1], w3: dc.float32[S1, S2], b3: dc.float32[S2]):
    x1 = relu(input @ w1 + b1)
    x2 = relu(x1 @ w2 + b2)
    x3 = softmax(x2 @ w3 + b3)  # Softmax call can be omitted if necessary
    return x3


def initialize(C_in, N, S0, S1, S2):
    from numpy.random import default_rng
    rng = default_rng(42)

    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float32)
    b1 = rng.random((mlp_sizes[0], ), dtype=np.float32)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float32)
    b2 = rng.random((mlp_sizes[1], ), dtype=np.float32)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float32)
    b3 = rng.random((mlp_sizes[2], ), dtype=np.float32)

    return input, w1, b1, w2, b2, w3, b3


def relu_np(x):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
def softmax_np(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
def mlp_np(input, w1, b1, w2, b2, w3, b3):
    x = relu_np(input @ w1 + b1)
    x = relu_np(x @ w2 + b2)
    x = softmax_np(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


def run_mlp(device_type: dace.dtypes.DeviceType):
    '''
    Runs conv2d_bias for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    C_in, N, S0, S1, S2 = 3, 8, 30000, 2000, 2000
    input, w1, b1, w2, b2, w3, b3 = initialize(C_in, N, S0, S1, S2)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = mlp_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(input, w1, b1, w2, b2, w3, b3, N=N, S0=S0, S1=S1, S2=S2, C_in=C_in)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = mlp_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.standard import Reduce
        Reduce.default_implementation = "FPGAPartialReduction"
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N, S0=S0, S1=S1, S2=S2, C_in=C_in))
        out = sdfg(input, w1, b1, w2, b2, w3, b3)

    # Compute ground truth and validate
    out_ref = mlp_np(input, w1, b1, w2, b2, w3, b3)
    assert np.allclose(out, out_ref)
    return sdfg


def test_cpu():
    run_mlp(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_mlp(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Intel, compilation error")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_mlp(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_mlp(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_mlp(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_mlp(dace.dtypes.DeviceType.FPGA)
