# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (20, 20), "small": (40, 60), "medium": (100, 200), "large": (500, 1000), "extra-large": (1000, 2000)}

N = dace.symbol('N', dtype=dace.int64)


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val - ref)
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


def numpy_kernel(TSTEPS, N, u):

    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in range(1, TSTEPS + 1):
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = v[0, 1:N - 1]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] -
                             a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]

        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] -
                             d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


@dace.program
def adi_kernel(TSTEPS: dace.int64, u: dace.float64[N, N]):

    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / np.float64(N)
    DY = 1.0 / np.float64(N)
    DT = 1.0 / np.float64(TSTEPS)
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in range(1, TSTEPS + 1):
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = v[0, 1:N - 1]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] -
                             a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]

        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j in range(1, N - 1):
            p[1:N - 1, j] = -f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] -
                             d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


def initialize(N, datatype=np.float64):
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)
    return u


def run_adi(device_type: dace.dtypes.DeviceType):
    '''
    Runs ADI for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small)
    TSTEPS, N = sizes["small"]
    u = initialize(N)
    dace_u = u.copy()

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = adi_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(TSTEPS=TSTEPS, u=dace_u, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = adi_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()
        # In this case, we want to generate the top-level state as an host-based state,
        # not an FPGA kernel. We need to explicitly indicate that
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(TSTEPS=TSTEPS, u=dace_u)

    # Compute ground truth and Validate result
    numpy_kernel(TSTEPS, N, u)
    assert (np.allclose(dace_u, u) or relerror(dace_u, u) < 1e-10)
    return sdfg


def test_cpu():
    run_adi(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_adi(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Intel FPGA argument overflow")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_adi(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_adi(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_adi(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_adi(dace.dtypes.DeviceType.FPGA)
