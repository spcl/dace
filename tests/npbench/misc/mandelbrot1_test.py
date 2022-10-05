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

XN, YN, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'N'])


@dc.program
def linspace(start: dc.float64, stop: dc.float64, X: dc.float64[N]):
    dist = (stop - start) / (N - 1)
    for i in dc.map[0:N]:
        X[i] = start + i * dist


@dc.program
def mandelbrot_kernel(xmin: dc.float64, xmax: dc.float64, ymin: dc.float64, ymax: dc.float64, maxiter: dc.int64,
                      horizon: dc.float64):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = np.ndarray((XN, ), dtype=np.float64)
    Y = np.ndarray((YN, ), dtype=np.float64)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    # C = X + np.reshape(Y, (YN, 1)) * 1j
    C = np.ndarray((YN, XN), dtype=np.complex128)
    for i, j in dc.map[0:YN, 0:XN]:
        C[i, j] = X[j] + Y[i] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex128)
    for n in range(maxiter):
        I = np.less(np.absolute(Z), horizon)
        N[I] = n
        for j, k in dc.map[0:YN, 0:XN]:
            if I[j, k]:
                Z[j, k] = Z[j, k]**2 + C[j, k]
    N[N == maxiter - 1] = 0
    return Z, N


def ground_truth(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)
    C = X + Y[:, None] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex128)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter - 1] = 0
    return Z, N


def run_mandelbrot1(device_type: dace.dtypes.DeviceType):
    '''
    Runs mandelbrot1 for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    xmin, xmax, XN, ymin, ymax, YN, maxiter, horizon = -1.75, 0.25, 125, -1.00, 1.00, 125, 60, 2.0

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = mandelbrot_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        Z, N = sdfg(xmin, xmax, ymin, ymax, maxiter, horizon, XN=XN, YN=YN)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = mandelbrot_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(XN=XN, YN=YN))
        Z, N = sdfg(xmin, xmax, ymin, ymax, maxiter, horizon)

    # Compute ground truth and validate
    Z_ref, N_ref = ground_truth(xmin, xmax, ymin, ymax, XN, YN, maxiter)
    assert np.linalg.norm(Z - Z_ref) / np.linalg.norm(Z_ref) < 1e-6
    assert np.linalg.norm(N - N_ref) / np.linalg.norm(N_ref) < 1e-6
    return sdfg


@pytest.mark.skip(reason="Parsing error")
def test_cpu():
    run_mandelbrot1(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="Parsing error")
@pytest.mark.gpu
def test_gpu():
    run_mandelbrot1(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Parsing error")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_mandelbrot1(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_mandelbrot1(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_mandelbrot1(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_mandelbrot1(dace.dtypes.DeviceType.FPGA)
