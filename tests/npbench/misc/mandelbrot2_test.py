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

XN, YN, M, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'M', 'N'])


@dc.program
def mgrid(X: dc.int64[M, N], Y: dc.int64[M, N]):
    for i in range(M):
        X[i, :] = i
    for j in range(N):
        Y[:, j] = j


@dc.program
def linspace(start: dc.float64, stop: dc.float64, X: dc.float64[N]):
    dist = (stop - start) / (N - 1)
    for i in dace.map[0:N]:
        X[i] = start + i * dist


@dc.program
def mandelbrot_kernel(xmin: dc.float64, xmax: dc.float64, ymin: dc.float64, ymax: dc.float64, maxiter: dc.int64,
                      horizon: dc.float64):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi = np.ndarray((XN, YN), dtype=np.int64)
    Yi = np.ndarray((XN, YN), dtype=np.int64)
    mgrid(Xi, Yi)
    X = np.ndarray((XN, ), dtype=np.float64)
    Y = np.ndarray((YN, ), dtype=np.float64)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    C = np.ndarray((XN, YN), dtype=np.complex128)
    for i, j in dc.map[0:XN, 0:YN]:
        C[i, j] = X[i] + Y[j] * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xiv = np.reshape(Xi, (XN * YN, ))
    Yiv = np.reshape(Yi, (XN * YN, ))
    Cv = np.reshape(C, (XN * YN, ))

    Z = np.zeros(Cv.shape, np.complex128)
    I = np.ndarray((XN * YN, ), dtype=np.bool_)
    length = XN * YN
    k = 0
    while length > 0 and k < maxiter:

        # Compute for relevant points only
        Z[:length] = np.multiply(Z[:length], Z[:length])
        Z[:length] = np.add(Z[:length], Cv[:length])

        # Failed convergence
        I[:length] = np.absolute(Z[:length]) > horizon
        for j in range(length):
            if I[j]:
                N_[Xiv[j], Yiv[j]] = k + 1
        for j in range(length):
            if I[j]:
                Z_[Xiv[j], Yiv[j]] = Z[j]

        # Keep going with those who have not diverged yet
        I[:length] = np.logical_not(I[:length])  # np.negative(I, I) not working any longer
        count = 0

        for j in range(length):
            if I[j]:
                Z[count] = Z[j]
                Xiv[count] = Xiv[j]
                Yiv[count] = Yiv[j]
                Cv[count] = Cv[j]
                count += 1
        length = count
        k += 1
    return Z_.T, N_.T


def ground_truth(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi.shape = Yi.shape = C.shape = xn * yn

    Z = np.zeros(C.shape, np.complex128)
    for i in range(itermax):
        if not len(Z):
            break

        # Compute for relevant points only
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)

        # Failed convergence
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]

        # Keep going with those who have not diverged yet
        np.logical_not(I, I)  # np.negative(I, I) not working any longer
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    return Z_.T, N_.T


def run_mandelbrot2(device_type: dace.dtypes.DeviceType):
    '''
    Runs mandelbrot2 for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    xmin, xmax, XN, ymin, ymax, YN, maxiter, horizon = -2.00, 0.50, 200, -1.25, 1.25, 200, 40, 2.0

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
    assert np.allclose(Z, Z_ref)
    assert np.allclose(N, N_ref)
    return sdfg


@pytest.mark.skip(reason="Parsing error")
def test_cpu():
    run_mandelbrot2(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="Parsing error")
@pytest.mark.gpu
def test_gpu():
    run_mandelbrot2(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Parsing error")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_mandelbrot2(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_mandelbrot2(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_mandelbrot2(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_mandelbrot2(dace.dtypes.DeviceType.FPGA)
