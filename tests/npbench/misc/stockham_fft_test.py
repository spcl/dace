# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

R, K, M1, M2 = (dc.symbol(s, dtype=dc.int64, integer=True, positive=True) for s in ('R', 'K', 'M1', 'M2'))
N = R**K


@dc.program
def mgrid1(X: dc.uint32[R, R], Y: dc.uint32[R, R]):
    for i in range(R):
        X[i, :] = i
    for j in range(R):
        Y[:, j] = j


@dc.program
def mgrid2(X: dc.uint32[R, N], Y: dc.uint32[R, N]):
    for i in range(R):
        X[i, :] = i
    for j in range(R**K):
        Y[:, j] = j


@dc.program
def stockham_fft_kernel(x: dc.complex128[R**K], y: dc.complex128[R**K]):

    # Generate DFT matrix for radix R.
    # Define transient variable for matrix.
    # i_coord, j_coord = np.mgrid[0:R, 0:R]
    i_coord = np.ndarray((R, R), dtype=np.uint32)
    j_coord = np.ndarray((R, R), dtype=np.uint32)
    mgrid1(i_coord, j_coord)
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat[:] = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
    # Move input x to output y
    # to avoid overwriting the input.
    y[:] = x[:]

    # ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]
    ii_coord = np.ndarray((R, N), dtype=np.uint32)
    jj_coord = np.ndarray((R, N), dtype=np.uint32)
    mgrid2(ii_coord, jj_coord)

    tmp_perm = np.empty_like(y)
    D = np.empty_like(y)
    tmp = np.empty_like(y)

    # Main Stockham loop
    for i in range(K):

        # Stride permutation
        yv = np.reshape(y, (R**i, R, R**(K - i - 1)))
        # tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        tmp_perm[:] = np.reshape(np.transpose(yv, axes=(1, 0, 2)), (N, ))
        # Twiddle Factor multiplication
        # D = np.empty((R, R ** i, R ** (K-i-1)), dtype=np.complex128)
        Dv = np.reshape(D, (R, R**i, R**(K - i - 1)))
        tmpv = np.reshape(tmp, (R**(K - i - 1), R, R**i))
        tmpv[0] = np.exp(-2.0j * np.pi * ii_coord[:, :R**i] * jj_coord[:, :R**i] / R**(i + 1))
        for k in range(R**(K - i - 1)):
            # D[:, :, k] = tmp
            Dv[:, :, k] = np.reshape(tmpv[0], (R, R**i, 1))
        # tmp_twid = np.reshape(tmp_perm, (N, )) * np.reshape(D, (N, ))
        tmp_twid = tmp_perm * D
        # Product with Butterfly
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R**(K - 1))), (N, ))


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(R, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    N = R**K
    X = rng_complex((N, ), rng)
    Y = np.zeros_like(X, dtype=np.complex128)

    return N, X, Y


def ground_truth(N, R, K, x, y):

    # Generate DFT matrix for radix R.
    # Define transient variable for matrix.
    i_coord, j_coord = np.mgrid[0:R, 0:R]
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
    # Move input x to output y
    # to avoid overwriting the input.
    y[:] = x[:]

    ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]

    # Main Stockham loop
    for i in range(K):

        # Stride permutation
        yv = np.reshape(y, (R**i, R, R**(K - i - 1)))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        # Twiddle Factor multiplication
        D = np.empty((R, R**i, R**(K - i - 1)), dtype=np.complex128)
        tmp = np.exp(-2.0j * np.pi * ii_coord[:, :R**i] * jj_coord[:, :R**i] / R**(i + 1))
        D[:] = np.repeat(np.reshape(tmp, (R, R**i, 1)), R**(K - i - 1), axis=2)
        tmp_twid = np.reshape(tmp_perm, (N, )) * np.reshape(D, (N, ))
        # Product with Butterfly
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R**(K - 1))), (N, ))


def run_stockham_fft(device_type: dace.dtypes.DeviceType):
    '''
    Runs SPMV for the given device
    :return: the SDFG
    '''

    # Initialize data (small size)
    R, K = 2, 15
    N, x, y = initialize(R, K)
    y_ref = np.copy(y)
    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = stockham_fft_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg.expand_library_nodes()
        sdfg(x=x, y=y, N=N, R=R, K=K)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = stockham_fft_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemm
        Gemm.default_implementation = "FPGA1DSystolic"
        sdfg.expand_library_nodes()

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

        sdfg.specialize(dict(N=N, R=R, K=K))
        sdfg(x=x, y=y)

    # Compute ground truth and Validate result
    ground_truth(N, R, K, x, y_ref)
    assert np.allclose(y, y_ref)
    return sdfg


@pytest.mark.skip(reason="Error in expansion")
def test_cpu():
    run_stockham_fft(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="Runtime error")
@pytest.mark.gpu
def test_gpu():
    run_stockham_fft(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Missing free symbol")
@fpga_test(assert_ii_1=False)
def test_fpga():
    run_stockham_fft(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_stockham_fft(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_stockham_fft(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_stockham_fft(dace.dtypes.DeviceType.FPGA)