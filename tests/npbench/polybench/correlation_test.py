# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize

# Data set sizes
# M, N
sizes = {"mini": (28, 32), "small": (80, 100), "medium": (240, 260), "large": (1200, 1400), "extra-large": (2600, 3000)}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def correlation_kernel(float_n: dc.float64, data: dc.float64[N, M]):

    mean = np.mean(data, axis=0)
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.mean(np.subtract(data, mean)**2, axis=0))
    stddev[stddev <= 0.1] = 1.0
    # data -= mean
    np.subtract(data, mean, out=data)
    # data /= np.sqrt(float_n) * stddev
    np.divide(data, np.sqrt(float_n) * stddev, out=data)
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        # corr[i, i+1:M] = np.transpose(data[:, i+1:M]) @ data[:, i]
        corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
        corr[i + 1:M, i] = corr[i, i + 1:M]

    return corr


def initialize(M, N, datatype=np.float64):
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=datatype)

    return float_n, data


def ground_truth(M, float_n, data):

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]

    return corr


def run_correlation(device_type: dace.dtypes.DeviceType):
    '''
    Runs correlation for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    M, N = sizes["small"]
    float_n, data = initialize(M, N)
    float_n_ref = np.copy(float_n)
    data_ref = np.copy(data)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = correlation_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        corr = sdfg(float_n, data, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        pass  # Not Yet Implemented

    # Compute ground truth and validate result

    corr_ref = ground_truth(M, float_n_ref, data_ref)
    assert np.allclose(corr, corr_ref)
    return sdfg


def test_cpu():
    run_correlation(dace.dtypes.DeviceType.CPU)


@pytest.mark.skip(reason="GPU Error")
@pytest.mark.gpu
def test_gpu():
    run_correlation(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_correlation(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_correlation(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_correlation(dace.dtypes.DeviceType.FPGA)
