# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val - ref)
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


M, N = (dace.symbol(s, dtype=dace.int64) for s in ('M', 'N'))


@dace.program
def compute(array_1: dace.int64[M, N], array_2: dace.int64[M, N], a: dace.int64, b: dace.int64, c: dace.int64):
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c


def initialize(M, N):
    from numpy.random import default_rng
    rng = default_rng(42)
    array_1 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    array_2 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    a = np.int64(4)
    b = np.int64(3)
    c = np.int64(9)
    return array_1, array_2, a, b, c


def run_compute(device_type: dace.dtypes.DeviceType):
    '''
    Runs compute for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    M, N = (2000, 2000)
    array_1, array_2, a, b, c = initialize(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = compute.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        val = sdfg(array_1=array_1, array_2=array_2, a=a, b=b, c=c, M=M, N=N)

    # Compute ground truth and Validate result
    ref = compute.f(array_1, array_2, a, b, c)
    assert (np.allclose(val, ref) or relerror(val, ref) < 1e-10)
    return sdfg


def test_cpu():
    run_compute(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_compute(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_compute(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_compute(dace.dtypes.DeviceType.GPU)
