# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize


N, npt = (dace.symbol(s, dtype=dace.int64) for s in ('N', 'npt'))


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val-ref)
    return np.linalg.norm(val-ref) / np.linalg.norm(ref)


@dace.program
def dace_azimint_naive(data: dace.float64[N], radius: dace.float64[N]):
    rmax = np.amax(radius)
    res = np.zeros((npt, ), dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        on_values = 0
        tmp = np.float64(0)
        for j in dace.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values
    return res


def numpy_azimint_naive(data, radius, npt):
    rmax = radius.max()
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    return res


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    data, radius = rng.random((N, )), rng.random((N, ))
    return data, radius


def run_azimint_naive(device_type: dace.dtypes.DeviceType):
    '''
    Runs azimint-naive for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    N, npt = (400000, 1000)
    data, radius = initialize(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = dace_azimint_naive.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        val = sdfg(data=data, radius=radius, N=N, npt=npt)

    # Compute ground truth and Validate result
    ref = numpy_azimint_naive(data, radius, npt)
    assert (np.allclose(val, ref) or relerror(val, ref) < 1e-10)
    return sdfg


def test_cpu():
    run_azimint_naive(dace.dtypes.DeviceType.CPU)


# NOTE: Doesn't work yet with GPU-auto-optimize
# @pytest.mark.gpu
@pytest.mark.skip
def test_gpu():
    run_azimint_naive(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_azimint_naive(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_azimint_naive(dace.dtypes.DeviceType.GPU)
