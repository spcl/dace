# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize

N = dace.symbol('N', dtype=dace.int64)


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val - ref)
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


@dace.program
def arc_distance(theta_1: dace.float64[N], phi_1: dace.float64[N], theta_2: dace.float64[N], phi_2: dace.float64[N]):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    temp = np.sin((theta_2 - theta_1) / 2)**2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2)**2
    distance_matrix = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))
    return distance_matrix


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    t0, p0, t1, p1 = rng.random((N, )), rng.random((N, )), rng.random((N, )), rng.random((N, ))
    return t0, p0, t1, p1


def run_arc_distance(device_type: dace.dtypes.DeviceType):
    '''
    Runs arc-distance for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    N = 100000
    t0, p0, t1, p1 = initialize(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = arc_distance.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        val = sdfg(theta_1=t0, phi_1=p0, theta_2=t1, phi_2=p1, N=N)

    # Compute ground truth and Validate result
    ref = arc_distance.f(t0, p0, t1, p1)
    assert (np.allclose(val, ref) or relerror(val, ref) < 1e-10)
    return sdfg


def test_cpu():
    run_arc_distance(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_arc_distance(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_arc_distance(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_arc_distance(dace.dtypes.DeviceType.GPU)
