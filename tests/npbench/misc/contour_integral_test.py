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


NR, NM, slab_per_bc = (dace.symbol(s, dtype=dace.int64) for s in ('NR', 'NM', 'slab_per_bc'))


@dace.program
def dace_contour_integral(Ham: dace.complex128[slab_per_bc + 1, NR, NR], int_pts: dace.complex128[32],
                          Y: dace.complex128[NR, NM]):
    P0 = np.zeros((NR, NM), dtype=np.complex128)
    P1 = np.zeros((NR, NM), dtype=np.complex128)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), dtype=np.complex128)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        X = np.linalg.solve(Tz, Y)
        if np.absolute(z) < 1.0:
            X[:] = -X
        P0 += X
        P1 += z * X

    return P0, P1


def numpy_contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y):
    P0 = np.zeros((NR, NM), dtype=np.complex128)
    P1 = np.zeros((NR, NM), dtype=np.complex128)
    for z in int_pts:
        Tz = np.zeros((NR, NR), dtype=np.complex128)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        if NR == NM:
            X = np.linalg.inv(Tz)
        else:
            X = np.linalg.solve(Tz, Y)
        if abs(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X

    return P0, P1


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(NR, NM, slab_per_bc, num_int_pts):
    from numpy.random import default_rng
    rng = default_rng(42)
    Ham = rng_complex((slab_per_bc + 1, NR, NR), rng)
    int_pts = rng_complex((num_int_pts, ), rng)
    Y = rng_complex((NR, NM), rng)
    return Ham, int_pts, Y


def run_contour_integral(device_type: dace.dtypes.DeviceType):
    '''
    Runs contour-integral for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    NR, NM, slab_per_bc, num_int_pts = (50, 150, 2, 32)
    Ham, int_pts, Y = initialize(NR, NM, slab_per_bc, num_int_pts)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = dace_contour_integral.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        val0, val1 = sdfg(Ham=Ham, int_pts=int_pts, Y=Y, NR=NR, NM=NM, slab_per_bc=slab_per_bc)

    # Compute ground truth and Validate result
    ref0, ref1 = numpy_contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y)
    assert (np.allclose(val0, ref0) or relerror(val0, ref0) < 1e-10)
    assert (np.allclose(val1, ref1) or relerror(val1, ref1) < 1e-10)
    return sdfg


def test_cpu():
    run_contour_integral(dace.dtypes.DeviceType.CPU)


# NOTE: Doesn't work yet with GPU-auto-optimize
# @pytest.mark.gpu
@pytest.mark.skip
def test_gpu():
    run_contour_integral(dace.dtypes.DeviceType.GPU)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_contour_integral(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_contour_integral(dace.dtypes.DeviceType.GPU)
