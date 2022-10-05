# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

N, bins, npt = (dace.symbol(s, dtype=dace.int64) for s in ('N', 'bins', 'npt'))


def relerror(val, ref):
    if np.linalg.norm(ref) == 0:
        return np.linalg.norm(val - ref)
    return np.linalg.norm(val - ref) / np.linalg.norm(ref)


@dace.program
def get_bin_edges(a: dace.float64[N], bin_edges: dace.float64[bins + 1]):
    a_min = np.amin(a)
    a_max = np.amax(a)
    delta = (a_max - a_min) / bins
    for i in dace.map[0:bins]:
        bin_edges[i] = a_min + i * delta

    bin_edges[bins] = a_max  # Avoid roundoff error on last point


@dace.program
def compute_bin(x: dace.float64, bin_edges: dace.float64[bins + 1]):
    # assuming uniform bins for now
    a_min = bin_edges[0]
    a_max = bin_edges[bins]
    return dace.int64(bins * (x - a_min) / (a_max - a_min))


@dace.program
def histogram(a: dace.float64[N], bin_edges: dace.float64[bins + 1]):
    hist = np.ndarray((bins, ), dtype=np.int64)
    hist[:] = 0
    get_bin_edges(a, bin_edges)

    for i in dace.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += 1

    return hist


@dace.program
def histogram_weights(a: dace.float64[N], bin_edges: dace.float64[bins + 1], weights: dace.float64[N]):
    hist = np.ndarray((bins, ), dtype=weights.dtype)
    hist[:] = 0
    get_bin_edges(a, bin_edges)

    for i in dace.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += weights[i]

    return hist


@dace.program
def dace_azimint_hist(data: dace.float64[N], radius: dace.float64[N]):
    bin_edges_u = np.ndarray((npt + 1, ), dtype=np.float64)
    histu = histogram(radius, bin_edges_u)
    bin_edges_w = np.ndarray((npt + 1, ), dtype=np.float64)
    histw = histogram_weights(radius, bin_edges_w, data)
    return histw / histu


def numpy_azimint_hist(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    histw = np.histogram(radius, npt, weights=data)[0]
    return histw / histu


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    data, radius = rng.random((N, )), rng.random((N, ))
    return data, radius


def run_azimint_hist(device_type: dace.dtypes.DeviceType):
    '''
    Runs azimint-hist for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench S size)
    N, npt = (400000, 1000)
    data, radius = initialize(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply autopot
        sdfg = dace_azimint_hist.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        val = sdfg(data=data, radius=radius, N=N, npt=npt)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = dace_azimint_hist.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N, npt=npt))
        val = sdfg(data=data, radius=radius)

    # Compute ground truth and Validate result
    ref = numpy_azimint_hist(data, radius, npt)
    # NOTE: On GPU there are very small errors on the boundaries of the bins which propagate to larger errors.
    err = 1e-10
    if device_type is dace.dtypes.DeviceType.GPU:
        err = 1e-3
    assert (np.allclose(val, ref) or relerror(val, ref) < err)
    return sdfg


def test_cpu():
    run_azimint_hist(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_azimint_hist(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="FPGA Transform error")
@fpga_test(assert_ii_1=False)
def test_fpga():
    run_azimint_hist(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_azimint_hist(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_azimint_hist(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_azimint_hist(dace.dtypes.DeviceType.FPGA)
