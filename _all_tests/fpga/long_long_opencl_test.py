# Simple test case to check that for the intel fpga the openCL type long gets used instead of C type long long

import dace.dtypes
import numpy as np
import dace as dc
import argparse
from dace.fpga_testing import intel_fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def simple_add_kernel(A: dc.int64[N], B: dc.int64[N]):
    B += A


def initialize(N, datatype=np.int64):
    A = np.fromfunction(lambda i: (i + 372036854775807), (N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3372036854775807), (N, ), dtype=datatype)
    return A, B


def ground_truth(A, B):
    B += A


def run_simple_add(device_type: dace.dtypes.DeviceType):
    '''
    Runs simple add for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = 120
    A, B = initialize(N)
    A_ref = np.copy(A)
    B_ref = np.copy(B)

    # Parse SDFG and apply FPGA friendly optimization
    sdfg = simple_add_kernel.to_sdfg(simplify=True)
    applied = sdfg.apply_transformations([FPGATransformSDFG])
    assert applied == 1

    sdfg.specialize(dict(N=N))
    sdfg(A=A, B=B)

    # Compute ground truth and validate
    ground_truth(A_ref, B_ref)
    assert np.allclose(B, B_ref)
    return sdfg


@intel_fpga_test(assert_ii_1=False)
def test_fpga():
    return run_simple_add(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "fpga":
        run_simple_add(dace.dtypes.DeviceType.FPGA)
