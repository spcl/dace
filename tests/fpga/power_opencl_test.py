# Simple test case to check how ** is translated to openCL

import dace.dtypes
import numpy as np
import dace as dc
import argparse
from dace.fpga_testing import intel_fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def power_test_kernel(A: dc.float64[N], B: dc.float64[N], C: dc.int32[N]):
    for i in range(N):
        B[i] = B[i]**A[i]
        A[i] = A[i]**C[i]
        A[i] = A[i]**2.3
        B[i] = B[i]**3


def initialize(N, datatype=np.float64):

    A = np.full((N, ), 1.2, dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3.1), (N, ), dtype=datatype)
    C = np.full((N, ), 2, dtype=np.int32)
    return A, B, C


def ground_truth(A, B, C):
    for i in range(120):
        B[i] = B[i]**A[i]
        A[i] = A[i]**C[i]
        A[i] = A[i]**2.3
        B[i] = B[i]**3


def run_power_test(device_type: dace.dtypes.DeviceType):
    '''
    Runs simple power calculations for Intel FPGAS
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = 120
    A, B, C = initialize(N)
    A_ref = np.copy(A)
    B_ref = np.copy(B)
    C_ref = np.copy(C)

    # Parse SDFG and apply FPGA friendly optimization
    sdfg = power_test_kernel.to_sdfg(simplify=True)
    applied = sdfg.apply_transformations([FPGATransformSDFG])
    assert applied == 1
    sdfg.specialize(dict(N=N))
    sdfg(A=A, B=B, C=C)

    # Compute ground truth and validate
    ground_truth(A_ref, B_ref, C_ref)

    assert np.allclose(B, B_ref)
    assert np.allclose(A, A_ref)
    return sdfg


@intel_fpga_test(assert_ii_1=False)
def test_fpga():
    return run_power_test(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "fpga":
        run_power_test(dace.dtypes.DeviceType.FPGA)
