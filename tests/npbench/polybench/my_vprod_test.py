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


#Data set sizes
#N
sizes = {
    "mini": 42,
    "small": 124,
    "medium": 410,
    "large": 2100,
    "extra-large": 2200
}

#dace sympols
N = dc.symbol("N", dtype=dc.int64)


@dc.program
def vprod_kernel(x: dc.float64[N], y: dc.float64[N]):
    return x @ y


def initialize(N, datatype=np.float64):
    fn = datatype(N)
    x = np.empty(N, dtype=datatype)
    y = np.empty(N, dtype=datatype)

    for i in range (N):
        x[i] = 1 + (i / fn)
        y[i] = (4 * i) / fn

    return x, y


def run_vprod(device_type: dace.dtypes.DeviceType):
    '''
    Runs vprod for the given device
    :return: the SDFG
    '''

    #initialize data
    N = sizes["mini"]
    x, y = initialize(N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = vprod_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        z = sdfg(x, y, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = vprod_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # --- Copied from bicg_test.py ---
        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()

       # sm_applied = sdfg.apply_transformations_repeated([InlineSDFG, StreamingMemory],
       #                                                  [{}, {
       #                                                      'storage': dace.StorageType.FPGA_Local
       #                                                  }],
       #                                                  print_report=True)
       # assert sm_applied == 5  # 3 inlines and 3 Streaming memories  TODO: check this, changed from 8 to 5

        ###########################
        # FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # specialize the SDFG (needed by the GEMV expansion)
        sdfg.specialize(dict(N=N))
        print("finished optimizing sdfg")
        z = sdfg(x, y)
        print("executed sdfg")

    # Compute ground truth and Validate result
    z_ref = vprod_kernel.f(x, y)
    assert np.allclose(z, z_ref)
    print("assertion finished")
    return sdfg

def test_cpu():
    run_vprod(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_vprod(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_vprod(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='fpga', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_vprod(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_vprod(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_vprod(dace.dtypes.DeviceType.FPGA)
