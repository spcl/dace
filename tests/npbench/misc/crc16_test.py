# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
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

N = dace.symbol('N')
poly: dace.uint16 = 0x8408


# Adapted from https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6
@dace.program
def crc16_kernel(data: dace.uint8[N]):
    '''
    CRC-16-CCITT Algorithm
    '''
    crc: dace.uint16 = 0xFFFF
    for i in range(N):
        b = data[i]
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    data = rng.integers(0, 256, size=(N, ), dtype=np.uint8)
    return data


def ground_truth(data, poly=0x8408):
    '''
    CRC-16-CCITT Algorithm
    '''
    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


def run_crc16(device_type: dace.dtypes.DeviceType):
    '''
    Runs CRC16 for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    N = 1600
    data = initialize(N)
    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = crc16_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        out = sdfg(data, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = crc16_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        out = sdfg(data)

    # Compute ground truth and validate
    out_ref = ground_truth(data)
    assert np.allclose(out, out_ref)
    return sdfg


def test_cpu():
    run_crc16(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_crc16(dace.dtypes.DeviceType.GPU)


@pytest.mark.skip(reason="Operand type in binary expressions")
@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_crc16(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_crc16(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_crc16(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_crc16(dace.dtypes.DeviceType.FPGA)
