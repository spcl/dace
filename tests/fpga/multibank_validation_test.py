# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGError, InvalidSDFGNodeError, validate
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace import subsets
from dace.sdfg import nodes as nd

# A test to check the changes to the validation required for the support for HBM and DDR


def assert_validation_failure(sdfg, exceptiontype):
    ok = False
    try:
        sdfg.validate()
    except exceptiontype as msg:
        ok = True
    assert ok


def multibank_deep_scope(mem_type):
    @dace.program
    def deep_scope(input: dace.int32[12, 10], output: dace.int32[12, 10]):
        for k in dace.map[0:10]:
            for j in dace.map[0:2]:
                for z in dace.map[0:2]:
                    with dace.tasklet:
                        _read << input[k + j + z, 0]
                        _write >> output[k + j * z, 0]
                        _write = _read + 1

    sdfg = deep_scope.to_sdfg()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nd.MapEntry):
                node.map.schedule = dtypes.ScheduleType.Unrolled
    sdfg.arrays["input"].location["memorytype"] = mem_type
    sdfg.arrays["output"].location["memorytype"] = mem_type
    sdfg.arrays["input"].location["bank"] = "0:12"
    sdfg.arrays["output"].location["bank"] = "12:24"
    sdfg.apply_fpga_transformations(validate=False)
    sdfg.validate()


def multibank_multi_tasklet(mem_type):
    @dace.program
    def multi_tasklet(input: dace.int32[12, 10], output: dace.int32[12, 10]):
        with dace.tasklet:
            m << input[0:2, 4]
            n >> output[0:4, 5]
            n = m

    sdfg = multi_tasklet.to_sdfg()
    sdfg.validate()
    sdfg.arrays["input"].location["memorytype"] = mem_type
    sdfg.arrays["output"].location["memorytype"] = mem_type
    sdfg.arrays["input"].location["bank"] = "0:12"
    sdfg.arrays["output"].location["bank"] = "12:24"
    sdfg.apply_fpga_transformations(validate=False)
    assert_validation_failure(sdfg, InvalidSDFGNodeError)

    @dace.program
    def singletasklet(input: dace.int32[2, 10], output: dace.int32[2, 10]):
        with dace.tasklet:
            m << input[0, 0:10]
            n >> output[1, 0:10]
            n = m

    sdfg = singletasklet.to_sdfg()
    sdfg.arrays["input"].location["memorytype"] = mem_type
    sdfg.arrays["output"].location["memorytype"] = mem_type
    sdfg.arrays["input"].location["bank"] = "0:2"
    sdfg.arrays["output"].location["bank"] = "2:4"
    sdfg.apply_fpga_transformations()
    sdfg.validate()


def multibank_unsound_location(mem_type_1, mem_type_2):
    sdfg = dace.SDFG("jdj")
    sdfg.add_array("a", [4, 3], dtypes.int32, dtypes.StorageType.FPGA_Global)
    sdfg.add_array("b", [4], dtypes.int32, dtypes.StorageType.FPGA_Global)
    state = sdfg.add_state("dummy")
    sdfg.validate()
    sdfg.arrays["a"].location["memorytype"] = ":"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["memorytype"] = mem_type_1
    sdfg.arrays["a"].location["bank"] = "2:5"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.add_constant("k", 1)
    sdfg.arrays["a"].location["memorytype"] = mem_type_1
    sdfg.arrays["a"].location["bank"] = "k:5"
    sdfg.validate()
    sdfg.constants_prop.clear()
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["memorytype"] = mem_type_1
    sdfg.arrays["a"].location["bank"] = "2:2"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["memorytype"] = mem_type_1
    sdfg.arrays["a"].location["bank"] = "0:4"
    sdfg.validate()
    sdfg.arrays["b"].location["memorytype"] = mem_type_1
    sdfg.arrays["b"].location["bank"] = "0:4"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["b"].location["memorytype"] = mem_type_2
    sdfg.arrays["b"].location["bank"] = "abc"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["b"].location["memorytype"] = mem_type_2
    sdfg.arrays["b"].location["bank"] = "1"
    sdfg.validate()
    sdfg.arrays["b"].location["memorytype"] = mem_type_1
    sdfg.arrays["b"].location["bank"] = "4"
    sdfg.validate()


def test_multibank_deep_scope_hbm():
    multibank_deep_scope("hbm")


def test_multibank_deep_scope_ddr():
    multibank_deep_scope("ddr")


def test_multibank_multi_tasklet_hbm():
    multibank_multi_tasklet("hbm")


def test_multibank_multi_tasklet_ddr():
    multibank_multi_tasklet("ddr")


def test_multibank_unsound_location_hmb2ddr():
    multibank_unsound_location("hbm", "ddr")


def test_multibank_unsound_location():
    multibank_unsound_location("ddr", "hbm")


if __name__ == "__main__":
    test_multibank_deep_scope_hbm()
    test_multibank_deep_scope_ddr()
    test_multibank_multi_tasklet_hbm()
    test_multibank_multi_tasklet_ddr()
    test_multibank_unsound_location_hmb2ddr()
    test_multibank_unsound_location()
