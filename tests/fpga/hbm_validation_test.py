# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGError, InvalidSDFGNodeError, validate
from dace import subsets as sbs, dtypes, memlet as mem
import dace
import numpy as np
from dace import subsets
from dace.sdfg import nodes as nd

# A test to check the changes to the validation required for the support for HBM
# The three functions will be automatically called by pytest

def assert_validation_failure(sdfg, exceptiontype):
    ok = False
    try:
        sdfg.validate()
    except exceptiontype as msg:
        ok = True
    assert ok

def deepscope_test():
    @dace.program
    def deepscope(input: dace.int32[12, 10], output: dace.int32[12, 10]):
        for k in dace.map[0:10]:
            for j in dace.map[0:2]:
                for z in dace.map[0:2]:
                    with dace.tasklet:
                        _read << input[k + j + z, 0]
                        _write >> output[k + j * z, 0]
                        _write = _read + 1

    sdfg = deepscope.to_sdfg()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nd.MapEntry):
                node.map.schedule = dtypes.ScheduleType.Unrolled
    sdfg.arrays["input"].location["bank"] = "hbm.0:12"
    sdfg.arrays["output"].location["bank"] = "hbm.12:24"
    sdfg.apply_fpga_transformations(validate=False)
    sdfg.validate()


def multitasklet_test():
    @dace.program
    def multitasklet(input: dace.int32[12, 10], output: dace.int32[12, 10]):
        with dace.tasklet:
            m << input[0:2, 4]
            n >> output[0:4, 5]
            m = n

    sdfg = multitasklet.to_sdfg()
    sdfg.validate()
    sdfg.arrays["input"].location["bank"] = "hbm.0:12"
    sdfg.arrays["output"].location["bank"] = "hbm.12:24"
    sdfg.apply_fpga_transformations(validate=False)
    assert_validation_failure(sdfg, InvalidSDFGNodeError)


def unsound_location_test():
    sdfg = dace.SDFG("jdj")
    sdfg.add_array("a", [4, 3], dtypes.int32, dtypes.StorageType.FPGA_Global)
    sdfg.add_array("b", [4], dtypes.int32, dtypes.StorageType.FPGA_Global)
    state = sdfg.add_state("dummy")
    sdfg.validate()
    sdfg.arrays["a"].location["bank"] = ":"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["bank"] = "2:5"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["bank"] = "hbm.2:5"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["bank"] = "hbm.k:5"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.add_constant("k", 1)
    sdfg.arrays["a"].location["bank"] = "hbm.k:5"
    sdfg.validate()
    sdfg.constants_prop.clear()
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["bank"] = "hbm.2:2"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["a"].location["bank"] = "hbm.0:4"
    sdfg.validate()
    sdfg.arrays["b"].location["bank"] = "hbm.0:4"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["b"].location["bank"] = "ddr.abc"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["b"].location["bank"] = "ddr.1"
    sdfg.validate()
    sdfg.arrays["b"].location["bank"] = "wut.32"
    assert_validation_failure(sdfg, InvalidSDFGError)
    sdfg.arrays["b"].location["bank"] = "ddr8"
    assert_validation_failure(sdfg, InvalidSDFGError)

if __name__ == "__main__":
    deepscope_test()
    multitasklet_test()
    unsound_location_test()