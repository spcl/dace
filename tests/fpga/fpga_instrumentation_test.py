# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg.utils import is_fpga_kernel
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace import config
import numpy as np
import re

def make_sdfg(make_tmp_local: bool):
    """
    Creates an SDFG that has a left and a right branch writing into two
    respective temporary arrays, which are both read by subsequent map.
    If `male_tmp_local` is set, the temporary arrays will be made local, such
    that DaCe will generate a single kernel for the state. Otherwise, DaCe
    will generate three separate kernels.
    """

    sdfg = dace.SDFG("instrumentation_test")
    sdfg.add_array("in0", (16, ), dace.float32)
    sdfg.add_array("in1", (16, ), dace.float32)
    sdfg.add_array("in2", (16, ), dace.float32)
    sdfg.add_array("tmp0", (16, ), dace.float32, transient=True)
    sdfg.add_array("tmp1", (16, ), dace.float32, transient=True)
    sdfg.add_array("out0", (16, ), dace.float32)
    sdfg.add_array("out1", (16, ), dace.float32)

    state = sdfg.add_state("instrumentation_test")

    in0 = state.add_read("in0")
    in1 = state.add_read("in1")
    tmp0 = state.add_access("tmp0")
    tmp1 = state.add_access("tmp1")
    out0 = state.add_write("out0")

    # Left branch subgraph
    entry_left, exit_left = state.add_map("left_map", {"i": "0:16"})
    tasklet_left = state.add_tasklet("left_tasklet", {"_in"}, {"_tmp"}, "_tmp = _in + 1")
    state.add_memlet_path(in0, entry_left, tasklet_left, dst_conn="_in", memlet=dace.Memlet("in0[i]"))
    state.add_memlet_path(tasklet_left, exit_left, tmp0, src_conn="_tmp", memlet=dace.Memlet("tmp0[i]"))

    # Right branch subgraph
    entry_right, exit_right = state.add_map("right_map", {"i": "0:16"})
    tasklet_right = state.add_tasklet("right_tasklet", {"_in"}, {"_tmp"}, "_tmp = _in + 1")
    state.add_memlet_path(in1, entry_right, tasklet_right, dst_conn="_in", memlet=dace.Memlet("in1[i]"))
    state.add_memlet_path(tasklet_right, exit_right, tmp1, src_conn="_tmp", memlet=dace.Memlet("tmp1[i]"))

    # Bottom subgraph
    entry_after, exit_after = state.add_map("after_map", {"i": "0:16"})
    tasklet_after = state.add_tasklet("after_tasklet", {"_tmp0", "_tmp1"}, {"_c"}, "_c = 2 * (_tmp0 + _tmp1)")
    state.add_memlet_path(tmp0, entry_after, tasklet_after, dst_conn="_tmp0", memlet=dace.Memlet("tmp0[i]"))
    state.add_memlet_path(tmp1, entry_after, tasklet_after, dst_conn="_tmp1", memlet=dace.Memlet("tmp1[i]"))
    state.add_memlet_path(tasklet_after, exit_after, out0, src_conn="_c", memlet=dace.Memlet("out0[i]"))

    # Extra independent subgraph (will be a PE on Xilinx, kernel on Intel)
    in2 = state.add_read("in2")
    out1 = state.add_write("out1")
    entry_extra, exit_extra = state.add_map("extra_map", {"i": "0:16"})
    tasklet_extra = state.add_tasklet("extra_tasklet", {"_in"}, {"_out"}, "_out = _in * _in")
    state.add_memlet_path(in2, entry_extra, tasklet_extra, dst_conn="_in", memlet=dace.Memlet("in2[i]"))
    state.add_memlet_path(tasklet_extra, exit_extra, out1, src_conn="_out", memlet=dace.Memlet("out1[i]"))

    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1

    if make_tmp_local:
        made_local = 0
        for name, desc in sdfg.arrays.items():
            if "tmp" in name:
                desc.storage = dace.StorageType.FPGA_Local
                made_local += 1
        assert made_local == 2

    for s in sdfg.states():
        if is_fpga_kernel(sdfg, s):
            s.instrument = dace.InstrumentationType.FPGA
            break
    else:
        raise RuntimeError("FPGA state was not found.")

    return sdfg


def run_program(sdfg):

    in0 = np.zeros((16, ), np.float32)
    in1 = np.ones((16, ), np.float32)
    in2 = np.ones((16, ), np.float32)
    out0 = np.empty((16, ), np.float32)
    out1 = np.empty((16, ), np.float32)

    sdfg(in0=in0, in1=in1, in2=in2, out0=out0, out1=out1)

    assert np.allclose(out0, 2 * ((in0 + 1) + (in1 + 1)))
    assert np.allclose(out1, in2 * in2)


@fpga_test()
def test_instrumentation_single():
    sdfg = make_sdfg(True)
    run_program(sdfg)
    report = sdfg.get_latest_report()
    # There should be three runtimes: One for the kernel, and two for the state
    if dace.Config.get("compiler", "fpga", "vendor") == "xilinx":
        # For Xilinx, processing elements live within a single kernel
        expected_num_kernels = 1
    elif dace.Config.get("compiler", "fpga", "vendor") == "intel_fpga":
        # For Intel, each processing element is a distinct kernel
        expected_num_kernels = 2
    assert len(re.findall(r"[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+", str(report))) == 2 + expected_num_kernels
    return sdfg


@fpga_test()
def test_instrumentation_multiple():
    sdfg = make_sdfg(False)
    with config.set_temporary("compiler", "fpga", "concurrent_kernel_detection", value=True):
        run_program(sdfg)
    report = sdfg.get_latest_report()
    # There should be five runtimes: One for each kernel, and two for the state
    assert len(re.findall(r"[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+", str(report))) == 6
    return sdfg


if __name__ == "__main__":
    test_instrumentation_multiple(None)
    test_instrumentation_single(None)
