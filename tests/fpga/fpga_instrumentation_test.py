# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.codegen.targets.fpga import is_fpga_kernel
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
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
    sdfg.add_array("a", (16, ), dace.float32)
    sdfg.add_array("b", (16, ), dace.float32)
    sdfg.add_array("tmp0", (16, ), dace.float32, transient=True)
    sdfg.add_array("tmp1", (16, ), dace.float32, transient=True)
    sdfg.add_array("c", (16, ), dace.float32)

    state = sdfg.add_state("instrumentation_test")

    a = state.add_read("a")
    b = state.add_read("b")
    tmp0 = state.add_access("tmp0")
    tmp1 = state.add_access("tmp1")
    c = state.add_write("c")

    entry_left, exit_left = state.add_map("left_map", {"i": "0:16"})
    tasklet_left = state.add_tasklet("left_tasklet", {"_a"}, {"_tmp"},
                                     "_tmp = _a + 1")
    state.add_memlet_path(a,
                          entry_left,
                          tasklet_left,
                          dst_conn="_a",
                          memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(tasklet_left,
                          exit_left,
                          tmp0,
                          src_conn="_tmp",
                          memlet=dace.Memlet("tmp0[i]"))

    entry_right, exit_right = state.add_map("right_map", {"i": "0:16"})
    tasklet_right = state.add_tasklet("right_tasklet", {"_b"}, {"_tmp"},
                                      "_tmp = _b + 1")
    state.add_memlet_path(b,
                          entry_right,
                          tasklet_right,
                          dst_conn="_b",
                          memlet=dace.Memlet("b[i]"))
    state.add_memlet_path(tasklet_right,
                          exit_right,
                          tmp1,
                          src_conn="_tmp",
                          memlet=dace.Memlet("tmp1[i]"))

    entry_after, exit_after = state.add_map("after_map", {"i": "0:16"})
    tasklet_after = state.add_tasklet("after_tasklet", {"_tmp0", "_tmp1"},
                                      {"_c"}, "_c = 2 * (_tmp0 + _tmp1)")
    state.add_memlet_path(tmp0,
                          entry_after,
                          tasklet_after,
                          dst_conn="_tmp0",
                          memlet=dace.Memlet("tmp0[i]"))
    state.add_memlet_path(tmp1,
                          entry_after,
                          tasklet_after,
                          dst_conn="_tmp1",
                          memlet=dace.Memlet("tmp1[i]"))
    state.add_memlet_path(tasklet_after,
                          exit_after,
                          c,
                          src_conn="_c",
                          memlet=dace.Memlet("c[i]"))

    assert sdfg.apply_transformations(FPGATransformSDFG) == 1
    assert sdfg.apply_transformations(InlineSDFG) == 1

    made_local = 0
    if make_tmp_local:
        for name, desc in sdfg.arrays.items():
            if "tmp" in name:
                desc.storage = dace.StorageType.FPGA_Local
                made_local += 1
    assert not make_tmp_local or made_local == 2

    for s in sdfg.states():
        if is_fpga_kernel(sdfg, s):
            s.instrument = dace.InstrumentationType.FPGA
            break
    else:
        raise RuntimeError("FPGA state was not found.")

    return sdfg


def run_program(sdfg):

    a = np.zeros((16, ), np.float32)
    b = np.ones((16, ), np.float32)
    c = np.empty((16, ), np.float32)

    sdfg(a=a, b=b, c=c)

    assert all(c == 2 * ((a + 1) + (b + 1)))


@fpga_test()
def test_instrumentation_single():
    sdfg = make_sdfg(True)
    run_program(sdfg)
    report = sdfg.get_latest_report()
    # There should be three runtimes: One for the kernel, and two for the state
    assert len(
        re.findall(r"[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+",
                   str(report))) == 3
    return sdfg


@fpga_test()
def test_instrumentation_multiple():
    sdfg = make_sdfg(False)
    run_program(sdfg)
    report = sdfg.get_latest_report()
    # There should be five runtimes: One for each kernel, and two for the state
    assert len(
        re.findall(r"[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+",
                   str(report))) == 5
    return sdfg


if __name__ == "__main__":
    test_instrumentation_multiple(None)
    test_instrumentation_single(None)
