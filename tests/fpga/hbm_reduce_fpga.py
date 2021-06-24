# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace import subsets
import dace
import numpy as np

# A test checking wcr-reduction with HBM arrays as inputs and output


def create_hbm_reduce_sdfg(banks=2, name="red_hbm"):
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = dace.SDFG(name)
    state = sdfg.add_state('red_hbm', True)

    in1 = sdfg.add_array("in1", [banks, N, M], dace.float32)
    in2 = sdfg.add_array("in2", [banks, N, M], dace.float32)
    out = sdfg.add_array("out", [banks, N], dace.float32)
    in1[1].location["bank"] = f"hbm.0:{banks}"
    in2[1].location["bank"] = f"hbm.{banks}:{2*banks}"
    out[1].location["bank"] = f"hbm.{2*banks}:{3*banks}"

    readin1 = state.add_read("in1")
    readin2 = state.add_read("in2")
    out_write = state.add_write("out")
    tmpin1_memlet = dace.Memlet(f"in1[k, i, j]")
    tmpin2_memlet = dace.Memlet(f"in2[k, i, j]")
    tmpout_memlet = dace.Memlet(f"out[k, i]", wcr="lambda x,y: x+y")

    outer_entry, outer_exit = state.add_map("vadd_outer_map",
                                            dict(k=f'0:{banks}'))
    map_entry, map_exit = state.add_map("vadd_inner_map", dict(i="0:N",
                                                               j="0:M"))
    tasklet = state.add_tasklet("mul", dict(__in1=None, __in2=None),
                                dict(__out=None), '__out = __in1 * __in2')
    outer_entry.map.schedule = dace.ScheduleType.Unrolled

    state.add_memlet_path(readin1,
                          outer_entry,
                          map_entry,
                          tasklet,
                          memlet=tmpin1_memlet,
                          dst_conn="__in1")
    state.add_memlet_path(readin2,
                          outer_entry,
                          map_entry,
                          tasklet,
                          memlet=tmpin2_memlet,
                          dst_conn="__in2")
    state.add_memlet_path(tasklet,
                          map_exit,
                          outer_exit,
                          out_write,
                          memlet=tmpout_memlet,
                          src_conn="__out")

    sdfg.apply_fpga_transformations()
    return sdfg


def createTestSet(N, M, banks):
    in1 = np.random.rand(*[banks, N, M]).astype('f')
    in2 = np.random.rand(*[banks, N, M]).astype('f')
    expected = np.sum(in1 * in2, axis=2, dtype=np.float32)
    out = np.zeros((banks, N), dtype=np.float32)
    return (in1, in2, expected, out)


def exec_test(N, M, banks, name):
    in1, in2, expected, target = createTestSet(N, M, banks)
    sdfg = create_hbm_reduce_sdfg(banks, name)
    sdfg(in1=in1, in2=in2, out=target, N=N, M=M)
    assert np.allclose(expected, target, rtol=1e-6)
    del sdfg


if __name__ == '__main__':
    exec_test(2, 3, 2, "red_2x3_2b")
    exec_test(10, 50, 4, "red_10x50_4b")
    exec_test(1, 50, 1, "red_1x50_1b")
    exec_test(1, 40, 8, "red_1x40_8b")
    exec_test(2, 40, 6, "red_2x40_6b")
