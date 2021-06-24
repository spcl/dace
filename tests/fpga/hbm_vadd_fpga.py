# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace import subsets
import dace
import numpy as np

# A test executing vector addition with multidimensional arrays using HBM.


def create_vadd_multibank_sdfg(bank_count_per_array=2,
                               ndim=1,
                               unroll_map_inside=False,
                               sdfgname="vadd_hbm"):
    N = dace.symbol("N")
    M = dace.symbol("M")
    S = dace.symbol("S")

    sdfg = dace.SDFG(sdfgname)
    state = sdfg.add_state('vadd_hbm', True)
    shape = [bank_count_per_array, N]
    accstr = "i"
    inner_map_range = dict()
    inner_map_range["i"] = "0:N"
    if (ndim >= 2):
        shape = [bank_count_per_array, N, M]
        accstr = "i, j"
        inner_map_range["j"] = "0:M"
    if (ndim >= 3):
        shape = [bank_count_per_array, N, M, S]
        accstr = "i, j, t"
        inner_map_range["t"] = "0:S"

    in1 = sdfg.add_array("in1", shape, dace.float32)
    in2 = sdfg.add_array("in2", shape, dace.float32)
    out = sdfg.add_array("out", shape, dace.float32)

    in1[1].location["bank"] = f"hbm.0:{bank_count_per_array}"
    in2[1].location["bank"] = f"hbm.{bank_count_per_array}:{2*bank_count_per_array}"
    out[1].location["bank"] = f"hbm.{2*bank_count_per_array}:{3*bank_count_per_array}"

    read_in1 = state.add_read("in1")
    read_in2 = state.add_read("in2")
    out_write = state.add_write("out")

    tmpin1_memlet = dace.Memlet(f"in1[k, {accstr}]")
    tmpin2_memlet = dace.Memlet(f"in2[k, {accstr}]")
    tmpout_memlet = dace.Memlet(f"out[k, {accstr}]")

    outer_entry, outer_exit = state.add_map("vadd_outer_map",
                                            dict(k=f'0:{bank_count_per_array}'))
    map_entry, map_exit = state.add_map("vadd_inner_map", inner_map_range)
    tasklet = state.add_tasklet("addandwrite", dict(__in1=None, __in2=None),
                                dict(__out=None), '__out = __in1 + __in2')
    outer_entry.map.schedule = dace.ScheduleType.Unrolled

    if (unroll_map_inside):
        state.add_memlet_path(read_in1,
                              map_entry,
                              outer_entry,
                              tasklet,
                              memlet=tmpin1_memlet,
                              dst_conn="__in1")
        state.add_memlet_path(read_in2,
                              map_entry,
                              outer_entry,
                              tasklet,
                              memlet=tmpin2_memlet,
                              dst_conn="__in2")
        state.add_memlet_path(tasklet,
                              outer_exit,
                              map_exit,
                              out_write,
                              memlet=tmpout_memlet,
                              src_conn="__out")
    else:
        state.add_memlet_path(read_in1,
                              outer_entry,
                              map_entry,
                              tasklet,
                              memlet=tmpin1_memlet,
                              dst_conn="__in1")
        state.add_memlet_path(read_in2,
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


def createTestSet(dim, size1D, banks):
    shape = [banks]
    for i in range(dim):
        shape.append(size1D)
    in1 = np.random.rand(*shape)
    in2 = np.random.rand(*shape)
    in1 = in1.astype(np.float32)
    in2 = in2.astype(np.float32)
    expected = in1 + in2
    out = np.empty(shape, dtype=np.float32)
    return (in1, in2, expected, out)


def exec_test(dim, size1D, banks, testname, unroll_map_inside=False):
    in1, in2, expected, target = createTestSet(dim, size1D, banks)
    sdfg = create_vadd_multibank_sdfg(banks, dim, unroll_map_inside, testname)
    if (dim == 1):
        sdfg(in1=in1, in2=in2, out=target, N=size1D)
    elif (dim == 2):
        sdfg(in1=in1, in2=in2, out=target, N=size1D, M=size1D)
    else:
        sdfg(in1=in1, in2=in2, out=target, N=size1D, M=size1D, S=size1D)
    assert np.allclose(expected, target, rtol=1e-6)
    del sdfg


if __name__ == '__main__':
    exec_test(1, 50, 2, "vadd_2b1d")  #2 banks, 1 dimensional
    exec_test(2, 50, 2, "vadd_2b2d")  #2 banks 2 dimensional
    exec_test(3, 10, 2, "vadd_2b3d")  #2 banks 3 dimensional
    exec_test(1, 50, 8, "vadd_8b1d", True)  #8 banks 1d, 1 pipeline
