"""
A test executing vector addition with multidimensional arrays using HBM.
"""

from dace import subsets
import dace
import numpy as np

def create_vadd_multibank_sdfg(bankcountPerArray = 2, ndim = 1, unroll_map_inside = False, sdfgname="vadd_hbm"):
    N = dace.symbol("N")
    M = dace.symbol("M")
    S = dace.symbol("S")

    sdfg = dace.SDFG(sdfgname)
    state = sdfg.add_state('vadd_hbm', True)
    shape = [bankcountPerArray, N]
    accstr = "i"
    innermaprange = dict()
    innermaprange["i"] = "0:N"
    if(ndim >= 2):
        shape = [bankcountPerArray, N, M]
        accstr = "i, j"
        innermaprange["j"] = "0:M"
    if(ndim >= 3):
        shape = [bankcountPerArray, N, M, S]
        accstr = "i, j, t"
        innermaprange["t"] = "0:S"

    in1 = sdfg.add_array("in1", shape, dace.float32)
    in2 = sdfg.add_array("in2", shape, dace.float32)
    out = sdfg.add_array("out", shape, dace.float32)

    in1[1].location["hbmbank"] = subsets.Range.from_string(f"0:{bankcountPerArray}")
    in2[1].location["hbmbank"] = subsets.Range.from_string(f"{bankcountPerArray}:{2*bankcountPerArray}")
    out[1].location["hbmbank"] = subsets.Range.from_string(f"{2*bankcountPerArray}:{3*bankcountPerArray}")
    
    readin1 = state.add_read("in1")
    readin2 = state.add_read("in2")
    outwrite = state.add_write("out")

    tmpin1_memlet = dace.Memlet(f"in1[k, {accstr}]")
    tmpin2_memlet = dace.Memlet(f"in2[k, {accstr}]")
    tmpout_memlet = dace.Memlet(f"out[k, {accstr}]")

    outer_entry, outer_exit = state.add_map("vadd_outer_map", dict(k=f'0:{bankcountPerArray}'))
    map_entry, map_exit = state.add_map("vadd_inner_map", innermaprange)
    tasklet = state.add_tasklet("addandwrite", dict(__in1=None, __in2=None), 
        dict(__out=None), '__out = __in1 + __in2')
    outer_entry.map.schedule = dace.ScheduleType.Unrolled

    if(unroll_map_inside):
        state.add_memlet_path(readin1, map_entry, outer_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
        state.add_memlet_path(readin2, map_entry, outer_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
        state.add_memlet_path(tasklet, outer_exit, map_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")
    else:
        state.add_memlet_path(readin1, outer_entry, map_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
        state.add_memlet_path(readin2, outer_entry, map_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
        state.add_memlet_path(tasklet, map_exit, outer_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")

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
    #in1 = np.ones(shape, dtype=np.float32)
    #in2 = np.ones(shape, dtype=np.float32)
    expected = in1 + in2
    out = np.empty(shape, dtype=np.float32)
    return (in1, in2, expected, out)

def exec_test(dim, size1D, banks, testname, unroll_map_inside=False):
    in1, in2, expected, target = createTestSet(dim, size1D, banks)
    sdfg = create_vadd_multibank_sdfg(banks, dim, unroll_map_inside, testname)
    if(dim == 1):
        sdfg(in1=in1, in2=in2, out=target, N=size1D)
    elif(dim==2):
        sdfg(in1=in1, in2=in2, out=target, N=size1D, M=size1D)
    else:
        sdfg(in1=in1, in2=in2, out=target, N=size1D, M=size1D, S=size1D)
    assert np.allclose(expected, target, rtol=1e-6)
    del sdfg

if __name__ == '__main__':
    exec_test(1, 50, 2, "vadd_2b1d") #2 banks, 1 dimensional
    exec_test(2, 50, 2, "vadd_2b2d") #2 banks 2 dimensional
    exec_test(3, 10, 2, "vadd_2b3d") #2 banks 3 dimensional
    exec_test(1, 50, 8, "vadd_8b1d", True) #8 banks 1d, 1 pipeline
