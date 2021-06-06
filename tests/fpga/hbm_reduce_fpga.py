"""
A test checking wcr with HBM arrays as inputs and output
"""

from dace import subsets
import dace
import numpy as np

def create_hbm_reduce_sdfg(banks=2):
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = dace.SDFG('red_hbm')
    state = sdfg.add_state('red_hbm', True)

    in1 = sdfg.add_array("in1", [banks, N, M], dace.float32)
    in2 = sdfg.add_array("in2", [banks, N, M], dace.float32)
    out = sdfg.add_array("out", [banks, N], dace.float32)
    in1[1].location["hbmbank"] = subsets.Range.from_string(f"0:{banks}")
    in2[1].location["hbmbank"] = subsets.Range.from_string(f"{banks}:{2*banks}")
    out[1].location["hbmbank"] = subsets.Range.from_string(f"{2*banks}:{3*banks}")
    
    readin1 = state.add_read("in1")
    readin2 = state.add_read("in2")
    outwrite = state.add_write("out")
    tmpin1_memlet = dace.Memlet(f"in1[k, i, j]")
    tmpin2_memlet = dace.Memlet(f"in2[k, i, j]")
    tmpout_memlet = dace.Memlet(f"out[k, i]", wcr="lambda x,y: x+y")

    outer_entry, outer_exit = state.add_map("vadd_outer_map", dict(k=f'0:{banks}'))
    map_entry, map_exit = state.add_map("vadd_inner_map", dict(i=str(N), j=str(M)))
    tasklet = state.add_tasklet("mul", dict(__in1=None, __in2=None), 
        dict(__out=None), '__out = __in1 * __in2')
    outer_entry.map.schedule = dace.ScheduleType.Unrolled

    state.add_memlet_path(readin1, outer_entry, map_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
    state.add_memlet_path(readin2, outer_entry, map_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
    state.add_memlet_path(tasklet, map_exit, outer_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")

    sdfg.fill_scope_connectors()
    sdfg.apply_fpga_transformations(validate=False)
    return sdfg

def createTestSet(N, M):
    in1 = np.random.rand([N, M])
    in2 = np.random.rand([N, M])
    expected = np.sum(in1 * in2, axis=1)
    out = np.empty([N])
    return (in1, in2, expected, out)

if __name__ == '__main__':
    N = dace.symbol("N")
    M = dace.symbol("M")
    in1, in2, expected, target = createTestSet(50, 1000)
    sdfg = create_hbm_reduce_sdfg(2)
    sdfg(in1=in1, in2=in2, out=target)
    assert np.allclose(expected, target, rtol=1e-6)
    del sdfg