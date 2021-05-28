import numpy
import dace
import dace.sdfg.nodes
from IPython.display import Code
import dace.sdfg.hbm_multibank_expansion as expander
from dace import subsets
import dace.cli.sdfv as sdfv

def create_vadd_multibank_sdfg(ndim = 1, nCutPerDim = 2, unroll_map_inside = False):
    N = dace.symbol("N")
    M = dace.symbol("M")
    S = dace.symbol("S")

    sdfg = dace.SDFG('vadd_hbm')
    state = sdfg.add_state('vadd_hbm', True)
    shape = [N]
    accstr = "i"
    innermaprange = dict()
    innermaprange["i"] = f"0:N//{nCutPerDim}"
    bankcountPerArray = nCutPerDim ** ndim
    if(ndim >= 2):
        shape = [N, M]
        accstr = "i, j"
        innermaprange["j"] = f"0:M//{nCutPerDim}"
    if(ndim >= 3):
        shape = [N, M, S]
        accstr = "i, j, t"
        innermaprange["t"] = f"0:S//{nCutPerDim}"

    in1 = sdfg.add_array("in1", shape, dace.float32)
    in2 = sdfg.add_array("in2", shape, dace.float32)
    out = sdfg.add_array("out", shape, dace.float32)

    in1[1].location["hbmbank"] = subsets.Range.from_string(f"0:{bankcountPerArray}")
    in2[1].location["hbmbank"] = subsets.Range.from_string(f"{bankcountPerArray}:{2*bankcountPerArray}")
    out[1].location["hbmbank"] = subsets.Range.from_string(f"{2*bankcountPerArray}:{3*bankcountPerArray}")
    in1[1].location["hbmalignment"] = 'even'
    in2[1].location["hbmalignment"] = 'even'
    out[1].location["hbmalignment"] = 'even'
    
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
    outer_entry.map.unroll = True

    if(unroll_map_inside):
        state.add_memlet_path(readin1, map_entry, outer_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
        state.add_memlet_path(readin2, map_entry, outer_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
        state.add_memlet_path(tasklet, outer_exit, map_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")
    else:
        state.add_memlet_path(readin1, outer_entry, map_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
        state.add_memlet_path(readin2, outer_entry, map_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
        state.add_memlet_path(tasklet, map_exit, outer_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")

    sdfg.fill_scope_connectors()
    sdfg.apply_fpga_transformations()
    return sdfg

if __name__ == '__main__':
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = create_vadd_multibank_sdfg(3, 2)
    expander.expand_hbm_multiarrays(sdfg)
    #sdfg.validate()
    #sdfg.view()
    code = Code(sdfg.generate_code()[0].code, language='cpp')
    print(code)