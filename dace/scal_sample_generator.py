import dace
import dace.sdfg.nodes
from IPython.display import Code
import dace.sdfg.hbm_multibank_expansion as expander
from dace import subsets
import dace.libraries.blas as blas

def create_scal_multibank(banks):
    N = dace.symbol("N")

    sdfg = dace.SDFG('vadd_hbm')
    state = sdfg.add_state('vadd_hbm', True)

    in1 = sdfg.add_array("in1", [N], dace.float32)
    out = sdfg.add_array("out", [N], dace.float32)
    in1[1].location["hbmbank"] = subsets.Range.from_string(f"0:{banks}")
    out[1].location["hbmbank"] = subsets.Range.from_string(f"{2*banks}:{3*banks}")
    in1[1].location["hbmalignment"] = 'even'
    out[1].location["hbmalignment"] = 'even'
    alpha = sdfg.add_array("alpha", [1], dace.float32)

    readin1 = state.add_read("in1")
    outwrite = state.add_write("out")
    readalpha = state.add_read("alpha")

    tmpin1_memlet = dace.Memlet(f"in1[k, i]")
    tmpout_memlet = dace.Memlet(f"out[k, i]")
    alpha_memlet = dace.Memlet(f"alpha[0]")

    outer_entry, outer_exit = state.add_map("scal_outer_map", dict(k=f'0:{banks}'))
    map_entry, map_exit = state.add_map("scal_inner_map", dict(i=f"0:(N//{banks})"))
    tasklet = state.add_tasklet("mulandwrite", dict(__in1=None, alpha_in=None), 
        dict(__out=None), '__out = __in1 * alpha_in')
    outer_entry.map.unroll = True

    state.add_memlet_path(readin1, outer_entry, map_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
    state.add_memlet_path(tasklet, map_exit, outer_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")
    state.add_memlet_path(readalpha, outer_entry, map_entry, tasklet, memlet=alpha_memlet, dst_conn="alpha_in")

    sdfg.apply_fpga_transformations()
    return sdfg

sdfg = create_scal_multibank(3)
expander.expand_hbm_multiarrays(sdfg)
sdfg.view()