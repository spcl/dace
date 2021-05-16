import dace
import dace.sdfg.nodes
from IPython.display import Code
import dace.sdfg.hbm_multibank_expansion as expander

import dace.cli.sdfv as sdfv


def create_vadd_sdfg():
    N = dace.symbol("N")

    sdfg = dace.SDFG('vadd_hbm')

    state = sdfg.add_state('vadd_hbm', True)

    in1 = sdfg.add_array("in1", [N], dace.float32)
    in2 = sdfg.add_array("in2", [N], dace.float32)
    out = sdfg.add_array("out", [N], dace.float32)

    in1[1].location["hbmbank"] = "0:1"
    in2[1].location["hbmbank"] = "2:3"
    out[1].location["hbmbank"] = "4:5"
    
    readin1 = state.add_read("in1")
    readin2 = state.add_read("in2")
    outwrite = state.add_write("out")

    tmpin1_memlet = dace.Memlet("in1[k, i]")
    tmpin2_memlet = dace.Memlet("in2[k, i]")
    tmpout_memlet = dace.Memlet("out[k, i]")

    outer_entry, outer_exit = state.add_map("vadd_outer_map", dict(k='0:2'))
    map_entry, map_exit = state.add_map("vadd_inner_map", dict(i='0:(N//2)'))
    tasklet = state.add_tasklet("addandwrite", dict(__in1=None, __in2=None), 
        dict(__out=None), '__out = __in1 + __in2')

    state.add_memlet_path(readin1, outer_entry, map_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
    state.add_memlet_path(readin2, outer_entry, map_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
    state.add_memlet_path(tasklet, map_exit, outer_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")

    sdfg.fill_scope_connectors()
  #  sdfg.apply_fpga_transformations() #modify todo
    return sdfg

def create_vadd_sdfg_scopememlets():
    N = dace.symbol("N")

    sdfg = dace.SDFG('vadd_hbm')

    state = sdfg.add_state('vadd_hbm', True)

    in1 = sdfg.add_array("in1", [N], dace.float32)
    in2 = sdfg.add_array("in2", [N], dace.float32)
    out = sdfg.add_array("out", [N], dace.float32)

    in1[1].location["hbmbank"] = "0:1"
    in2[1].location["hbmbank"] = "2:3"
    out[1].location["hbmbank"] = "4:5"
    
    readin1 = state.add_read("in1")
    readin2 = state.add_read("in2")
    outwrite = state.add_write("out")

    tmpin1_memlet = dace.Memlet("in1[k, i]")
    tmpin2_memlet = dace.Memlet("in2[k, i]")
    tmpout_memlet = dace.Memlet("out[k, i]")

    outer_entry, outer_exit = state.add_map("vadd_outer_map", dict(k='0:2'))
    map_entry, map_exit = state.add_map("vadd_inner_map", dict(i='0:(N//2)'))
    tasklet = state.add_tasklet("addandwrite", dict(__in1=None, __in2=None), 
        dict(__out=None), '__out = __in1 + __in2')

    #The outer map gets the inner map
    state.add_memlet_path(readin1, map_entry, outer_entry, tasklet, memlet=tmpin1_memlet, dst_conn="__in1")
    state.add_memlet_path(readin2, map_entry, outer_entry, tasklet, memlet=tmpin2_memlet, dst_conn="__in2")
    state.add_memlet_path(tasklet, outer_exit, map_exit, outwrite, memlet=tmpout_memlet, src_conn="__out")

    sdfg.fill_scope_connectors()
  #  sdfg.apply_fpga_transformations() #modify todo
    return sdfg

def create_vadd_sdfg_without_hbm():
    N = dace.symbol("N")

    @dace.program
    def vadd(in1 : dace.float32[N], in2 : dace.float32[N], out : dace.float32[N]):
        for i in dace.map[0:N]:
            out[i] = in1[i] + in2[i]
    
    sdfg = vadd.to_sdfg()
    sdfg.apply_fpga_transformations()
    sdfg.fill_scope_connectors()

    return sdfg
    

if __name__ == '__main__':
    #sdfg = create_vadd_sdfg_without_hbm()
    #sdfg = create_vadd_sdfg()
    sdfg = create_vadd_sdfg_scopememlets()
    expander.expand_hbm_multiarrays(sdfg)
    sdfv.view(sdfg)
    #code = Code(sdfg.generate_code()[2].code, language='cpp')
  #  print(code)