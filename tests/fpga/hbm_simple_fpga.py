import numpy
import dace
import dace.sdfg.nodes
from IPython.display import Code
from dace import subsets
import dace.libraries.blas as blas
import dace.sdfg.hbm_helper

def create_vadd_multibank_sdfg(bankcountPerArray = 2, ndim = 1, unroll_map_inside = False):
    N = dace.symbol("N")
    M = dace.symbol("M")
    S = dace.symbol("S")

    sdfg = dace.SDFG('vadd_hbm')
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

    sdfg.fill_scope_connectors()
    sdfg.apply_fpga_transformations(validate=False)
    return sdfg

def bug():
    N = dace.symbol("N")
    M = dace.symbol("M")
    oldlib = dace.SDFG("gemv_test")
    state = oldlib.add_state("main", True)
    A = oldlib.add_array("A", [M, N], dace.float32)
    vin = oldlib.add_array("vin", [N], dace.float32)
    vout = oldlib.add_array("vout", [M], dace.float32)
    Aread = state.add_read("A")
    vinRead = state.add_read("vin")
    voutWrite = state.add_write("vout")
    libnode = blas.Gemv("myGemv")
    libnode.implementation = "FPGA_Accumulate"
    state.add_node(libnode)
    state.add_memlet_path(Aread, libnode, memlet=dace.Memlet.from_array("A", A[1]), dst_conn="_A")
    state.add_memlet_path(vinRead, libnode, memlet=dace.Memlet.from_array("vin", vin[1]), dst_conn="_x")
    state.add_memlet_path(libnode, voutWrite, memlet=dace.Memlet.from_array("vout", vout[1]), src_conn="_y")
        
    oldlib.expand_library_nodes()
    #oldlib.view()
    oldlib.apply_fpga_transformations()
    code = Code(oldlib.generate_code()[2].code, language='cpp')

if __name__ == '__main__':
    sdfg = create_vadd_multibank_sdfg(2, 1)
    #sdfg.validate()
    sdfg.view()
    #code = Code(sdfg.generate_code()[0].code, language='cpp')
    #print(code)
    #bug()
