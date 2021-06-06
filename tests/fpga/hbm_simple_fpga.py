import numpy
import dace
import dace.sdfg.nodes
from IPython.display import Code
from dace import subsets
import dace.libraries.blas as blas
from tests.fpga.hbm_vadd_fpga import create_vadd_multibank_sdfg

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
    #sdfg.view()
    code = Code(sdfg.generate_code()[0].code, language='cpp')
    print(code)
    #bug()
