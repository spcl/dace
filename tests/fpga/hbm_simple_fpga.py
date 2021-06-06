from dace import dtypes
import numpy
import dace
import dace.sdfg.nodes
from IPython.display import Code
from dace import subsets
import dace.libraries.blas as blas
from tests.fpga.hbm_vadd_fpga import create_vadd_multibank_sdfg
from tests.fpga.hbm_reduce_fpga import create_hbm_reduce_sdfg

if __name__ == '__main__':
    sdfg = create_hbm_reduce_sdfg(2)
    #sdfg.validate()
    sdfg.view()
    #code = Code(sdfg.generate_code()[2].code, language='cpp')
    #print(code)
    #bug()
