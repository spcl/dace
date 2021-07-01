# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.transformation.interstate import GPUTransformSDFG
from dace import nodes
from dace.libraries.standard import Reduce

# Define data type to use
N = dace.symbol('N')
dtype = dace.float64
np_dtype = np.float64


@dace.program
def sum(A: dtype[N], sumA: dtype[1]):
    dace.reduce(lambda a, b: a + b, A, sumA, identity=0)

@pytest.mark.gpu
def test_sGPU_CPU_library():

    Reduce.default_implementation = 'CUDA (device)'
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'sGPU_CPU_lib_add'
    sdfg.apply_transformations(GPUTransformSDFG)
    for n, s in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.LibraryNode):
            n.expand(sdfg,sdfg.nodes()[0])
     

    np.random.seed(0)
    n = 1200
    sumA = cuda.pinned_array(shape=1, dtype = np_dtype)
    sumA.fill(0)
    A = cuda.pinned_array(shape=n, dtype = np_dtype)
    Aa = np.random.rand(n)
    A[:] = Aa[:]

    sdfg(A=A, sumA=sumA, N=n)
    res = np.sum(A)
    assert np.isclose(sumA, res, atol=0, rtol=1e-7)
    
    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_sGPU_CPU_library()