# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.sdfg import nodes
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.transformation.dataflow import GPUMultiTransformMap, GPUTransformMap
N = dace.symbol('N')
M = dace.symbol('M')

n = 1200

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

@dace.program
def red(inbuff:dtype[N], outbuff: dtype[N]):
    dace.nccl.AllReduce(inbuff, outbuff, operation='ncclSum')

@dace.program
def sum(N: dtype, gpu_A: dtype[N], out: dtype[N], reduction_output: dtype[N]):
    # reduction_output = np.zeros((N,), dtype=dtype)
    
    for gpu in dace.map[0:3]:
        # dace.comm.Scatter(gpu_A, reduction_output)
        # dace.reduce(lambda a, b: a + b, gpu_A, reduction_output, identity=0)
        dace.nccl.AllReduce(gpu_A, reduction_output, operation='ncclSum')
        # red(gpu_A, reduction_output)
        # if gpu==0:
        #     out[:] = reduction_output[:]
    # return reduction_output

# Original

# @dace.program
# def sum(N: dtype, reduction_output: dtype[N]):
#     # reduction_output = np.zeros((N,), dtype=dtype)
    
#     # out = np.zeros((N,), dtype=dtype)
#     for gpu in dace.map[0:3]:
#         gpu_A = np.zeros((N,), dtype=dtype)
#         for i in dace.map[0:N]:
#             gpu_A[i] = gpu
#         # dace.comm.Scatter(gpu_A, reduction_output)
#         # dace.reduce(lambda a, b: a + b, gpu_A, reduction_output, identity=0)
#         # dace.nccl.AllReduce(gpu_A, reduction_output, operation='ncclSum')
#         red(gpu_A, reduction_output)
#         if gpu==0:
#             out = reduction_output[:]

#     return out

# no loop version
    
# @dace.program
# def sum(N: dtype, reduction_output: dtype[N]):
#     # reduction_output = np.zeros((N,), dtype=dtype)
    
#     # for gpu in dace.map[0:3]:
#     gpu=0
#     gpu_A = np.zeros((N,), dtype=dtype)
#     out = np.zeros((N,), dtype=dtype)
#     for i in dace.map[0:N]:
#         gpu_A[i] = gpu
#     # dace.comm.Scatter(gpu_A, reduction_output)
#     # dace.reduce(lambda a, b: a + b, gpu_A, reduction_output, identity=0)
#     dace.nccl.AllReduce(gpu_A, reduction_output, operation='ncclSum')
#     if gpu==0:
#         out[:] = reduction_output[:]

#     return out

@pytest.mark.gpu
def test_reduction_mGPU_GPU0_sum():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    sdfg.name = 'nccl_reduce_multimap'
    sdfg.view()
    # sdfg.arrays['gpu_sumA'].location={'gpu':0}
    # sdfg.arrays['gpu_sumA'].storage=dtypes.StorageType.GPU_Global
    sdfg.apply_transformations(GPUTransformMap, options={'gpu_id':0})  
    sdfg.apply_transformations(GPUMultiTransformMap)  # options={'number_of_gpus':4})

    # np.random.seed(0)
    # sumA = cuda.pinned_array(shape=1, dtype = np_dtype)
    # sumA.fill(0)
    # A = cuda.pinned_array(shape=n, dtype = np_dtype)
    # Aa = np.random.rand(n)
    # A[:] = Aa[:]

    # sdfg(A=A, sumA=sumA, N=n)
    # res = np.sum(A)
    # assert np.isclose(sumA, res, atol=0, rtol=1e-7)

    program_objects = sdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/nccl/' + sdfg.name
    program_folder = compiler.generate_program_folder(sdfg, program_objects,
                                                      out_path)

if __name__ == "__main__":
    test_reduction_mGPU_GPU0_sum()