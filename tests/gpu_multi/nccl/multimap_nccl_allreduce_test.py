# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.sdfg import nodes, infer_types
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
def sum(N: dtype, out: dtype[N]):
    for gpu in dace.map[0:3]:
        reduction_output = dace.define_local([N], dtype)
        gpu_A = np.zeros((N,), dtype=dtype)
        for i in dace.map[0:N]:
            gpu_A[i] = gpu
        red(gpu_A, reduction_output)
        if gpu==0:
            out[:] = reduction_output[:]

@pytest.mark.gpu
def test_nccl_allreduce():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    state = sdfg.start_state
    gpu_map = state.nodes()[0]
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    red_out = state.nodes()[3]
    red_out.desc(sdfg).storage = dtypes.StorageType.GPU_Global
    red_out.desc(sdfg).location['gpu']=0
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)

    sdfg.name = 'nccl_allreduce_multimap'
    # sdfg.view()
    # sdfg.arrays['gpu_sumA'].location={'gpu':0}
    # sdfg.arrays['gpu_sumA'].storage=dtypes.StorageType.GPU_Global
    # sdfg.apply_transformations(GPUTransformMap, options={'gpu_id':0})  
    # sdfg.apply_transformations(GPUMultiTransformMap)  # options={'number_of_gpus':4})

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
    test_nccl_allreduce()