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

# Define data type to use
dtype = dace.float64
np_dtype = np.float64

@dace.program
def red(inbuff:dtype[N], outbuff: dtype[N]):
    dace.nccl.AllReduce(lambda a,b: a+b, inbuff, outbuff)

@dace.program
def sum(N: dtype, out: dtype[N]):
    for gpu in dace.map[0:3]:
        reduction_output = dace.ndarray([N], dtype=dtype, storage=dace.StorageType.GPU_Global)
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

    N.set(15)
    out = cuda.pinned_array(shape=N, dtype = np_dtype)
    out.fill(0)

    sdfg(N, out=out)

    print(np.unique(out))
    assert np.unique(out)==np.array([3])
    

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/nccl/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

if __name__ == "__main__":
    test_nccl_allreduce()