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
def red(inbuff: dtype[N], outbuff: dtype[N]):
    dace.nccl.AllReduce(lambda a, b: a + b, inbuff, outbuff)


@dace.program
def sum(N: dtype, out: dtype[N]):
    for gpu in dace.map[0:3]:
        reduction_output = dace.ndarray([N], dtype=dtype)
        gpu_A = dace.ndarray([N], dtype=dtype)
        for i in dace.map[0:N]:
            gpu_A[i] = gpu
        red(gpu_A, reduction_output)
        if gpu == 0:
            out[:] = reduction_output[:]


@pytest.mark.gpu
def test_nccl_allreduce():
    sdfg: dace.SDFG = sum.to_sdfg(strict=True)
    state = sdfg.start_state
    gpu_map = state.nodes()[0]
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)

    sdfg.name = 'nccl_allreduce_multimap'

    n = 15
    out = cuda.pinned_array(shape=n, dtype=np_dtype)
    out.fill(0)

    sdfg(n, out=out)

    print(np.unique(out))
    assert np.unique(out) == np.array([3])

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/nccl/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)
    # Compile the code and get the shared library path
    # shared_library = compiler.configure_and_compile(program_folder, sdfg.name)


if __name__ == "__main__":
    test_nccl_allreduce()