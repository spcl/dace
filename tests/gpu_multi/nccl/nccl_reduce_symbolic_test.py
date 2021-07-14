# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl

N = dace.symbol('N')
num_gpus = dace.symbol('num_gpus')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def reduction_test(out: dtype[num_gpus, N]):
    for root_gpu in dace.map[0:num_gpus]:
        for gpu in dace.map[0:num_gpus]:
            reduction_output = dace.ndarray([N], dtype=dtype)
            gpu_A = dace.ndarray([N], dtype=dtype)
            for i in dace.map[0:N]:
                gpu_A[i] = root_gpu
            dace.nccl.Reduce(lambda a, b: a + b,
                             gpu_A,
                             reduction_output,
                             root_gpu,
                             use_group_calls=False)
            if gpu == root_gpu:
                out[root_gpu, :] = reduction_output[:]


def find_map_by_param(state: dace.SDFGState, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    try:
        return next(n for n in state.nodes()
                    if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

    except StopIteration:
        return False


@pytest.mark.gpu
def test_nccl_reduce_symbolic():
    ng = 4
    n = 2
    sdfg: dace.SDFG = reduction_test.to_sdfg(strict=True)
    state = sdfg.start_state
    outer_map = find_map_by_param(state, 'root_gpu')
    if outer_map:
        outer_map.schedule = dtypes.ScheduleType.Sequential
    gpu_map = find_map_by_param(state, 'gpu')
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.specialize(dict(num_gpus=ng))
    sdfg.name = 'nccl_reduce_symbolic'

    out = cuda.pinned_array(shape=[ng, n], dtype=np_dtype)
    out.fill(0)

    sdfg(out=out, N=n)

    res = np.array([0, 4, 8, 12])
    assert (np.unique(out) == res).all()

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/nccl/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_nccl_reduce_symbolic()