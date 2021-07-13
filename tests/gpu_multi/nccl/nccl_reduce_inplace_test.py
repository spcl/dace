# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.sdfg import nodes, infer_types
from dace import dtypes


N = dace.symbol('N')
root_device = dace.symbol('root_device')
num_gpus = dace.symbol('num_gpus')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def reduction_test(out: dtype[N]):
    for gpu in dace.map[0:num_gpus]:
        gpu_A = dace.ndarray([N], dtype=dtype)
        for i in dace.map[0:N]:
            gpu_A[i] = gpu
        dace.nccl.Reduce(lambda a, b: a + b,
                         gpu_A,
                         root = root_device,
                         use_group_calls=False)
        if gpu == root_device:
            out[:] = gpu_A[:]


def find_map_by_param(state: dace.SDFGState, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    try:
        return next(n for n in state.nodes()
                    if isinstance(n, dace.nodes.MapEntry) and pname in n.params)

    except StopIteration:
        return False


@pytest.mark.gpu
def test_nccl_reduce_inplace():
    ng = 3
    n = 15
    sdfg: dace.SDFG = reduction_test.to_sdfg(strict=True)
    state = sdfg.start_state
    gpu_map = find_map_by_param(state, 'gpu')
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.specialize(dict(root=0, num_gpus=ng))

    sdfg.name = 'nccl_reduce_inplace'

    out = cuda.pinned_array(shape=n, dtype=np_dtype)
    out.fill(0)

    sdfg(out=out, N=n)

    res = sum(range(ng))
    assert np.unique(out)[0] == res

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/nccl/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_nccl_reduce_inplace()