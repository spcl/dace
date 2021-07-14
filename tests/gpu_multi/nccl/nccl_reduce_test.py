# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from numba import cuda
from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.config import Config

N = dace.symbol('N')
root_device = dace.symbol('root_device')
num_gpus = dace.symbol('num_gpus')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def reduction_test(out: dtype[N]):
    for gpu in dace.map[0:num_gpus]:
        reduction_output = dace.ndarray([N], dtype=dtype)
        gpu_A = dace.ndarray([N], dtype=dtype)
        for i in dace.map[0:N]:
            gpu_A[i] = gpu
        dace.nccl.Reduce(lambda a, b: a + b,
                         gpu_A,
                         reduction_output,
                         root_device,
                         use_group_calls=False)
        if gpu == root_device:
            out[:] = reduction_output[:]


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_nccl_reduce():
    ng = Config.get('compiler', 'cuda', 'max_number_gpus')
    n = 15
    sdfg: dace.SDFG = reduction_test.to_sdfg(strict=True)
    gpu_map = find_map_by_param(sdfg, 'gpu')
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.specialize(dict(root_device=0, num_gpus=ng))

    sdfg.name = 'nccl_reduce'

    out = cuda.pinned_array(shape=n, dtype=np_dtype)
    out.fill(0)

    sdfg(out=out, N=n)

    assert np.unique(out)[0] == sum(range(ng))

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/nccl/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_nccl_reduce()