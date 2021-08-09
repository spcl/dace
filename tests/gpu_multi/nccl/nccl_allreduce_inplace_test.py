# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.config import Config

N = dace.symbol('N')
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
        dace.comm.nccl.AllReduce(lambda a, b: a + b, gpu_A)
        if gpu == 0:
            out[:] = gpu_A[:]


@pytest.mark.multigpu
def test_nccl_allreduce_inplace():
    ng = Config.get('compiler', 'cuda', 'max_number_gpus')
    n = 15
    sdfg: dace.SDFG = reduction_test.to_sdfg(strict=True)
    state = sdfg.start_state
    gpu_map = state.nodes()[0]
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.specialize(dict(num_gpus=ng))
    sdfg.name = 'nccl_allreduce_inplace'

    out = np.ndarray(shape=n, dtype=np_dtype)
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
    test_nccl_allreduce_inplace()