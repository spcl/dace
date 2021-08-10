# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.config import Config

num_gpus = dace.symbol('num_gpus')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def nccl_send_recv():
    out = dace.ndarray([num_gpus, 2], dtype)
    pinned_out = dace.ndarray([num_gpus, 2],
                              dtype,
                              storage=dace.StorageType.CPU_Pinned)
    for gpu_id in dace.map[0:num_gpus]:
        # Transients
        send_buffer = dace.ndarray([2],
                                   dtype,
                                   storage=dace.StorageType.GPU_Global)
        recv_buffer = dace.ndarray([2],
                                   dtype,
                                   storage=dace.StorageType.GPU_Global)

        # Init transients
        for i in dace.map[0:2]:
            send_buffer[i] = gpu_id
        group_handle = dace.define_local_scalar(
            dace.int32, storage=dace.StorageType.GPU_Global)
        if gpu_id == 0:
            dace.comm.nccl.Send(send_buffer, 1, group_handle=group_handle)
            dace.comm.nccl.Recv(recv_buffer, 1, group_handle=group_handle)
        else:
            dace.comm.nccl.Send(send_buffer, 0, group_handle=group_handle)
            dace.comm.nccl.Recv(recv_buffer, 0, group_handle=group_handle)

        pinned_out[gpu_id, :] = recv_buffer[:]

    out[:] = pinned_out[:]
    return out


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_data_desc(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return next(d for s, n, d in sdfg.arrays_recursive() if n == name)


@pytest.mark.multigpu
def test_nccl_send_recv():
    ng = Config.get('compiler', 'cuda', 'max_number_gpus')
    if ng < 2:
        raise ValueError('This test needs to run with at least 2 GPUs.')
    else:
        ng = 2
    sdfg: dace.SDFG = nccl_send_recv.to_sdfg(strict=True)
    gpu_map = find_map_by_param(sdfg, 'gpu_id')
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.specialize(dict(num_gpus=ng))

    out = sdfg()
    res = np.array([0, 1])

    assert np.allclose(np.unique(out), res), f'\nout: {out}\nres: {res}\n'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/nccl/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_nccl_send_recv()
