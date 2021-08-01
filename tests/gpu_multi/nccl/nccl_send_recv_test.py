# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.config import Config

right_neighbor = dace.symbol('right_neighbor')
left_neighbor = dace.symbol('left_neighbor')
gpu_id = dace.symbol('gpu_id')
num_gpus = dace.symbol('num_gpus')

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def nccl_send_recv():
    out = dace.ndarray([num_gpus], dtype)
    pinned_out = dace.ndarray([num_gpus],
                              dtype,
                              storage=dace.StorageType.CPU_Pinned)
    for gpu_id in dace.map[0:num_gpus]:
        send_buffer = dace.ndarray([1], dtype)
        recv_buffer = dace.ndarray([1], dtype)
        ring_sum = dace.ndarray([1], dtype)
        # right_neighbor = dace.ndarray([1], dtype)
        # left_neighbor = dace.ndarray([1], dtype)

        # right_neighbor = gpu_id + 1 if gpu_id < num_gpus - 1 else 0
        # left_neighbor = gpu_id - 1 if gpu_id > 0 else num_gpus - 1
        for i in dace.map[0:1]:
            send_buffer[i] = gpu_id
        for i in range(num_gpus):
            if gpu_id == 0:
                dace.nccl.Send(send_buffer,
                               gpu_id + 1,
                               group_calls=dtypes.NcclGroupCalls.Start)
                dace.nccl.Recv(recv_buffer,
                               num_gpus - 1,
                               group_calls=dtypes.NcclGroupCalls.End)
            elif gpu_id == num_gpus - 1:
                dace.nccl.Send(send_buffer,
                               0,
                               group_calls=dtypes.NcclGroupCalls.Start)
                dace.nccl.Recv(recv_buffer,
                               gpu_id - 1,
                               group_calls=dtypes.NcclGroupCalls.End)
            else:
                dace.nccl.Send(send_buffer,
                               gpu_id + 1,
                               group_calls=dtypes.NcclGroupCalls.Start)
                dace.nccl.Recv(recv_buffer,
                               gpu_id - 1,
                               group_calls=dtypes.NcclGroupCalls.End)
            for i in dace.map[0:1]:
                ring_sum[i] = recv_buffer[i] + ring_sum[i]
                send_buffer[i] = recv_buffer[i]
        pinned_out[gpu_id] = ring_sum[:]
    out[:] = pinned_out[:]
    return out


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@pytest.mark.multigpu
def test_nccl_send_recv():
    ng = Config.get('compiler', 'cuda', 'max_number_gpus')
    sdfg: dace.SDFG = nccl_send_recv.to_sdfg(strict=True)
    gpu_map = find_map_by_param(sdfg, 'gpu_id')
    gpu_map.schedule = dtypes.ScheduleType.GPU_Multidevice
    infer_types.set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.specialize(dict(num_gpus=ng))

    # out = sdfg()
    # res = np.sum(range(num_gpus))

    # assert np.allclose(out, res), f'\nout: {out}\nres: {res}\n'

    program_objects = sdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/nccl/' + sdfg.name
    program_folder = compiler.generate_program_folder(sdfg, program_objects,
                                                      out_path)


if __name__ == "__main__":
    test_nccl_send_recv()