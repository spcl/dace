# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Explicitly multi GPU distributed Jacobi-2D sample."""
import dace
import numpy as np
import os
import timeit
from typing import List

from dace.transformation.dataflow import MapFusion, RedundantArray, RedundantSecondArray
from dace.transformation.interstate import InlineSDFG, StateFusion
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg import nodes, infer_types
from dace import dtypes
import dace.libraries.nccl as nccl
from dace.libraries.nccl import utils as nutil

d_int = dace.int64
d_float = dace.float64
np_float = np.float64

TSTEPS = dace.symbol('TSTEPS', dtype=d_int, integer=True, positive=True)
N = dace.symbol('N', dtype=d_int, integer=True, positive=True)
lNy = dace.symbol('lNy', dtype=d_int, integer=True, positive=True)
Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
pi = dace.symbol('pi', dtype=dace.int32, integer=True, nonnegative=True)
size = dace.symbol('size', dtype=dace.int32, integer=True, positive=True)

Ny = Py * lNy


@dace.program
def jacobi_2d_shared(A: dace.float64[Ny, N], B: dace.float64[Ny, N]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


@dace.program
def j2d_overlap_nccl(A: d_float[Ny, N], B: d_float[Ny, N]):
    for rank in dace.map[0:size]:
        # Local extended domain
        lA = np.zeros((lNy + 2, N), dtype=A.dtype)
        lB = np.zeros((lNy + 2, N), dtype=B.dtype)

        lA[1:-1, :] = A[rank * lNy:(rank + 1) * lNy, :]
        lB[1:-1, :] = B[rank * lNy:(rank + 1) * lNy, :]
        group_handle_0 = dace.define_local_scalar(d_int)
        group_handle_1 = dace.define_local_scalar(d_int)
        group_handle_2 = dace.define_local_scalar(d_int)
        group_handle_3 = dace.define_local_scalar(d_int)
        group_handle_4 = dace.define_local_scalar(d_int)
        group_handle_5 = dace.define_local_scalar(d_int)
        if rank == 0:
            dace.comm.nccl.Send(lA[-2],
                                peer=rank + 1,
                                group_handle=group_handle_0)
            dace.comm.nccl.Recv(lA[-1],
                                peer=rank + 1,
                                group_handle=group_handle_0)
        elif rank == size - 1:
            dace.comm.nccl.Send(lA[1],
                                peer=rank - 1,
                                group_handle=group_handle_0)
            dace.comm.nccl.Recv(lA[0],
                                peer=rank - 1,
                                group_handle=group_handle_0)
        else:
            dace.comm.nccl.Send(lA[1],
                                peer=rank - 1,
                                group_handle=group_handle_0)
            dace.comm.nccl.Recv(lA[-1],
                                peer=rank + 1,
                                group_handle=group_handle_0)
            dace.comm.nccl.Send(lA[-2],
                                peer=rank + 1,
                                group_handle=group_handle_1)
            dace.comm.nccl.Recv(lA[0],
                                peer=rank - 1,
                                group_handle=group_handle_1)
        for t in range(1, TSTEPS):
            if rank == 0:
                # boundary
                lB[-2, 1:-1] = 0.2 * (lA[-2, 1:-1] + lA[-2, :-2] + lA[-2, 2:] +
                                      lA[-1, 1:-1] + lA[-3, 1:-1])
                # exchange
                dace.comm.nccl.Send(lB[-2],
                                    peer=rank + 1,
                                    group_handle=group_handle_2)
                dace.comm.nccl.Recv(lB[-1],
                                    peer=rank + 1,
                                    group_handle=group_handle_2)
                # interior
                lB[2:-2,
                   1:-1] = 0.2 * (lA[2:-2, 1:-1] + lA[2:-2, :-2] + lA[2:-2, 2:]
                                  + lA[3:-1, 1:-1] + lA[1:-3, 1:-1])
                # boundary
                lA[-2, 1:-1] = 0.2 * (lB[-2, 1:-1] + lB[-2, :-2] + lB[-2, 2:] +
                                      lB[-1, 1:-1] + lB[-3, 1:-1])
                # exchange
                # if t < TSTEPS - 1:
                dace.comm.nccl.Send(lA[-2],
                                    peer=rank + 1,
                                    group_handle=group_handle_3)
                dace.comm.nccl.Recv(lA[-1],
                                    peer=rank + 1,
                                    group_handle=group_handle_3)
                # interior
                lA[2:-2,
                   1:-1] = 0.2 * (lB[2:-2, 1:-1] + lB[2:-2, :-2] + lB[2:-2, 2:]
                                  + lB[3:-1, 1:-1] + lB[1:-3, 1:-1])
            elif rank == size - 1:
                # boundary
                lB[1, 1:-1] = 0.2 * (lA[1, 1:-1] + lA[1, :-2] + lA[1, 2:] +
                                     lA[2, 1:-1] + lA[0, 1:-1])
                # exchange
                dace.comm.nccl.Send(lB[1],
                                    peer=rank - 1,
                                    group_handle=group_handle_2)
                dace.comm.nccl.Recv(lB[0],
                                    peer=rank - 1,
                                    group_handle=group_handle_2)
                # interior
                lB[2:-2,
                   1:-1] = 0.2 * (lA[2:-2, 1:-1] + lA[2:-2, :-2] + lA[2:-2, 2:]
                                  + lA[3:-1, 1:-1] + lA[1:-3, 1:-1])
                # boundary
                lA[1, 1:-1] = 0.2 * (lB[1, 1:-1] + lB[1, :-2] + lB[1, 2:] +
                                     lB[2, 1:-1] + lB[0, 1:-1])
                # exchange
                # if t < TSTEPS - 1:
                dace.comm.nccl.Send(lA[1],
                                    peer=rank - 1,
                                    group_handle=group_handle_3)
                dace.comm.nccl.Recv(lA[0],
                                    peer=rank - 1,
                                    group_handle=group_handle_3)
                # interior
                lA[2:-2,
                   1:-1] = 0.2 * (lB[2:-2, 1:-1] + lB[2:-2, :-2] + lB[2:-2, 2:]
                                  + lB[3:-1, 1:-1] + lB[1:-3, 1:-1])
            else:
                # boundary
                lB[1, 1:-1] = 0.2 * (lA[1, 1:-1] + lA[1, :-2] + lA[1, 2:] +
                                     lA[2, 1:-1] + lA[0, 1:-1])
                lB[-2, 1:-1] = 0.2 * (lA[-2, 1:-1] + lA[-2, :-2] + lA[-2, 2:] +
                                      lA[-1, 1:-1] + lA[-3, 1:-1])
                # exchange
                dace.comm.nccl.Send(lB[1],
                                    peer=rank - 1,
                                    group_handle=group_handle_2)
                dace.comm.nccl.Recv(lB[-1],
                                    peer=rank + 1,
                                    group_handle=group_handle_2)
                dace.comm.nccl.Send(lB[-2],
                                    peer=rank + 1,
                                    group_handle=group_handle_3)
                dace.comm.nccl.Recv(lB[0],
                                    peer=rank - 1,
                                    group_handle=group_handle_3)

                # interior
                lB[2:-2,
                   1:-1] = 0.2 * (lA[2:-2, 1:-1] + lA[2:-2, :-2] + lA[2:-2, 2:]
                                  + lA[3:-1, 1:-1] + lA[1:-3, 1:-1])

                # boundary
                lA[1, 1:-1] = 0.2 * (lB[1, 1:-1] + lB[1, :-2] + lB[1, 2:] +
                                     lB[2, 1:-1] + lB[0, 1:-1])
                lA[-2, 1:-1] = 0.2 * (lB[-2, 1:-1] + lB[-2, :-2] + lB[-2, 2:] +
                                      lB[-1, 1:-1] + lB[-3, 1:-1])

                # exchange
                # if t < TSTEPS - 1:
                dace.comm.nccl.Send(lA[1],
                                    peer=rank - 1,
                                    group_handle=group_handle_4)
                dace.comm.nccl.Recv(lA[-1],
                                    peer=rank + 1,
                                    group_handle=group_handle_4)
                dace.comm.nccl.Send(lA[-2],
                                    peer=rank + 1,
                                    group_handle=group_handle_5)
                dace.comm.nccl.Recv(lA[0],
                                    peer=rank - 1,
                                    group_handle=group_handle_5)
                # interior
                lA[2:-2,
                   1:-1] = 0.2 * (lB[2:-2, 1:-1] + lB[2:-2, :-2] + lB[2:-2, 2:]
                                  + lB[3:-1, 1:-1] + lB[1:-3, 1:-1])

        A[rank * lNy:(rank + 1) * lNy] = lA[1:-1, :]
        B[rank * lNy:(rank + 1) * lNy] = lB[1:-1, :]


def init_data(N, datatype):

    A = np.fromfunction(lambda i, j: i * (j + 2) / N,
                        shape=(N, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N,
                        shape=(N, N),
                        dtype=datatype)

    return A, B


def time_to_ms(raw):
    return int(round(raw * 1000))


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_data_desc(sdfg: dace.SDFG, name: str) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return next(d for s, n, d in sdfg.arrays_recursive() if n == name)


if __name__ == "__main__":
    ts, n = 1, 16
    number_of_gpus = 4
    sdfg = j2d_overlap_nccl.to_sdfg(strict=True)
    gpu_map = find_map_by_param(sdfg, 'rank')
    gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    sdfg.specialize(
        dict(Py=number_of_gpus,
             size=number_of_gpus,
             lNy=n // number_of_gpus,
             TSTEPS=ts,
             N=n))
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.apply_transformations_repeated([RedundantArray, RedundantSecondArray])
    sdfg.apply_strict_transformations()

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/jacobi/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)

    print('GPU: start')
    A, B = init_data(n, np_float)
    sdfg(A, B)
    print('GPU: done')

    print('CPU: start')
    refA, refB = init_data(n, np_float)
    shared_sdfg = jacobi_2d_shared.to_sdfg()
    shared_sdfg.specialize(
        dict(Py=number_of_gpus,
             size=number_of_gpus,
             lNy=n // number_of_gpus,
             TSTEPS=ts,
             N=n))
    shared_sdfg(A=refA, B=refB)
    print('CPU: done')

    print("=======Validation=======")
    assert (np.allclose(A, refA)), f'\ndiff: {np.linalg.norm(A-refA)}'
    assert (np.allclose(B, refB)), f'\ndiff: {np.linalg.norm(B-refB)}'
    # assert (np.allclose(A, refA)), f'A:\n{repr(A)}\nrefA:\n{repr(refA)}'
    # assert (np.allclose(B, refB)), f'A:\n{repr(B)}\nrefA:\n{repr(refB)}'
    print("OK")
