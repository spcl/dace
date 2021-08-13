# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import dace

import dace.libraries.nccl as nccl
from dace.libraries.standard import Reduce
from dace.transformation.dataflow import RedundantSecondArray, RedundantArray
from dace.sdfg.infer_types import (
    set_default_schedule_storage_types_and_location, infer_connector_types)

N, H, W, C, N_gpu, H_gpu, W_gpu, C_gpu, NN, number_of_gpus = (dace.symbol(
    s, dtype=dace.int64) for s in ('N', 'H', 'W', 'C', 'N_gpu', 'H_gpu',
                                   'W_gpu', 'C_gpu', 'NN', 'number_of_gpus'))
dc_dtype = dace.float32
np_dtype = np.float32

n, h, w, c = 16, 4, 4, 8
ng = 4
outer_dim = [N, H, W, C]
inner_dim = [N, H, W, C_gpu]
outer_red_dim = [1, H, W, C]
inner_red_dim = [1, H, W, C_gpu]


@dace.program
def batchnorm2d_model_parallelism_cpu(x: dc_dtype[N, H, W, C]):
    x_pinned = dace.ndarray(outer_dim, dc_dtype)
    mean = dace.ndarray(outer_red_dim, dc_dtype)
    std = dace.ndarray(outer_red_dim, dc_dtype)
    red = dace.ndarray(outer_red_dim, dc_dtype)

    x_pinned[:] = x[:]

    for gpu_id in dace.map[0:number_of_gpus]:
        x_gpu = dace.ndarray(inner_dim, dc_dtype)
        x_tmp = dace.ndarray(inner_dim, dc_dtype)
        x_mean = dace.ndarray(inner_red_dim, dc_dtype)
        x_std = dace.ndarray(inner_red_dim, dc_dtype)

        x_gpu[:, :, :, :] = x_pinned[:, :, :,
                                     C_gpu * gpu_id:C_gpu * (gpu_id + 1)]

        dace.reduce(lambda a, b: a + b, x_gpu, x_mean, axis=(0), identity=0)
        red[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)] = x_mean[:]

        x_mean[:] = x_mean[:] / NN
        x_gpu[:] = x_gpu - x_mean
        x_tmp[:] = x_gpu * x_gpu

        dace.reduce(lambda a, b: a + b, x_tmp, x_std, axis=(0), identity=0)
        x_std[:] = np.sqrt(x_std / NN)
        x_gpu[:] = x_gpu / np.sqrt(x_std + 1e-5)

        mean[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)] = x_mean[:]
        std[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)] = x_std[:]
        x_pinned[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)] = x_gpu[:]

    x[:] = x_pinned[:]
    return mean, std, red


@dace.program
def batchnorm2d_model_parallelism_numpy(x: dc_dtype[N, H, W, C]):
    x_pinned = dace.ndarray([N, H, W, C], dc_dtype)
    x_pinned[:] = x[:]
    for gpu_id in dace.map[0:number_of_gpus]:
        x_gpu = dace.ndarray([N, H, W, C_gpu], dtype=dc_dtype)
        x_mean = dace.ndarray([H, W, C_gpu], dtype=dc_dtype)
        x_std = dace.ndarray([H, W, C_gpu], dtype=dc_dtype)
        x_gpu[:, :, :, :] = x_pinned[:, :, :,
                                     C_gpu * gpu_id:C_gpu * (gpu_id + 1)]
        x_mean[:] = np.mean(x_gpu, axis=0)
        x_std[:] = np.sqrt(
            np.sum((x_gpu - x_mean) * (x_gpu - x_mean), axis=0) / NN)
        x_pinned[:, :, :, C_gpu * gpu_id:C_gpu *
                 (gpu_id + 1)] = (x_gpu - x_mean) / np.sqrt(x_std + 1e-5)
    x[:] = x_pinned[:]


@dace.program
def batchnorm2d(x: dc_dtype[N, H, W, C]):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.ndarray((1, H, W, C), dtype=np.float32)
    mean[:] = np.mean(x, axis=0)
    # std = np.std(x, axis=0, keepdims=True)
    std = np.ndarray((1, H, W, C), dtype=np.float32)
    # std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / np.float32(S0))
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / np.float32(N))
    # return (x - mean) / np.sqrt(std + eps)
    return (x - mean) / np.sqrt(std + 1e-5)


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


def find_library_nodes(
        sdfg: dace.SDFG,
        lib_type: dace.sdfg.nodes.LibraryNode) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, lib_type)]


@pytest.mark.multigpu
def test_batchnorm2d_model_parallelism_cpu():
    sdfg: dace.SDFG = batchnorm2d_model_parallelism_cpu.to_sdfg(strict=True)
    # multi_gpu_map = find_map_by_param(sdfg, 'gpu_id')
    # multi_gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    # lib_nodes = find_library_nodes(sdfg, Reduce)
    # lib_nodes[0].implementation = 'CUDA (device)'
    # lib_nodes[1].implementation = 'CUDA (device)'

    sdfg.specialize(
        dict(number_of_gpus=ng,
             N=n,
             H=h,
             W=w,
             C=c,
             C_gpu=c // ng,
             NN=np.float32(n)))
    set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([RedundantSecondArray, RedundantArray])
    sdfg.apply_strict_transformations()

    np.random.seed(0)
    # X = np.ndarray(shape=[n, h, w, c], dtype=np_dtype)
    # X[:] = np.random.rand(n, h, w, c)[:]
    X = np.arange(n * h * w * c, dtype=np_dtype).reshape([n, h, w, c])
    Z = np.copy(X)

    print('CPU')
    mean, std, red = sdfg(X)
    print('CPU done')
    print(f'\rred:\n{repr(red)}\n\nmean:\n{repr(mean)}\n\nstd:\n{repr(std)}\n')

    bnsdfg: dace.SDFG = batchnorm2d.to_sdfg()
    lib_nodes = find_library_nodes(bnsdfg, Reduce)
    lib_nodes[0].implementation = 'pure'
    lib_nodes[1].implementation = 'pure'

    print('CPU')
    res = bnsdfg(Z, N=n, H=h, W=w, C=c)
    print('CPU done')
    assert np.allclose(
        X, Z
    ), f'\nout:\n{repr(X[:,0,0,0])}\n{repr(X[:,-1,-1,-1])}\nres:\n{repr(res[:,0,0,0])}\n{repr(res[:,-1,-1,-1])}\n'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/batchnorm/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_batchnorm2d_model_parallelism_cpu()
    # test_batchnorm2d_model_parallelism_numpy()
    # test_batchnorm2d()
