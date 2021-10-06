# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import dace

import dace.libraries.nccl as nccl
from dace.libraries.standard import Reduce
from dace.transformation.interstate import GPUTransformSDFG
from dace.transformation.dataflow import RedundantSecondArray, RedundantArray, MapFusion
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg.infer_types import (
    set_default_schedule_storage_types_and_location, infer_connector_types)

N, H, W, C, N_gpu, H_gpu, W_gpu, C_gpu, NN, number_of_gpus = (dace.symbol(
    s, dtype=dace.int64) for s in ('N', 'H', 'W', 'C', 'N_gpu', 'H_gpu',
                                   'W_gpu', 'C_gpu', 'NN', 'number_of_gpus'))
dc_dtype = dace.float32
np_dtype = np.float32


@dace.program
def batchnorm2d_data_parallelism_gpu(x_gpu: dc_dtype[N_gpu, H, W, C],
                                     N: dace.int32):
    x_tmp = dace.ndarray([N_gpu, H, W, C], dtype=dc_dtype)
    x_mean = dace.ndarray([1, H, W, C], dtype=dc_dtype)
    x_std = dace.ndarray([1, H, W, C], dtype=dc_dtype)
    dace.reduce(lambda a, b: a + b, x_gpu, x_mean, axis=(0), identity=0)
    dace.comm.nccl.allreduce(lambda a, b: a + b, x_mean, x_mean)
    fn = np.float32(N)
    x_mean[:] = x_mean[:] / fn
    x_gpu[:] = x_gpu - x_mean
    x_tmp[:] = x_gpu * x_gpu
    dace.reduce(lambda a, b: a + b, x_tmp, x_std, axis=(0), identity=0)
    dace.comm.nccl.allreduce(lambda a, b: a + b, x_std, x_std)
    x_std[:] = np.sqrt(x_std / fn)
    x_gpu[:] = x_gpu / np.sqrt(x_std + 1e-5)


@dace.program
def batchnorm2d_data_parallelism(x: dc_dtype[N, H, W, C]):
    for gpu_id in dace.map[0:number_of_gpus]:
        x_gpu = dace.ndarray([N_gpu, H, W, C],
                             dtype=dc_dtype,
                             storage=dace.StorageType.GPU_Global)
        x_gpu[:] = x[N_gpu * gpu_id:N_gpu * (gpu_id + 1)]
        x_tmp = dace.ndarray([N_gpu, H, W, C], dtype=dc_dtype)
        x_mean = dace.ndarray([H, W, C], dtype=dc_dtype)
        x_std = dace.ndarray([H, W, C], dtype=dc_dtype)
        dace.reduce(lambda a, b: a + b, x_gpu, x_mean, axis=(0), identity=0)
        dace.comm.nccl.allreduce(lambda a, b: a + b, x_mean, x_mean)
        x_gpu[:] = x_gpu - x_mean / NN
        x_tmp[:] = x_gpu * x_gpu
        dace.reduce(lambda a, b: a + b, x_tmp, x_std, axis=(0), identity=0)
        dace.comm.nccl.allreduce(lambda a, b: a + b, x_std, x_std)
        x_gpu[:] = x_gpu / np.sqrt(np.sqrt(x_std / NN) + 1e-5)
        x[N_gpu * gpu_id:N_gpu * (gpu_id + 1)] = x_gpu[:]


@pytest.mark.skip
def test_batchnorm2d_dp_gpu():
    bngpusdfg: dace.SDFG = batchnorm2d_data_parallelism_gpu.to_sdfg(strict=True)
    bngpusdfg.apply_transformations(GPUTransformSDFG)
    # bngpusdfg.view()
    # rnsdfg: dace.SDFG = resnet_basicblock_gpu.to_sdfg()
    # rnsdfg.view()

    program_objects = bngpusdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/batchnorm/' + bngpusdfg.name
    program_folder = compiler.generate_program_folder(bngpusdfg,
                                                      program_objects, out_path)


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
def test_batchnorm2d_data_parallelism():
    n, h, w, c = 16, 128, 128, 64
    # n, h, w, c = 16, 4, 4, 8
    ng = 4

    sdfg: dace.SDFG = batchnorm2d_data_parallelism.to_sdfg(strict=True)
    sdfg.name = sdfg.name + '_inline'
    multi_gpu_map = find_map_by_param(sdfg, 'gpu_id')
    multi_gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    lib_nodes = find_library_nodes(sdfg, Reduce)
    lib_nodes[0].implementation = 'CUDA (device)'
    lib_nodes[1].implementation = 'CUDA (device)'
    # sdfg.apply_transformations(GPUTransformSDFG)
    sdfg.specialize(
        dict(number_of_gpus=ng,
             N=n,
             H=h,
             W=w,
             C=c,
             N_gpu=n // ng,
             NN=np.float32(n)))
    set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([MapFusion])
    sdfg.apply_transformations_repeated([RedundantSecondArray, RedundantArray])
    sdfg.apply_strict_transformations()

    np.random.seed(0)
    X = np.ndarray(shape=[n, h, w, c], dtype=np_dtype)
    X[:] = np.random.rand(n, h, w, c)[:]
    # X = np.arange(n * h * w * c, dtype=np_dtype).reshape([n, h, w, c])
    Z = np.copy(X)

    print('GPU')
    sdfg(X)
    print('GPU done')

    bnsdfg: dace.SDFG = batchnorm2d.to_sdfg()
    lib_nodes = find_library_nodes(bnsdfg, Reduce)
    lib_nodes[0].implementation = 'pure'
    lib_nodes[1].implementation = 'pure'

    print('CPU')
    res = bnsdfg(Z, N=n, H=h, W=w, C=c)
    print('CPU done')
    assert np.allclose(X, res), f'\ndiff: {np.linalg.norm(X-res)}'
    # , f'\nout:\n{X[0][0][0]}\nres:\n{res[0][0][0]}\n'

    # sdfg.name += '_topo'
    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/batchnorm/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_batchnorm2d_data_parallelism()
    # test_batchnorm2d_dp_gpu()
