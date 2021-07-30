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

cpu_storage = dace.StorageType.CPU_Pinned

outer_dim = [N, H, W, C]
inner_dim = [N, H, W, C_gpu]
outer_red_dim = [H, W, C]
inner_red_dim = [H, W, C_gpu]
size = n * h * w * c
shape = [n, h, w, c]


@dace.program
def ndarray_reduce_gpu(x: dc_dtype[N, H, W, C]):
    x_pinned = dace.ndarray(outer_dim, dc_dtype, storage=cpu_storage)
    red = dace.ndarray(outer_red_dim, dc_dtype, storage=cpu_storage)
    x_pinned[:] = x[:]
    for gpu_id in dace.map[0:number_of_gpus]:
        x_gpu = dace.ndarray(inner_dim, dtype=dc_dtype)
        x_mean = dace.ndarray(inner_red_dim, dtype=dc_dtype)

        x_gpu[:] = x_pinned[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)]

        dace.reduce(lambda a, b: a + b, x_gpu, x_mean, axis=(0), identity=0)

        red[:, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)] = x_mean[:]

    return red


def find_map(sdfg: dace.SDFG, condition=None) -> dace.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, s in sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.MapEntry) and (
                    condition is None or condition(n, s)))


def find_library_nodes(
        sdfg: dace.SDFG,
        lib_type: dace.sdfg.nodes.LibraryNode) -> dace.nodes.MapEntry:
    """ Finds the first access node by the given data name. """
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, lib_type)]


@pytest.mark.multigpu
def test_ndarray_reduce():
    sdfg: dace.SDFG = ndarray_reduce_gpu.to_sdfg(strict=True)
    map_entry = find_map(sdfg)
    map_entry.schedule = dace.ScheduleType.GPU_Multidevice
    lib_nodes = find_library_nodes(sdfg, Reduce)
    lib_nodes[0].implementation = 'CUDA (device)'

    sdfg.specialize(dict(number_of_gpus=ng, N=n, H=h, W=w, C=c, C_gpu=c // ng))
    set_default_schedule_storage_types_and_location(sdfg, None)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([RedundantSecondArray, RedundantArray])
    sdfg.apply_strict_transformations()

    np.random.seed(0)
    X = np.ndarray(shape=shape, dtype=np_dtype)
    X = np.arange(size, dtype=np_dtype).reshape(shape)
    # X[:] = np.random.rand(size)[:]
    Z = np.copy(X)

    print('GPU')
    out = sdfg(X)
    print('GPU done')
    res = np.sum(Z, axis=0)
    assert np.allclose(out, res), f'\out:\n{repr(out)}\n\nres:\n{repr(res)}\n'

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/basic/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_ndarray_reduce()