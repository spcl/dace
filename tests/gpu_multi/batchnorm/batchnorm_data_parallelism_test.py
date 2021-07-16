import numpy as np
import dace
import dace.libraries.nccl as nccl
from dace.transformation.interstate import GPUTransformSDFG
from numba import cuda
import pytest

N, H, W, C, N_gpu, H_gpu, W_gpu, C_gpu = (dace.symbol(s, dtype=dace.int64)
                                          for s in ('N', 'H', 'W', 'C', 'N_gpu',
                                                    'H_gpu', 'W_gpu', 'C_gpu'))
dc_dtype = dace.float32
np_dtype = np.float32

number_of_gpus = dace.symbol('number_of_gpus')


@dace.program
def batchnorm2d_data_parallelism_gpu(x_gpu: dc_dtype[N_gpu, H, W, C],
                                     n: dace.int64):
    x_tmp = dace.ndarray([N_gpu, H, W, C], dtype=dc_dtype)
    x_mean = dace.ndarray([1, H, W, C], dtype=dc_dtype)
    x_std = dace.ndarray([1, H, W, C], dtype=dc_dtype)
    dace.reduce(lambda a, b: a + b, x_gpu, x_mean, axis=(0), identity=0)
    dace.nccl.allreduce(lambda a, b: a + b, x_mean, x_mean)
    fn = np.float32(n)
    x_mean[:] = x_mean[:] / fn
    x_gpu[:] = x_gpu - x_mean
    x_tmp[:] = x_gpu * x_gpu
    dace.reduce(lambda a, b: a + b, x_tmp, x_std, axis=(0), identity=0)
    dace.nccl.allreduce(lambda a, b: a + b, x_std, x_std)
    x_std[:] = np.sqrt(x_std / fn)
    x_gpu[:] = x_gpu / np.sqrt(x_std + 1e-5)


@dace.program
def batchnorm2d_data_parallelism(x: dc_dtype[N, H, W, C]):
    for gpu_id in dace.map[0:number_of_gpus]:
        x_gpu = dace.ndarray([N_gpu, H, W, C],
                             dtype=dc_dtype,
                             storage=dace.StorageType.GPU_Global)
        x_gpu[:] = x[N_gpu * gpu_id:N_gpu * (gpu_id + 1)]
        batchnorm2d_data_parallelism_gpu(x_gpu, N)
        x[N_gpu * gpu_id:N_gpu * (gpu_id + 1)] = x_gpu[:]


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


@pytest.mark.multigpu
def test_batchnorm2d_data_parallelism():
    sdfg: dace.SDFG = batchnorm2d_data_parallelism.to_sdfg(strict=True)
    state = sdfg.start_state
    source = state.source_nodes()[0]
    multi_gpu_map = state.successors(source)[0]
    multi_gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    # sdfg.apply_transformations(GPUTransformSDFG)
    sdfg.specialize(dict(number_of_gpus=4))

    n = 16
    h = 128
    w = 128
    c = 64

    size = 256
    np.random.seed(0)
    X = cuda.pinned_array(shape=[n, h, w, c], dtype=np_dtype)
    X[:] = np.random.rand(n, h, w, c)[:]
    Z = np.copy(X)

    print('GPU')

    sdfg(X)
    print('GPU done')

    bnsdfg: dace.SDFG = batchnorm2d.to_sdfg()

    print('CPU')
    bnsdfg(Z)
    print('CPU done')
    assert np.allclose(X, Z)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/batchnorm/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_batchnorm2d_data_parallelism()
    # test_batchnorm2d_dp_gpu()