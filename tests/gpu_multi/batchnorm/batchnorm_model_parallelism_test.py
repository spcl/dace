import numpy as np
import dace
import dace.libraries.nccl as nccl
from numba import cuda
from dace.transformation.interstate import GPUTransformSDFG
import pytest

N, H, W, C, N_gpu, H_gpu, W_gpu, C_gpu = (dace.symbol(s, dtype=dace.int64)
                                          for s in ('N', 'H', 'W', 'C', 'N_gpu',
                                                    'H_gpu', 'W_gpu', 'C_gpu'))
dc_dtype = dace.float32
np_dtype = np.float32

number_of_gpus = 4


@dace.program
def batchnorm2d_model_parallelism_gpu(x_gpu: dc_dtype[N, H, W, C_gpu],
                                      num_samples: dace.int64):
    # x_tmp = dace.ndarray([1, H, W, C_gpu], dtype=dc_dtype, storage=dace.StorageType.GPU_Global)
    x_tmp = dace.ndarray([N, H, W, C_gpu],
                         dtype=dc_dtype,
                         storage=dace.StorageType.GPU_Global)
    x_mean = dace.ndarray([1, H, W, C_gpu],
                          dtype=dc_dtype,
                          storage=dace.StorageType.GPU_Global)
    x_std = dace.ndarray([1, H, W, C_gpu],
                         dtype=dc_dtype,
                         storage=dace.StorageType.GPU_Global)
    dace.reduce(lambda a, b: a + b, x_gpu, x_mean, axis=(0), identity=0)
    fn = np.float32(num_samples)
    x_mean[:] = x_mean[:] / fn
    x_gpu[:] = x_gpu - x_mean
    x_tmp[:] = x_gpu * x_gpu
    dace.reduce(lambda a, b: a + b, x_tmp, x_std, axis=(0), identity=0)
    x_std[:] = np.sqrt(x_std / fn)
    x_gpu[:] = (x_gpu - x_mean) / np.sqrt(x_std + 1e-5)


@dace.program
def batchnorm2d_model_parallelism(x: dc_dtype[N, H, W, C]):
    for gpu_id in dace.map[0:number_of_gpus]:
        x_gpu = dace.ndarray([N, H, W, C_gpu],
                             dtype=dc_dtype,
                             storage=dace.StorageType.GPU_Global)
        x_gpu[:, :, :, :] = x[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)]
        batchnorm2d_model_parallelism_gpu(x_gpu, N)
        x[:, :, :, C_gpu * gpu_id:C_gpu * (gpu_id + 1)] = x_gpu[:, :, :, :]


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
def test_batchnorm2d_model_parallelism():
    sdfg: dace.SDFG = batchnorm2d_model_parallelism.to_sdfg(strict=True)
    state = sdfg.start_state
    source = state.source_nodes()[0]
    multi_gpu_map = state.successors(source)[0]
    multi_gpu_map.schedule = dace.ScheduleType.GPU_Multidevice
    # sdfg.apply_transformations(GPUTransformSDFG)
    n = 16
    h = 128
    w = 128
    c = 64
    ng = 4
    sdfg.specialize(dict(number_of_gpus=ng, N=n, H=h, W=w, C=c, C_gpu=c // ng))

    np.random.seed(0)
    X = cuda.pinned_array(shape=[n, h, w, c], dtype=np_dtype)
    # X = np.empty(shape=[n, h, w, c], dtype=np_dtype)
    X[:] = np.random.rand(n, h, w, c)[:]
    Z = np.copy(X)

    print('GPU')
    sdfg(X)
    print('GPU done')

    bnsdfg: dace.SDFG = batchnorm2d.to_sdfg()

    print('CPU')
    bnsdfg(Z, N=n, H=h, W=w, C=c)
    print('CPU done')
    assert np.allclose(X, Z)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/batchnorm/' + sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_batchnorm2d_model_parallelism()
    # test_batchnorm2d()
