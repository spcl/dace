import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG

N, H, W, C, N_gpu, H_gpu, W_gpu, C_gpu = (dace.symbol(s, dtype=dace.int64)
                                           for s in ('N', 'H', 'W', 'C','N_gpu','H_gpu','W_gpu','C_gpu'))
dc_dtype = dace.float32
np_dtype = np.float32

@dace.program
def gpu_sum_data_parallelism(x_gpu: dc_dtype[N_gpu, H, W, C]):
    x_tmp = dace.define_local([1, H, W, C], dtype=dc_dtype)
    x_tmp_2 = dace.define_local([N_gpu, H, W, C], dtype=dc_dtype)
    x_mean = dace.define_local([1, H, W, C], dtype=dc_dtype)
    x_std = dace.define_local([1, H, W, C], dtype=dc_dtype)
    dace.reduce(lambda a, b: a+b, x_gpu, x_tmp, axis = (0), identity=0)
    x_tmp[:] = x_tmp[:]/np.float(N)
    dace.nccl.Allreduce(x_tmp, x_mean, operation = 'ncclSum')
    x_tmp_2[:] = (x_gpu - x_mean)*(x_gpu - x_mean)/np_dtype(N)
    dace.reduce(lambda a,b: a+b, x_tmp_2, x_tmp, axis=(0), identity=0)
    dace.nccl.Allreduce(x_tmp_2, x_mean, operation = 'ncclSum')
    

    

@dace.program
def batchnorm2d_data_parallelism(x: dc_dtype[N, H, W, C], ngpus: dace.int64):
    for gpu_id in dace.map[0,ngpus]:
        x_gpu = dace.define_local([N_gpu, H, W, C], dtype=dc_dtype)
        x_mean = dace.define_local([1, H, W, C], dtype=dc_dtype)
        x_std = dace.define_local([1, H, W, C], dtype=dc_dtype)
        x_gpu[:] = x[(N//ngpus)*gpu_id:((1+N)//ngpus)*gpu_id]
        x_mean = gpu_sum(x_gpu)





@dace.program
def batchnorm2d(x: dc_dtype[N, H, W, C]):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.ndarray((1, H, W, C), dtype=np.float32)
    mean[:] = np.mean(x, axis=0)
    # std = np.std(x, axis=0, keepdims=True)
    std = np.ndarray((1, H, W, C), dtype=np.float32)
    # std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / np.float32(N))
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / np.float32(N))
    # return (x - mean) / np.sqrt(std + eps)
    return (x - mean) / np.sqrt(std + 1e-5)


def test_batchnorm2d():
    bnsdfg: dace.SDFG = batchnorm2d.to_sdfg(strict=True)
    bnsdfg.apply_transformations(GPUTransformSDFG)
    bnsdfg.view()
    # rnsdfg: dace.SDFG = resnet_basicblock_gpu.to_sdfg()
    # rnsdfg.view()

    program_objects = bnsdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/batchnorm/' + bnsdfg.name
    program_folder = compiler.generate_program_folder(bnsdfg, program_objects,
                                                      out_path)


if __name__ == "__main__":
    test_batchnorm2d()