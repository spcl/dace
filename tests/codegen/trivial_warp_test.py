import dace
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow.add_thread_block_map import AddThreadBlockMap
from dace.transformation.dataflow.add_warp_map import AddWarpMap
import cupy

def test_vector_copy_with_warps():
    N = dace.symbol("N")

    @dace.program
    def dace_vcopy_kernel(
                    A: dace.float32[N] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.float32[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
            A[i] = B[i]

    sdfg = dace_vcopy_kernel.to_sdfg()
    sdfg.simplify()
    sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)

    sdfg.apply_transformations(
        AddThreadBlockMap,
        options={"thread_block_size_x":32, "thread_block_size_y":1},
        validate=True,
        validate_all=True
    )
    sdfg.apply_transformations(
        AddWarpMap,
        options={"warp_dims":[32]},
        validate=True,
        validate_all=True
    )

    N_value = (1 * 1024**2)

    A = cupy.ones(N_value, dtype=cupy.float32)
    B = cupy.full(N_value, 2, dtype=cupy.float32)

    sdfg(A=A, B=B, N=N_value)

    assert cupy.all(A == 2)

if __name__ == '__main__':
    test_vector_copy_with_warps()