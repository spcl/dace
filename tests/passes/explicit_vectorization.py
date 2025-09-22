import dace
from dace.transformation.passes.explicit_vectorization import ExplicitVectorizationPipelineGPU

N = dace.symbol('N')




@dace.program
def vadd(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        B[i, j] = 3 * B[i, j] + 2.0

def test_simple():
    sdfg = vadd.to_sdfg()
    sdfg.validate()
    ExplicitVectorizationPipelineGPU(vector_width=4).apply_pass(sdfg, {})
    sdfg.save("vectorized_vadd.sdfg")
    sdfg.validate()

if __name__ == "__main__":
    test_simple()