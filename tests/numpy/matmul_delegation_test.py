import numpy as np
import dace

N, K, M = 24, 12, 48


@dace.program
def matmul_delegation_test(matrix0: dace.float32[N, K],
                           matrix1: dace.float32[K, M],
                           vector0: dace.float32[M], vector1: dace.float32[N],
                           result: dace.float32[1]):
    # GEMM -> GEMV -> dot product
    result[0] = ((matrix0 @ matrix1) @ vector0) @ vector1


if __name__ == '__main__':
    matrix0 = np.random.rand(N, K).astype(np.float32)
    matrix1 = np.random.rand(K, M).astype(np.float32)
    vector0 = np.random.rand(M).astype(np.float32)
    vector1 = np.random.rand(N).astype(np.float32)
    result = np.empty([1], dtype=np.float32)

    matmul_delegation_test(
        matrix0=matrix0,
        matrix1=matrix1,
        vector0=vector0,
        vector1=vector1,
        result=result)

    reference = ((matrix0 @ matrix1) @ vector0) @ vector1
    rel_error = (result - reference) / reference
    if rel_error > 1e-5:
        raise ValueError("Result mismatch: {} (expected {})".format(
            result, reference))
    else:
        print("Linear algebra multiplication delegation test verified.")
