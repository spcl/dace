import dace
import random
import cupy as cp

from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp
from IPython.display import Code
from dace.config import Config


def test_1():

    vec_size = 66
    @dace.program
    def vector_copy1(A: dace.float64[vec_size] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[vec_size] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:vec_size] @ dace.dtypes.ScheduleType.GPU_Device:
                A[i] = B[i]

    sdfg = vector_copy1.to_sdfg()

    # Initialize random CUDA arrays
    A = cp.zeros(vec_size, dtype=cp.float64)  # Output array
    B = cp.random.rand(vec_size).astype(cp.float64)  # Random input array

    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 1: Vectors are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = vector_copy1.to_sdfg()
    sdfg(A=A, B=B)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 1: 1D vector copy simple':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 1: 1D vector copy simple':<70}\033[91m[FAILED]\033[0m")


def test_2():

    N = dace.symbol('N')
    @dace.program
    def vector_copy2(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
                A[i] = B[i]

    sdfg = vector_copy2.to_sdfg()

    n = random.randint(3, 100)
    # Initialize random CUDA arrays
    A = cp.zeros(n, dtype=cp.float64)  # Output array
    B = cp.random.rand(n).astype(cp.float64)  # Random input array

    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 2: Vectors are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = vector_copy2.to_sdfg()
    sdfg(A=A, B=B, N=n)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 2: 1D vector copy with symbolic size':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 2: 1D vector copy with symbolic size':<70}\033[91m[FAILED]\033[0m")


def test_3():
    @dace.program
    def vector_copy3(A: dace.float64[64] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[64] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:64:32] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                A[i + j] = B[i + j]

    sdfg = vector_copy3.to_sdfg()

    # Initialize random CUDA arrays
    A = cp.zeros(64, dtype=cp.float64)  # Output array
    B = cp.random.rand(64).astype(cp.float64)  # Random input array

    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 3: Vectors are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = vector_copy3.to_sdfg()
    sdfg(A=A, B=B)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 3: 1D vector copy with threadblocking':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 3: 1D vector copy with threadblocking':<70}\033[91m[FAILED]\033[0m")


def test_4():

    N = dace.symbol('N')

    @dace.program
    def vector_copy4(A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N:32] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:32] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                if i + j < N:
                    A[i + j] = B[i + j]
    
    n = random.randint(40, 150)
    # Initialize random CUDA arrays
    A = cp.zeros(n, dtype=cp.float64)  # Output array
    B = cp.random.rand(n).astype(cp.float64)  # Random input array


    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 4: Vectors are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = vector_copy4.to_sdfg()
    sdfg(A=A, B=B, N=n)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 4: 1D vector copy with threadblocking & smybolic size':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 4: 1D vector copy with threadblocking & smybolic size':<70}\033[91m[FAILED]\033[0m")


def test_5():
    @dace.program
    def matrix_copy1(A: dace.float64[64,64] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[64,64] @ dace.dtypes.StorageType.GPU_Global):
        for i, j in dace.map[0:64, 0:64] @ dace.dtypes.ScheduleType.GPU_Device:
                A[i][j] = B[i][j]
    # Preview SDFG
    sdfg = matrix_copy1.to_sdfg()
    

    # Initialize random CUDA arrays
    A = cp.zeros((64,64), dtype=cp.float64)  # Output array
    B = cp.random.rand(64,64).astype(cp.float64)  # Random input array


    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 5: Matrices are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = matrix_copy1.to_sdfg()
    sdfg(A=A, B=B)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 5: Simple Matrix Copy':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 5: Simple Matrix Copy':<70}\033[91m[FAILED]\033[0m")


def test_6():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def matrix_copy2(A: dace.float64[M,N] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[M,N] @ dace.dtypes.StorageType.GPU_Global):
        for i, j in dace.map[0:M, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
            A[i][j] = B[i][j]
    # Preview SDFG
    sdfg = matrix_copy2.to_sdfg()
    
    n = random.randint(40, 150)
    m = random.randint(40, 150)
    # Initialize random CUDA arrays
    A = cp.zeros((m,n), dtype=cp.float64)  # Output array
    B = cp.random.rand(m,n).astype(cp.float64)  # Random input array


    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 6: Matrices are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = matrix_copy2.to_sdfg()
    sdfg(A=A, B=B, M=m, N=n)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 6: Matrix Copy with symbolic sizes':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 6: Matrix Copy with symbolic sizes':<70}\033[91m[FAILED]\033[0m")


def test_7():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def matrix_copy3(A: dace.float64[M,N] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[M,N] @ dace.dtypes.StorageType.GPU_Global):
        for i, j in dace.map[0:M:32, 0:N:32] @ dace.dtypes.ScheduleType.GPU_Device:
            for ii, jj in dace.map[0:32, 0:32] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                if i + ii < M and j + jj < N:
                    A[i + ii, j + jj] = B[i + ii, j + jj]
    # Preview SDFG
    sdfg = matrix_copy3.to_sdfg()
    
    n = random.randint(40, 150)
    m = random.randint(40, 150)
    # Initialize random CUDA arrays
    A = cp.zeros((m,n), dtype=cp.float64)  # Output array
    B = cp.random.rand(m,n).astype(cp.float64)  # Random input array


    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 6: Matrices are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = matrix_copy3.to_sdfg()
    sdfg(A=A, B=B, M=m, N=n)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 7: Matrix Copy with threadblocking & symbolic sizes':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 7: Matrix Copy with threadblocking & symbolic sizes':<70}\033[91m[FAILED]\033[0m")


def test_8():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def matrix_copy3(A: dace.float64[M,N] @ dace.dtypes.StorageType.GPU_Global, B: dace.float64[M,N] @ dace.dtypes.StorageType.GPU_Global):
        for i, j in dace.map[0:M:32, 0:N:32] @ dace.dtypes.ScheduleType.GPU_Device:
            for ii, jj  in dace.map[0:32, 0:32] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                sB = dace.define_local([32,32], dace.float64, storage=dace.StorageType.GPU_Shared)
                sB[ii, jj] = B[i + ii, j + jj]
                A[i + ii, j + jj] = sB[ii, jj]


    # Preview SDFG
    sdfg = matrix_copy3.to_sdfg()
    
    n = random.randint(40, 150)
    m = random.randint(40, 150)
    # Initialize random CUDA arrays
    A = cp.zeros((m,n), dtype=cp.float64)  # Output array
    B = cp.random.rand(m,n).astype(cp.float64)  # Random input array


    equal_at_start = cp.all(A == B)
    if equal_at_start:
        print(f"{'Test 8: Matrices are equal at start. Test is skipped.':<70}\033[93m[WARNING]\033[0m")
        return

    sdfg = matrix_copy3.to_sdfg()
    sdfg(A=A, B=B, M=m, N=n)
    equal_at_end = cp.all(A == B)

    if equal_at_end:
        print(f"{'Test 8: Matrix Copy with shared memory':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 8: Matrix Copy with shared memory':<70}\033[91m[FAILED]\033[0m")


def test_9():
    
    N = dace.symbol('N')

    @dace.program
    def notskewed(A: dace.float32[N] @ dace.dtypes.StorageType.GPU_Global, 
                  B: dace.float32[N] @ dace.dtypes.StorageType.GPU_Global, 
                  C: dace.float32[N] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
            for j in dace.map[i:(i+32)] @ dace.ScheduleType.GPU_ThreadBlock:
                C[j] = A[j] + B[j]

    # Preview SDFG
    sdfg = notskewed.to_sdfg()
    
    n = random.randint(40, 150)
    # Initialize random CUDA arrays
    A = cp.random.rand(n).astype(cp.float32)  # Output array
    B = cp.random.rand(n).astype(cp.float32)  # Random input array
    C = cp.zeros((n), dtype=cp.float32)
    C_ref = cp.zeros((n), dtype=cp.float32)


    C_ref = A + B
    sdfg(A=A, B=B, C=C, N=n)


    if cp.all(C == C_ref):
        print(f"{'Test 9: Not skewed vadd3':<70}\033[92m[PASSED]\033[0m")
    else:
        print(f"{'Test 9: Not skewed vadd3':<70}\033[91m[FAILED]\033[0m")





def selected():
    test_1()
    test_4()
    test_5()  

def all():
    test_1()
    test_2()
    test_3() 
    test_4()
    test_5() 
    test_6()
    test_7()
    test_8()
    test_9()

if __name__ == '__main__':


    print("\n" + "="*80)
    print(f"Tests started: You are using the {Config.get('compiler', 'cuda', 'implementation')} CUDA implementation.")
    print("="*80 + "\n")

    all()

    print("\n" + "="*80)
    print(f"Tests ended.")
    print("="*80 + "\n")