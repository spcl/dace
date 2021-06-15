# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Example that defines a CUBLAS C++ tasklet. """
import dace as dp
import numpy as np
import os

# First, add libraries to link (CUBLAS) to configuration
if os.name == 'nt':
    dp.Config.append('compiler', 'cpu', 'libs', value='cublas.lib')
else:
    dp.Config.append('compiler', 'cpu', 'libs', value='libcublas.so')
######################################################################

# Create symbols
M = dp.symbol('M')
K = dp.symbol('K')
N = dp.symbol('N')
M.set(25)
K.set(26)
N.set(27)

# Create a GPU SDFG with a custom C++ tasklet
sdfg = dp.SDFG('cublastest')
state = sdfg.add_state()

# Add arrays
sdfg.add_array('A', [M, K], dtype=dp.float64)
sdfg.add_array('B', [K, N], dtype=dp.float64)
sdfg.add_array('C', [M, N], dtype=dp.float64)

# Add transient GPU arrays
sdfg.add_transient('gA', [M, K], dp.float64, dp.StorageType.GPU_Global)
sdfg.add_transient('gB', [K, N], dp.float64, dp.StorageType.GPU_Global)
sdfg.add_transient('gC', [M, N], dp.float64, dp.StorageType.GPU_Global)

# Add custom C++ tasklet to graph
tasklet = state.add_tasklet(
    # Tasklet name (can be arbitrary)
    name='gemm',
    # Inputs and output names (will be obtained as raw pointers)
    inputs={'a', 'b'},
    outputs={'c'},
    # Custom code (on invocation)
    code='''
    // Set the current stream to match DaCe (for correct synchronization)
    cublasSetStream(handle, __dace_current_stream);
    
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha, 
                a, M, b, K, 
                &beta,
                c, M);
    ''',
    # Language (C++ in this case)
    language=dp.Language.CPP)

# Add CPU arrays, GPU arrays, and connect to tasklet
A = state.add_read('A')
B = state.add_read('B')
C = state.add_write('C')
gA = state.add_access('gA')
gB = state.add_access('gB')
gC = state.add_access('gC')

# Memlets cover all data
state.add_edge(gA, None, tasklet, 'a', dp.Memlet.simple('gA', '0:M, 0:K'))
state.add_edge(gB, None, tasklet, 'b', dp.Memlet.simple('gB', '0:K, 0:N'))
state.add_edge(tasklet, 'c', gC, None, dp.Memlet.simple('gC', '0:M, 0:N'))

# Between two arrays we use a convenience function, `add_nedge`, which is
# short for "no-connector edge", i.e., `add_edge(u, None, v, None, memlet)`.
state.add_nedge(A, gA, dp.Memlet.simple('gA', '0:M, 0:K'))
state.add_nedge(B, gB, dp.Memlet.simple('gB', '0:K, 0:N'))
state.add_nedge(gC, C, dp.Memlet.simple('C', '0:M, 0:N'))

# Add CUBLAS initialization and teardown code
# Global code (top of file, can be used for includes and global variables)
sdfg.append_global_code('''#include <cublas_v2.h>
cublasHandle_t handle;''')
# Initialization code (called in __dace_init())
sdfg.append_init_code('cublasCreate(&handle);')
# Teardown code (called in __dace_exit())
sdfg.append_exit_code('cublasDestroy(handle);')

######################################################################

# Validate GPU SDFG
sdfg.validate()

######################################################################

if __name__ == '__main__':
    # Initialize arrays. We are using column-major order to support CUBLAS!
    A = np.ndarray([M.get(), K.get()], dtype=np.float64, order='F')
    B = np.ndarray([K.get(), N.get()], dtype=np.float64, order='F')
    C = np.ndarray([M.get(), N.get()], dtype=np.float64, order='F')
    C_ref = np.ndarray([M.get(), N.get()], dtype=np.float64, order='F')

    A[:] = np.random.rand(M.get(), K.get())
    B[:] = np.random.rand(K.get(), N.get())
    C[:] = np.random.rand(M.get(), N.get())

    C_ref[:] = A @ B

    # We can safely call numpy with arrays allocated on the CPU, since they
    # will be copied.
    sdfg(A=A, B=B, C=C, M=M, N=N, K=K)

    diff = np.linalg.norm(C - C_ref) / (M.get() * N.get())
    print('Difference:', diff)

    if diff > 1e-5:
        print(A, '\n * \n', B)
        print('C output   :', C)
        print('C reference:', C_ref)

    exit(0 if diff <= 1e-5 else 1)
