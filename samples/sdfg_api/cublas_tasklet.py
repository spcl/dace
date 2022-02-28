# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Example that defines a CUBLAS C++ tasklet. """
import dace as dc
import numpy as np
import os

# First, add libraries to link (CUBLAS) to configuration
if os.name == 'nt':
    dc.Config.append('compiler', 'cpu', 'libs', value='cublas.lib')
else:
    dc.Config.append('compiler', 'cpu', 'libs', value='libcublas.so')
######################################################################

# Create symbols
M = dc.symbol('M')
K = dc.symbol('K')
N = dc.symbol('N')

# Create a GPU SDFG with a custom C++ tasklet
sdfg = dc.SDFG('cublastest')
state = sdfg.add_state()

# Add arrays
sdfg.add_array('A', [M, K], dtype=dc.float64)
sdfg.add_array('B', [K, N], dtype=dc.float64)
sdfg.add_array('C', [M, N], dtype=dc.float64)

# Add transient GPU arrays
sdfg.add_transient('gA', [M, K], dc.float64, dc.StorageType.GPU_Global)
sdfg.add_transient('gB', [K, N], dc.float64, dc.StorageType.GPU_Global)
sdfg.add_transient('gC', [M, N], dc.float64, dc.StorageType.GPU_Global)

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
    language=dc.Language.CPP)

# Add CPU arrays, GPU arrays, and connect to tasklet
A = state.add_read('A')
B = state.add_read('B')
C = state.add_write('C')
gA = state.add_access('gA')
gB = state.add_access('gB')
gC = state.add_access('gC')

# Memlets can be constructed with a single string (for convenience, not recommended), or with keyword arguments
# that specify the data and subset as objects
state.add_edge(gA, None, tasklet, 'a', dc.Memlet('gA[0:M, 0:K]'))
state.add_edge(gB, None, tasklet, 'b', dc.Memlet('gB[0:K, 0:N]'))
state.add_edge(tasklet, 'c', gC, None, dc.Memlet('gC[0:M, 0:N]'))
# NOTE: The memlets above cover the entire arrays because CUBLAS may access any address

# Between two arrays we use a convenience function, `add_nedge`, which is
# short for "no-connector edge", i.e., `add_edge(u, None, v, None, memlet)`.
state.add_nedge(A, gA, dc.Memlet(data='gA', subset='0:M, 0:K'))
state.add_nedge(B, gB, dc.Memlet(data='gB', subset='0:K, 0:N'))
state.add_nedge(gC, C, dc.Memlet(data='C', subset='0:M, 0:N'))

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
    M = 25
    K = 26
    N = 27

    # Initialize arrays. We are using column-major order to support CUBLAS!
    A = np.ndarray([M, K], dtype=np.float64, order='F')
    B = np.ndarray([K, N], dtype=np.float64, order='F')
    C = np.ndarray([M, N], dtype=np.float64, order='F')
    C_ref = np.ndarray([M, N], dtype=np.float64, order='F')

    A[:] = np.random.rand(M, K)
    B[:] = np.random.rand(K, N)
    C[:] = np.random.rand(M, N)

    C_ref[:] = A @ B

    # We can safely call numpy with arrays allocated on the CPU, since they
    # will be copied.
    sdfg(A=A, B=B, C=C, M=M, N=N, K=K)

    diff = np.linalg.norm(C - C_ref) / (M * N)
    print('Difference:', diff)

    if diff > 1e-5:
        print(A, '\n * \n', B)
        print('C output   :', C)
        print('C reference:', C_ref)

    exit(0 if diff <= 1e-5 else 1)
