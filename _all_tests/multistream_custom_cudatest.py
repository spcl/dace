# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np
import os
import pytest

# Create symbols
N = dp.symbol('N')

# Create a GPU SDFG with a custom C++ tasklet
sdfg = dp.SDFG('cublas_multistream_test')
state = sdfg.add_state()

# Add arrays
sdfg.add_array('A', [N, N], dtype=dp.float64)
sdfg.add_array('B', [N, N], dtype=dp.float64)
sdfg.add_array('C', [N, N], dtype=dp.float64)

# Add transient GPU arrays
sdfg.add_transient('gA', [N, N], dp.float64, dp.StorageType.GPU_Global)
sdfg.add_transient('gB', [N, N], dp.float64, dp.StorageType.GPU_Global)
sdfg.add_transient('gC', [N, N], dp.float64, dp.StorageType.GPU_Global)
sdfg.add_transient('gTmp', [N, N], dp.float64, dp.StorageType.GPU_Global)

# Add custom C++ tasklet to graph
tasklet = state.add_tasklet(
    # Tasklet name (can be arbitrary)
    name='gemm1',
    # Inputs and output names (will be obtained as raw pointers)
    inputs={'a', 'b'},
    outputs={'c'},
    # Custom code (on invocation)
    code='''
    double alpha = 1.0, beta = 0.0;
    cublasSetStream(handle, __dace_current_stream);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, 
                a, N, b, N, 
                &beta,
                c, N);
    ''',
    # Language (C++ in this case)
    language=dp.Language.CPP)

tasklet2 = state.add_tasklet(name='gemm2',
                             inputs={'a', 'b'},
                             outputs={'c'},
                             code='''
    double alpha = 1.0, beta = 0.0;
    cublasSetStream(handle, __dace_current_stream);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, 
                a, N, b, N, 
                &beta,
                c, N);
    ''',
                             language=dp.Language.CPP)

# Add CPU arrays, GPU arrays, and connect to tasklet
A = state.add_read('A')
B = state.add_read('B')
C = state.add_write('C')

gA = state.add_access('gA')
gB = state.add_access('gB')
gTmp = state.add_access('gTmp')
gC = state.add_access('gC')

# Memlets cover all data
state.add_edge(gA, None, tasklet2, 'a', dp.Memlet.simple('gA', '0:N, 0:N'))
state.add_edge(gA, None, tasklet, 'a', dp.Memlet.simple('gA', '0:N, 0:N'))
state.add_edge(gB, None, tasklet, 'b', dp.Memlet.simple('gB', '0:N, 0:N'))
state.add_edge(tasklet, 'c', gTmp, None, dp.Memlet.simple('gTmp', '0:N, 0:N'))

state.add_edge(gTmp, None, tasklet2, 'b', dp.Memlet.simple('gTmp', '0:N, 0:N'))
state.add_edge(tasklet2, 'c', gC, None, dp.Memlet.simple('gC', '0:N, 0:N'))

# Between two arrays we use a convenience function, `add_nedge`, which is
# short for "no-connector edge", i.e., `add_edge(u, None, v, None, memlet)`.
state.add_nedge(A, gA, dp.Memlet.simple('gA', '0:N, 0:N'))
state.add_nedge(B, gB, dp.Memlet.simple('gB', '0:N, 0:N'))
state.add_nedge(gC, C, dp.Memlet.simple('gC', '0:N, 0:N'))

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


@pytest.mark.gpu
def test_multistream_custom():
    N = 27
    # First, add libraries to link (CUBLAS) to configuration
    oldconf = dp.Config.get('compiler', 'cpu', 'libs')
    if os.name == 'nt':
        dp.Config.append('compiler', 'cpu', 'libs', value='cublas.lib')
    else:
        dp.Config.append('compiler', 'cpu', 'libs', value='libcublas.so')

    # Initialize arrays. We are using column-major order to support CUBLAS!
    A = np.ndarray([N, N], dtype=np.float64, order='F')
    B = np.ndarray([N, N], dtype=np.float64, order='F')
    C = np.ndarray([N, N], dtype=np.float64, order='F')

    A[:] = np.random.rand(N, N)
    B[:] = np.random.rand(N, N)
    C[:] = np.random.rand(N, N)

    out_ref = A @ A @ B

    # We can safely call numpy with arrays allocated on the CPU, since they
    # will be copied.
    sdfg(A=A, B=B, C=C, N=N)

    # Revert config change
    dp.Config.set('compiler', 'cpu', 'libs', value=oldconf)

    diff = np.linalg.norm(C - out_ref)
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test_multistream_custom()
