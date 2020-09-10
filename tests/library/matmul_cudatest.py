# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
from dace.codegen.compiler import CompilerConfigurationError, CompilationError
import dace.libraries.blas as blas
import itertools
import numpy as np
import sys
import warnings

###############################################################################


def make_sdfg(implementation,
              dtype,
              storage=dace.StorageType.Default,
              data_layout='CCC'):
    m = dace.symbol("m")
    n = dace.symbol("n")
    k = dace.symbol("k")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("mm_{}_{}".format(dtype.type.__name__, data_layout))
    state = sdfg.add_state("dataflow")

    # Data layout is a 3-character string with either C (for row major)
    # or F (for column major) matrices for x, y, and z respectively.
    xstrides = (k, 1) if data_layout[0] == 'C' else (1, m)
    ystrides = (n, 1) if data_layout[1] == 'C' else (1, k)
    zstrides = (n, 1) if data_layout[2] == 'C' else (1, m)

    sdfg.add_array("x" + suffix, [m, k],
                   dtype,
                   storage=storage,
                   transient=transient,
                   strides=xstrides)
    sdfg.add_array("y" + suffix, [k, n],
                   dtype,
                   storage=storage,
                   transient=transient,
                   strides=ystrides)
    sdfg.add_array("result" + suffix, [m, n],
                   dtype,
                   storage=storage,
                   transient=transient,
                   strides=zstrides)

    x = state.add_read("x" + suffix)
    y = state.add_read("y" + suffix)
    result = state.add_write("result" + suffix)

    node = blas.nodes.matmul.MatMul("matmul", dtype)

    state.add_memlet_path(x,
                          node,
                          dst_conn="_a",
                          memlet=Memlet.simple(x, "0:m, 0:k"))
    state.add_memlet_path(y,
                          node,
                          dst_conn="_b",
                          memlet=Memlet.simple(y, "0:k, 0:n"))
    state.add_memlet_path(node,
                          result,
                          src_conn="_c",
                          memlet=Memlet.simple(result, "0:m, 0:n"))

    if storage != dace.StorageType.Default:
        sdfg.add_array("x", [m, k], dtype)
        sdfg.add_array("y", [k, n], dtype)
        sdfg.add_array("result", [m, n], dtype)

        init_state = sdfg.add_state("copy_to_device")
        sdfg.add_edge(init_state, state, dace.InterstateEdge())

        x_host = init_state.add_read("x")
        y_host = init_state.add_read("y")
        x_device = init_state.add_write("x" + suffix)
        y_device = init_state.add_write("y" + suffix)
        init_state.add_memlet_path(x_host,
                                   x_device,
                                   memlet=Memlet.simple(x_host, "0:m, 0:k"))
        init_state.add_memlet_path(y_host,
                                   y_device,
                                   memlet=Memlet.simple(y_host, "0:k, 0:n"))

        finalize_state = sdfg.add_state("copy_to_host")
        sdfg.add_edge(state, finalize_state, dace.InterstateEdge())

        result_device = finalize_state.add_write("result" + suffix)
        result_host = finalize_state.add_read("result")
        finalize_state.add_memlet_path(result_device,
                                       result_host,
                                       memlet=Memlet.simple(
                                           result_device, "0:m, 0:n"))

    return sdfg


###############################################################################


def _test_matmul(implementation,
                 dtype,
                 impl_name,
                 storage,
                 data_layout='CCC',
                 eps=1e-4):
    sdfg = make_sdfg(impl_name, dtype, storage, data_layout)
    csdfg = sdfg.compile()

    m, n, k = 32, 31, 30

    x = np.ndarray([m, k], dtype=dtype.type, order=data_layout[0])
    y = np.ndarray([k, n], dtype=dtype.type, order=data_layout[1])
    z = np.ndarray([m, n], dtype=dtype.type, order=data_layout[2])

    x[:] = np.random.rand(m, k)
    y[:] = np.random.rand(k, n)
    z[:] = 0

    csdfg(x=x, y=y, result=z, m=m, n=n, k=k)

    ref = np.dot(x, y)

    if dtype == dace.float16 and np.linalg.norm(z) == 0:
        print('No computation performed, half-precision probably not '
              'supported, skipping test.')
        return

    diff = np.linalg.norm(ref - z)
    if diff >= eps:
        print("Unexpected result returned from matrix multiplication: "
              "diff %f" % diff)
        sys.exit(1)

    print("Test ran successfully for {}.".format(implementation))


def test_types():
    # Try different data types
    _test_matmul('cuBLAS double',
                 dace.float64,
                 'cuBLAS',
                 dace.StorageType.GPU_Global,
                 eps=1e-6)
    _test_matmul('cuBLAS half',
                 dace.float16,
                 'cuBLAS',
                 dace.StorageType.GPU_Global,
                 eps=1)
    _test_matmul('cuBLAS scmplx', dace.complex64, 'cuBLAS',
                 dace.StorageType.GPU_Global)
    _test_matmul('cuBLAS dcmplx',
                 dace.complex128,
                 'cuBLAS',
                 dace.StorageType.GPU_Global,
                 eps=1e-6)


def test_layouts():
    # Try all data layouts
    for dl in map(lambda t: ''.join(t),
                  itertools.product(*([['C', 'F']] * 3))):
        _test_matmul('cuBLAS float ' + dl,
                     dace.float32,
                     'cuBLAS',
                     dace.StorageType.GPU_Global,
                     data_layout=dl)


def test_batchmm():
    b, m, n, k = tuple(dace.symbol(k) for k in 'bmnk')

    @dace.program
    def bmmtest(A: dace.float64[b, m, k], B: dace.float64[b, k, n],
                C: dace.float64[b, m, n]):
        C[:] = A @ B

    sdfg = bmmtest.to_sdfg()
    sdfg.apply_gpu_transformations()
    csdfg = sdfg.compile()

    b, m, n, k = 3, 32, 31, 30

    x = np.random.rand(b, m, k)
    y = np.random.rand(b, k, n)
    z = np.zeros([b, m, n], np.float64)
    csdfg(A=x, B=y, C=z, b=b, m=m, n=n, k=k)

    ref = x @ y

    diff = np.linalg.norm(ref - z)
    print('Difference:', diff)
    assert diff < 1e-6


###############################################################################

if __name__ == '__main__':
    import os
    blas.default_implementation = "cuBLAS"
    try:
        test_batchmm()
        test_types()
        test_layouts()
    except SystemExit as ex:
        print('\n', flush=True)
        # Skip all teardown to avoid crashes affecting exit code
        os._exit(ex.code)
    os._exit(0)
###############################################################################
