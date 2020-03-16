import dace
from dace.memlet import Memlet
from dace.codegen.compiler import CompilerConfigurationError, CompilationError
import dace.libraries.blas as blas
import numpy as np
import sys
import warnings

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):
    m = dace.symbol("m")
    n = dace.symbol("n")
    k = dace.symbol("k")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("cublasgemm_{}".format(dtype))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x" + suffix, [m, k],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("y" + suffix, [k, n],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("result" + suffix, [m, n],
                   dtype,
                   storage=storage,
                   transient=transient)

    x = state.add_read("x" + suffix)
    y = state.add_read("y" + suffix)
    result = state.add_write("result" + suffix)

    node = blas.nodes.matmul.MatMul("matmul", dtype)
    node.implementation = implementation

    state.add_memlet_path(x,
                          node,
                          dst_conn="_a",
                          memlet=Memlet.simple(x, "0:m, 0:k"))
    state.add_memlet_path(y,
                          node,
                          dst_conn="_b",
                          memlet=Memlet.simple(y, "0:k, 0:n"))
    # TODO: remove -1 once this no longer triggers a write in the codegen.
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
                                   memlet=Memlet.simple(x_host,
                                                        "0:m, 0:k"))
        init_state.add_memlet_path(y_host,
                                   y_device,
                                   memlet=Memlet.simple(y_host,
                                                        "0:k, 0:n"))

        finalize_state = sdfg.add_state("copy_to_host")
        sdfg.add_edge(state, finalize_state, dace.InterstateEdge())

        result_device = finalize_state.add_write("result" + suffix)
        result_host = finalize_state.add_read("result")
        finalize_state.add_memlet_path(result_device,
                                       result_host,
                                       memlet=Memlet.simple(result_device,
                                                            "0:m, 0:n"))

    return sdfg


###############################################################################


def _test_matmul(implementation, dtype, sdfg):
    try:
        csdfg = sdfg.compile()
    except (CompilerConfigurationError, CompilationError):
        warnings.warn(
            'Configuration/compilation failed, library missing or '
            'misconfigured, skipping test for {}.'.format(implementation))
        return

    m, n, k = 32, 31, 30

    x = np.ndarray([m, k], dtype=dtype)
    y = np.ndarray([k, n], dtype=dtype)
    result = np.ndarray([m, n], dtype=dtype)

    x[:] = 2.5
    y[:] = 2
    result[:] = 0

    csdfg(x=x, y=y, result=result, m=m, n=n, k=k)

    ref = np.dot(x, y)

    diff = np.linalg.norm(ref - result)
    if diff >= 1e-6:
        print("Unexpected result returned from dot product: "
              "diff %f" % diff)
        sys.exit(1)

    print("Test ran successfully for {}.".format(implementation))


def test_matmul():
    _test_matmul(
        "32-bit cuBLAS", np.float32,
        make_sdfg("cuBLAS", dace.float32, dace.StorageType.GPU_Global))
    _test_matmul(
        "64-bit cuBLAS", np.float64,
        make_sdfg("cuBLAS", dace.float64, dace.StorageType.GPU_Global))


###############################################################################

if __name__ == "__main__":
    test_matmul()
###############################################################################
