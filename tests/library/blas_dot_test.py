# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
import dace.libraries.blas as blas
import numpy as np
import sys
import warnings

###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("dot_product_{}_{}".format(implementation, dtype))
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x" + suffix, [n],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("y" + suffix, [n],
                   dtype,
                   storage=storage,
                   transient=transient)
    sdfg.add_array("result" + suffix, [1],
                   dtype,
                   storage=storage,
                   transient=transient)

    x = state.add_read("x" + suffix)
    y = state.add_read("y" + suffix)
    result = state.add_write("result" + suffix)

    dot_node = blas.nodes.dot.Dot("dot")
    dot_node.implementation = implementation

    state.add_memlet_path(x,
                          dot_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(y,
                          dot_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y, "0:n", num_accesses=n))
    # TODO: remove -1 once this no longer triggers a write in the codegen.
    state.add_memlet_path(dot_node,
                          result,
                          src_conn="_result",
                          memlet=Memlet.simple(result, "0", num_accesses=-1))

    if storage != dace.StorageType.Default:

        sdfg.add_array("x", [n], dtype)
        sdfg.add_array("y", [n], dtype)
        sdfg.add_array("result", [1], dtype)

        init_state = sdfg.add_state("copy_to_device")
        sdfg.add_edge(init_state, state, dace.InterstateEdge())

        x_host = init_state.add_read("x")
        y_host = init_state.add_read("y")
        x_device = init_state.add_write("x" + suffix)
        y_device = init_state.add_write("y" + suffix)
        init_state.add_memlet_path(x_host,
                                   x_device,
                                   memlet=Memlet.simple(x_host,
                                                        "0:n",
                                                        num_accesses=n))
        init_state.add_memlet_path(y_host,
                                   y_device,
                                   memlet=Memlet.simple(y_host,
                                                        "0:n",
                                                        num_accesses=n))

        finalize_state = sdfg.add_state("copy_to_host")
        sdfg.add_edge(state, finalize_state, dace.InterstateEdge())

        result_device = finalize_state.add_write("result" + suffix)
        result_host = finalize_state.add_read("result")
        finalize_state.add_memlet_path(result_device,
                                       result_host,
                                       memlet=Memlet.simple(result_device,
                                                            "0",
                                                            num_accesses=1))

    return sdfg


###############################################################################


def _test_dot(implementation, dtype, sdfg):
    try:
        dot = sdfg.compile()
    except (CompilerConfigurationError, CompilationError):
        warnings.warn(
            'Configuration/compilation failed, library missing or '
            'misconfigured, skipping test for {}.'.format(implementation))
        return

    size = 32

    x = np.ndarray(size, dtype=dtype)
    y = np.ndarray(size, dtype=dtype)
    result = np.ndarray(1, dtype=dtype)

    x[:] = 2.5
    y[:] = 2
    result[0] = 0

    dot(x=x, y=y, result=result, n=size)

    ref = np.dot(x, y)

    diff = abs(result[0] - ref)
    assert diff < 1e-6 * ref

    print("Test ran successfully for {}.".format(implementation))


def test_dot():
    _test_dot("32-bit pure SDFG", np.float32, make_sdfg("pure", dace.float32))
    _test_dot("64-bit pure SDFG", np.float64, make_sdfg("pure", dace.float64))
    _test_dot("32-bit MKL", np.float32, make_sdfg("MKL", dace.float32))
    _test_dot("64-bit MKL", np.float64, make_sdfg("MKL", dace.float64))
    _test_dot("32-bit cuBLAS", np.float32,
              make_sdfg("cuBLAS", dace.float32, dace.StorageType.GPU_Global))
    _test_dot("64-bit cuBLAS", np.float64,
              make_sdfg("cuBLAS", dace.float64, dace.StorageType.GPU_Global))


###############################################################################

if __name__ == "__main__":
    test_dot()
###############################################################################
