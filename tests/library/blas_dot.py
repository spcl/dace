import dace
from dace.memlet import Memlet
import dacelibs.blas as blas
import numpy as np
import sys

n = dace.symbol("n")

###############################################################################


def test_dot(implementation, dtype):

    sdfg = dace.SDFG("dot_product")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x", [n], dtype)
    sdfg.add_array("y", [n], dtype)
    sdfg.add_array("result", [1], dtype)

    x = state.add_read("x")
    y = state.add_read("y")
    result = state.add_write("result")

    dot_node = blas.nodes.dot.Dot("dot", dtype)
    dot_node.implementation = implementation

    state.add_memlet_path(
        x,
        dot_node,
        dst_conn="_x",
        memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(
        y,
        dot_node,
        dst_conn="_y",
        memlet=Memlet.simple(y, "0:n", num_accesses=n))
    state.add_memlet_path(
        dot_node,
        result,
        src_conn="_result",
        memlet=Memlet.simple(result, "0", num_accesses=1))

    dot = sdfg.compile()

    x = dace.ndarray([n], dtype=dtype)
    y = dace.ndarray([n], dtype=dtype)
    result = dace.ndarray([1], dtype=dtype)

    n.set(32)
    x[:] = 2.5
    y[:] = 2

    dot(x=x, y=y, result=result, n=n.get())

    ref = np.dot(x, y)

    diff = abs(result[0] - ref)
    if diff >= 1e-6 * ref:
        print("Unexpected result returned from dot product.")
        sys.exit(1)

    print("Test ran successfully for implementation \"" + implementation +
          "\" for " + str(dtype))


###############################################################################

# test_dot("cuBLAS", dace.float32)
# test_dot("cuBLAS", dace.float64)
test_dot("pure", dace.float32)
test_dot("pure", dace.float64)
test_dot("MKL", dace.float32)
test_dot("MKL", dace.float64)

###############################################################################
