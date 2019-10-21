import dace
from dace.memlet import Memlet
import dacelibs.blas as blas
import numpy as np

dtype = dace.float32

###############################################################################

def run(compiled_sdfg):

    x = dace.ndarray([n], dtype=dtype)
    y = dace.ndarray([n], dtype=dtype)
    result = dace.ndarray([1], dtype=dtype)

    n.set(32)
    x[:] = -3
    y[:] = 8

    compiled_sdfg(x=x, y=y, result=result, n=n.get())

    diff = abs(result[0] - 32 * 5)
    if diff >= 1e-6 * 32 * 5:
        print("Unexpected result returned from dot product.")
        sys.exit(1)

###############################################################################

n = dace.symbol("n")

sdfg = dace.SDFG("dot_product")
state = sdfg.add_state("dataflow")

sdfg.add_array("x", [n], dtype)
sdfg.add_array("y", [n], dtype)
sdfg.add_array("result", [1], dtype)

x = state.add_read("x")
y = state.add_read("y")
result = state.add_write("result")

dot_node = blas.nodes.Dot(dtype)

state.add_memlet_path(
    x, dot_node, dst_conn="x", memlet=Memlet.simple(x, "0:n"))
state.add_memlet_path(
    y, dot_node, dst_conn="y", memlet=Memlet.simple(y, "0:n"))
state.add_memlet_path(
    dot_node, result, src_conn="result", memlet=Memlet.simple(result, "0"))

sdot = sdfg.compile()

run(sdot)

###############################################################################
