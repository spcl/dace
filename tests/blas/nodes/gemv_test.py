import numpy as np
import argparse
import scipy
import dace
from dace.memlet import Memlet

import dace.libraries.blas as blas


# ---------- ----------
# Pure graph program (CPU)
# ---------- ----------
def pure_graph(dtype, transposed, expansion, veclen):
    n = dace.symbol("n")
    m = dace.symbol("m")

    sdfg = dace.SDFG("gemv")

    if veclen != 1:
        raise NotImplementedError("Vectorization not implemented for pure.")

    # alpha and beta are symbols
    sdfg.add_symbol("alpha", dtype)
    sdfg.add_symbol("beta", dtype)

    state = sdfg.add_state("gemv_compute")

    A_rows = n
    A_cols = m
    x_size = n if transposed else m
    y_size = m if transposed else n

    sdfg.add_array('A', shape=[A_rows, A_cols], dtype=dtype)
    sdfg.add_array('x', shape=[x_size], dtype=dtype)
    sdfg.add_array('y', shape=[y_size], dtype=dtype)

    A = state.add_read("A")
    x = state.add_read("x")
    result = state.add_write("y")

    gemv_node = blas.Gemv("gemv",
                          dtype=dace.float32,
                          transA=transposed)
    gemv_node.implementation = expansion

    state.add_memlet_path(A,
                          gemv_node,
                          dst_conn="_A",
                          memlet=Memlet.simple(
                              A, "0:{}, 0:{}".format(A_rows, A_cols)))
    state.add_memlet_path(x,
                          gemv_node,
                          dst_conn="_x",
                          memlet=Memlet.simple(x,
                                               "0:{}".format(x_size)))
    y = state.add_read("y")
    state.add_memlet_path(y,
                          gemv_node,
                          dst_conn="_y",
                          memlet=Memlet.simple(y,
                                               "0:{}".format(y_size)))
    state.add_memlet_path(gemv_node,
                          result,
                          src_conn="_y",
                          memlet=Memlet.simple(result, "0:{}".format(y_size)))
    return sdfg


# ---------- ----------
# Intel FPGA graph
# ---------- ----------
def fpga_graph(dtype, transposed, expansion, vec_width):

    vtype = dace.vector(dtype, vec_width)

    sdfg = dace.SDFG("gemv_fpga_test")

    n = dace.symbol("n")
    m = dace.symbol("m")

    m /= vec_width

    # alpha and beta are symbols
    sdfg.add_symbol("alpha", dtype)
    sdfg.add_symbol("beta", dtype)

    A_rows = n
    A_cols = m
    x_size = n if transposed else m
    y_size = m if transposed else n

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")

    sdfg.add_array("A", shape=[n, m], dtype=vtype)
    sdfg.add_array("x", shape=[x_size], dtype=vtype if not transposed else dtype)
    sdfg.add_array("y", shape=[y_size], dtype=vtype if transposed else dtype)

    in_host_A = copy_in_state.add_read("A")
    in_host_x = copy_in_state.add_read("x")
    in_host_y = copy_in_state.add_read("y")

    sdfg.add_array("device_A", shape=[A_rows, A_cols], dtype=vtype, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_x", shape=[x_size], dtype=vtype if not transposed else dtype, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)
    sdfg.add_array("device_y", shape=[y_size], dtype=vtype if transposed else dtype, storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    in_device_A = copy_in_state.add_write("device_A")
    in_device_x = copy_in_state.add_write("device_x")
    in_device_y = copy_in_state.add_write("device_y")

    copy_in_state.add_memlet_path(
        in_host_A, in_device_A,
        memlet=Memlet.simple(in_host_A, "0:{}, 0:{}".format(A_rows, A_cols))
    )
    copy_in_state.add_memlet_path(
        in_host_x, in_device_x,
        memlet=Memlet.simple(in_host_x, "0:{}".format(x_size))
    )
    copy_in_state.add_memlet_path(
        in_host_y, in_device_y,
        memlet=Memlet.simple(in_host_y, "0:{}".format(y_size))
    )

    ###########################################################################
    # Copy data from FPGA

    copy_out_state = sdfg.add_state("copy_to_host")

    out_device = copy_out_state.add_read("device_y")
    out_host = copy_out_state.add_write("y")

    copy_out_state.add_memlet_path(
        out_device, out_host,
        memlet=Memlet.simple(out_host, "0:{}".format(y_size))
    )

    ########################################################################
    # FPGA State

    fpga_state = sdfg.add_state("gemv_computation")
    # This should not be an FPGA kernel, rather the gemv_expanded nested SDFG should

    A = fpga_state.add_read("device_A")
    x = fpga_state.add_read("device_x")
    y_out = fpga_state.add_write("device_y")

    gemv_node = blas.Gemv("gemv",
                          dtype=dace.float32,
                          transA=transposed,
                          alpha=alpha,
                          beta=beta)
    gemv_node.implementation = expansion

    fpga_state.add_memlet_path(A,
                               gemv_node,
                               dst_conn="_A",
                               memlet=Memlet.simple(A, "0:{}, 0:{}".format(n, m)))

    fpga_state.add_memlet_path(x,
                               gemv_node,
                               dst_conn="_x",
                               memlet=Memlet.simple(x, "0:{}".format("{}".format(x_size))))

    if beta != 0:
        y_in = fpga_state.add_read("device_y")
        fpga_state.add_memlet_path(y_in,
                                   gemv_node,
                                   dst_conn="_y",
                                   memlet=Memlet.simple(y_in, "0:{}".format(y_size)))
    fpga_state.add_memlet_path(gemv_node,
                               y_out,
                               src_conn="_y",
                               memlet=Memlet.simple(y_out, "0:{}".format(y_size)))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state,
                  dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state,
                  dace.sdfg.sdfg.InterstateEdge())

    gemv_node.expand(sdfg, fpga_state, tile_size_x=16, tile_size_y=16)

    sdfg.fill_scope_connectors()
    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=512)
    parser.add_argument("M", type=int, nargs="?", default=512)
    parser.add_argument("alpha", type=int, nargs="?", default=1)
    parser.add_argument("beta", type=int, nargs="?", default=0)
    parser.add_argument("--transposed",
                        action="store_true",
                        default=False,
                        help="Compute GEMV with transposed matrix")
    parser.add_argument("--target", dest="target", default="pure")
    parser.add_argument("--vectorize", dest="vectorize", default=1, type=int)

    args = parser.parse_args()
    n = args.N
    m = args.M
    alpha = args.alpha
    beta = args.beta
    transposed = args.transposed
    if args.target == "pure":
        sdfg = pure_graph(dace.float32, transposed, "pure", args.vectorize)
    elif args.target == "intel_fpga":
        sdfg = fpga_graph(dace.float32, transposed, "IntelFPGA", args.vectorize)
    elif args.target == "tiles_by_column":
        if not transposed and args.vectorize > 1:
            raise NotImplementedError(
                "Non-transposed vectorized tile-by-column NYI.")
        sdfg = fpga_graph(dace.float32, transposed, "TilesByColumn", args.vectorize)
    else:
        print("Unsupported target")
        exit(-1)

    A = np.random.rand(n, m).astype(np.float32)
    x = np.random.rand(n if transposed else m).astype(np.float32)
    y = np.random.rand(m if transposed else n).astype(np.float32)

    y_copy = np.copy(y)

    sdfg(A=A, x=x, y=y, n=n, m=m, alpha=alpha, beta=beta)

    ref = scipy.linalg.blas.sgemv(alpha, A, x, beta, y_copy, trans=transposed)

    diff = np.linalg.norm(y - ref) / (m if transposed else n)
    if diff >= 1e-5:
        print("Error")
    else:
        print("Ok")
