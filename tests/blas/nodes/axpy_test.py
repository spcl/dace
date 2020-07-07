import dace
from dace.memlet import Memlet
import dace.libraries.blas as blas
import numpy as np
import scipy
from tqdm import tqdm

# # ---------- ----------
# # SETUP
# # ---------- ----------
# platform = "CPU"

# innerVecWidth = 1
# outerVecWidth = 1

# # ---------- ----------
# # GRAPH
# # ---------- ----------
# n = dace.symbol("n")
# a = dace.symbol("a")


# test_sdfg = dace.SDFG("saxpy_test")
# test_state = test_sdfg.add_state("test_state")


# test_sdfg.add_symbol(a.name, dace.float32)

# test_sdfg.add_array('x1', shape=[n], dtype=dace.float32)
# test_sdfg.add_array('y1', shape=[n], dtype=dace.float32)
# test_sdfg.add_array('z1', shape=[n], dtype=dace.float32)

# x_in = test_state.add_read('x1')
# y_in = test_state.add_read('y1')
# z_out = test_state.add_write('z1')

# saxpy_node = blas.axpy.Axpy("saxpy", dace.float32 , vecWidth=innerVecWidth)
# saxpy_node.implementation = 'simple'

# # connect the blas node
# test_state.add_memlet_path(
#     x_in, saxpy_node,
#     dst_conn='_x',
#     memlet=Memlet.simple(x_in, "0:n", num_accesses=n, veclen=outerVecWidth)
# )
# test_state.add_memlet_path(
#     y_in, saxpy_node,
#     dst_conn='_y',
#     memlet=Memlet.simple(y_in, "0:n", num_accesses=n, veclen=outerVecWidth)
# )

# test_state.add_memlet_path(
#     saxpy_node, z_out,
#     src_conn='_res',
#     memlet=Memlet.simple(z_out, "0:n", num_accesses=n, veclen=outerVecWidth)
# )


# test_sdfg.expand_library_nodes()

    
# # ---------- ----------
# # COMPILE
# # ---------- ----------
# compiledSDFG = test_sdfg.compile(optimizer=False)


# # ---------- ----------
# # RUN
# # ---------- ----------
# testSize = 2048
# a = np.random.randint(testSize, size=testSize).astype(np.float32)
# b = np.random.randint(testSize, size=testSize).astype(np.float32)

# c = np.zeros(testSize).astype(np.float32)
# scaling = np.float32(2.0)

# b_ref = np.copy(b)

# print("\n----- COMPILE & RUN -----")
# compiledSDFG(x1=a, y1=b, a=scaling, z1=c, n=np.int32(testSize))

# ref_result = scipy.linalg.blas.saxpy(a, b_ref, a=scaling)

# print("\n----- VERIFY -----")
# print("Vec a:\t", a[0:10])
# print("Vec b:\t", b_ref[0:10])
# print("Scale:\t", scaling)
# print("Res:\t", c[0:10])
# print("Ref:\t", ref_result[0:10])


# ref_norm = np.linalg.norm(c - ref_result)
# passed = ref_norm < 0.001

# if not passed:
#     raise RuntimeError('AXPY simple implementation wrong test results')

# print("\n----- PASSED -----")




# ---------- ----------
# Ref result
# ---------- ----------
def reference_result(x_in, y_in, alpha):
    pass


# ---------- ----------
# CPU library graph program
# ---------- ----------
def cpu_graph(implementation):
    pass


def test_cpu(graph):
    pass


# ---------- ----------
# GPU Cuda graph program
# ---------- ----------
def gpu_graph():
    pass


def test_gpu(graph):
    pass


# ---------- ----------
# Pure graph program
# ---------- ----------
def pure_graph():
    pass


def test_pure(graph):
    pass


# ---------- ----------
# FPGA graph program
# ---------- ----------
def fpga_graph():
    pass


def test_fpga(graph):
    pass



