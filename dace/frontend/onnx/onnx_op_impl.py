from typing import Any, Callable, Tuple, List, Union

import dace
import numpy as np
from dace.frontend.onnx.onnx_op_repository import onnx_op, onnx_op_with_name, onnx_op_program
from dace.graph.nodes import AccessNode
from dace.libraries.blas.nodes import Gemm
from dace.memlet import Memlet


@onnx_op
def Transpose(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
              outputs: List[str], *, perm):

    A = inputs[0]
    B = outputs[0]
    A_arr = sdfg.arrays[A]

    if not len(A_arr.shape) == 2 or (perm is not None and perm != [1, 0]):
        raise NotImplementedError

    _, B_arr = sdfg.add_transient(B, [A_arr.shape[1], A_arr.shape[0]],
                                  A_arr.dtype)

    @dace.program
    def transpose(A_p: A_arr, B_p: B_arr):
        B_p[:] = np.transpose(A_p)

    nsdfg = transpose.to_sdfg()
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A_p'}, {'B_p'})
    read_A = state.add_read(A)
    write_B = state.add_write(B)

    state.add_memlet_path(read_A,
                          nsdfg_node,
                          dst_conn='A_p',
                          memlet=Memlet.from_array(A, A_arr))
    state.add_memlet_path(nsdfg_node,
                          write_B,
                          src_conn='B_p',
                          memlet=Memlet.from_array(B, B_arr))


@onnx_op_with_name("Gemm")
def GemmOp(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
           outputs: List[str], *, alpha, beta, transA, transB):

    A, B = inputs[0:2]
    C = None
    if len(inputs) == 3:
        C = inputs[2]

    Y = outputs[0]

    A_arr = sdfg.arrays[A]
    B_arr = sdfg.arrays[B]
    dtype = A_arr.dtype
    if C is not None:
        C_arr = sdfg.arrays[C]

    rA = state.add_read(A)
    rB = state.add_read(B)
    if C is not None:
        rC = state.add_read(C)

    tasklet = Gemm('_Gemm_',
                   dtype,
                   transA=bool(transA),
                   transB=bool(transB),
                   alpha=alpha,
                   beta=beta)
    state.add_node(tasklet)
    if C is None:
        tasklet.remove_in_connector("_c")

    state.add_edge(rA, None, tasklet, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, tasklet, '_b', dace.Memlet.from_array(B, B_arr))
    if C is not None:
        state.add_edge(rC, None, tasklet, '_c',
                       dace.Memlet.from_array(C, C_arr))

    Y_shape = tasklet.infer_output_shapes(sdfg, state)["_y"]

    _, Y_arr = sdfg.add_transient(Y, Y_shape, dtype)
    wY = state.add_write(Y)
    state.add_edge(tasklet, '_y', wY, None, dace.Memlet.from_array(Y, Y_arr))


@onnx_op_program
def Add(A: "D[M]", B: "D[M]", C: "D[M]"):
    C[:] = A + B


@onnx_op_program
def MatMul(A: "D[M, K]", B: "D[K, N]", C: "D[M, N]"):
    C[:] = A @ B


@onnx_op_program
def Relu(X: "D[M]", Y: "D[M]"):
    Y[:] = elementwise(X, lambda x: max(x, __dtype_D(0)))
