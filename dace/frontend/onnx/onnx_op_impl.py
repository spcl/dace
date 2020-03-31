from typing import Any, Callable, Tuple, List, Union

import dace
import numpy as np
from dace.frontend.onnx.onnx_op_repository import onnx_op, onnx_op_with_name
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


@onnx_op
def MatMul(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
           outputs: List[str]):

    A, B = inputs
    C = outputs[0]

    A_arr = sdfg.arrays[A]
    B_arr = sdfg.arrays[B]

    # TODO onnx spec states that we should support n-d matmul
    assert len(A_arr.shape) <= 2
    assert len(B_arr.shape) <= 2
    assert A_arr.shape[-1] == B_arr.shape[0]

    _, C_arr = sdfg.add_transient(C, [A_arr.shape[0], B_arr.shape[-1]],
                                  A_arr.dtype)

    @dace.program
    def matmul(A_p: A_arr, B_p: B_arr, C_p: C_arr):
        C_p[:] = A_p @ B_p

    nsdfg = matmul.to_sdfg()
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A_p', 'B_p'}, {'C_p'})
    read_A = state.add_read(A)
    read_B = state.add_read(B)
    write_C = state.add_write(C)

    state.add_memlet_path(read_A,
                          nsdfg_node,
                          dst_conn='A_p',
                          memlet=Memlet.from_array(A, A_arr))
    state.add_memlet_path(read_B,
                          nsdfg_node,
                          dst_conn='B_p',
                          memlet=Memlet.from_array(B, B_arr))
    state.add_memlet_path(nsdfg_node,
                          write_C,
                          src_conn='C_p',
                          memlet=Memlet.from_array(C, C_arr))


@onnx_op
def Add(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
        outputs: List[str]):

    A, B = inputs
    C = outputs[0]

    A_arr = sdfg.arrays[A]
    B_arr = sdfg.arrays[B]
    _, C_arr = sdfg.add_transient(C, A_arr.shape, A_arr.dtype)

    @dace.program
    def add(A_p: A_arr, B_p: B_arr, C_p: C_arr):
        C_p[:] = A_p + B_p

    nsdfg = add.to_sdfg()
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'A_p', 'B_p'}, {'C_p'})
    read_A = state.add_read(A)
    read_B = state.add_read(B)
    write_C = state.add_write(C)

    state.add_memlet_path(read_A,
                          nsdfg_node,
                          dst_conn='A_p',
                          memlet=Memlet.from_array(A, A_arr))
    state.add_memlet_path(read_B,
                          nsdfg_node,
                          dst_conn='B_p',
                          memlet=Memlet.from_array(B, B_arr))
    state.add_memlet_path(nsdfg_node,
                          write_C,
                          src_conn='C_p',
                          memlet=Memlet.from_array(C, C_arr))


@onnx_op
def Relu(sdfg: dace.SDFG, state: dace.SDFGState, inputs: List[str],
         outputs: List[str]):
    x = sdfg.arrays[inputs[0]]

    sdfg.add_transient(outputs[0], x.shape, x.dtype)

    state.add_mapped_tasklet(
        name="_Relu_",
        map_ranges={
            '__i{}'.format(i): '0:{}'.format(n)
            for i, n in enumerate(x.shape)
        },
        inputs={
            '__inp':
            Memlet.simple(
                inputs[0],
                ','.join(['__i{}'.format(i) for i in range(len(x.shape))]))
        },
        code='__out = max(dace.{}(0), __inp)'.format(x.dtype.to_string()),
        outputs={
            '__out':
            Memlet.simple(
                outputs[0],
                ','.join(['__i{}'.format(i) for i in range(len(x.shape))]))
        },
        external_edges=True)
