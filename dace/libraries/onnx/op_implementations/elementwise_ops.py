# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Elementwise and mathematical ONNX operations.

This module contains pure implementations of elementwise mathematical operations including:
- Basic arithmetic: Add, Sub, Mul, Div, Pow
- Unary math functions: Log, Exp, Sqrt, Sin, Cos, Tanh, Erf, Neg, Reciprocal
- Activation functions: Relu, LeakyRelu, Sigmoid, Softplus
- Utility operations: Clip

All operations support broadcasting where applicable.
"""

import copy
import typing

import dace
import numpy as np
from dace import SDFG, SDFGState
from dace.sdfg.nodes import Node

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.common import broadcast_indices
from dace.libraries.onnx.op_implementations.utils import (op_implementation, out_desc_with_name, program_for_node,
                                                          python_pure_op_implementation)
from dace.sdfg.utils import in_desc_with_name, in_edge_with_name, out_desc_with_name
from dace.transformation.onnx.replacement import onnx_constant_or_none

# ============================================================================
# Unary Mathematical Operations
# ============================================================================


@python_pure_op_implementation
def Log(input, output):
    """ONNX Log operation implementation.

    Computes the natural logarithm of the input tensor element-wise.

    :param input: Input tensor of any numeric type.
    :param output: Output tensor with the same shape and type as input.
    """
    output[:] = np.log(input)


@python_pure_op_implementation
def Exp(input, output):
    """ONNX Exp operation implementation.

    Computes the exponential of the input tensor element-wise.

    :param input: Input tensor of any numeric type.
    :param output: Output tensor with the same shape and type as input.
    """
    output[:] = np.exp(input)


@python_pure_op_implementation
def Sqrt(X, Y):
    """ONNX Sqrt operation implementation.

    Computes the square root of the input tensor element-wise.

    :param X: Input tensor of any numeric type.
    :param Y: Output tensor with the same shape and type as X.
    """
    Y[:] = dace.elementwise(lambda x: sqrt(x), X)


@python_pure_op_implementation
def Sin(input, output):
    output[:] = np.sin(input)


@python_pure_op_implementation
def Cos(input, output):
    output[:] = np.cos(input)


@python_pure_op_implementation
def Tanh(input, output):
    output[:] = dace.elementwise(lambda x: tanh(x), input)


@python_pure_op_implementation
def Erf(input, output):
    output[:] = dace.elementwise(lambda x: erf(x), input)


@python_pure_op_implementation
def Neg(X, Y):
    Y[:] = -X


@python_pure_op_implementation(string=lambda X: "lambda x: dace.{}(1) / x".format(X.dtype.to_string()))
def Reciprocal(X, Y):
    Y[:] = dace.elementwise(string, X)


@python_pure_op_implementation
def Softplus(X, Y):
    Y[:] = np.log(1 + np.exp(X))


@python_pure_op_implementation(dtype=lambda X: X.dtype)
def Sigmoid(X, Y):
    Y[:] = dace.elementwise(lambda x: dtype(1) / (dtype(1) + exp(-x)), X)


# ============================================================================
# Binary Arithmetic Operations
# ============================================================================


@op_implementation(op="Pow", name="pure")
class PurePow(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # Special case for constant exponents
        y_value = None
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "Y").src.data in sdfg._parent_onnx_model.clean_weights:
                y_value = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "Y").src.data].numpy()
        except ValueError:
            pass

        if y_value is not None and y_value.ndim == 0:
            y_value = int(y_value)

            def prog(X, Z):
                Z[:] = X**y_value

            return program_for_node(prog, sdfg, state, node)

        # General case
        def prog(X, Y, Z):
            Z[:] = X**Y

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Add", name="pure")
class PureAdd(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        A_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "A"))
        B_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "B"))
        C_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "C"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("A", A_desc)
        nsdfg.add_datadesc("B", B_desc)
        nsdfg.add_datadesc("C", C_desc)
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["C"].transient = False

        # Create mapped tasklet for element-wise addition with broadcasting
        map_ranges = {f"i{i}": f"0:{C_desc.shape[i]}" for i in range(len(C_desc.shape))}
        index_str = ", ".join(map_ranges.keys())

        # Generate broadcasting-aware indexing for inputs
        A_indices = broadcast_indices(A_desc.shape, C_desc.shape)
        A_index_str = ", ".join(A_indices) if A_indices else "0"

        B_indices = broadcast_indices(B_desc.shape, C_desc.shape)
        B_index_str = ", ".join(B_indices) if B_indices else "0"

        tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                                                 map_ranges=map_ranges,
                                                                 inputs={
                                                                     "__A": dace.Memlet(f"A[{A_index_str}]"),
                                                                     "__B": dace.Memlet(f"B[{B_index_str}]")
                                                                 },
                                                                 code="__C = __A + __B",
                                                                 outputs={"__C": dace.Memlet(f"C[{index_str}]")},
                                                                 external_edges=True)

        return nsdfg


@op_implementation(op="Sub", name="pure")
class PureSub(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        A_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "A"))
        B_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "B"))
        C_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "C"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("A", A_desc)
        nsdfg.add_datadesc("B", B_desc)
        nsdfg.add_datadesc("C", C_desc)
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["C"].transient = False

        # Create mapped tasklet for element-wise subtraction with broadcasting
        map_ranges = {f"i{i}": f"0:{C_desc.shape[i]}" for i in range(len(C_desc.shape))}
        index_str = ", ".join(map_ranges.keys())

        # Generate broadcasting-aware indexing for inputs
        A_indices = broadcast_indices(A_desc.shape, C_desc.shape)
        A_index_str = ", ".join(A_indices) if A_indices else "0"

        B_indices = broadcast_indices(B_desc.shape, C_desc.shape)
        B_index_str = ", ".join(B_indices) if B_indices else "0"

        tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                                                 map_ranges=map_ranges,
                                                                 inputs={
                                                                     "__A": dace.Memlet(f"A[{A_index_str}]"),
                                                                     "__B": dace.Memlet(f"B[{B_index_str}]")
                                                                 },
                                                                 code="__C = __A - __B",
                                                                 outputs={"__C": dace.Memlet(f"C[{index_str}]")},
                                                                 external_edges=True)

        return nsdfg


@op_implementation(op="Mul", name="pure")
class PureMul(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        A_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "A"))
        B_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "B"))
        C_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "C"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("A", A_desc)
        nsdfg.add_datadesc("B", B_desc)
        nsdfg.add_datadesc("C", C_desc)
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["C"].transient = False

        # Create mapped tasklet for element-wise multiplication with broadcasting
        map_ranges = {f"i{i}": f"0:{C_desc.shape[i]}" for i in range(len(C_desc.shape))}
        index_str = ", ".join(map_ranges.keys())

        # Generate broadcasting-aware indexing for inputs
        A_indices = broadcast_indices(A_desc.shape, C_desc.shape)
        A_index_str = ", ".join(A_indices) if A_indices else "0"

        B_indices = broadcast_indices(B_desc.shape, C_desc.shape)
        B_index_str = ", ".join(B_indices) if B_indices else "0"

        tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                                                 map_ranges=map_ranges,
                                                                 inputs={
                                                                     "__A": dace.Memlet(f"A[{A_index_str}]"),
                                                                     "__B": dace.Memlet(f"B[{B_index_str}]")
                                                                 },
                                                                 code="__C = __A * __B",
                                                                 outputs={"__C": dace.Memlet(f"C[{index_str}]")},
                                                                 external_edges=True)

        return nsdfg


@op_implementation(op="Div", name="pure")
class PureDiv(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        A_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "A"))
        B_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "B"))
        C_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "C"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("A", A_desc)
        nsdfg.add_datadesc("B", B_desc)
        nsdfg.add_datadesc("C", C_desc)
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["C"].transient = False

        # Create mapped tasklet for element-wise division with broadcasting
        map_ranges = {f"i{i}": f"0:{C_desc.shape[i]}" for i in range(len(C_desc.shape))}
        index_str = ", ".join(map_ranges.keys())

        # Generate broadcasting-aware indexing for inputs
        A_indices = broadcast_indices(A_desc.shape, C_desc.shape)
        A_index_str = ", ".join(A_indices) if A_indices else "0"

        B_indices = broadcast_indices(B_desc.shape, C_desc.shape)
        B_index_str = ", ".join(B_indices) if B_indices else "0"

        tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                                                 map_ranges=map_ranges,
                                                                 inputs={
                                                                     "__A": dace.Memlet(f"A[{A_index_str}]"),
                                                                     "__B": dace.Memlet(f"B[{B_index_str}]")
                                                                 },
                                                                 code="__C = __A / __B",
                                                                 outputs={"__C": dace.Memlet(f"C[{index_str}]")},
                                                                 external_edges=True)

        return nsdfg


# ============================================================================
# Activation Functions and Clipping
# ============================================================================


@python_pure_op_implementation(cast_lambda=lambda X: "lambda x: max(x, dace.{}(0))".format(X.dtype.to_string()))
def Relu(X, Y):
    Y[:] = dace.elementwise(cast_lambda, X)


@python_pure_op_implementation(
    cast_lambda=lambda node, X: "lambda x: (max(x, dace.{dtype}(0)) + {alpha} * min(x, dace.{dtype}(0)))".format(
        dtype=X.dtype.to_string(), alpha=node.alpha))
def LeakyRelu(X, Y):
    Y[:] = dace.elementwise(cast_lambda, X)


@op_implementation(op="Clip", name="pure")
class PureClip(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        min_node = next(state.in_edges_by_connector(node, 'min')).src
        max_node = next(state.in_edges_by_connector(node, 'max')).src
        # TODO other cases
        return (onnx_constant_or_none(sdfg, min_node) is not None and onnx_constant_or_none(sdfg, max_node) is not None)

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        min_node = next(state.in_edges_by_connector(node, 'min')).src
        max_node = next(state.in_edges_by_connector(node, 'max')).src
        minval = onnx_constant_or_none(sdfg, min_node)
        maxval = onnx_constant_or_none(sdfg, max_node)

        input_dtype = in_desc_with_name(node, state, sdfg, "input").dtype
        minstr = f"dace.{input_dtype.to_string()}({minval})"
        maxstr = f"dace.{input_dtype.to_string()}({maxval})"

        lfunc = f"lambda x: min(max(x, {minstr}), {maxstr})"

        def prog(input, output):
            output[:] = dace.elementwise(lfunc, input)

        return program_for_node(prog, sdfg, state, node)
