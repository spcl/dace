# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Elementwise and mathematical ONNX operations.

This module contains pure implementations of elementwise mathematical operations including:
- Basic arithmetic: Add, Sub, Mul, Div, Pow
- Unary math functions: Log, Exp, Sqrt, Sin, Cos, Tan, Tanh, Erf, Neg, Reciprocal
- Activation functions: Relu, LeakyRelu, Sigmoid, Softplus
- Comparison operations: Equal, GreaterOrEqual
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
from dace.util import in_desc_with_name, in_edge_with_name
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.util import in_desc_with_name, out_desc_with_name

# ============================================================================
# Unary Mathematical Operations
# ============================================================================


@python_pure_op_implementation
def Log(input, output):
    """
    ONNX Log operation implementation.

    Computes the natural logarithm of the input tensor element-wise.

    Args:
        input: Input tensor of any numeric type.
        output: Output tensor with the same shape and type as input.
    """
    output[:] = np.log(input)


@python_pure_op_implementation
def Exp(input, output):
    """
    ONNX Exp operation implementation.

    Computes the exponential of the input tensor element-wise.

    Args:
        input: Input tensor of any numeric type.
        output: Output tensor with the same shape and type as input.
    """
    output[:] = np.exp(input)


@python_pure_op_implementation
def Sqrt(X, Y):
    """
    ONNX Sqrt operation implementation.

    Computes the square root of the input tensor element-wise.

    Args:
        X: Input tensor of any numeric type.
        Y: Output tensor with the same shape and type as X.
    """
    Y[:] = dace.elementwise(lambda x: sqrt(x), X)


@python_pure_op_implementation
def Sin(input, output):
    output[:] = np.sin(input)


@python_pure_op_implementation
def Cos(input, output):
    output[:] = np.cos(input)


@python_pure_op_implementation
def Tan(input, output):
    output[:] = np.tan(input)


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
        # Always applicable - supports both constant and dynamic min/max
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        input_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "input"))
        output_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "output"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("input", input_desc)
        nsdfg.add_datadesc("output", output_desc)
        nsdfg.arrays["input"].transient = False
        nsdfg.arrays["output"].transient = False

        # Check if min and max are provided
        has_min = 'min' in node.in_connectors
        has_max = 'max' in node.in_connectors

        # Try to get constant values if available
        min_const = None
        max_const = None

        if has_min:
            try:
                min_edge = next(state.in_edges_by_connector(node, 'min'))
                min_const = onnx_constant_or_none(sdfg, min_edge.src)
            except (StopIteration, AttributeError):
                pass

        if has_max:
            try:
                max_edge = next(state.in_edges_by_connector(node, 'max'))
                max_const = onnx_constant_or_none(sdfg, max_edge.src)
            except (StopIteration, AttributeError):
                pass

        # Determine clipping strategy based on what's available
        dtype_str = input_desc.dtype.to_string()

        if min_const is not None and max_const is not None:
            # Both min and max are constants - use simple elementwise
            minstr = f"dace.{dtype_str}({min_const})"
            maxstr = f"dace.{dtype_str}({max_const})"
            code = f"__output = min(max(__input, {minstr}), {maxstr})"

            # Create mapped tasklet
            map_ranges = {f"i{i}": f"0:{input_desc.shape[i]}" for i in range(len(input_desc.shape))}
            index_str = ", ".join(map_ranges.keys())

            tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(
                name=node.label + "_tasklet",
                map_ranges=map_ranges,
                inputs={"__input": dace.Memlet(f"input[{index_str}]")},
                code=code,
                outputs={"__output": dace.Memlet(f"output[{index_str}]")},
                external_edges=True)
        else:
            # At least one of min/max is dynamic - need to handle them as arrays
            if has_min and min_const is None:
                min_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "min"))
                nsdfg.add_datadesc("min", min_desc)
                nsdfg.arrays["min"].transient = False

            if has_max and max_const is None:
                max_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "max"))
                nsdfg.add_datadesc("max", max_desc)
                nsdfg.arrays["max"].transient = False

            # Build the clipping code using Python ternary operators
            # Note: We can't use min/max functions due to conflicts with array parameter names 'min' and 'max'
            if not has_min and not has_max:
                # No clipping needed - just copy
                code = "__output = __input"
                inputs = {"__input": None}  # Will be filled below
            elif not has_min:
                # Only max clipping
                if max_const is not None:
                    maxstr = f"dace.{dtype_str}({max_const})"
                    code = f"__output = __input if __input < {maxstr} else {maxstr}"
                    inputs = {"__input": None}
                else:
                    code = "__output = __input if __input < __max_val else __max_val"
                    inputs = {"__input": None, "__max": None}
            elif not has_max:
                # Only min clipping
                if min_const is not None:
                    minstr = f"dace.{dtype_str}({min_const})"
                    code = f"__output = __input if __input > {minstr} else {minstr}"
                    inputs = {"__input": None}
                else:
                    code = "__output = __input if __input > __min_val else __min_val"
                    inputs = {"__input": None, "__min": None}
            else:
                # Both min and max, at least one is dynamic
                # Use two-step clipping: clip to min, then clip to max
                if min_const is not None:
                    minstr = f"dace.{dtype_str}({min_const})"
                    if max_const is not None:
                        maxstr = f"dace.{dtype_str}({max_const})"
                        code = f"_temp = __input if __input > {minstr} else {minstr}\n__output = _temp if _temp < {maxstr} else {maxstr}"
                        inputs = {"__input": None}
                    else:
                        code = f"_temp = __input if __input > {minstr} else {minstr}\n__output = _temp if _temp < __max_val else __max_val"
                        inputs = {"__input": None, "__max": None}
                else:
                    if max_const is not None:
                        maxstr = f"dace.{dtype_str}({max_const})"
                        code = f"_temp = __input if __input > __min_val else __min_val\n__output = _temp if _temp < {maxstr} else {maxstr}"
                        inputs = {"__input": None, "__min": None}
                    else:
                        code = "_temp = __input if __input > __min_val else __min_val\n__output = _temp if _temp < __max_val else __max_val"
                        inputs = {"__input": None, "__min": None, "__max": None}

            # Create mapped tasklet with broadcasting support
            map_ranges = {f"i{i}": f"0:{input_desc.shape[i]}" for i in range(len(input_desc.shape))}
            index_str = ", ".join(map_ranges.keys())

            # Build input memlets
            final_inputs = {"__input": dace.Memlet(f"input[{index_str}]")}

            if "__min" in inputs and has_min and min_const is None:
                # Handle broadcasting for min
                min_indices = broadcast_indices(min_desc.shape, input_desc.shape)
                min_index_str = ", ".join(min_indices) if min_indices else "0"
                final_inputs["__min_val"] = dace.Memlet(f"min[{min_index_str}]")

            if "__max" in inputs and has_max and max_const is None:
                # Handle broadcasting for max
                max_indices = broadcast_indices(max_desc.shape, input_desc.shape)
                max_index_str = ", ".join(max_indices) if max_indices else "0"
                final_inputs["__max_val"] = dace.Memlet(f"max[{max_index_str}]")

            tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(
                name=node.label + "_tasklet",
                map_ranges=map_ranges,
                inputs=final_inputs,
                code=code,
                outputs={"__output": dace.Memlet(f"output[{index_str}]")},
                external_edges=True)

        return nsdfg


# ============================================================================
# Comparison Operations
# ============================================================================


@python_pure_op_implementation
def Equal(A, B, C):
    """
    ONNX Equal operation implementation.

    Returns element-wise equality comparison between two tensors.

    Args:
        A: First input tensor.
        B: Second input tensor.
        C: Output tensor containing boolean values.
    """
    C[:] = np.equal(A, B)


@python_pure_op_implementation
def GreaterOrEqual(A, B, C):
    """
    ONNX GreaterOrEqual operation implementation.

    Returns element-wise A >= B comparison.

    Args:
        A: First input tensor.
        B: Second input tensor.
        C: Output tensor containing boolean values.
    """
    C[:] = np.greater_equal(A, B)
