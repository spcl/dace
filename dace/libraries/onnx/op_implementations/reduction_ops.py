# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Reduction operations for ONNX.

This module contains implementations of reduction operations including:
- ReduceSum, ReduceMean: Standard reductions over specified axes
- ReduceMax, ReduceMin: Min/max reductions
- ReduceL2: L2 norm reduction
- CumSum: Cumulative sum along an axis
- Sum: Element-wise sum of multiple inputs

"""

import copy
import typing

import dace
import numpy as np
from dace import SDFG, SDFGState
from dace.sdfg.nodes import Node
from dace.util import (in_desc_with_name, in_edge_with_name, iterables_equal, out_desc_with_name)

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.common import (generate_reduction_tasklet_code, setup_reduction_sdfg)
from dace.libraries.onnx.op_implementations.utils import (empty_sdfg_for_node, in_desc_with_name, op_implementation,
                                                          out_desc_with_name, program_for_node)

# ============================================================================
# Cumulative Sum
# ============================================================================


@op_implementation(op="CumSum", name="pure")
class PureCumSum(ONNXForward):

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg, nstate, input_nodes, output_nodes = empty_sdfg_for_node(sdfg, state, node, add_access_nodes=True)

        x_desc = in_desc_with_name(node, state, sdfg, "x")
        axis_desc = in_desc_with_name(node, state, sdfg, "axis")
        y_desc = out_desc_with_name(node, state, sdfg, "y")

        x_idx_expr = " + ".join([f"i{i} * {s}" for i, s in enumerate(x_desc.strides)])
        y_idx_expr = " + ".join([f"i{i} * {s}" for i, s in enumerate(y_desc.strides)])

        num_dims = len(x_desc.shape)

        # Get exclusive and reverse flags
        exclusive = int(node.exclusive) if hasattr(node, 'exclusive') else 0
        reverse = int(node.reverse) if hasattr(node, 'reverse') else 0

        # Generate code based on exclusive and reverse flags
        code = ""

        if reverse:
            # For reverse cumsum, iterate from high to low on the axis
            for i, val in enumerate(y_desc.shape):
                code += f"int is_axis{i} = ({i} == ({num_dims} + __axis) % {num_dims});\n"

            # Reverse iteration: start from the end
            for i, val in enumerate(y_desc.shape):
                code += f"for (int i{i} = {val} - 1; i{i} >= 0; i{i}--) {{\n"

            if exclusive:
                # Reverse exclusive: y[i] = sum(x[i+1:])
                # So for the last element on axis, y = 0; otherwise y = x + y[next]
                y_next_idx_expr = " + ".join([f"(i{i} + is_axis{i}) * {s}" for i, s in enumerate(y_desc.strides)])
                code += f"if (" + ' || '.join(
                    [f"(i{i} < {y_desc.shape[i]} - 1 && is_axis{i})" for i in range(num_dims)]) + ") {\n"
                code += f"__y[{y_idx_expr}] = __y[{y_next_idx_expr}];\n"
                code += "} else {\n"
                code += f"__y[{y_idx_expr}] = 0;\n"
                code += "}\n"
            else:
                # Reverse inclusive: y[i] = sum(x[i:])
                y_next_idx_expr = " + ".join([f"(i{i} + is_axis{i}) * {s}" for i, s in enumerate(y_desc.strides)])
                code += f"__y[{y_idx_expr}] = __x[{x_idx_expr}];\n"
                code += f"if (" + ' || '.join(
                    [f"(i{i} < {y_desc.shape[i]} - 1 && is_axis{i})" for i in range(num_dims)]) + ") {\n"
                code += f"__y[{y_idx_expr}] += __y[{y_next_idx_expr}];\n"
                code += "}\n"

            for _ in y_desc.shape:
                code += "}\n"
        else:
            # For forward cumsum, iterate from low to high on the axis
            for i, val in enumerate(y_desc.shape):
                code += f"for (int i{i} = 0; i{i} < {val}; i{i}++) {{\n"
                code += f"int is_axis{i} = ({i} == ({num_dims} + __axis) % {num_dims});\n"

            if exclusive:
                # Forward exclusive: y[i] = sum(x[0:i])
                # So for first element on axis, y = 0; otherwise y = y[prev] + x[prev]
                y_prev_idx_expr = " + ".join([f"(i{i} - is_axis{i}) * {s}" for i, s in enumerate(y_desc.strides)])
                x_prev_idx_expr = " + ".join([f"(i{i} - is_axis{i}) * {s}" for i, s in enumerate(x_desc.strides)])
                code += f"if (" + ' || '.join([f"(i{i} > 0 && is_axis{i})" for i in range(num_dims)]) + ") {\n"
                code += f"__y[{y_idx_expr}] = __y[{y_prev_idx_expr}] + __x[{x_prev_idx_expr}];\n"
                code += "} else {\n"
                code += f"__y[{y_idx_expr}] = 0;\n"
                code += "}\n"
            else:
                # Forward inclusive: y[i] = sum(x[0:i+1])
                y_prev_idx_expr = " + ".join([f"(i{i} - is_axis{i}) * {s}" for i, s in enumerate(y_desc.strides)])
                code += f"__y[{y_idx_expr}] = __x[{x_idx_expr}];\n"
                code += f"if (" + ' || '.join([f"(i{i} > 0 && is_axis{i})" for i in range(num_dims)]) + ") {\n"
                code += f"__y[{y_idx_expr}] += __y[{y_prev_idx_expr}];\n"
                code += "}\n"

            for _ in y_desc.shape:
                code += "}\n"

        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs={
                "__x": dace.pointer(x_desc.dtype),
                "__axis": axis_desc.dtype,
            },
            outputs={"__y": dace.pointer(y_desc.dtype)},
            language=dace.Language.CPP,
            code=code,
        )

        nstate.add_edge(input_nodes["x"], None, tasklet, "__x", dace.Memlet.from_array("x", x_desc))
        nstate.add_edge(input_nodes["axis"], None, tasklet, "__axis", dace.Memlet.from_array("axis", axis_desc))
        nstate.add_edge(tasklet, "__y", output_nodes["y"], None, dace.Memlet.from_array("y", y_desc))

        return nsdfg


# ============================================================================
# ReduceMean Operations
# ============================================================================


@op_implementation(op="ReduceMean", name="pure")
class PureReduceMeanCPP(ONNXForward):

    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Avoid this expansion if the backward pass will be constructed
        # TODO pass the backward flag to the functions
        return False

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Set up the common SDFG structure
        nsdfg, nstate, data_desc, reduced_desc, data_read, reduced_write, axes_node, axes_desc, num_reduce_axes = setup_reduction_sdfg(
            node, state, sdfg, "reduce_mean")

        # Generate tasklet code for mean reduction
        keepdims = getattr(node, 'keepdims', 1)
        tasklet_code = generate_reduction_tasklet_code(data_desc, reduced_desc, num_reduce_axes, keepdims, 'mean')

        # Create tasklet and connect it
        uid = state.node_id(node)
        tasklet = nstate.add_tasklet(f'reduce_mean_{uid}', {
            'inp': dace.pointer(data_desc.dtype),
            'axes_arr': dace.pointer(axes_desc.dtype)
        }, {'out': dace.pointer(reduced_desc.dtype)},
                                     tasklet_code,
                                     language=dace.Language.CPP)

        # Add edges for axes input, data input and output
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_read.data))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_node.data))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_write.data))

        return nsdfg


@op_implementation(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):
    '''
        ReduceMean expansion
    '''

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # Check that all the inputs (even the optional ones) are present and constant
        # optional inputs
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        # Current constraints: axes must be explict. Axes must be zero
        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # We treat both cases where axes is an attribute and where it is an input
        # Since can be applied is true, we know that axes is present and valid
        axes = None
        # TODO: avoid catching Exceptions
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                raise NotImplementedError(
                    "PureReduceMean in the case where there are multiple axes as input connectors is not implemented yet."
                )
        else:
            # Axes is an attribute of the node
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.mean(data, axis=axes)

        result = program_for_node(prog, sdfg, state, node)
        return result


# ============================================================================
# ReduceSum Operations
# ============================================================================


@op_implementation(op="ReduceSum", name="pure")
class PureReduceSumCPP(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Avoid this expansion if the backward pass will be contructed
        # TODO pass the backward flag to the functions
        return False

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Set up the common SDFG structure
        nsdfg, nstate, data_desc, reduced_desc, data_read, reduced_write, axes_node, axes_desc, num_reduce_axes = setup_reduction_sdfg(
            node, state, sdfg, "reduce_sum")

        # Generate tasklet code for sum reduction
        keepdims = getattr(node, 'keepdims', 1)
        tasklet_code = generate_reduction_tasklet_code(data_desc, reduced_desc, num_reduce_axes, keepdims, 'sum')

        # Create tasklet and connect it
        uid = state.node_id(node)
        tasklet = nstate.add_tasklet(f'reduce_sum_{uid}', {
            'inp': dace.pointer(data_desc.dtype),
            'axes_arr': dace.pointer(axes_desc.dtype)
        }, {'out': dace.pointer(reduced_desc.dtype)},
                                     tasklet_code,
                                     language=dace.Language.CPP)

        # Add edges for axes input, data input and output
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_read.data))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_node.data))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_write.data))

        return nsdfg


@op_implementation(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):
    '''
        ReduceSum expansion
    '''

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # Check that all the inputs (even the optional ones) are present and constant
        # optional inputs
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        # Current constraints: axes must be explict.
        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # We treat both cases where axes is an attribute and where it is an input
        # Since can be applied is true, we know that axes is present and valid
        axes = None
        # TODO: avoid catching Exceptions
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                raise NotImplementedError(
                    "PureReduceSum in the case where there are multiple axes as input connectors is not implemented yet."
                )
        else:
            # Axes is an attribute of the node
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.sum(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


# ============================================================================
# ReduceMax and ReduceMin Operations
# ============================================================================


@op_implementation(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):
    '''
        ReduceMax expansion
    '''

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # Check that all the inputs (even the optional ones) are present and constant
        # optional inputs
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        # Current constraints: axes must be explict. Axes must be zero
        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # We treat both cases where axes is an attribute and where it is an input
        # Since can be applied is true, we know that axes is present and valid
        axes = None
        # TODO: avoid catching Exceptions
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass
        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                raise NotImplementedError(
                    "PureReduceSum in the case where there are multiple axes as input connectors is not implemented yet."
                )
        else:
            # Axes is an attribute of the node
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.max(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Set up the common SDFG structure
        nsdfg, nstate, data_desc, reduced_desc, data_read, reduced_write, axes_node, axes_desc, num_reduce_axes = setup_reduction_sdfg(
            node, state, sdfg, "reduce_min")

        # Generate tasklet code for min reduction
        keepdims = getattr(node, 'keepdims', 1)
        tasklet_code = generate_reduction_tasklet_code(data_desc, reduced_desc, num_reduce_axes, keepdims, 'min')

        # Create tasklet and connect it
        uid = state.node_id(node)
        tasklet = nstate.add_tasklet(f'reduce_min_{uid}', {
            'inp': dace.pointer(data_desc.dtype),
            'axes_arr': dace.pointer(axes_desc.dtype)
        }, {'out': dace.pointer(reduced_desc.dtype)},
                                     tasklet_code,
                                     language=dace.Language.CPP)

        # Add edges for axes input, data input and output
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_read.data))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_node.data))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_write.data))

        return nsdfg


# ============================================================================
# Sum (Multi-input sum)
# ============================================================================


@op_implementation(op="Sum", name="pure")
class PureSum(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # check that all shapes are arrays, and that the shapes are all equal
        shape = None
        for edge in node.iter_inputs_in_onnx_order(state):
            desc = in_desc_with_name(node, state, sdfg, edge.dst_conn)
            if shape is None:
                shape = desc.shape

            if not iterables_equal(shape, desc.shape):
                return False

        if not iterables_equal(shape, out_desc_with_name(node, state, sdfg, "sum").shape):
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.name)
        input_names = []
        for e in node.iter_inputs_in_onnx_order(state):
            new_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, e.dst_conn))
            new_desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, new_desc)
            input_names.append(e.dst_conn)

        new_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "sum"))
        new_desc.transient = False
        nsdfg.add_datadesc("sum", new_desc)

        nstate = nsdfg.add_state()
        # we know all shapes are equal to the output shape
        shape = out_desc_with_name(node, state, sdfg, "sum").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        tasklet, _, _ = nstate.add_mapped_tasklet(
            node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs={f"__{inp}": dace.Memlet(f"{inp}[{index_str}]")
                    for inp in input_names},
            code=f"__sum = {' + '.join(f'__{inp}' for inp in input_names)}",
            outputs={"__sum": dace.Memlet(f"sum[{index_str}]")},
            external_edges=True)

        tasklet.in_connectors = {f"__{inp}": in_desc_with_name(node, state, sdfg, inp).dtype for inp in input_names}
        tasklet.out_connectors = {"__sum": out_desc_with_name(node, state, sdfg, "sum").dtype}
        return nsdfg


# ============================================================================
# ReduceL2 Operations
# ============================================================================


@op_implementation(op="ReduceL2", name="pure")
class PureReduceL2(ONNXForward):
    """
    Pure implementation of ONNX ReduceL2 operator.

    Computes the L2 norm (sqrt of sum of squares) of input elements along specified axes.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Check that axes are present and constant
        is_axes_present = True
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        if not is_axes_present and hasattr(node, "axes"):
            is_axes_present = True

        if not is_axes_present:
            return False

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get axes
        axes = None
        try:
            if hasattr(sdfg, "_parent_onnx_model") and in_edge_with_name(
                    node, state, "axes").src.data in sdfg._parent_onnx_model.clean_weights:
                axes = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()
        except ValueError:
            pass

        if axes is not None:
            if len(axes) == 1:
                axes = axes[0]
            else:
                raise NotImplementedError("PureReduceL2 with multiple axes as input connectors is not implemented yet.")
        else:
            axes = node.axes if hasattr(node, "axes") else None

        def prog(data, reduced):
            reduced[:] = np.sqrt(np.sum(np.square(data), axis=axes))

        return program_for_node(prog, sdfg, state, node)
