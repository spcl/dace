# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Normalization and activation operations for ONNX.

This module contains implementations of normalization and activation operations including:
- Softmax, LogSoftmax: Softmax normalization
- LayerNormalization: Layer normalization
- Dropout: Dropout regularization
- Sigmoid: Sigmoid activation (class-based implementation)

"""

import copy
import typing

import dace
import numpy as np
from dace import SDFG, SDFGState, nodes
from dace.sdfg.nodes import Node
from dace.util import in_desc_with_name, out_desc_with_name

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.op_implementations.common import strides_from_shape
from dace.libraries.onnx.op_implementations.utils import (in_desc_with_name, op_implementation, out_desc_with_name,
                                                          program_for_node, python_pure_op_implementation)

# ============================================================================
# Softmax Operations
# ============================================================================


@op_implementation(op="Softmax", name="pure")
class PureSoftmax(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
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

        # Add access nodes
        input_read = nstate.add_read("input")
        output_write = nstate.add_write("output")

        # Get axis for softmax computation
        axis = getattr(node, 'axis', -1)
        if axis < 0:
            axis = len(input_desc.shape) + axis

        # Create intermediate arrays for the computation
        uid = state.node_id(node)

        # max_values: stores the maximum values along the axis
        max_values_desc = copy.deepcopy(input_desc)
        max_values_desc.transient = True
        # Reduce the axis dimension to 1 for max_values
        max_values_desc_shape = list(max_values_desc.shape)
        max_values_desc_shape[axis] = 1
        max_values_desc.shape = max_values_desc_shape
        max_values_desc.total_size = int(np.prod(max_values_desc_shape))
        max_values_desc.strides = strides_from_shape(max_values_desc_shape)
        nsdfg.add_datadesc(f"max_values_{uid}", max_values_desc)

        # exp_values: stores exp(input - max_values)
        exp_values_desc = copy.deepcopy(input_desc)
        exp_values_desc.transient = True
        nsdfg.add_datadesc(f"exp_values_{uid}", exp_values_desc)

        # sum_exp: stores the sum of exp_values along the axis
        sum_exp_desc = copy.deepcopy(input_desc)
        sum_exp_desc.transient = True
        # Reduce the axis dimension to 1 for sum_exp
        sum_exp_desc_shape = list(sum_exp_desc.shape)
        sum_exp_desc_shape[axis] = 1
        sum_exp_desc.shape = sum_exp_desc_shape
        sum_exp_desc.total_size = int(np.prod(sum_exp_desc_shape))
        sum_exp_desc.strides = strides_from_shape(sum_exp_desc_shape)
        nsdfg.add_datadesc(f"sum_exp_{uid}", sum_exp_desc)

        # sub_values: stores the result of subtracting max_values from input
        sub_values_desc = copy.deepcopy(input_desc)
        sub_values_desc.transient = True
        nsdfg.add_datadesc(f"sub_values_{uid}", sub_values_desc)

        # Step 1: ReduceMax along the specified axis
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXReduceMax
        reduce_max_node = ONNXReduceMax(f"reduce_max_{uid}", keepdims=1)
        reduce_max_node.axes = axis
        nstate.add_node(reduce_max_node)
        reduce_max_node.add_in_connector("data")
        reduce_max_node.add_in_connector("axes")
        reduce_max_node.add_out_connector("reduced")

        # Create axes array for ReduceMax
        axes_name, axes_desc = nsdfg.add_array(f"axes_{uid}", [1], dace.int64, transient=True)
        axes_access = nstate.add_access(axes_name)
        axes_tasklet = nstate.add_tasklet(f"init_axes_{uid}", {}, {"out"}, f"out = {axis};", language=dace.Language.CPP)
        nstate.add_edge(axes_tasklet, "out", axes_access, None, dace.Memlet(f"{axes_name}"))

        max_values_access = nstate.add_access(f"max_values_{uid}")
        nstate.add_edge(input_read, None, reduce_max_node, "data", nsdfg.make_array_memlet("input"))
        nstate.add_edge(axes_access, None, reduce_max_node, "axes", nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(reduce_max_node, "reduced", max_values_access, None,
                        nsdfg.make_array_memlet(f"max_values_{uid}"))

        # Step 2: Subtract max_values from input (input - max_values)
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXSub
        sub_node = ONNXSub(f"sub_{uid}")
        nstate.add_node(sub_node)
        sub_node.add_in_connector("A")
        sub_node.add_in_connector("B")
        sub_node.add_out_connector("C")

        sub_values_access = nstate.add_access(f"sub_values_{uid}")
        nstate.add_edge(input_read, None, sub_node, "A", nsdfg.make_array_memlet("input"))
        nstate.add_edge(max_values_access, None, sub_node, "B", nsdfg.make_array_memlet(f"max_values_{uid}"))
        nstate.add_edge(sub_node, "C", sub_values_access, None, nsdfg.make_array_memlet(f"sub_values_{uid}"))

        # Step 3: Apply exponential (exp(input - max_values))
        exp_values_access = nstate.add_access(f"exp_values_{uid}")
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXExp
        exp_node = ONNXExp(f"exp_{uid}")
        nstate.add_node(exp_node)
        exp_node.add_in_connector("input")
        exp_node.add_out_connector("output")

        nstate.add_edge(sub_values_access, None, exp_node, "input", nsdfg.make_array_memlet(f"sub_values_{uid}"))
        nstate.add_edge(exp_node, "output", exp_values_access, None, nsdfg.make_array_memlet(f"exp_values_{uid}"))

        # Step 4: ReduceSum along the specified axis to get sum of exponentials
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXReduceSum
        reduce_sum_node = ONNXReduceSum(f"reduce_sum_{uid}", keepdims=1)
        reduce_sum_node.axes = axis
        nstate.add_node(reduce_sum_node)
        reduce_sum_node.add_in_connector("data")
        reduce_sum_node.add_in_connector("axes")
        reduce_sum_node.add_out_connector("reduced")

        # Reuse the same axes array
        sum_exp_access = nstate.add_access(f"sum_exp_{uid}")
        nstate.add_edge(exp_values_access, None, reduce_sum_node, "data", nsdfg.make_array_memlet(f"exp_values_{uid}"))
        nstate.add_edge(axes_access, None, reduce_sum_node, "axes", nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(reduce_sum_node, "reduced", sum_exp_access, None, nsdfg.make_array_memlet(f"sum_exp_{uid}"))

        # Step 5: Divide exp_values by sum_exp (exp_values / sum_exp)
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXDiv
        div_node = ONNXDiv(f"div_{uid}")
        nstate.add_node(div_node)
        div_node.add_in_connector("A")
        div_node.add_in_connector("B")
        div_node.add_out_connector("C")

        nstate.add_edge(exp_values_access, None, div_node, "A", nsdfg.make_array_memlet(f"exp_values_{uid}"))
        nstate.add_edge(sum_exp_access, None, div_node, "B", nsdfg.make_array_memlet(f"sum_exp_{uid}"))
        nstate.add_edge(div_node, "C", output_write, None, nsdfg.make_array_memlet("output"))

        return nsdfg


softmax_compute = dict(axis=lambda node, input: list(range(len(input.shape)))[node.axis:])


@python_pure_op_implementation(**softmax_compute)
def LogSoftmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=True)
    max_sub = input - maximum
    exponent = np.exp(max_sub)
    sum = np.add.reduce(exponent, axis=axis, keepdims=True)
    log_sum = np.log(sum)
    output[:] = max_sub - log_sum


# ============================================================================
# Sigmoid Activation
# ============================================================================


@op_implementation(op="Sigmoid", name="pure")
class PureSigmoid(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_data = list(state.in_edges_by_connector(node, "X"))[0].src.data
        dtype = sdfg.arrays[input_data].dtype

        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: dtype(1) / (dtype(1) + exp(-x)), X)

        result = program_for_node(prog, sdfg, state, node)
        return result


# ============================================================================
# Layer Normalization
# ============================================================================


@op_implementation(op="LayerNormalization", name="pure")
class PureLayerNormalization(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        X_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "X"))
        scale_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "Scale"))
        B_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "B"))
        Y_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "Y"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("X", X_desc)
        nsdfg.add_datadesc("Scale", scale_desc)
        nsdfg.add_datadesc("B", B_desc)
        nsdfg.add_datadesc("Y", Y_desc)
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["Scale"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Add access nodes
        X_read = nstate.add_read("X")
        scale_read = nstate.add_read("Scale")
        B_read = nstate.add_read("B")
        Y_write = nstate.add_write("Y")

        # Check if optional outputs exist
        has_mean = len(list(state.out_edges_by_connector(node, "Mean"))) > 0
        has_inv_std_dev = len(list(state.out_edges_by_connector(node, "InvStdDev"))) > 0
        mean_write = None
        inv_std_dev_write = None

        if has_mean:
            mean_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "Mean"))
            nsdfg.add_datadesc("Mean", mean_desc)
            nsdfg.arrays["Mean"].transient = False
            mean_write = nstate.add_write("Mean")

        if has_inv_std_dev:
            inv_std_dev_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "InvStdDev"))
            nsdfg.add_datadesc("InvStdDev", inv_std_dev_desc)
            nsdfg.arrays["InvStdDev"].transient = False
            inv_std_dev_write = nstate.add_write("InvStdDev")

        # Get axis and epsilon
        axis = node.axis if hasattr(node, 'axis') else -1
        epsilon = node.epsilon if hasattr(node, 'epsilon') else 1e-5
        stash_type = node.stash_type if hasattr(node, 'stash_type') else 1

        # Create tasklet that performs the layer normalization
        tasklet_inputs = {
            "__X": dace.pointer(X_desc.dtype),
            "__Scale": dace.pointer(scale_desc.dtype),
            "__B": dace.pointer(B_desc.dtype),
        }
        tasklet_outputs = {
            "__Y": dace.pointer(Y_desc.dtype),
        }
        if has_mean:
            tasklet_outputs["__Mean"] = dace.pointer(mean_desc.dtype)
        if has_inv_std_dev:
            tasklet_outputs["__InvStdDev"] = dace.pointer(inv_std_dev_desc.dtype)

        # Generate code for multi-dimensional normalization
        rank = len(X_desc.shape)
        if axis < 0:
            axis = rank + axis

        # Generate map ranges for the outer dimensions (before axis)
        outer_map_ranges = {f"i{i}": f"0:{X_desc.shape[i]}" for i in range(axis)}

        # Generate map ranges for the inner dimensions (axis and after)
        inner_map_ranges = {f"i{i}": f"0:{X_desc.shape[i]}" for i in range(axis, rank)}

        # Calculate size of normalization dimensions
        norm_size = int(np.prod([X_desc.shape[i] for i in range(axis, rank)]))

        # Determine computation type based on stash_type
        compute_type = "dace::" + ("float32" if stash_type == 1 else X_desc.dtype.to_string())

        # Generate code for the tasklet
        code = f"""
        // Outer loop over dimensions before axis
        {chr(10).join([f'for (int i{i} = 0; i{i} < {X_desc.shape[i]}; i{i}++) {{' for i in range(axis)])}

        // Calculate mean over normalization dimensions
        {compute_type} sum = 0.0;
        {chr(10).join([f'for (int i{i} = 0; i{i} < {X_desc.shape[i]}; i{i}++) {{' for i in range(axis, rank)])}
            sum += __X[{'+'.join([f'i{i} * {X_desc.strides[i]}' for i in range(rank)])}];
        {chr(10).join(['}' for _ in range(axis, rank)])}
        {compute_type} mean = sum / {norm_size};
        """

        if has_mean:
            code += f"""
            // Store mean
            __Mean[{'+'.join([f'i{i} * {mean_desc.strides[i]}' for i in range(axis)])}] = mean;
            """

        code += f"""
        // Calculate variance
        {compute_type} sq_sum = 0.0;
        {chr(10).join([f'for (int i{i} = 0; i{i} < {X_desc.shape[i]}; i{i}++) {{' for i in range(axis, rank)])}
            {compute_type} diff = __X[{'+'.join([f'i{i} * {X_desc.strides[i]}' for i in range(rank)])}] - mean;
            sq_sum += diff * diff;
        {chr(10).join(['}' for _ in range(axis, rank)])}
        {compute_type} variance = sq_sum / {norm_size};
        {compute_type} inv_std_dev = 1.0 / sqrt(variance + {epsilon});
        """

        if has_inv_std_dev:
            code += f"""
            // Store inverse standard deviation
            __InvStdDev[{'+'.join([f'i{i} * {inv_std_dev_desc.strides[i]}' for i in range(axis)])}] = inv_std_dev;
            """

        code += f"""
        // Normalize and apply scale and bias
        {chr(10).join([f'for (int i{i} = 0; i{i} < {X_desc.shape[i]}; i{i}++) {{' for i in range(axis, rank)])}
            int x_idx = {'+'.join([f'i{i} * {X_desc.strides[i]}' for i in range(rank)])};
            int y_idx = {'+'.join([f'i{i} * {Y_desc.strides[i]}' for i in range(rank)])};
            // Scale and B only have dimensions for normalization axes
            int scale_idx = {'+'.join([f'i{i + axis} * {scale_desc.strides[i]}' for i in range(len(scale_desc.shape))])};
            int b_idx = {'+'.join([f'i{i + axis} * {B_desc.strides[i]}' for i in range(len(B_desc.shape))])};
            // Compute normalized value in the computation type
            {compute_type} normalized = (__X[x_idx] - mean) * inv_std_dev;
            // Cast final result back to output type
            __Y[y_idx] = normalized * __Scale[scale_idx] + __B[b_idx];
        {chr(10).join(['}' for _ in range(axis, rank)])}
        {chr(10).join(['}' for _ in range(axis)])}
        """

        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs=tasklet_inputs,
                                     outputs=tasklet_outputs,
                                     code=code,
                                     language=dace.Language.CPP)

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", dace.Memlet.from_array("X", X_desc))
        nstate.add_edge(scale_read, None, tasklet, "__Scale", dace.Memlet.from_array("Scale", scale_desc))
        nstate.add_edge(B_read, None, tasklet, "__B", dace.Memlet.from_array("B", B_desc))
        nstate.add_edge(tasklet, "__Y", Y_write, None, dace.Memlet.from_array("Y", Y_desc))
        if has_mean:
            nstate.add_edge(tasklet, "__Mean", mean_write, None, dace.Memlet.from_array("Mean", mean_desc))
        if has_inv_std_dev:
            nstate.add_edge(tasklet, "__InvStdDev", inv_std_dev_write, None,
                            dace.Memlet.from_array("InvStdDev", inv_std_dev_desc))

        return nsdfg


# ============================================================================
# Dropout
# ============================================================================


@op_implementation(op="Dropout", name="pure")
class PureDropout(ONNXForward):
    """ Dropout implementation with support for training and inference modes.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Get input descriptor
        data = in_desc_with_name(node, state, sdfg, "data")

        # Check if optional inputs are present
        has_ratio = "ratio" in node.in_connectors
        has_training_mode = "training_mode" in node.in_connectors

        # Check data type
        if data.dtype not in [dace.float16, dace.float32, dace.float64]:
            return False

        # If ratio is provided as input, it should be a scalar
        if has_ratio:
            ratio = in_desc_with_name(node, state, sdfg, "ratio")
            if ratio.total_size != 1:
                return False

        # If training_mode is provided as input, it should be a scalar boolean
        if has_training_mode:
            training_mode = in_desc_with_name(node, state, sdfg, "training_mode")
            if training_mode.total_size != 1:
                return False

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        # Get descriptors
        data = in_desc_with_name(node, state, sdfg, "data")
        output = out_desc_with_name(node, state, sdfg, "output")

        # Check for optional mask output
        has_mask_output = "mask" in node.out_connectors
        mask = out_desc_with_name(node, state, sdfg, "mask") if has_mask_output else None

        # Check for optional inputs
        has_ratio_input = "ratio" in node.in_connectors
        has_training_mode_input = "training_mode" in node.in_connectors

        ratio_desc = in_desc_with_name(node, state, sdfg, "ratio") if has_ratio_input else None
        training_mode_desc = in_desc_with_name(node, state, sdfg, "training_mode") if has_training_mode_input else None

        # Get dropout ratio (from attribute or will be provided as input)
        # ONNX spec: default ratio is 0.5 if not specified
        dropout_ratio = getattr(node, 'ratio', 0.5) if not has_ratio_input else None

        # Get seed if specified (for reproducible dropout)
        seed = getattr(node, 'seed', None)

        # Calculate total elements
        total_elements = data.total_size

        # Get data type
        dtype = data.dtype
        dtype_str = str(dtype).replace("dace.", "")

        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors
        nsdfg.add_datadesc("data", copy.deepcopy(data))
        nsdfg.add_datadesc("output", copy.deepcopy(output))

        if has_mask_output:
            nsdfg.add_datadesc("mask", copy.deepcopy(mask))

        if has_ratio_input:
            nsdfg.add_datadesc("ratio", copy.deepcopy(ratio_desc))

        if has_training_mode_input:
            nsdfg.add_datadesc("training_mode", copy.deepcopy(training_mode_desc))

        # Set arrays as non-transient
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["output"].transient = False
        if has_mask_output:
            nsdfg.arrays["mask"].transient = False
        if has_ratio_input:
            nsdfg.arrays["ratio"].transient = False
        if has_training_mode_input:
            nsdfg.arrays["training_mode"].transient = False

        # Add access nodes
        data_read = nstate.add_read("data")
        output_write = nstate.add_write("output")
        mask_write = nstate.add_write("mask") if has_mask_output else None
        ratio_read = nstate.add_read("ratio") if has_ratio_input else None
        training_mode_read = nstate.add_read("training_mode") if has_training_mode_input else None

        # Generate C++ code for dropout
        # Note: This implementation uses a simple linear congruential generator for portability
        # In production, you might want to use a better random number generator

        code = f"""
        #include <cstdint>
        #include <ctime>

        // Get dropout ratio
        {dtype_str} ratio = {dropout_ratio if not has_ratio_input else '__ratio'};

        // Get training mode (default to false if not specified)
        bool training_mode = {('__training_mode' if has_training_mode_input else 'false')};

        // If in inference mode, just copy input to output
        if (!training_mode) {{
            for (int i = 0; i < {total_elements}; i++) {{
                __output[i] = __data[i];
                {"__mask[i] = true;" if has_mask_output else ""}
            }}
        }} else {{
            // Training mode: apply dropout

            // Initialize random seed
            static uint64_t rng_state = {seed if seed is not None else 'uint64_t(std::time(nullptr))'};

            // Scale factor for remaining values (1 / (1 - ratio))
            {dtype_str} scale = ({dtype_str})(1.0 / (1.0 - ratio));

            // Apply dropout
            for (int i = 0; i < {total_elements}; i++) {{
                // Simple LCG for random number generation
                // This generates a random number in [0, 1)
                rng_state = (rng_state * 1664525ULL + 1013904223ULL);
                double random_val = double(rng_state) / double(UINT64_MAX);

                // Dropout: keep if random value is greater than ratio
                bool keep = (random_val >= ratio);

                if (keep) {{
                    // Scale the kept values
                    __output[i] = __data[i] * scale;
                    {"__mask[i] = true;" if has_mask_output else ""}
                }} else {{
                    // Drop the value
                    __output[i] = 0;
                    {"__mask[i] = false;" if has_mask_output else ""}
                }}
            }}
        }}
        """

        # Create tasklet inputs and outputs
        tasklet_inputs = {
            "__data": dace.pointer(data.dtype),
        }
        tasklet_outputs = {
            "__output": dace.pointer(output.dtype),
        }

        if has_ratio_input:
            tasklet_inputs["__ratio"] = ratio_desc.dtype
        if has_training_mode_input:
            tasklet_inputs["__training_mode"] = training_mode_desc.dtype
        if has_mask_output:
            tasklet_outputs["__mask"] = dace.pointer(mask.dtype)

        # Create the tasklet
        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs=tasklet_inputs,
                                     outputs=tasklet_outputs,
                                     code=code,
                                     language=dace.Language.CPP)

        # Connect the tasklet with memlets
        nstate.add_edge(data_read, None, tasklet, "__data", dace.Memlet.from_array("data", data))

        if has_ratio_input:
            nstate.add_edge(ratio_read, None, tasklet, "__ratio", dace.Memlet.from_array("ratio", ratio_desc))

        if has_training_mode_input:
            nstate.add_edge(training_mode_read, None, tasklet, "__training_mode",
                            dace.Memlet.from_array("training_mode", training_mode_desc))

        nstate.add_edge(tasklet, "__output", output_write, None, dace.Memlet.from_array("output", output))

        if has_mask_output:
            nstate.add_edge(tasklet, "__mask", mask_write, None, dace.Memlet.from_array("mask", mask))

        return nsdfg
