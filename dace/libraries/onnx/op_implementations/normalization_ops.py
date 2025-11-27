# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Normalization operations for ONNX.

This module contains implementations of normalization operations including:
- Softmax, LogSoftmax: Softmax normalization
- LayerNormalization: Layer normalization
- Dropout: Dropout regularization
"""

import copy
import typing

import dace
import numpy as np
from dace import SDFG, SDFGState, nodes
from dace.sdfg.utils import in_desc_with_name, out_desc_with_name

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.op_implementations.utils import (in_desc_with_name, op_implementation, out_desc_with_name,
                                                          python_pure_op_implementation)

# ============================================================================
# Softmax Operations
# ============================================================================

softmax_compute = dict(axis=lambda node, input: tuple(range(len(input.shape)))[node.axis:])


@python_pure_op_implementation(**softmax_compute)
def Softmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=True)
    exp_values = np.exp(input - maximum)
    sum_exp = np.add.reduce(exp_values, axis=axis, keepdims=True)
    output[:] = exp_values / sum_exp


@python_pure_op_implementation(**softmax_compute)
def LogSoftmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=True)
    max_sub = input - maximum
    exponent = np.exp(max_sub)
    sum = np.add.reduce(exponent, axis=axis, keepdims=True)
    log_sum = np.log(sum)
    output[:] = max_sub - log_sum


# ============================================================================
# Layer Normalization
# ============================================================================


def _layernorm_axis(node, X):
    axis = node.axis if hasattr(node, 'axis') and node.axis >= 0 else len(X.shape) + node.axis
    return tuple(range(axis, len(X.shape)))


def _layernorm_norm_size(node, X):
    axis = node.axis if hasattr(node, 'axis') and node.axis >= 0 else len(X.shape) + node.axis
    return np.prod([X.shape[i] for i in range(axis, len(X.shape))])


layernorm_compute = dict(axis=_layernorm_axis,
                         epsilon=lambda node: getattr(node, 'epsilon', 1e-5),
                         norm_size=_layernorm_norm_size)


@python_pure_op_implementation(**layernorm_compute)
def LayerNormalization(X, Scale, B, Y):
    sum_x = np.add.reduce(X, axis=axis, keepdims=True)
    mean = sum_x / norm_size
    diff = X - mean
    sum_sq = np.add.reduce(diff * diff, axis=axis, keepdims=True)
    variance = sum_sq / norm_size
    inv_std = 1.0 / np.sqrt(variance + epsilon)
    normalized = diff * inv_std
    Y[:] = normalized * Scale + B


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
