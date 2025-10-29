# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import functools
import typing

import numpy as np

import dace
from dace import SDFGState, SDFG, dtypes
from dace.sdfg import nodes, propagation
from dace.transformation.dataflow import MapExpansion, MapCollapse
from dace.sdfg.nodes import Node
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes.onnx_op import ONNXOp
from dace.libraries.onnx.op_implementations.utils import op_implementation, program_for_node
from dace.util import in_desc_with_name, out_desc_with_name, in_edge_with_name, out_edge_with_name


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@op_implementation(op="MaxPool", name="pure")
class PureMaxPool2D(ONNXForward):
    """Pure implementation of 2D MaxPool operation."""

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        """Check if this implementation can be applied to the given node.

        Args:
            node: The MaxPool ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            True if the implementation can be applied, False otherwise
        """
        X = in_desc_with_name(node, state, sdfg, "X")

        if "Indices" in {e.src_conn for e in state.out_edges(node)}:
            return False

        image_dims = len(X.shape) - 2

        # Only do 2D for now
        if image_dims != 2:
            return False

        if node.pads is not None and (len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and len(node.strides) != image_dims:
            return False

        if node.auto_pad != 'NOTSET':
            return False

        if node.ceil_mode != 0 or node.storage_order != 0:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or len(node.dilations) != image_dims):
            return False
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        """Generate the forward pass implementation for MaxPool2D.

        Args:
            node: The MaxPool ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            A nested SDFG implementing the MaxPool operation
        """
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        image_dims = len(X.shape) - 2
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        strides = node.strides if node.strides is not None else [1 for _ in range(image_dims)]
        pads = node.pads if node.pads is not None else [0 for _ in range(image_dims * 2)]
        stride_x, stride_y = strides
        assert pads[0] == pads[2] and pads[1] == pads[3]
        pad_x, pad_y, _, _ = pads
        filter_hx, filter_hy = node.kernel_shape
        input_size_x, input_size_y = X.shape[2:]
        output_size_x, output_size_y = Y.shape[2:]

        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors
        nsdfg.add_datadesc("X", copy.deepcopy(X))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y))
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Add access nodes
        X_read = nstate.add_read("X")
        Y_write = nstate.add_write("Y")

        # Create tasklet that performs the max pooling operation
        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs={"__X": dace.pointer(X.dtype)},
                                     outputs={"__Y": dace.pointer(Y.dtype)},
                                     code=f"""
            // Initialize output with minimum value
            for (int b = 0; b < {batch_size}; b++) {{
                for (int c = 0; c < {num_channels}; c++) {{
                    for (int out_x = 0; out_x < {output_size_x}; out_x++) {{
                        for (int out_y = 0; out_y < {output_size_y}; out_y++) {{
                            __Y[b * {Y.strides[0]} + c * {Y.strides[1]} + out_x * {Y.strides[2]} + out_y * {Y.strides[3]}] = {dtypes.min_value(Y.dtype)};
                        }}
                    }}
                }}
            }}

            // Main max pooling computation
            for (int b = 0; b < {batch_size}; b++) {{
                for (int c = 0; c < {num_channels}; c++) {{
                    for (int out_x = 0; out_x < {output_size_x}; out_x++) {{
                        for (int out_y = 0; out_y < {output_size_y}; out_y++) {{
                            for (int hx = 0; hx < {filter_hx}; hx++) {{
                                for (int hy = 0; hy < {filter_hy}; hy++) {{
                                    int sx = hx + out_x * {stride_x} - {pad_x};
                                    int sy = hy + out_y * {stride_y} - {pad_y};

                                    if (0 <= sx && sx < {input_size_x} && 0 <= sy && sy < {input_size_y}) {{
                                        float input_val = __X[b * {X.strides[0]} + c * {X.strides[1]} + sx * {X.strides[2]} + sy * {X.strides[3]}];
                                        float& output_val = __Y[b * {Y.strides[0]} + c * {Y.strides[1]} + out_x * {Y.strides[2]} + out_y * {Y.strides[3]}];
                                        output_val = max(output_val, input_val);
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """,
                                     language=dace.Language.CPP)

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", dace.Memlet.from_array("X", X))
        nstate.add_edge(tasklet, "__Y", Y_write, None, dace.Memlet.from_array("Y", Y))

        return nsdfg


@op_implementation(op="Conv", name="pure")
class PureConv2D(ONNXForward):
    """Convolution implementation with support for grouped and depthwise convolutions."""

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        """Check if this implementation can be applied to the given node.

        Args:
            node: The Conv ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            True if the implementation can be applied, False otherwise
        """
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        num_filters = W.shape[0]
        num_channels = X.shape[1]

        if (X.dtype not in [dace.float16, dace.float32, dace.float64]
                or W.dtype not in [dace.float16, dace.float32, dace.float64]):
            return False

        # Only do 2D for now
        if len(X.shape) != 4 or len(W.shape) != 4:
            return False

        # Check group convolution constraints
        groups = node.group if node.group is not None else 1
        if groups < 1:
            return False

        # For grouped convolution:
        # - Input channels must be divisible by groups
        # - Output channels (num_filters) must be divisible by groups
        # - Weight shape[1] should be num_channels // groups
        if num_channels % groups != 0:
            return False
        if num_filters % groups != 0:
            return False
        if W.shape[1] != num_channels // groups:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or len(node.dilations) != image_dims):
            return False

        if node.pads is not None and (len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and (len(node.strides) != image_dims):
            return False

        if B is not None and B.shape[0] != num_filters:
            return False

        # Support NOTSET (explicit pads), SAME_UPPER, SAME_LOWER, and VALID
        if node.auto_pad not in ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID', None]:
            return False

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        """Generate the forward pass implementation for Conv2D.

        Args:
            node: The Conv ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            A nested SDFG implementing the Conv operation
        """
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # Check if bias is present in input connectors
        B = in_desc_with_name(node, state, sdfg, "B") if "B" in node.in_connectors else None

        if node.kernel_shape is not None:
            filter_hx, filter_hy = node.kernel_shape
        else:
            filter_hx, filter_hy = W.shape[2:]

        num_filters = W.shape[0]
        num_channels = X.shape[1]
        batch_size = X.shape[0]

        # Get number of groups (default to 1 for standard convolution)
        groups = node.group if node.group is not None else 1
        channels_per_group = num_channels // groups
        filters_per_group = num_filters // groups

        input_size_x, input_size_y = X.shape[2:]
        output_size_y, output_size_x = Y.shape[2:]
        stride_y, stride_x = node.strides or [1, 1]

        # Compute padding based on auto_pad mode
        auto_pad = node.auto_pad if node.auto_pad is not None else 'NOTSET'
        if auto_pad == 'NOTSET':
            # Use explicit pads
            pad_x, pad_y, _, _ = node.pads or [0, 0, 0, 0]
        elif auto_pad == 'VALID':
            # No padding
            pad_x, pad_y = 0, 0
        elif auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
            # Compute padding to make output size = ceil(input_size / stride)
            # ONNX formula: output_size = ceil(input_size / stride)
            # total_pad = (output_size - 1) * stride + kernel_size - input_size

            # Note: Due to naming conventions in the original code:
            # - input_size_x = H, input_size_y = W
            # - output_size_y = H, output_size_x = W (swapped!)
            # - stride_y = H stride, stride_x = W stride (swapped!)
            # - filter_hx = H kernel, filter_hy = W kernel
            # - pad_x = H pad, pad_y = W pad

            # Standard SAME padding formula:
            # For H dimension: total_pad_H = (output_H - 1) * stride_H + kernel_H - input_H
            # For W dimension: total_pad_W = (output_W - 1) * stride_W + kernel_W - input_W

            total_pad_x = max(0, (output_size_y - 1) * stride_y + filter_hx - input_size_x)  # H dimension
            total_pad_y = max(0, (output_size_x - 1) * stride_x + filter_hy - input_size_y)  # W dimension

            if auto_pad == 'SAME_UPPER':
                # Prefer padding at the beginning (top/left)
                pad_x = total_pad_x // 2
                pad_y = total_pad_y // 2
            else:  # SAME_LOWER
                # Prefer padding at the end (bottom/right)
                pad_x = (total_pad_x + 1) // 2
                pad_y = (total_pad_y + 1) // 2
        else:
            # Should not happen due to can_be_applied check
            pad_x, pad_y = 0, 0

        dtype = X.dtype

        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors
        nsdfg.add_datadesc("X", copy.deepcopy(X))
        nsdfg.add_datadesc("W", copy.deepcopy(W))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            nsdfg.add_datadesc("B", copy.deepcopy(B))

        # Set arrays as non-transient since they are inputs/outputs
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["W"].transient = False
        nsdfg.arrays["Y"].transient = False
        if B is not None:
            nsdfg.arrays["B"].transient = False

        # Add access nodes
        X_read = nstate.add_read("X")
        W_read = nstate.add_read("W")
        Y_write = nstate.add_write("Y")
        if B is not None:
            B_read = nstate.add_read("B")

        # Generate C++ code for the grouped convolution
        code = f"""
        // Initialize output
        {f'''
        // Initialize with bias
        for (int b = 0; b < {batch_size}; b++) {{
            for (int m = 0; m < {num_filters}; m++) {{
                for (int out_x = 0; out_x < {output_size_x}; out_x++) {{
                    for (int out_y = 0; out_y < {output_size_y}; out_y++) {{
                        __Y[b * {Y.strides[0]} + m * {Y.strides[1]} + out_x * {Y.strides[2]} + out_y * {Y.strides[3]}] = __B[m];
                    }}
                }}
            }}
        }}
        ''' if B is not None else f'''
        // Zero-initialize output
        for (int b = 0; b < {batch_size}; b++) {{
            for (int m = 0; m < {num_filters}; m++) {{
                for (int out_x = 0; out_x < {output_size_x}; out_x++) {{
                    for (int out_y = 0; out_y < {output_size_y}; out_y++) {{
                        __Y[b * {Y.strides[0]} + m * {Y.strides[1]} + out_x * {Y.strides[2]} + out_y * {Y.strides[3]}] = 0;
                    }}
                }}
            }}
        }}
        '''}

        // Main grouped convolution computation
        for (int b = 0; b < {batch_size}; b++) {{
            for (int g = 0; g < {groups}; g++) {{
                // Each group processes a subset of input/output channels
                int in_channel_start = g * {channels_per_group};
                int out_channel_start = g * {filters_per_group};

                for (int m = 0; m < {filters_per_group}; m++) {{
                    int out_channel = out_channel_start + m;

                    for (int out_x = 0; out_x < {output_size_x}; out_x++) {{
                        for (int out_y = 0; out_y < {output_size_y}; out_y++) {{
                            // Only convolve with channels in the same group
                            for (int c = 0; c < {channels_per_group}; c++) {{
                                int in_channel = in_channel_start + c;

                                for (int hx = 0; hx < {filter_hx}; hx++) {{
                                    for (int hy = 0; hy < {filter_hy}; hy++) {{
                                        int sx = hx + out_x * {stride_x} - {pad_x};
                                        int sy = hy + out_y * {stride_y} - {pad_y};

                                        if (0 <= sx && sx < {input_size_x} && 0 <= sy && sy < {input_size_y}) {{
                                            // Note: Weight tensor layout for grouped conv:
                                            // [num_filters, channels_per_group, filter_hx, filter_hy]
                                            float filter = __W[out_channel * {W.strides[0]} + c * {W.strides[1]} + hx * {W.strides[2]} + hy * {W.strides[3]}];
                                            float image = __X[b * {X.strides[0]} + in_channel * {X.strides[1]} + sx * {X.strides[2]} + sy * {X.strides[3]}];
                                            __Y[b * {Y.strides[0]} + out_channel * {Y.strides[1]} + out_x * {Y.strides[2]} + out_y * {Y.strides[3]}] += filter * image;
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

        # Create tasklet inputs and outputs
        tasklet_inputs = {
            "__X": dace.pointer(X.dtype),
            "__W": dace.pointer(W.dtype),
        }
        tasklet_outputs = {
            "__Y": dace.pointer(Y.dtype),
        }

        if B is not None:
            tasklet_inputs["__B"] = dace.pointer(B.dtype)

        # Create the tasklet
        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs=tasklet_inputs,
                                     outputs=tasklet_outputs,
                                     code=code,
                                     language=dace.Language.CPP)

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", dace.Memlet.from_array("X", X))
        nstate.add_edge(W_read, None, tasklet, "__W", dace.Memlet.from_array("W", W))
        if B is not None:
            nstate.add_edge(B_read, None, tasklet, "__B", dace.Memlet.from_array("B", B))
        nstate.add_edge(tasklet, "__Y", Y_write, None, dace.Memlet.from_array("Y", Y))

        return nsdfg


@op_implementation(op="BatchNormalization", name="pure")
class PureBatchNormalization(ONNXForward):
    """Pure implementation of BatchNormalization operation."""

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        """Check if this implementation can be applied to the given node.

        Args:
            node: The BatchNormalization ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            True if the implementation can be applied, False otherwise
        """
        X = in_desc_with_name(node, state, sdfg, "X")
        # BatchNormalization supports 2D, 3D, 4D, 5D inputs
        # Input shape: (N, C) or (N, C, D1, ..., Dn) where n >= 0
        if len(X.shape) < 2:
            return False

        if "in_mean" in node.in_connectors and "input_mean" not in node.in_connectors:
            # Replace the old names with the new ones
            node.add_in_connector("input_mean", node.in_connectors["in_mean"])
            node.remove_in_connector("in_mean")

        if "in_var" in node.in_connectors and "input_var" not in node.in_connectors:
            # Replace the old names with the new ones
            node.add_in_connector("input_var", node.in_connectors["in_var"])
            node.remove_in_connector("in_var")

        # Check for the new output names
        if not {"scale", "B", "input_mean", "input_var"}.issubset(node.in_connectors):
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        """Generate the forward pass implementation for BatchNormalization.

        Args:
            node: The BatchNormalization ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            A nested SDFG implementing the BatchNormalization operation
        """
        shape = copy.deepcopy(in_desc_with_name(node, state, sdfg, "X").shape)
        reduce_axes = list(shape)
        num_channels = reduce_axes.pop(1)

        N = _prod(reduce_axes)
        # Compute broadcast shape based on input dimensions
        # For 2D input (N, C): broadcast_shape = [C]
        # For 3D input (N, C, D): broadcast_shape = [C, 1]
        # For 4D input (N, C, H, W): broadcast_shape = [C, 1, 1]
        # etc.
        broadcast_shape = [num_channels] + [1] * (len(shape) - 2)
        dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        eps = node.epsilon
        momentum = node.momentum
        inv_momentum = 1 - node.momentum

        axis = tuple(i for i in range(len(shape)) if i != 1)

        # Check if training_mode attribute exists
        if not hasattr(node, "training_mode"):
            # By default, set to False (inference mode)
            node.training_mode = False

        if node.training_mode:
            # TRAINING: compute batch statistics and update running statistics (EMA like PyTorch)
            def prog(input_mean, scale, input_var, B, X, Y, running_mean, running_var):
                # Batch mean, variance over axis=(0,2,3) for NCHW (your `axis`/`N` already set)
                batch_mean = np.add.reduce(X, axis=axis) / N

                batch_mean_broadcastable = dace.define_local(broadcast_shape, dtype)
                batch_mean_broadcastable[:] = batch_mean
                X_minus_mean = X - batch_mean_broadcastable

                batch_var = np.add.reduce(X_minus_mean * X_minus_mean, axis=axis) / N
                batch_var_eps = np.reshape(batch_var + eps, broadcast_shape)

                inv_std = dace.elementwise(lambda x: dace.float32(1.0) / sqrt(x), batch_var_eps)
                normalized = X_minus_mean * inv_std

                scale_reshaped = np.reshape(scale, broadcast_shape)
                bias_reshaped = np.reshape(B, broadcast_shape)
                Y[:] = normalized * scale_reshaped + bias_reshaped

                # FIXED: PyTorch EMA
                # running = (1 - momentum) * running + momentum * batch
                running_mean[:] = input_mean * (1.0 - momentum) + batch_mean * momentum
                running_var[:] = input_var * (1.0 - momentum) + batch_var * momentum

            new_sdfg = program_for_node(prog, sdfg, state, node)

            # Keep your "write-back" edges as-is
            new_state = sdfg.add_state_after(sdfg.nodes()[0])
            rm_name = out_edge_with_name(node, state, "running_mean").data.data
            new_state.add_edge(new_state.add_read(rm_name), None,
                               new_state.add_read(in_edge_with_name(node, state, "input_mean").data.data), None,
                               sdfg.make_array_memlet(rm_name))
            rv_name = out_edge_with_name(node, state, "running_var").data.data
            new_state.add_edge(new_state.add_read(rv_name), None,
                               new_state.add_read(in_edge_with_name(node, state, "input_var").data.data), None,
                               sdfg.make_array_memlet(rv_name))
        else:
            # EVAL: use provided running statistics; DO NOT recompute mean/var
            def prog(input_mean, scale, input_var, B, X, Y):
                mean_b = dace.define_local(broadcast_shape, dtype)
                var_b = dace.define_local(broadcast_shape, dtype)
                mean_b[:] = input_mean
                var_b[:] = input_var

                X_minus_mean = X - mean_b
                inv_std = dace.elementwise(lambda x: dace.float32(1.0) / sqrt(x + eps), var_b)

                normalized = X_minus_mean * inv_std
                scale_b = np.reshape(scale, broadcast_shape)
                bias_b = np.reshape(B, broadcast_shape)
                Y[:] = normalized * scale_b + bias_b

            new_sdfg = program_for_node(prog, sdfg, state, node)

        return new_sdfg


@op_implementation(op="GlobalAveragePool", name="pure")
class PureGlobalAveragePool(ONNXForward):
    """Pure implementation of GlobalAveragePool operation."""

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        """Check if this implementation can be applied to the given node.

        Args:
            node: The GlobalAveragePool ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            Always True for this implementation
        """
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        """Generate the forward pass implementation for GlobalAveragePool.

        Args:
            node: The GlobalAveragePool ONNX node
            state: The SDFG state containing the node
            sdfg: The parent SDFG

        Returns:
            A nested SDFG implementing the GlobalAveragePool operation
        """
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXReduceMean

        # Get input and output descriptors
        X_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "X"))
        Y_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "Y"))

        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors
        nsdfg.add_datadesc("X", X_desc)
        nsdfg.add_datadesc("Y", Y_desc)
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Add access nodes
        X_read = nstate.add_read("X")
        Y_write = nstate.add_write("Y")

        # Create axes array for reduction over spatial dimensions (2, 3)
        axes_name = "axes"
        rank = len(X_desc.shape)  # e.g., (N, C, H, W) -> 4
        axes_values = list(range(2, rank))
        axes_arr_dtype = dace.int64
        axes_arr_shape = [len(axes_values)]
        _, axes_desc = nsdfg.add_array(axes_name, axes_arr_shape, axes_arr_dtype, transient=True)
        axes_node = nstate.add_access(axes_name)

        # Add a tasklet to initialize the axes array
        axes_init_tasklet = nstate.add_tasklet("init_axes",
                                               set(), {"out": dace.pointer(axes_arr_dtype)},
                                               "\n".join(
                                                   [f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
                                               language=dace.Language.CPP)
        nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"{axes_name}[0:{len(axes_values)}]"))

        # Create ONNXReduceMean node
        reduce_mean_op = ONNXReduceMean("reduce_mean", keepdims=1)
        reduce_mean_op.axes = axes_values
        nstate.add_node(reduce_mean_op)
        reduce_mean_op.add_in_connector("data")
        reduce_mean_op.add_in_connector("axes")
        reduce_mean_op.add_out_connector("reduced")

        # Connect the ReduceMean operation
        nstate.add_edge(X_read, None, reduce_mean_op, "data", nsdfg.make_array_memlet("X"))
        nstate.add_edge(axes_node, None, reduce_mean_op, "axes", nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(reduce_mean_op, "reduced", Y_write, None, nsdfg.make_array_memlet("Y"))

        return nsdfg
