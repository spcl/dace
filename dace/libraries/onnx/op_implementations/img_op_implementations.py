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
from dace.libraries.onnx.op_implementations.utils import python_pure_op_implementation


def _2d_sliding_window_index_expr(x_or_y, stride, pad, kernel_size):
    return f"out_{x_or_y} * {stride} + h{x_or_y} - {pad}"


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@op_implementation(op="MaxPool", name="pure")
class PureMaxPool2D(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")

        if "Indices" in {e.src_conn for e in state.out_edges(node)}:
            return False

        image_dims = len(X.shape) - 2

        # only do 2D for now
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
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        image_dims = len(X.shape) - 2
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        pads = node.pads if node.pads is not None else [0 for _ in range(image_dims) * 2]
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
        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
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
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", 
                       dace.Memlet.from_array("X", X))
        nstate.add_edge(tasklet, "__Y", Y_write, None,
                       dace.Memlet.from_array("Y", Y))

        return nsdfg


@op_implementation(op="Conv", name="pure")
class PureConv2D(ONNXForward):
    """ The "trivial" convolution implementation, i.e. two nested maps.
    """

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
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

        # only do 2D for now
        if len(X.shape) != 4 or len(W.shape) != 4:
            return False

        if node.group != 1:
            return False

        if num_channels != W.shape[1]:
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

        if node.auto_pad != 'NOTSET':
            return False

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
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

        input_size_x, input_size_y = X.shape[2:]
        output_size_y, output_size_x = Y.shape[2:]
        stride_y, stride_x = node.strides or [1, 1]
        pad_x, pad_y, _, _ = node.pads or [0, 0, 0, 0]

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

        # Generate C++ code for the convolution
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
        // Zero initialize output
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

        // Main convolution computation
        for (int b = 0; b < {batch_size}; b++) {{
            for (int m = 0; m < {num_filters}; m++) {{
                for (int out_x = 0; out_x < {output_size_x}; out_x++) {{
                    for (int out_y = 0; out_y < {output_size_y}; out_y++) {{
                        for (int cin = 0; cin < {num_channels}; cin++) {{
                            for (int hx = 0; hx < {filter_hx}; hx++) {{
                                for (int hy = 0; hy < {filter_hy}; hy++) {{
                                    int sx = hx + out_x * {stride_x} - {pad_x};
                                    int sy = hy + out_y * {stride_y} - {pad_y};
                                    
                                    if (0 <= sx && sx < {input_size_x} && 0 <= sy && sy < {input_size_y}) {{
                                        float filter = __W[m * {W.strides[0]} + cin * {W.strides[1]} + hx * {W.strides[2]} + hy * {W.strides[3]}];
                                        float image = __X[b * {X.strides[0]} + cin * {X.strides[1]} + sx * {X.strides[2]} + sy * {X.strides[3]}];
                                        __Y[b * {Y.strides[0]} + m * {Y.strides[1]} + out_x * {Y.strides[2]} + out_y * {Y.strides[3]}] += filter * image;
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
        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs=tasklet_inputs,
            outputs=tasklet_outputs,
            code=code,
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", 
                       dace.Memlet.from_array("X", X))
        nstate.add_edge(W_read, None, tasklet, "__W",
                       dace.Memlet.from_array("W", W))
        if B is not None:
            nstate.add_edge(B_read, None, tasklet, "__B",
                           dace.Memlet.from_array("B", B))
        nstate.add_edge(tasklet, "__Y", Y_write, None,
                       dace.Memlet.from_array("Y", Y))

        return nsdfg

@op_implementation(op="BatchNormalization", name="pure")
class PureBatchNormalization(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        if len(X.shape) != 4:
            return False

        # only for training for now
        if not {"out_mean", "out_var", "saved_mean", "saved_var"}.issubset(
                node.out_connectors):
            return False
        if not {"scale", "B"}.issubset(node.in_connectors):
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        shape = copy.deepcopy(in_desc_with_name(node, state, sdfg, "X").shape)
        reduce_axes = list(shape)
        num_channels = reduce_axes.pop(1)

        N = _prod(reduce_axes)
        broadcast_shape = [num_channels, 1, 1]
        dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        eps = node.epsilon
        momentum = node.momentum
        inv_momentum = 1 - node.momentum

        axis = tuple(i for i in range(len(shape)) if i != 1)

        def prog(X, scale, B, in_mean, in_var, Y, out_mean, out_var,
                 saved_mean, saved_var):
            saved_mean[:] = np.add.reduce(X, axis=axis) / N

            saved_mean_broadcastable = dace.define_local(
                broadcast_shape, dtype)
            # this copy will get removed after parsing -- using reshape here would be nicer
            # but it messes with statefusion
            saved_mean_broadcastable[:] = saved_mean

            X_minus_mean = (X - saved_mean_broadcastable)

            saved_var[:] = np.add.reduce(X_minus_mean * X_minus_mean,
                                         axis=axis) / N
            saved_var_eps = np.reshape(saved_var + eps, broadcast_shape)

            normalized = X_minus_mean * dace.elementwise(
                lambda x: dace.float32(1.0) / sqrt(x), saved_var_eps)

            scale_reshaped = np.reshape(scale, broadcast_shape)
            bias_reshaped = np.reshape(B, broadcast_shape)
            Y[:] = normalized * scale_reshaped + bias_reshaped

            out_mean[:] = in_mean * momentum + saved_mean * inv_momentum
            out_var[:] = in_var * momentum + saved_var * inv_momentum

        new_sdfg = program_for_node(prog, sdfg, state, node)

        # write the mean and var back to the parameters so that they are updated
        # this is a bit of a hack, but the ONNX spec is currently not really working for training
        new_state = sdfg.add_state_after(sdfg.nodes()[0])
        mean_data_name = out_edge_with_name(node, state, "out_mean").data.data
        read_mean = new_state.add_read(mean_data_name)
        write_mean = new_state.add_read(
            in_edge_with_name(node, state, "in_mean").data.data)
        new_state.add_edge(read_mean, None, write_mean, None,
                           sdfg.make_array_memlet(mean_data_name))

        var_data_name = out_edge_with_name(node, state, "out_var").data.data
        read_var = new_state.add_read(var_data_name)
        write_var = new_state.add_read(
            in_edge_with_name(node, state, "in_var").data.data)
        new_state.add_edge(read_var, None, write_var, None,
                           sdfg.make_array_memlet(var_data_name))

        return new_sdfg
    
@op_implementation(op="BatchNormalization", name="pure")
class PureBatchNormalization(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        # Avoid this expansion if the backward pass will be contructed
        # TODO pass the backward flag to the functions
        return False

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXAdd, ONNXSub, ONNXMul, ONNXDiv, ONNXSqrt, ONNXReduceMean, ONNXUnsqueeze
        
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        X_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "X"))
        scale_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "scale"))
        B_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "B"))
        input_mean_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "input_mean"))
        input_var_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "input_var"))
        Y_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "Y"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("X", X_desc)
        nsdfg.add_datadesc("scale", scale_desc)
        nsdfg.add_datadesc("B", B_desc)
        nsdfg.add_datadesc("input_mean", input_mean_desc)
        nsdfg.add_datadesc("input_var", input_var_desc)
        nsdfg.add_datadesc("Y", Y_desc)
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["scale"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["input_mean"].transient = False
        nsdfg.arrays["input_var"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Check if we're in training mode
        training_mode = getattr(node, 'training_mode', 0)
        epsilon = getattr(node, 'epsilon', 1e-5)
        momentum = getattr(node, 'momentum', 0.9)

        # Check if optional outputs exist
        has_running_mean = len(list(state.out_edges_by_connector(node, "running_mean"))) > 0
        has_running_var = len(list(state.out_edges_by_connector(node, "running_var"))) > 0

        if has_running_mean:
            running_mean_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "running_mean"))
            nsdfg.add_datadesc("running_mean", running_mean_desc)
            nsdfg.arrays["running_mean"].transient = False

        if has_running_var:
            running_var_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "running_var"))
            nsdfg.add_datadesc("running_var", running_var_desc)
            nsdfg.arrays["running_var"].transient = False

        # Get dimensions
        rank = len(X_desc.shape)
        channel_dim = 1  # C is typically the second dimension (N x C x D1 x D2 ...)
        
        # Create axes for reduction (all dimensions except channel dimension)
        reduce_axes = [i for i in range(rank) if i != channel_dim]
        
        # Create axes array for reduction operations
        axes_name = "reduce_axes"
        axes_shape = [len(reduce_axes)]
        axes_dtype = dace.int64
        nsdfg.add_array(axes_name, axes_shape, axes_dtype, transient=True)
        axes_access = nstate.add_access(axes_name)
        
        # Initialize axes array
        axes_init_tasklet = nstate.add_tasklet(
            "init_axes",
            set(),
            {"out": dace.pointer(axes_dtype)},
            "\n".join([f"out[{i}] = {reduce_axes[i]};" for i in range(len(reduce_axes))]),
            language=dace.Language.CPP
        )
        nstate.add_edge(axes_init_tasklet, "out", axes_access, None, 
                       dace.Memlet(f"{axes_name}[0:{len(reduce_axes)}]"))

        # Add access nodes for inputs
        X_read = nstate.add_read("X")
        scale_read = nstate.add_read("scale")
        B_read = nstate.add_read("B")
        input_mean_read = nstate.add_read("input_mean")
        input_var_read = nstate.add_read("input_var")
        Y_write = nstate.add_write("Y")

        if training_mode:
            # Training mode - compute statistics and normalize
            
            # Step 1: Compute mean across batch and spatial dimensions
            mean_name = "current_mean"
            mean_shape = [X_desc.shape[channel_dim]]
            mean_desc = dace.data.Array(X_desc.dtype, mean_shape)
            mean_desc.transient = True
            nsdfg.add_datadesc(mean_name, mean_desc)
            mean_access = nstate.add_access(mean_name)
            
            mean_op = ONNXReduceMean("compute_mean", keepdims=0)
            nstate.add_node(mean_op)
            mean_op.add_in_connector("data")
            mean_op.add_in_connector("axes")
            mean_op.add_out_connector("reduced")
            
            nstate.add_edge(X_read, None, mean_op, "data", 
                           nsdfg.make_array_memlet("X"))
            nstate.add_edge(axes_access, None, mean_op, "axes", 
                           nsdfg.make_array_memlet(axes_name))
            nstate.add_edge(mean_op, "reduced", mean_access, None, 
                           nsdfg.make_array_memlet(mean_name))
            
            # Step 2: Compute variance
            # First, unsqueeze mean to match input shape for proper broadcasting
            mean_unsqueezed_name = "mean_unsqueezed"
            mean_unsqueezed_shape = [1 for _ in range(rank)]
            mean_unsqueezed_shape[channel_dim] = X_desc.shape[channel_dim]  # Keep original channel dimension
            mean_unsqueezed_desc = dace.data.Array(X_desc.dtype, mean_unsqueezed_shape, transient=True)
            nsdfg.add_datadesc(mean_unsqueezed_name, mean_unsqueezed_desc)
            mean_unsqueezed_access = nstate.add_access(mean_unsqueezed_name)
            
            # Create axes for unsqueezing mean (add dimensions for batch and spatial dims)
            mean_axes_name = "mean_unsqueeze_axes"
            mean_axes_shape = [rank - 1]  # All dimensions except channel
            mean_axes_dtype = dace.int64
            nsdfg.add_array(mean_axes_name, mean_axes_shape, mean_axes_dtype, transient=True)
            mean_axes_access = nstate.add_access(mean_axes_name)
            
            # Initialize mean unsqueeze axes (0, 2, 3, ... for NCHW format)
            mean_axes_init_tasklet = nstate.add_tasklet(
                "init_mean_axes",
                set(),
                {"out": dace.pointer(mean_axes_dtype)},
                "\n".join([f"out[{i}] = {i if i < channel_dim else i + 1};" for i in range(rank - 1)]),
                language=dace.Language.CPP
            )
            nstate.add_edge(mean_axes_init_tasklet, "out", mean_axes_access, None, 
                           dace.Memlet(f"{mean_axes_name}[0:{rank - 1}]"))
            
            from dace.libraries.onnx.nodes.onnx_op_registry import ONNXUnsqueeze
            mean_unsqueeze_op = ONNXUnsqueeze("unsqueeze_mean")
            nstate.add_node(mean_unsqueeze_op)
            mean_unsqueeze_op.add_in_connector("data")
            mean_unsqueeze_op.add_in_connector("axes")
            mean_unsqueeze_op.add_out_connector("expanded")
            
            nstate.add_edge(mean_access, None, mean_unsqueeze_op, "data", 
                           nsdfg.make_array_memlet(mean_name))
            nstate.add_edge(mean_axes_access, None, mean_unsqueeze_op, "axes", 
                           nsdfg.make_array_memlet(mean_axes_name))
            nstate.add_edge(mean_unsqueeze_op, "expanded", mean_unsqueezed_access, None, 
                           nsdfg.make_array_memlet(mean_unsqueezed_name))
            
            # Now subtract unsqueezed mean from input
            centered_name = "centered"
            centered_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            centered_desc.transient = True
            nsdfg.add_datadesc(centered_name, centered_desc)
            centered_access = nstate.add_access(centered_name)
            
            sub_op = ONNXSub("subtract_mean")
            nstate.add_node(sub_op)
            sub_op.add_in_connector("A")
            sub_op.add_in_connector("B")
            sub_op.add_out_connector("C")
            
            nstate.add_edge(X_read, None, sub_op, "A", 
                           nsdfg.make_array_memlet("X"))
            nstate.add_edge(mean_unsqueezed_access, None, sub_op, "B", 
                           nsdfg.make_array_memlet(mean_unsqueezed_name))
            nstate.add_edge(sub_op, "C", centered_access, None, 
                           nsdfg.make_array_memlet(centered_name))
            
            # Square the centered values
            squared_name = "squared"
            squared_desc = dace.data.Array(X_desc.dtype, X_desc.shape, transient=True)
            nsdfg.add_datadesc(squared_name, squared_desc)
            squared_access = nstate.add_access(squared_name)
            
            square_op = ONNXMul("square")
            nstate.add_node(square_op)
            square_op.add_in_connector("A")
            square_op.add_in_connector("B")
            square_op.add_out_connector("C")
            
            nstate.add_edge(centered_access, None, square_op, "A", 
                           nsdfg.make_array_memlet(centered_name))
            nstate.add_edge(centered_access, None, square_op, "B", 
                           nsdfg.make_array_memlet(centered_name))
            nstate.add_edge(square_op, "C", squared_access, None, 
                           nsdfg.make_array_memlet(squared_name))
            
            # Compute variance
            var_name = "current_var"
            var_shape = [X_desc.shape[channel_dim]]
            var_desc = dace.data.Array(X_desc.dtype, var_shape, transient=True)
            nsdfg.add_datadesc(var_name, var_desc)
            var_access = nstate.add_access(var_name)
            
            var_op = ONNXReduceMean("compute_var", keepdims=0)
            nstate.add_node(var_op)
            var_op.add_in_connector("data")
            var_op.add_in_connector("axes")
            var_op.add_out_connector("reduced")
            
            nstate.add_edge(squared_access, None, var_op, "data", 
                           nsdfg.make_array_memlet(squared_name))
            nstate.add_edge(axes_access, None, var_op, "axes", 
                           nsdfg.make_array_memlet(axes_name))
            nstate.add_edge(var_op, "reduced", var_access, None, 
                           nsdfg.make_array_memlet(var_name))
            
            # Update running statistics if needed
            if has_running_mean:
                running_mean_write = nstate.add_write("running_mean")
                # running_mean = momentum * input_mean + (1 - momentum) * current_mean
                running_mean_op = ONNXAdd("update_running_mean")
                nstate.add_node(running_mean_op)
                running_mean_op.add_in_connector("A")
                running_mean_op.add_in_connector("B")
                running_mean_op.add_out_connector("C")
                
                # Create momentum * input_mean
                momentum_mean_name = "momentum_mean"
                momentum_mean_desc = dace.data.Array(X_desc.dtype, mean_shape, transient=True)
                nsdfg.add_datadesc(momentum_mean_name, momentum_mean_desc)
                momentum_mean_access = nstate.add_access(momentum_mean_name)
                
                momentum_op = ONNXMul("scale_momentum")
                nstate.add_node(momentum_op)
                momentum_op.add_in_connector("A")
                momentum_op.add_in_connector("B")
                momentum_op.add_out_connector("C")
                
                nstate.add_edge(input_mean_read, None, momentum_op, "A", nsdfg.make_array_memlet("input_mean"))
                # Create momentum constant
                momentum_const_name = "momentum_const"
                momentum_const_desc = dace.data.Scalar(X_desc.dtype)
                momentum_const_desc.transient = True
                nsdfg.add_datadesc(momentum_const_name, momentum_const_desc)
                momentum_const_access = nstate.add_access(momentum_const_name)
                
                momentum_const_init = nstate.add_tasklet(
                    "init_momentum",
                    set(),
                    {"out": dace.pointer(X_desc.dtype)},
                    f"out[0] = {momentum};",
                    language=dace.Language.CPP
                )
                nstate.add_edge(momentum_const_init, "out", momentum_const_access, None, dace.Memlet(f"{momentum_const_name}[0]"))
                
                nstate.add_edge(momentum_const_access, None, momentum_op, "B", nsdfg.make_array_memlet(momentum_const_name))
                nstate.add_edge(momentum_op, "C", momentum_mean_access, None, nsdfg.make_array_memlet(momentum_mean_name))
                
                # Create (1 - momentum) * current_mean
                one_minus_momentum_mean_name = "one_minus_momentum_mean"
                one_minus_momentum_mean_desc = dace.data.Array(X_desc.dtype, mean_shape, transient=True)
                nsdfg.add_datadesc(one_minus_momentum_mean_name, one_minus_momentum_mean_desc)
                one_minus_momentum_mean_access = nstate.add_access(one_minus_momentum_mean_name)
                
                one_minus_momentum_op = ONNXMul("scale_one_minus_momentum")
                nstate.add_node(one_minus_momentum_op)
                one_minus_momentum_op.add_in_connector("A")
                one_minus_momentum_op.add_in_connector("B")
                one_minus_momentum_op.add_out_connector("C")
                
                nstate.add_edge(mean_access, None, one_minus_momentum_op, "A", nsdfg.make_array_memlet(mean_name))
                
                # Create (1 - momentum) constant
                one_minus_momentum_const_name = "one_minus_momentum_const"
                one_minus_momentum_const_desc = dace.data.Scalar(X_desc.dtype)
                one_minus_momentum_const_desc.transient = True
                nsdfg.add_datadesc(one_minus_momentum_const_name, one_minus_momentum_const_desc)
                one_minus_momentum_const_access = nstate.add_access(one_minus_momentum_const_name)
                
                one_minus_momentum_const_init = nstate.add_tasklet(
                    "init_one_minus_momentum",
                    set(),
                    {"out": dace.pointer(X_desc.dtype)},
                    f"out[0] = {1 - momentum};",
                    language=dace.Language.CPP
                )
                nstate.add_edge(one_minus_momentum_const_init, "out", one_minus_momentum_const_access, None, dace.Memlet(f"{one_minus_momentum_const_name}[0]"))
                
                nstate.add_edge(one_minus_momentum_const_access, None, one_minus_momentum_op, "B", nsdfg.make_array_memlet(one_minus_momentum_const_name))
                nstate.add_edge(one_minus_momentum_op, "C", one_minus_momentum_mean_access, None, nsdfg.make_array_memlet(one_minus_momentum_mean_name))
                
                # Add them together
                nstate.add_edge(momentum_mean_access, None, running_mean_op, "A", nsdfg.make_array_memlet(momentum_mean_name))
                nstate.add_edge(one_minus_momentum_mean_access, None, running_mean_op, "B", nsdfg.make_array_memlet(one_minus_momentum_mean_name))
                nstate.add_edge(running_mean_op, "C", running_mean_write, None, nsdfg.make_array_memlet("running_mean"))
            
            if has_running_var:
                running_var_write = nstate.add_write("running_var")
                # Similar logic for running variance
                running_var_op = ONNXAdd("update_running_var")
                nstate.add_node(running_var_op)
                running_var_op.add_in_connector("A")
                running_var_op.add_in_connector("B")
                running_var_op.add_out_connector("C")
                
                # Create momentum * input_var
                momentum_var_name = "momentum_var"
                momentum_var_desc = dace.data.Array(X_desc.dtype, var_shape)
                momentum_var_desc.transient = True
                nsdfg.add_datadesc(momentum_var_name, momentum_var_desc)
                momentum_var_access = nstate.add_access(momentum_var_name)
                
                momentum_var_op = ONNXMul("scale_momentum_var")
                nstate.add_node(momentum_var_op)
                momentum_var_op.add_in_connector("A")
                momentum_var_op.add_in_connector("B")
                momentum_var_op.add_out_connector("C")
                
                nstate.add_edge(input_var_read, None, momentum_var_op, "A", 
                               nsdfg.make_array_memlet("input_var"))
                nstate.add_edge(momentum_const_access, None, momentum_var_op, "B", 
                               nsdfg.make_array_memlet(momentum_const_name))
                nstate.add_edge(momentum_var_op, "C", momentum_var_access, None, 
                               nsdfg.make_array_memlet(momentum_var_name))
                
                # Create (1 - momentum) * current_var
                one_minus_momentum_var_name = "one_minus_momentum_var"
                one_minus_momentum_var_desc = dace.data.Array(X_desc.dtype, var_shape)
                one_minus_momentum_var_desc.transient = True
                nsdfg.add_datadesc(one_minus_momentum_var_name, one_minus_momentum_var_desc)
                one_minus_momentum_var_access = nstate.add_access(one_minus_momentum_var_name)
                
                one_minus_momentum_var_op = ONNXMul("scale_one_minus_momentum_var")
                nstate.add_node(one_minus_momentum_var_op)
                one_minus_momentum_var_op.add_in_connector("A")
                one_minus_momentum_var_op.add_in_connector("B")
                one_minus_momentum_var_op.add_out_connector("C")
                
                nstate.add_edge(var_access, None, one_minus_momentum_var_op, "A", 
                               nsdfg.make_array_memlet(var_name))
                nstate.add_edge(one_minus_momentum_const_access, None, one_minus_momentum_var_op, "B", 
                               nsdfg.make_array_memlet(one_minus_momentum_const_name))
                nstate.add_edge(one_minus_momentum_var_op, "C", one_minus_momentum_var_access, None, 
                               nsdfg.make_array_memlet(one_minus_momentum_var_name))
                
                # Add them together
                nstate.add_edge(momentum_var_access, None, running_var_op, "A", 
                               nsdfg.make_array_memlet(momentum_var_name))
                nstate.add_edge(one_minus_momentum_var_access, None, running_var_op, "B", 
                               nsdfg.make_array_memlet(one_minus_momentum_var_name))
                nstate.add_edge(running_var_op, "C", running_var_write, None, 
                               nsdfg.make_array_memlet("running_var"))
            
            # Use current mean and variance for normalization
            mean_for_norm = mean_access
            var_for_norm = var_access
            
        else:
            # Inference mode - use pre-computed statistics
            mean_for_norm = input_mean_read
            var_for_norm = input_var_read
        
        # Step 3: Normalize
        # For inference mode, we need to subtract mean from input first
        if not training_mode:
            # Unsqueeze mean to match input shape for proper broadcasting in inference mode
            mean_unsqueezed_name = "mean_unsqueezed_inference"
            mean_unsqueezed_shape = [1 for _ in range(rank)]
            mean_unsqueezed_shape[channel_dim] = X_desc.shape[channel_dim]  # Keep original channel dimension
            mean_unsqueezed_desc = dace.data.Array(X_desc.dtype, mean_unsqueezed_shape, transient=True)
            nsdfg.add_datadesc(mean_unsqueezed_name, mean_unsqueezed_desc)
            mean_unsqueezed_access = nstate.add_access(mean_unsqueezed_name)
            
            # Create axes for unsqueezing mean (add dimensions for batch and spatial dims)
            mean_axes_name = "mean_unsqueeze_axes_inference"
            mean_axes_shape = [rank - 1]  # All dimensions except channel
            mean_axes_dtype = dace.int64
            nsdfg.add_array(mean_axes_name, mean_axes_shape, mean_axes_dtype, transient=True)
            mean_axes_access = nstate.add_access(mean_axes_name)
            
            # Initialize mean unsqueeze axes (0, 2, 3, ... for NCHW format)
            mean_axes_init_tasklet = nstate.add_tasklet(
                "init_mean_axes_inference",
                set(),
                {"out": dace.pointer(mean_axes_dtype)},
                "\n".join([f"out[{i}] = {i if i < channel_dim else i + 1};" for i in range(rank - 1)]),
                language=dace.Language.CPP
            )
            nstate.add_edge(mean_axes_init_tasklet, "out", mean_axes_access, None, 
                           dace.Memlet(f"{mean_axes_name}[0:{rank - 1}]"))
            
            mean_unsqueeze_op = ONNXUnsqueeze("unsqueeze_mean_inference")
            nstate.add_node(mean_unsqueeze_op)
            mean_unsqueeze_op.add_in_connector("data")
            mean_unsqueeze_op.add_in_connector("axes")
            mean_unsqueeze_op.add_out_connector("expanded")
            
            nstate.add_edge(mean_for_norm, None, mean_unsqueeze_op, "data", 
                           nsdfg.make_array_memlet(mean_for_norm.desc.name))
            nstate.add_edge(mean_axes_access, None, mean_unsqueeze_op, "axes", 
                           nsdfg.make_array_memlet(mean_axes_name))
            nstate.add_edge(mean_unsqueeze_op, "expanded", mean_unsqueezed_access, None, 
                           nsdfg.make_array_memlet(mean_unsqueezed_name))
            
            # Subtract unsqueezed mean from input for inference mode
            centered_name = "centered"
            centered_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            centered_desc.transient = True
            nsdfg.add_datadesc(centered_name, centered_desc)
            centered_access = nstate.add_access(centered_name)
            
            sub_op = ONNXSub("subtract_mean_inference")
            nstate.add_node(sub_op)
            sub_op.add_in_connector("A")
            sub_op.add_in_connector("B")
            sub_op.add_out_connector("C")
            
            nstate.add_edge(X_read, None, sub_op, "A", 
                           nsdfg.make_array_memlet("X"))
            nstate.add_edge(mean_unsqueezed_access, None, sub_op, "B", 
                           nsdfg.make_array_memlet(mean_unsqueezed_name))
            nstate.add_edge(sub_op, "C", centered_access, None, 
                           nsdfg.make_array_memlet(centered_name))
        
        # Add epsilon to variance
        var_plus_epsilon_name = "var_plus_epsilon"
        var_plus_epsilon_desc = dace.data.Array(X_desc.dtype, var_for_norm.desc(nsdfg).shape, transient=True)
        nsdfg.add_datadesc(var_plus_epsilon_name, var_plus_epsilon_desc)
        var_plus_epsilon_access = nstate.add_access(var_plus_epsilon_name)
        
        epsilon_add_op = ONNXAdd("add_epsilon")
        nstate.add_node(epsilon_add_op)
        epsilon_add_op.add_in_connector("A")
        epsilon_add_op.add_in_connector("B")
        epsilon_add_op.add_out_connector("C")
        
        nstate.add_edge(var_for_norm, None, epsilon_add_op, "A", nsdfg.make_array_memlet(var_for_norm.data))
        
        # Create epsilon constant
        epsilon_const_name = "epsilon_const"
        epsilon_const_desc = dace.data.Scalar(X_desc.dtype, transient=True)
        nsdfg.add_datadesc(epsilon_const_name, epsilon_const_desc)
        epsilon_const_access = nstate.add_access(epsilon_const_name)
        
        epsilon_const_init = nstate.add_tasklet(
            "init_epsilon",
            set(),
            {"out": dace.pointer(X_desc.dtype)},
            f"out[0] = {epsilon};",
            language=dace.Language.CPP
        )
        nstate.add_edge(epsilon_const_init, "out", epsilon_const_access, None, 
                       dace.Memlet(f"{epsilon_const_name}[0]"))
        
        nstate.add_edge(epsilon_const_access, None, epsilon_add_op, "B", 
                       nsdfg.make_array_memlet(epsilon_const_name))
        nstate.add_edge(epsilon_add_op, "C", var_plus_epsilon_access, None, 
                       nsdfg.make_array_memlet(var_plus_epsilon_name))
        
        # Compute sqrt(var + epsilon)
        std_name = "std"
        std_desc = dace.data.Array(X_desc.dtype, var_plus_epsilon_desc.shape)
        std_desc.transient = True
        nsdfg.add_datadesc(std_name, std_desc)
        std_access = nstate.add_access(std_name)
        
        sqrt_op = ONNXSqrt("compute_std")
        nstate.add_node(sqrt_op)
        sqrt_op.add_in_connector("X")
        sqrt_op.add_out_connector("Y")
        
        nstate.add_edge(var_plus_epsilon_access, None, sqrt_op, "X", 
                       nsdfg.make_array_memlet(var_plus_epsilon_name))
        nstate.add_edge(sqrt_op, "Y", std_access, None, 
                       nsdfg.make_array_memlet(std_name))
        
        # Unsqueeze std to match input shape for proper broadcasting
        std_unsqueezed_name = "std_unsqueezed"
        std_unsqueezed_shape = [1 for _ in range(rank)]
        std_unsqueezed_shape[channel_dim] = X_desc.shape[channel_dim]  # Keep original channel dimension
        std_unsqueezed_desc = dace.data.Array(X_desc.dtype, std_unsqueezed_shape, transient=True)
        nsdfg.add_datadesc(std_unsqueezed_name, std_unsqueezed_desc)
        std_unsqueezed_access = nstate.add_access(std_unsqueezed_name)
        
        # Create axes for unsqueezing std (add dimensions for batch and spatial dims)
        std_axes_name = "std_unsqueeze_axes"
        std_axes_shape = [rank - 1]  # All dimensions except channel
        std_axes_dtype = dace.int64
        nsdfg.add_array(std_axes_name, std_axes_shape, std_axes_dtype, transient=True)
        std_axes_access = nstate.add_access(std_axes_name)
        
        # Initialize std unsqueeze axes (0, 2, 3, ... for NCHW format)
        std_axes_init_tasklet = nstate.add_tasklet(
            "init_std_axes",
            set(),
            {"out": dace.pointer(std_axes_dtype)},
            "\n".join([f"out[{i}] = {i if i < channel_dim else i + 1};" for i in range(rank - 1)]),
            language=dace.Language.CPP
        )
        nstate.add_edge(std_axes_init_tasklet, "out", std_axes_access, None, 
                       dace.Memlet(f"{std_axes_name}[0:{rank - 1}]"))
        
        std_unsqueeze_op = ONNXUnsqueeze("unsqueeze_std")
        nstate.add_node(std_unsqueeze_op)
        std_unsqueeze_op.add_in_connector("data")
        std_unsqueeze_op.add_in_connector("axes")
        std_unsqueeze_op.add_out_connector("expanded")
        
        nstate.add_edge(std_access, None, std_unsqueeze_op, "data", 
                       nsdfg.make_array_memlet(std_name))
        nstate.add_edge(std_axes_access, None, std_unsqueeze_op, "axes", 
                       nsdfg.make_array_memlet(std_axes_name))
        nstate.add_edge(std_unsqueeze_op, "expanded", std_unsqueezed_access, None, 
                       nsdfg.make_array_memlet(std_unsqueezed_name))
        
        # Normalize: (X - mean) / std
        normalized_name = "normalized"
        normalized_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
        normalized_desc.transient = True
        nsdfg.add_datadesc(normalized_name, normalized_desc)
        normalized_access = nstate.add_access(normalized_name)
        
        div_op = ONNXDiv("normalize")
        nstate.add_node(div_op)
        div_op.add_in_connector("A")
        div_op.add_in_connector("B")
        div_op.add_out_connector("C")
        
        # Use centered data (either from training or inference)
        centered_data = centered_access if training_mode else centered_access
        centered_name_to_use = centered_name
        
        nstate.add_edge(centered_data, None, div_op, "A", 
                       nsdfg.make_array_memlet(centered_name_to_use))
        nstate.add_edge(std_unsqueezed_access, None, div_op, "B", 
                       nsdfg.make_array_memlet(std_unsqueezed_name))
        nstate.add_edge(div_op, "C", normalized_access, None, 
                       nsdfg.make_array_memlet(normalized_name))
        
        # Apply scale and bias: scale * normalized + bias
        # First, unsqueeze scale and bias to match input dimensions for proper broadcasting
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXUnsqueeze
        
        # Unsqueeze scale to match input dimensions
        scale_unsqueezed_name = "scale_unsqueezed"
        # Create shape with 1s in newly added dimensions
        scale_unsqueezed_shape = [1 for _ in range(rank)]
        scale_unsqueezed_shape[channel_dim] = X_desc.shape[channel_dim]  # Keep original channel dimension
        scale_unsqueezed_desc = dace.data.Array(X_desc.dtype, scale_unsqueezed_shape, transient=True)
        nsdfg.add_datadesc(scale_unsqueezed_name, scale_unsqueezed_desc)
        scale_unsqueezed_access = nstate.add_access(scale_unsqueezed_name)
        
        # Create axes for unsqueezing scale (add dimensions for batch and spatial dims)
        scale_axes_name = "scale_unsqueeze_axes"
        scale_axes_shape = [rank - 1]  # All dimensions except channel
        scale_axes_dtype = dace.int64
        nsdfg.add_array(scale_axes_name, scale_axes_shape, scale_axes_dtype, transient=True)
        scale_axes_access = nstate.add_access(scale_axes_name)
        
        # Initialize scale unsqueeze axes (0, 2, 3, ... for NCHW format)
        scale_axes_init_tasklet = nstate.add_tasklet(
            "init_scale_axes",
            set(),
            {"out": dace.pointer(scale_axes_dtype)},
            "\n".join([f"out[{i}] = {i if i < channel_dim else i + 1};" for i in range(rank - 1)]),
            language=dace.Language.CPP
        )
        nstate.add_edge(scale_axes_init_tasklet, "out", scale_axes_access, None, 
                       dace.Memlet(f"{scale_axes_name}[0:{rank - 1}]"))
        
        scale_unsqueeze_op = ONNXUnsqueeze("unsqueeze_scale")
        nstate.add_node(scale_unsqueeze_op)
        scale_unsqueeze_op.add_in_connector("data")
        scale_unsqueeze_op.add_in_connector("axes")
        scale_unsqueeze_op.add_out_connector("expanded")
        
        nstate.add_edge(scale_read, None, scale_unsqueeze_op, "data", 
                       nsdfg.make_array_memlet("scale"))
        nstate.add_edge(scale_axes_access, None, scale_unsqueeze_op, "axes", 
                       nsdfg.make_array_memlet(scale_axes_name))
        nstate.add_edge(scale_unsqueeze_op, "expanded", scale_unsqueezed_access, None, 
                       nsdfg.make_array_memlet(scale_unsqueezed_name))
        
        # Unsqueeze bias to match input dimensions
        bias_unsqueezed_name = "bias_unsqueezed"
        # Create shape with 1s in newly added dimensions
        bias_unsqueezed_shape = [1 for _ in range(rank)]
        bias_unsqueezed_shape[channel_dim] = X_desc.shape[channel_dim]  # Keep original channel dimension
        bias_unsqueezed_desc = dace.data.Array(X_desc.dtype, bias_unsqueezed_shape, transient=True)
        nsdfg.add_datadesc(bias_unsqueezed_name, bias_unsqueezed_desc)
        bias_unsqueezed_access = nstate.add_access(bias_unsqueezed_name)
        
        bias_unsqueeze_op = ONNXUnsqueeze("unsqueeze_bias")
        nstate.add_node(bias_unsqueeze_op)
        bias_unsqueeze_op.add_in_connector("data")
        bias_unsqueeze_op.add_in_connector("axes")
        bias_unsqueeze_op.add_out_connector("expanded")
        
        nstate.add_edge(B_read, None, bias_unsqueeze_op, "data", 
                       nsdfg.make_array_memlet("B"))
        nstate.add_edge(scale_axes_access, None, bias_unsqueeze_op, "axes", 
                       nsdfg.make_array_memlet(scale_axes_name))
        nstate.add_edge(bias_unsqueeze_op, "expanded", bias_unsqueezed_access, None, 
                       nsdfg.make_array_memlet(bias_unsqueezed_name))
        
        # Apply scale
        scaled_name = "scaled"
        scaled_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
        scaled_desc.transient = True
        nsdfg.add_datadesc(scaled_name, scaled_desc)
        scaled_access = nstate.add_access(scaled_name)
        
        scale_op = ONNXMul("apply_scale")
        nstate.add_node(scale_op)
        scale_op.add_in_connector("A")
        scale_op.add_in_connector("B")
        scale_op.add_out_connector("C")
        
        nstate.add_edge(normalized_access, None, scale_op, "A", 
                       nsdfg.make_array_memlet(normalized_name))
        nstate.add_edge(scale_unsqueezed_access, None, scale_op, "B", 
                       nsdfg.make_array_memlet(scale_unsqueezed_name))
        nstate.add_edge(scale_op, "C", scaled_access, None, 
                       nsdfg.make_array_memlet(scaled_name))
        
        # Add bias
        bias_op = ONNXAdd("add_bias")
        nstate.add_node(bias_op)
        bias_op.add_in_connector("A")
        bias_op.add_in_connector("B")
        bias_op.add_out_connector("C")
        
        nstate.add_edge(scaled_access, None, bias_op, "A", 
                       nsdfg.make_array_memlet(scaled_name))
        nstate.add_edge(bias_unsqueezed_access, None, bias_op, "B", 
                       nsdfg.make_array_memlet(bias_unsqueezed_name))
        nstate.add_edge(bias_op, "C", Y_write, None, 
                       nsdfg.make_array_memlet("Y"))

        return nsdfg


@op_implementation(op="GlobalAveragePool", name="pure")
class PureGlobalAveragePool(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
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
        axes_values = [2, 3]  # Reduce over spatial dimensions
        axes_arr_shape = [len(axes_values)]
        axes_arr_dtype = dace.int64
        _, axes_desc = nsdfg.add_array(axes_name, axes_arr_shape, axes_arr_dtype, transient=True)
        axes_node = nstate.add_access(axes_name)

        # Add a tasklet to initialize the axes array
        axes_init_tasklet = nstate.add_tasklet(
            "init_axes",
            set(),
            {"out": dace.pointer(axes_arr_dtype)},
            "\n".join([f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
            language=dace.Language.CPP
        )
        nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"{axes_name}[0:{len(axes_values)}]"))

        # Create ONNXReduceMean node
        reduce_mean_op = ONNXReduceMean("reduce_mean", keepdims=1)
        nstate.add_node(reduce_mean_op)
        reduce_mean_op.add_in_connector("data")
        reduce_mean_op.add_in_connector("axes")
        reduce_mean_op.add_out_connector("reduced")

        # Connect the ReduceMean operation
        nstate.add_edge(X_read, None, reduce_mean_op, "data", 
                       nsdfg.make_array_memlet("X"))
        nstate.add_edge(axes_node, None, reduce_mean_op, "axes", 
                       nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(reduce_mean_op, "reduced", Y_write, None, 
                       nsdfg.make_array_memlet("Y"))

        return nsdfg