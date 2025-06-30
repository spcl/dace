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
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
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

        # Add access nodes
        X_read = nstate.add_read("X")
        scale_read = nstate.add_read("scale")
        B_read = nstate.add_read("B")
        input_mean_read = nstate.add_read("input_mean")
        input_var_read = nstate.add_read("input_var")
        Y_write = nstate.add_write("Y")

        # Check if we're in training mode
        training_mode = getattr(node, 'training_mode', 0)
        epsilon = getattr(node, 'epsilon', 1e-5)
        momentum = getattr(node, 'momentum', 0.9)

        # Check if optional outputs exist
        has_running_mean = len(list(state.out_edges_by_connector(node, "running_mean"))) > 0
        has_running_var = len(list(state.out_edges_by_connector(node, "running_var"))) > 0
        running_mean_write = None
        running_var_write = None

        if has_running_mean:
            running_mean_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "running_mean"))
            nsdfg.add_datadesc("running_mean", running_mean_desc)
            nsdfg.arrays["running_mean"].transient = False
            running_mean_write = nstate.add_write("running_mean")

        if has_running_var:
            running_var_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "running_var"))
            nsdfg.add_datadesc("running_var", running_var_desc)
            nsdfg.arrays["running_var"].transient = False
            running_var_write = nstate.add_write("running_var")

        # Create tasklet inputs and outputs
        tasklet_inputs = {
            "__X": dace.pointer(X_desc.dtype),
            "__scale": dace.pointer(scale_desc.dtype),
            "__B": dace.pointer(B_desc.dtype),
            "__input_mean": dace.pointer(input_mean_desc.dtype),
            "__input_var": dace.pointer(input_var_desc.dtype),
        }
        tasklet_outputs = {
            "__Y": dace.pointer(Y_desc.dtype),
        }
        if has_running_mean:
            tasklet_outputs["__running_mean"] = dace.pointer(running_mean_desc.dtype)
        if has_running_var:
            tasklet_outputs["__running_var"] = dace.pointer(running_var_desc.dtype)

        # Generate code for the tasklet
        code = []
        
        # Get dimensions
        rank = len(X_desc.shape)
        channel_dim = 1  # C is typically the second dimension (N x C x D1 x D2 ...)
        
        # Calculate size of dimensions to reduce over (all except channel)
        reduce_size = 1
        for i in range(rank):
            if i != channel_dim:
                reduce_size *= X_desc.shape[i]
        
        # Generate loops for all dimensions
        for i in range(rank):
            code.append(f"for (int i{i} = 0; i{i} < {X_desc.shape[i]}; i{i}++) {{")
        
        # Calculate indices
        x_idx = " + ".join([f"i{i} * {X_desc.strides[i]}" for i in range(rank)])
        y_idx = " + ".join([f"i{i} * {Y_desc.strides[i]}" for i in range(rank)])
        channel_idx = f"i{channel_dim}"
        
        if training_mode:
            # Training mode
            code.append(f"""
            // Calculate current mean and variance
            float bn_sum = 0.0f;
            float bn_sq_sum = 0.0f;
            for (int i = 0; i < {reduce_size}; i++) {{
                float bn_val = __X[{x_idx}];
                bn_sum += bn_val;
                bn_sq_sum += bn_val * bn_val;
            }}
            float bn_current_mean = bn_sum / {reduce_size};
            float bn_current_var = (bn_sq_sum / {reduce_size}) - (bn_current_mean * bn_current_mean);
            
            // Update running statistics
            float bn_running_mean_val = __input_mean[{channel_idx}] * {momentum} + bn_current_mean * (1 - {momentum});
            float bn_running_var_val = __input_var[{channel_idx}] * {momentum} + bn_current_var * (1 - {momentum});
            """)
            
            if has_running_mean:
                code.append(f"__running_mean[{channel_idx}] = bn_running_mean_val;")
            if has_running_var:
                code.append(f"__running_var[{channel_idx}] = bn_running_var_val;")
                
            code.append(f"""
            // Normalize using current statistics
            float bn_normalized = (__X[{x_idx}] - bn_current_mean) / sqrt(bn_current_var + {epsilon});
            __Y[{y_idx}] = bn_normalized * __scale[{channel_idx}] + __B[{channel_idx}];
            """)
        else:
            # Inference mode
            code.append(f"""
            // Normalize using input statistics
            float bn_normalized = (__X[{x_idx}] - __input_mean[{channel_idx}]) / sqrt(__input_var[{channel_idx}] + {epsilon});
            __Y[{y_idx}] = bn_normalized * __scale[{channel_idx}] + __B[{channel_idx}];
            """)
        
        # Close dimension loops
        for _ in range(rank):
            code.append("}")

        # Create tasklet
        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs=tasklet_inputs,
            outputs=tasklet_outputs,
            code="\n".join(code),
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", 
                       dace.Memlet.from_array("X", X_desc))
        nstate.add_edge(scale_read, None, tasklet, "__scale",
                       dace.Memlet.from_array("scale", scale_desc))
        nstate.add_edge(B_read, None, tasklet, "__B",
                       dace.Memlet.from_array("B", B_desc))
        nstate.add_edge(input_mean_read, None, tasklet, "__input_mean",
                       dace.Memlet.from_array("input_mean", input_mean_desc))
        nstate.add_edge(input_var_read, None, tasklet, "__input_var",
                       dace.Memlet.from_array("input_var", input_var_desc))
        nstate.add_edge(tasklet, "__Y", Y_write, None,
                       dace.Memlet.from_array("Y", Y_desc))
        if has_running_mean:
            nstate.add_edge(tasklet, "__running_mean", running_mean_write, None,
                           dace.Memlet.from_array("running_mean", running_mean_desc))
        if has_running_var:
            nstate.add_edge(tasklet, "__running_var", running_var_write, None,
                           dace.Memlet.from_array("running_var", running_var_desc))

        return nsdfg


@python_pure_op_implementation
def GlobalAveragePool(X, Y):
    Y[:] = np.mean(X, axis=[2, 3])


@op_implementation(op="GlobalAveragePool", name="pure")
class PureGlobalAveragePool(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        x_name = list(node.in_connectors)[0]
        y_name = list(node.out_connectors)[0]

        in_edge = state.in_edges(node)[0]
        out_edge = state.out_edges(node)[0]

        x_data = sdfg.data(in_edge.data.data)
        y_data = sdfg.data(out_edge.data.data)

        new_sdfg = SDFG(name='GlobalAveragePoolExpansion')
        new_state = new_sdfg.add_state()

        new_sdfg.add_array(name=x_name, shape=x_data.shape, dtype=x_data.dtype)
        new_sdfg.add_array(name=y_name, shape=y_data.shape, dtype=y_data.dtype)

        map_entry_1, map_exit_1 = new_state.add_map(name='grid_map',
                                                    ndrange={
                                                        'a':
                                                        f'0:{x_data.shape[0]}',
                                                        'b':
                                                        f'0:{x_data.shape[1]}'
                                                    })

        map_entry_2, map_exit_2 = new_state.add_map(
            name='block_map', ndrange={'c': f'0:{x_data.shape[2]}'})

        map_entry_3, map_exit_3 = new_state.add_map(
            name='thread_map', ndrange={'d': f'0:{x_data.shape[3]}'})

        input_access = new_state.add_access(x_name)
        output_access = new_state.add_access(y_name)

        tasklet = new_state.add_tasklet(name='reduce',
                                        inputs={'x'},
                                        outputs={'y'},
                                        code='y = x')

        new_state.add_memlet_path(input_access,
                                  map_entry_1,
                                  map_entry_2,
                                  map_entry_3,
                                  tasklet,
                                  dst_conn='x',
                                  memlet=dace.Memlet(data=x_name,
                                                     subset="a,b,c,d"))

        acc_transient = 'acc_transient1'
        new_sdfg.add_transient(name=acc_transient,
                               shape=(1, ),
                               dtype=x_data.dtype)
        acc_transient_access = new_state.add_access(acc_transient)
        acc_transient_access.setzero = True

        new_state.add_memlet_path(tasklet,
                                  map_exit_3,
                                  acc_transient_access,
                                  src_conn='y',
                                  memlet=dace.Memlet(data=acc_transient,
                                                     subset="0",
                                                     wcr='lambda a, b: a + b'))

        red = new_state.add_reduce(wcr='lambda a,b: a+b',
                                   axes=None,
                                   identity=0)
        red.name = "reduce_cuda_block"
        red.implementation = 'CUDA (block)'

        new_state.add_edge(acc_transient_access, None, red, None,
                           dace.Memlet(data=acc_transient, subset="0"))

        red_transient = 'red_transient1'
        new_sdfg.add_transient(name=red_transient,
                               shape=(1, ),
                               dtype=x_data.dtype)
        red_transient_access = new_state.add_access(red_transient)

        writeout_tasklet = new_state.add_tasklet(
            'writeout1', {'inp'}, {'out'},
            f'if c == 0: out = inp * {1. / (x_data.shape[2] * x_data.shape[3])}'
        )

        new_state.add_edge(red, None, red_transient_access, None,
                           dace.Memlet(data=red_transient, subset="0"))
        new_state.add_edge(red_transient_access, None, writeout_tasklet, 'inp',
                           dace.Memlet(data=red_transient, subset="0"))

        new_state.add_memlet_path(writeout_tasklet,
                                  map_exit_2,
                                  map_exit_1,
                                  output_access,
                                  src_conn='out',
                                  memlet=dace.Memlet(data=y_name,
                                                     subset="a,b",
                                                     dynamic=True,
                                                     volume=0))

        return new_sdfg
