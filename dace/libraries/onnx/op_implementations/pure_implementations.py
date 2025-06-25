import copy
import itertools
import logging
import typing

import dace
import numpy as np
from dace import SDFGState, SDFG, nodes, subsets, data
from dace.frontend.common import create_einsum_sdfg
from dace.sdfg.nodes import Node

from dace.libraries.onnx import converters
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.utils import op_implementation, program_for_node, empty_sdfg_for_node, \
    python_pure_op_implementation
from dace.transformation.onnx import constant_folding
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.util import iterables_equal, in_desc_with_name, out_desc_with_name, in_edge_with_name

log = logging.getLogger(__name__)


@python_pure_op_implementation
def Log(input, output):
    output[:] = np.log(input)


@python_pure_op_implementation
def Exp(input, output):
    output[:] = np.exp(input)


@python_pure_op_implementation
def Sqrt(X, Y):
    Y[:] = dace.elementwise(lambda x: sqrt(x), X)


@python_pure_op_implementation
def Pow(X, Y, Z):
    Z[:] = X**Y

@op_implementation(op="Concat", name="pure")
class PureConcat(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True
    
    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axis = node.axis
        
        num_inputs = len(state.in_edges(node))
        
        def inp_name(i):
            return f"inputs__{i}"
        out_name = "concat_result"
        
        inp_data = [in_desc_with_name(node, state, sdfg, inp_name(i)) for i in range(num_inputs)]
        out_data = out_desc_with_name(node, state, sdfg, out_name)
                
        nsdfg = dace.SDFG(node.label)
        
        inp_data_descs = [copy.deepcopy(desc) for desc in inp_data]        
        for i, desc in enumerate(inp_data_descs):
            desc.transient = False
            nsdfg.add_datadesc(inp_name(i), desc)
        out_data_desc = copy.deepcopy(out_data)
        out_data_desc.transient = False
        nsdfg.add_datadesc(out_name, out_data_desc)
        
        inp_shapes = [d.shape for d in inp_data]
        out_shape = out_data_desc.shape
        
        nstate = nsdfg.add_state()
        out_write = nstate.add_write(out_name)
        
        for inp_idx in range(num_inputs):
            inp_read = nstate.add_read(inp_name(inp_idx))
        
            tasklet = nstate.add_tasklet(
                f'concat_{inp_idx}', 
                {'inp': inp_data_descs[inp_idx].dtype},
                {'out': out_data_desc.dtype},
                "out = inp",
            )
            
            map_entry, map_exit = nstate.add_map(
                f"concat_map_{inp_idx}",
                {f"i{i}": f"0:{s}" for i, s in enumerate(inp_shapes[inp_idx])}
            )
        
            inp_access = [f'i{i}' for i, _ in enumerate(inp_shapes[inp_idx])]
            inp_access_str = ", ".join(inp_access)
            inp_memlet = dace.Memlet(f"{inp_name(inp_idx)}[{inp_access_str}]")
            
            stack_idx_offset = ""
            for i in range(inp_idx):
                stack_idx_offset += f" + ({inp_shapes[i][axis]})"
            
            out_access = [f'i{i}' for i, _ in enumerate(out_shape)]
            if stack_idx_offset:
                out_access[axis] += stack_idx_offset
            out_access_str = ", ".join(out_access)
            out_memlet = dace.Memlet(f"{out_name}[{out_access_str}]")
            
            nstate.add_memlet_path(
                inp_read, map_entry, tasklet,
                memlet=inp_memlet,
                dst_conn="inp"
            )
            nstate.add_memlet_path(
                tasklet, map_exit, out_write,
                memlet=out_memlet,
                src_conn="out"
            )
        
        return nsdfg


@op_implementation(op="Resize", name="pure")
class PureResize(ONNXForward):    
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        # Check if we have either scales or sizes (but not both)
        has_scales = len(list(state.in_edges_by_connector(node, 'scales'))) > 0
        has_sizes = len(list(state.in_edges_by_connector(node, 'sizes'))) > 0
        
        if has_scales == has_sizes:
            return False
            
        # Check interpolation mode
        mode = getattr(node, 'mode', 'nearest')
        if mode is not None and mode not in ['nearest', 'linear', 'cubic']:
            return False
            
        # Check nearest mode if using nearest interpolation
        if mode == 'nearest':
            nearest_mode = getattr(node, 'nearest_mode', 'round_prefer_floor')
            if nearest_mode is not None and nearest_mode not in ['round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']:
                return False
                
        # Check coordinate transformation mode
        coord_mode = getattr(node, 'coordinate_transformation_mode', 'half_pixel')
        if coord_mode is not None and coord_mode not in [
            'half_pixel',
            'half_pixel_symmetric',
            'pytorch_half_pixel', 
            'align_corners',
            'asymmetric',
            'tf_crop_and_resize'
        ]:
            return False
            
        # For tf_crop_and_resize, roi must be present
        if coord_mode == 'tf_crop_and_resize':
            has_roi = len(list(state.in_edges_by_connector(node, 'roi'))) > 0
            if not has_roi:
                return False
                
        # Check keep_aspect_ratio_policy if using sizes
        if has_sizes:
            policy = getattr(node, 'keep_aspect_ratio_policy', 'stretch')
            if policy is not None and policy not in ['stretch', 'not_larger', 'not_smaller']:
                return False
                
        # Check antialias
        antialias = getattr(node, 'antialias', 0)
        if antialias is not None and antialias not in [0, 1]:
            return False
            
        # Check exclude_outside
        exclude_outside = getattr(node, 'exclude_outside', 0)
        if exclude_outside is not None and exclude_outside not in [0, 1]:
            return False
            
        # Check extrapolation_value
        extrapolation_value = getattr(node, 'extrapolation_value', 0.0)
        if extrapolation_value is not None and not isinstance(extrapolation_value, (int, float)):
            return False
            
        # Check cubic coefficient
        if mode == 'cubic':
            cubic_coeff_a = getattr(node, 'cubic_coeff_a', -0.75)
            if cubic_coeff_a is not None and not isinstance(cubic_coeff_a, (int, float)):
                return False
            
        # Check axes if provided
        axes = getattr(node, 'axes', None)
        if axes is not None:
            if not isinstance(axes, (list, tuple)):
                return False
            # Check for duplicate axes
            if len(set(axes)) != len(axes):
                return False
            # Check for valid axis values
            rank = len(in_desc_with_name(node, state, sdfg, 'X').shape)
            for axis in axes:
                if not isinstance(axis, int) or axis < -rank or axis >= rank:
                    return False
                    
        # Check input shapes
        x_desc = in_desc_with_name(node, state, sdfg, 'X')
        rank = len(x_desc.shape)
        if has_scales:
            scales_desc = in_desc_with_name(node, state, sdfg, 'scales')
            if len(scales_desc.shape) != 1:
                return False
            if len(axes) if axes is not None else rank != scales_desc.shape[0]:
                return False
        if has_sizes:
            sizes_desc = in_desc_with_name(node, state, sdfg, 'sizes')
            if len(sizes_desc.shape) != 1:
                return False
            if len(axes) if axes is not None else rank != sizes_desc.shape[0]:
                return False
                
        # Check output shape
        y_desc = out_desc_with_name(node, state, sdfg, 'Y')
        if len(x_desc.shape) != len(y_desc.shape):
            return False
            
        return True
    
    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        
        inp_name = 'X'
        out_name = 'Y'
        
        nsdfg = dace.SDFG(node.label)
        
        # Add required input and output descriptors
        inp_data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, inp_name))
        inp_data_desc.transient = False
        nsdfg.add_datadesc(inp_name, inp_data_desc)
        
        out_data_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, out_name))
        out_data_desc.transient = False
        nsdfg.add_datadesc(out_name, out_data_desc)
        
        # Check for optional parameters
        has_scales = len(list(state.in_edges_by_connector(node, 'scales'))) > 0
        has_sizes = len(list(state.in_edges_by_connector(node, 'sizes'))) > 0
        has_roi = len(list(state.in_edges_by_connector(node, 'roi'))) > 0
        
        # Get axes to resize
        axes = node.axes or list(range(len(inp_data_desc.shape)))
            
        # Convert negative axes to positive
        axes = [ax if ax >= 0 else len(inp_data_desc.shape) + ax for ax in axes]
        
        # Add optional parameter descriptors if they exist
        if has_scales:
            scales_name = 'scales'
            scales_data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, scales_name))
            scales_data_desc.transient = False
            nsdfg.add_datadesc(scales_name, scales_data_desc)
            
        if has_sizes:
            sizes_name = 'sizes'
            sizes_data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, sizes_name))
            sizes_data_desc.transient = False
            nsdfg.add_datadesc(sizes_name, sizes_data_desc)
            
        if has_roi:
            roi_name = 'roi'
            roi_data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, roi_name))
            roi_data_desc.transient = False
            nsdfg.add_datadesc(roi_name, roi_data_desc)
        
        num_dims = len(inp_data_desc.shape)
        
        # setup inner SDFG
        nstate = nsdfg.add_state()
        
        inp_read = nstate.add_read(inp_name)
        out_write = nstate.add_write(out_name)
        
        # Add reads for optional parameters
        tasklet_inputs = {'__inp': dace.pointer(inp_data_desc.dtype)}
        if has_scales:
            scales_read = nstate.add_read(scales_name)
            tasklet_inputs['__scales'] = dace.pointer(scales_data_desc.dtype)
        if has_sizes:
            sizes_read = nstate.add_read(sizes_name)
            tasklet_inputs['__sizes'] = dace.pointer(sizes_data_desc.dtype)
        if has_roi:
            roi_read = nstate.add_read(roi_name)
            tasklet_inputs['__roi'] = dace.pointer(roi_data_desc.dtype)
        
        # Generate tasklet code for interpolation
        tasklet_code = []
        
        # Get interpolation parameters
        coord_mode = getattr(node, 'coordinate_transformation_mode', 'half_pixel')
        mode = getattr(node, 'mode', 'nearest')
        antialias = getattr(node, 'antialias', 0)
        exclude_outside = getattr(node, 'exclude_outside', 0)
        extrapolation_value = getattr(node, 'extrapolation_value', 0.0)
        
        # Add cubic interpolation helper functions if needed
        if mode == 'cubic':
            cubic_coeff_a = getattr(node, 'cubic_coeff_a', -0.75)
            tasklet_code.append(f"""
            // Cubic interpolation helper functions
            float cubic_weight(float x) {{
                float a = {cubic_coeff_a};
                float absx = abs(x);
                if (absx < 1.0) {{
                    return (a + 2.0) * absx * absx * absx - (a + 3.0) * absx * absx + 1.0;
                }} else if (absx < 2.0) {{
                    return a * absx * absx * absx - 5.0 * a * absx * absx + 8.0 * a * absx - 4.0 * a;
                }}
                return 0.0;
            }}
            """)
        
        # Loop over output dimensions
        tasklet_code.append("""
        // Loop over all output dimensions
        """)
        
        # Create nested loops for each dimension
        for i in range(len(out_data_desc.shape)):
            tasklet_code.append(f"for (int i{i} = 0; i{i} < {out_data_desc.shape[i]}; i{i}++) {{")
        
        # Calculate input indices
        tasklet_code.append("""
        // Calculate input indices for each dimension
        int inp_indices[{}];
        """.format(num_dims))
        
        # Declare all size variables at the beginning
        for i in range(num_dims):
            if i in axes:
                tasklet_code.append(f"float inp_size_{i};")
                tasklet_code.append(f"float out_size_{i};")
        
        for i in range(num_dims):
            tasklet_code.append(f"// Dimension {i}")
            if i in axes:
                axis_idx = axes.index(i)
                if has_scales:
                    tasklet_code.append(f"""
                    float scale_{i} = __scales[{axis_idx}];
                    inp_size_{i} = {inp_data_desc.shape[i]};
                    out_size_{i} = {out_data_desc.shape[i]};
                    float x_resized_{i} = i{i};
                    float x_original_{i};
                    """)
                    
                    # Add coordinate transformation based on mode
                    if coord_mode == 'half_pixel':
                        tasklet_code.append(f"""
                        x_original_{i} = (x_resized_{i} + 0.5) / scale_{i} - 0.5;
                        """)
                    elif coord_mode == 'half_pixel_symmetric':
                        tasklet_code.append(f"""
                        float adjustment_{i} = out_size_{i} / (out_size_{i} - 1);
                        float center_{i} = inp_size_{i} / 2;
                        float offset_{i} = center_{i} * (1 - adjustment_{i});
                        x_original_{i} = offset_{i} + (x_resized_{i} + 0.5) / scale_{i} - 0.5;
                        """)
                    elif coord_mode == 'pytorch_half_pixel':
                        tasklet_code.append(f"""
                        x_original_{i} = out_size_{i} > 1 ? (x_resized_{i} + 0.5) / scale_{i} - 0.5 : 0;
                        """)
                    elif coord_mode == 'align_corners':
                        tasklet_code.append(f"""
                        x_original_{i} = x_resized_{i} * (inp_size_{i} - 1) / (out_size_{i} - 1);
                        """)
                    elif coord_mode == 'asymmetric':
                        tasklet_code.append(f"""
                        x_original_{i} = x_resized_{i} / scale_{i};
                        """)
                    elif coord_mode == 'tf_crop_and_resize':
                        tasklet_code.append(f"""
                        float roi_start_{i} = __roi[{axis_idx}];
                        float roi_end_{i} = __roi[{len(axes) + axis_idx}];
                        if (out_size_{i} > 1) {{
                            x_original_{i} = roi_start_{i} * (inp_size_{i} - 1) + x_resized_{i} * (roi_end_{i} - roi_start_{i}) * (inp_size_{i} - 1) / (out_size_{i} - 1);
                        }} else {{
                            x_original_{i} = 0.5 * (roi_start_{i} + roi_end_{i}) * (inp_size_{i} - 1);
                        }}
                        """)
                    
                    # Add interpolation mode handling
                    if mode == 'nearest':
                        nearest_mode = getattr(node, 'nearest_mode', 'round_prefer_floor')
                        if nearest_mode == 'floor':
                            tasklet_code.append(f"inp_indices[{i}] = int(floor(x_original_{i}));")
                        elif nearest_mode == 'ceil':
                            tasklet_code.append(f"inp_indices[{i}] = int(ceil(x_original_{i}));")
                        else:  # round_prefer_floor or round_prefer_ceil
                            tasklet_code.append(f"inp_indices[{i}] = int(round(x_original_{i}));")
                    elif mode == 'linear':
                        tasklet_code.append(f"""
                        float x0_{i} = floor(x_original_{i});
                        float x1_{i} = ceil(x_original_{i});
                        float w0_{i} = x1_{i} - x_original_{i};
                        float w1_{i} = x_original_{i} - x0_{i};
                        inp_indices[{i}] = int(x0_{i});
                        inp_indices[{i} + {num_dims}] = int(x1_{i});  // Store second index for linear interpolation
                        """)
                    elif mode == 'cubic':
                        tasklet_code.append(f"""
                        float x0_{i} = floor(x_original_{i});
                        float x1_{i} = x0_{i} + 1;
                        float x2_{i} = x0_{i} + 2;
                        float x3_{i} = x0_{i} + 3;
                        float w0_{i} = cubic_weight(x_original_{i} - x0_{i});
                        float w1_{i} = cubic_weight(x_original_{i} - x1_{i});
                        float w2_{i} = cubic_weight(x_original_{i} - x2_{i});
                        float w3_{i} = cubic_weight(x_original_{i} - x3_{i});
                        inp_indices[{i}] = int(x0_{i});
                        inp_indices[{i} + {num_dims}] = int(x1_{i});  // Store indices for cubic interpolation
                        inp_indices[{i} + {2*num_dims}] = int(x2_{i});
                        inp_indices[{i} + {3*num_dims}] = int(x3_{i});
                        """)
                else:  # has_sizes
                    tasklet_code.append(f"""
                    inp_size_{i} = {inp_data_desc.shape[i]};
                    out_size_{i} = {out_data_desc.shape[i]};
                    inp_indices[{i}] = int(floor(i{i} * inp_size_{i} / out_size_{i}));
                    """)
            else:
                tasklet_code.append(f"inp_indices[{i}] = i{i};")
        
        # Calculate input index
        tasklet_code.append("""
        // Calculate input index
        int inp_idx = 0;
        """)
        for i in range(num_dims):
            tasklet_code.append(f"inp_idx += inp_indices[{i}] * {inp_data_desc.strides[i]};")
        
        # Calculate output index
        tasklet_code.append("""
        // Calculate output index
        int out_idx = 0;
        """)
        for i in range(num_dims):
            tasklet_code.append(f"out_idx += i{i} * {out_data_desc.strides[i]};")
        
        # Perform interpolation based on mode
        if mode == 'linear':
            tasklet_code.append(f"""
            // Linear interpolation
            float x0 = __inp [inp_idx];
            float x1 = __inp [inp_idx + {inp_data_desc.strides[axes[0]]}];  // Second index for linear interpolation
            float result = w0 * x0 + w1 * x1;
            """)
        elif mode == 'cubic':
            tasklet_code.append(f"""
            // Cubic interpolation
            float x0 = __inp [inp_idx];
            float x1 = __inp [inp_idx + {inp_data_desc.strides[axes[0]]}];
            float x2 = __inp [inp_idx + {2*inp_data_desc.strides[axes[0]]}];
            float x3 = __inp [inp_idx + {3*inp_data_desc.strides[axes[0]]}];
            float result = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
            """)
        else:  # nearest or default
            tasklet_code.append("""
            // Nearest neighbor interpolation
            float result = __inp [inp_idx];
            """)
        
        # Handle antialiasing if enabled
        if antialias == 1 and mode in ['linear', 'cubic']:
            tasklet_code.append("""
            // Apply antialiasing filter
            float scale = __scales[0];  // Assuming first axis is being resized
            if (scale < 1.0) {
                float filter_scale = max(1.0, 1.0 / scale);
                result *= filter_scale;
            }
            """)
        
        # Handle exclude_outside if enabled
        if exclude_outside == 1:
            tasklet_code.append(f"""
            // Handle exclude_outside
            bool is_outside = false;
            for (int i = 0; i < {num_dims}; i++) {{
                if (inp_indices[i] < 0 || inp_indices[i] >= {inp_data_desc.shape[0]}) {{
                    is_outside = true;
                    break;
                }}
            }}
            if (is_outside) {{
                result = 0.0;
            }}
            """)
        
        # Handle extrapolation_value for tf_crop_and_resize
        if coord_mode == 'tf_crop_and_resize':
            tasklet_code.append(f"""
            // Handle extrapolation for tf_crop_and_resize
            bool is_outside = false;
            for (int i = 0; i < {num_dims}; i++) {{
                if (inp_indices[i] < 0 || inp_indices[i] >= {inp_data_desc.shape[0]}) {{
                    is_outside = true;
                    break;
                }}
            }}
            if (is_outside) {{
                result = {extrapolation_value};
            }}
            """)
        
        # Write the result to output
        tasklet_code.append("""
        // Write output
        __out [out_idx] = result;
        """)
        
        # Close dimension loops
        for i in range(len(out_data_desc.shape)):
            tasklet_code.append("}")
        
        tasklet = nstate.add_tasklet(
            f'tasklet_reshape', 
            tasklet_inputs,
            {'__out': dace.pointer(out_data_desc.dtype)},
            "\n".join(tasklet_code),
            language=dace.Language.CPP
        )
        
        # Connect tasklet inputs
        nstate.add_edge(inp_read, None, tasklet, "__inp", 
                       dace.Memlet.from_array(inp_name, inp_data_desc))
        if has_scales:
            nstate.add_edge(scales_read, None, tasklet, "__scales",
                           dace.Memlet.from_array(scales_name, scales_data_desc))
        if has_sizes:
            nstate.add_edge(sizes_read, None, tasklet, "__sizes",
                           dace.Memlet.from_array(sizes_name, sizes_data_desc))
        if has_roi:
            nstate.add_edge(roi_read, None, tasklet, "__roi",
                           dace.Memlet.from_array(roi_name, roi_data_desc))
        
        # Connect tasklet output
        nstate.add_edge(tasklet, "__out", out_write, None,
                       dace.Memlet.from_array(out_name, out_data_desc))
        
        return nsdfg
    
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


@python_pure_op_implementation
def Add(A, B, C):
    C[:] = A + B


@python_pure_op_implementation
def Sub(A, B, C):
    C[:] = A - B


@python_pure_op_implementation
def Mul(A, B, C):
    C[:] = A * B


@python_pure_op_implementation
def Div(A, B, C):
    C[:] = A / B


@python_pure_op_implementation
def Where(condition, X, Y, output):
    output[:] = np.where(condition, X, Y)


@python_pure_op_implementation
def Erf(input, output):
    output[:] = dace.elementwise(lambda x: erf(x), input)
    
@op_implementation(op="CumSum", name="pure")
class PureCumSum(ONNXForward):
    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        if node.exclusive or node.reverse:
            raise NotImplementedError("CumSum with exclusive or reverse attributes is not implemented")

        nsdfg, nstate, input_nodes, output_nodes = empty_sdfg_for_node(sdfg,
                                                  state,
                                                  node,
                                                  add_access_nodes=True)

        x_desc = in_desc_with_name(node, state, sdfg, "x")
        axis_desc = in_desc_with_name(node, state, sdfg, "axis")
        y_desc = out_desc_with_name(node, state, sdfg, "y")

        x_idx_expr = " + ".join([f"i{i} * {s}" for i, s in enumerate(x_desc.strides)])
        y_idx_expr = " + ".join([f"i{i} * {s}" for i, s in enumerate(y_desc.strides)])

        num_dims = len(x_desc.shape)

        y_prev_idx_expr = " + ".join([
            f"(i{i} - is_axis{i}) * {s}"
            for i, s in enumerate(y_desc.strides)
        ])

        code = ""
        for i, val in enumerate(y_desc.shape):
            code += f"for (int i{i} = 0; i{i} < {val}; i{i}++) {{\n"
            code += f"int is_axis{i} = ({i} == ({num_dims} + __axis) % {num_dims});\n"
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
    
@op_implementation(op="Unsqueeze", name="pure")
class PureUnsqueeze(ONNXForward):
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
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        axes_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "axes"))
        expanded_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "expanded"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("data", data_desc)
        nsdfg.add_datadesc("axes", axes_desc)
        nsdfg.add_datadesc("expanded", expanded_desc)
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["axes"].transient = False
        nsdfg.arrays["expanded"].transient = False

        # Add access nodes
        data_read = nstate.add_read("data")
        axes_read = nstate.add_read("axes")
        expanded_write = nstate.add_write("expanded")

        # Create tasklet that performs the unsqueeze operation
        data_size = int(np.prod(data_desc.shape))
        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs={
                "__data": dace.pointer(data_desc.dtype),
                "__axes": dace.pointer(axes_desc.dtype),
            },
            outputs={"__unsqueezed": dace.pointer(expanded_desc.dtype)},
            code=f"""
            for (int i = 0; i < {data_size}; i++) {{
                __unsqueezed[i] = __data[i];
            }}
            """,
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(data_read, None, tasklet, "__data", 
                       dace.Memlet.from_array("data", data_desc))
        nstate.add_edge(axes_read, None, tasklet, "__axes",
                       dace.Memlet.from_array("axes", axes_desc))
        nstate.add_edge(tasklet, "__unsqueezed", expanded_write, None,
                       dace.Memlet.from_array("expanded", expanded_desc))

        return nsdfg
    
@op_implementation(op="Squeeze", name="pure")
class PureSqueeze(ONNXForward):
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
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        axes_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "axes"))
        squeezed_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "squeezed"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("data", data_desc)
        nsdfg.add_datadesc("axes", axes_desc)
        nsdfg.add_datadesc("squeezed", squeezed_desc)
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["axes"].transient = False
        nsdfg.arrays["squeezed"].transient = False

        # Add access nodes
        data_read = nstate.add_read("data")
        axes_read = nstate.add_read("axes")
        squeezed_write = nstate.add_write("squeezed")

        # Create tasklet that performs the squeeze operation
        data_size = int(np.prod(data_desc.shape))
        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs={
                "__data": dace.pointer(data_desc.dtype),
                "__axes": dace.pointer(axes_desc.dtype),
            },
            outputs={"__squeezed": dace.pointer(squeezed_desc.dtype)},
            code=f"""
            for (int i = 0; i < {data_size}; i++) {{
                __squeezed[i] = __data[i];
            }}
            """,
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(data_read, None, tasklet, "__data", 
                       dace.Memlet.from_array("data", data_desc))
        nstate.add_edge(axes_read, None, tasklet, "__axes",
                       dace.Memlet.from_array("axes", axes_desc))
        nstate.add_edge(tasklet, "__squeezed", squeezed_write, None,
                       dace.Memlet.from_array("squeezed", squeezed_desc))

        return nsdfg
    
@op_implementation(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get keepdims attribute (default 1)
        keepdims = getattr(node, 'keepdims', 1)
        
        # Get noop_with_empty_axes attribute (default 0)
        noop_with_empty_axes = getattr(node, 'noop_with_empty_axes', 0)

        # Create a new SDFG for the reduction with unique name
        uid = state.node_id(node)
        nsdfg = SDFG(f'reduce_mean_{uid}')
        nstate = nsdfg.add_state(f'reduce_mean_{uid}')

        # Get input and output arrays with deep copies and unique names
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        reduced_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "reduced"))
        
        # Add arrays to the SDFG with unique names
        data_name = f"data"
        reduced_name = f"reduced"
        nsdfg.add_datadesc(data_name, data_desc)
        nsdfg.add_datadesc(reduced_name, reduced_desc)
        nsdfg.arrays[data_name].transient = False
        nsdfg.arrays[reduced_name].transient = False

        # Create access nodes
        data_read = nstate.add_read(data_name)
        reduced_write = nstate.add_write(reduced_name)

        # Determine axes and num_axes
        axes_name = f"axes"
        if axes_name in node.in_connectors:
            [axes_edge] = state.in_edges_by_connector(node, 'axes')
            axes_desc = copy.deepcopy(sdfg.arrays[axes_edge.data.data])
            nsdfg.add_datadesc(axes_name, axes_desc)
            nsdfg.arrays[axes_name].transient = False
            axes_node = nstate.add_access(axes_name)
        else:
            axes = getattr(node, axes_name, None)
            if not axes:
                if noop_with_empty_axes:
                    axes_values = []
                else:
                    axes_values = list(range(len(data_desc.shape)))
            else:
                axes_values = list(axes)
            # Create axes_arr as an array with just axes_values
            axes_arr_shape = [len(axes_values)]
            axes_arr_dtype = dace.int64
            _, axes_desc = nsdfg.add_array(axes_name, axes_arr_shape, axes_arr_dtype)
            axes_node = nstate.add_access(axes_name)

            # Add a tasklet to initialize the axes array in the SDFG
            axes_init_tasklet = nstate.add_tasklet(
                f"init_axes",
                set(),
                {"out": dace.pointer(axes_arr_dtype)},
                "\n".join([f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
                language=dace.Language.CPP
            )
            nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"{axes_name}[0:{len(axes_values)}]"))

        is_axes_scalar = len(axes_desc.shape) == 1 and axes_desc.shape[0] == 1

        # Create the reduction tasklet with embedded constants
        shape_str = ', '.join(map(str, data_desc.shape))
        tasklet_code = (f"""
            constexpr long long input_dims = {len(data_desc.shape)};
            constexpr long long output_dims = {len(reduced_desc.shape)};
            constexpr long long num_reduce_dims = {len(axes_desc.shape)};
            constexpr long long num_non_reduce_dims = {len(data_desc.shape) - len(axes_desc.shape)};
            constexpr long long keepdims = {keepdims};
            long long reduce_dims [num_reduce_dims] = """ +
            ("{axes_arr};" if is_axes_scalar else f'{{{", ".join([f"axes_arr[{i}]" for i in range(len(axes_desc.shape))])}}};') +
            f""" 

            long long input_shape [input_dims] = {{{", ".join([str(data_desc.shape[i]) for i in range(len(data_desc.shape))])}}};
            long long output_shape [output_dims] = {{{", ".join([str(reduced_desc.shape[i]) for i in range(len(reduced_desc.shape))])}}};
            long long input_strides [input_dims] = {{{", ".join([str(data_desc.strides[i]) for i in range(len(data_desc.shape))])}}};
            long long output_strides [output_dims] = {{{", ".join([str(reduced_desc.strides[i]) for i in range(len(reduced_desc.shape))])}}};
            long long output_strides_input [input_dims];
            long long output_strides_input_idx = 0;
            for (long long i = 0; i < input_dims; i++) {{
                int is_reduce_dim = 0;
                for (long long j = 0; j < num_reduce_dims; j++) {{
                    if (reduce_dims[j] == i) {{
                        is_reduce_dim = 1;
                        break;
                    }}
                }}
                if (is_reduce_dim) {{
                    output_strides_input[i] = 0;
                    if (keepdims) {{
                        output_strides_input_idx++;
                    }}
                }} else {{
                    output_strides_input[i] = output_strides [output_strides_input_idx];
                    output_strides_input_idx++;
                }}
            }}
            
            // initialize output to zero
            """ +
            "\n".join([f"for (long long i{i} = 0; i{i} < output_shape[{i}]; i{i}++) {{" for i in range(len(reduced_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides [{i}] * i{i}" for i in range(len(reduced_desc.shape))]) + "] = 0.0;" + "\n" +
            "\n".join(["}" for _ in reduced_desc.shape]) + "\n" +
            """
            
            // Compute the number of elements to reduce over
            long long reduce_size = 1;
            for (long long i = 0; i < input_dims; i++) {{
                int is_reduce_dim = 0;
                for (long long j = 0; j < num_reduce_dims; j++) {{
                    if (reduce_dims[j] == i) {{
                        is_reduce_dim = 1;
                        break;
                    }}
                }}
                if (is_reduce_dim) {{
                    reduce_size *= input_shape[i];
                }}
            }}

            // Loop over all input elements
            """ + 
            "\n".join([f"for (long long i{i} = 0; i{i} < input_shape[{i}]; i{i}++) {{" for i in range(len(data_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))]) +"]" +
            " += inp [" + " + ".join([f"input_strides[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "] / reduce_size;" + "\n" +
            "\n".join(["}" for _ in data_desc.shape]) + "\n" +
            """
        """)
        tasklet = nstate.add_tasklet(
            f'reduce_mean_{uid}',
            {'inp': dace.pointer(data_desc.dtype), 'axes_arr': axes_desc.dtype},
            {'out': dace.pointer(reduced_desc.dtype)},
            tasklet_code,
            language=dace.Language.CPP)

        # Add edges for axes input, num_axes, data input and output
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_name))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_name))

        return nsdfg
    
@op_implementation(op="MatMul", name="pure")
class PureMatMul(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 1 or input1_dim == 1:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim), reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXEinsum
        einsum_node: onnx_op.ONNXOp = ONNXEinsum(node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0", nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1", nsdfg.make_array_memlet("B"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("Y"), None, nsdfg.make_array_memlet("Y"))

        return nsdfg


@op_implementation(op="Einsum", name="pure")
class PureEinsum(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        if "..." in node.equation:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        for e in node.iter_inputs_in_onnx_order(state):
            desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, e.dst_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, desc)
        for e in node.iter_outputs_in_onnx_order(state):
            desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, e.src_conn))
            desc.transient = False
            nsdfg.add_datadesc(e.src_conn, desc)

        create_einsum_sdfg(None,
                           nsdfg,
                           nstate,
                           node.equation.replace(" ", ""),
                           *(e.dst_conn for e in node.iter_inputs_in_onnx_order(state)),
                           output="Output")
        return nsdfg


@python_pure_op_implementation
def Identity(input, output):
    output[:] = input


@op_implementation(op="Expand", name="pure")
class PureExpand(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        try:
            in_desc = in_desc_with_name(node, state, sdfg, "input")
            out_desc = out_desc_with_name(node, state, sdfg, "output")
            in_desc_with_name(node, state, sdfg, "shape")
        except:
            return False

        in_shape = in_desc.shape
        out_shape = out_desc.shape
        if len(in_shape) > len(out_shape):
            return False

        # check that the shapes are broadcastable
        for i, o in zip(reversed(in_shape), reversed(out_shape)):
            if i != 1 and i != o:
                return False
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg, nstate, input_nodes, output_nodes = empty_sdfg_for_node(
            sdfg, state, node, add_access_nodes=True)
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")
        shape_desc = in_desc_with_name(node, state, sdfg, "shape")

        num_out_dims = len(output_desc.shape)
        num_in_dims = len(input_desc.shape)

        code = []
        for i in range(num_out_dims):
            code.append(
                f"for (int i{i} = 0; i{i} < {output_desc.shape[i]}; ++i{i}) {{")
        code.append("    int input_idx = 0;")
        for i in range(num_in_dims):
            j = i + num_out_dims - num_in_dims
            code.append(
                f"    if ({input_desc.shape[i]} != 1) input_idx += i{j} * {input_desc.strides[i]};"
            )

        output_idx_str = " + ".join(
            [f"i{i} * {s}" for i, s in enumerate(output_desc.strides)])

        code.append(f"    __output[{output_idx_str}] = __input[input_idx];")

        for _ in range(num_out_dims):
            code.append("}")

        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs={
                "__input": dace.pointer(input_desc.dtype),
                "__shape": dace.pointer(shape_desc.dtype)
            },
            outputs={"__output": dace.pointer(output_desc.dtype)},
            code="\n".join(code),
            language=dace.Language.CPP)

        nstate.add_edge(input_nodes["input"], None, tasklet, "__input",
                        dace.Memlet.from_array("input", input_desc))
        nstate.add_edge(input_nodes["shape"], None, tasklet, "__shape",
                        dace.Memlet.from_array("shape", shape_desc))
        nstate.add_edge(tasklet, "__output", output_nodes["output"], None,
                        dace.Memlet.from_array("output", output_desc))
        return nsdfg


@python_pure_op_implementation(string=lambda X: "lambda x: dace.{}(1) / x".format(X.dtype.to_string()))
def Reciprocal(X, Y):
    Y[:] = dace.elementwise(string, X)


@python_pure_op_implementation
def Tanh(input, output):
    output[:] = dace.elementwise(lambda x: tanh(x), input)


softmax_compute = dict(axis=lambda node, input: list(range(len(input.shape)))[node.axis:])


@python_pure_op_implementation(**softmax_compute)
def Softmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=True)
    exponent = np.exp(input - maximum)
    sum = np.add.reduce(exponent, axis=axis, keepdims=True)
    output[:] = exponent / sum


@python_pure_op_implementation(
    perm=lambda node, data: node.perm if node.perm is not None else list(reversed(range(len(data.shape)))))
def Transpose(data, transposed):
    transposed[:] = np.transpose(data, axes=perm)


@op_implementation(op="Cast", name="pure")
class PureCast(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:

        if (in_desc_with_name(node, state, sdfg, "input").dtype == out_desc_with_name(node, state, sdfg,
                                                                                      "output").dtype):
            return True

        target_type = node.to
        try:
            converters.onnx_tensor_type_to_typeclass(target_type)
        except ValueError:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")
        if (input_desc.dtype == output_desc.dtype):

            def prog(input, output):
                output[:] = input

            return program_for_node(prog, sdfg, state, node)
        else:

            nsdfg, nstate, _, _ = empty_sdfg_for_node(sdfg, state, node, add_access_nodes=False)

            shape = out_desc_with_name(node, state, sdfg, "output").shape
            map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
            index_str = f"{', '.join(map_ranges.keys())}"
            tasklet, _, _ = nstate.add_mapped_tasklet(node.label + "_tasklet",
                                                      map_ranges=map_ranges,
                                                      inputs={f"__input": dace.Memlet(f"input[{index_str}]")},
                                                      code=f"__output = __input",
                                                      outputs={"__output": dace.Memlet(f"output[{index_str}]")},
                                                      external_edges=True)

            return nsdfg


@op_implementation(op="Gemm", name="pure")
class PureGemm(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim), reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        if node.transA == 1:
            arg1 = ''.join(reversed(arg1))
        if node.transB == 1:
            arg2 = ''.join(reversed(arg2))

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to
        # make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Einsum: "A", "B" -> mm_result
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Decide on array names based on alpha and beta
        uid = state.node_id(node)
        mm_result = "Y"
        if node.alpha != 1 or node.beta != 0:
            mm_result = f"Ytmp_{uid}"
        scal_result = mm_result
        if node.alpha != 1:
            scal_result = f"scaled_{uid}"

        # Create arrays according to alpha and beta
        if node.alpha != 1 or node.beta != 0:
            Ytmp_desc = out_desc_with_name(node, state, sdfg, "Y")
            nsdfg.add_datadesc(f"Ytmp_{uid}", copy.deepcopy(Ytmp_desc))
            nsdfg.arrays[f"Ytmp_{uid}"].transient = True
        if node.beta != 0:
            beta_desc = out_desc_with_name(node, state, sdfg, "Y")
            nsdfg.add_datadesc(f"scaled_{uid}", copy.deepcopy(beta_desc))
            nsdfg.arrays[f"scaled_{uid}"].transient = True

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0", nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1", nsdfg.make_array_memlet("B"))
        mm_result_node = nstate.add_write(mm_result)
        nstate.add_edge(einsum_node, "Output", mm_result_node, None, nsdfg.make_array_memlet(mm_result))

        # Multiply by alpha: mm_result -> scal_result
        if node.alpha != 1:
            nstate.add_mapped_tasklet(
                node.label + '_alphascale',
                {
                    k: f'0:{Ytmp_desc.shape[i]}'
                    for i, k in enumerate(result)
                },
                dict(a=dace.Memlet(data=mm_result, subset=','.join(result))),
                f'o = a * dace.{Ytmp_desc.dtype}({node.alpha})',
                dict(o=dace.Memlet(data=scal_result, subset=','.join(result))),
                external_edges=True,
                input_nodes=dict(a=mm_result_node),
            )

        # Multiply by beta: scal_result, "C" -> "Y"
        if node.beta != 0:
            C_desc = in_desc_with_name(node, state, sdfg, "C")
            nsdfg.add_datadesc("C", copy.deepcopy(C_desc))
            nsdfg.arrays["C"].transient = False
            scal_result_node = next(n for n in nstate.sink_nodes()
                                    if isinstance(n, dace.nodes.AccessNode) and n.data == scal_result)
            beta_scale_code = f'o = s + c * dace.{C_desc.dtype}({node.beta})'
            if node.beta == 1:
                beta_scale_code = f'o = s + c'

            # Support broadcasting in C -> Y
            c_index = result[-len(C_desc.shape):]
            for c_shp, y_shp in zip(reversed(C_desc.shape), reversed(Y_desc.shape)):
                if c_shp != y_shp:
                    raise ValueError('Could not broadcast dimensions from C '
                                     'to Y in ONNXGemm')

            nstate.add_mapped_tasklet(
                node.label + '_betascale',
                {
                    k: f'0:{Y_desc.shape[i]}'
                    for i, k in enumerate(result)
                },
                dict(s=dace.Memlet(data=scal_result, subset=','.join(result)),
                     c=dace.Memlet(data="C", subset=','.join(c_index))),
                beta_scale_code,
                dict(o=dace.Memlet(data="Y", subset=','.join(result))),
                external_edges=True,
                input_nodes={scal_result: scal_result_node},
            )

        return nsdfg


@python_pure_op_implementation(cast_lambda=lambda X: "lambda x: max(x, dace.{}(0))".format(X.dtype.to_string()))
def Relu(X, Y):
    Y[:] = dace.elementwise(cast_lambda, X)


@python_pure_op_implementation(
    cast_lambda=lambda node, X: "lambda x: (max(x, dace.{dtype}(0)) + {alpha} * min(x, dace.{dtype}(0)))".format(
        dtype=X.dtype.to_string(), alpha=node.alpha))
def LeakyRelu(X, Y):
    Y[:] = dace.elementwise(cast_lambda, X)


@python_pure_op_implementation(
    shape=lambda reshaped: reshaped.shape,
    allowzero=lambda node: getattr(node, 'allowzero', 0))
def Reshape(data, reshaped):
    # If allowzero is 0 (default), we use numpy's reshape which doesn't allow zeros
    # If allowzero is 1, we need to handle zeros in the shape tensor
    if allowzero == 0:
        reshaped[:] = np.reshape(data, shape)
    else:
        # For allowzero=1, we need to handle zeros in the shape tensor
        # This means we need to preserve the original dimension size when a zero is encountered
        new_shape = list(shape)
        for i, dim in enumerate(new_shape):
            if dim == 0:
                new_shape[i] = data.shape[i]
        reshaped[:] = np.reshape(data, new_shape)


@python_pure_op_implementation(shape=lambda input, node: [prod(input.shape[:node.axis]), prod(input.shape[node.axis:])])
def Flatten(input, output):
    output[:] = input.reshape(shape)


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


@python_pure_op_implementation(**softmax_compute)
def LogSoftmax(input, output):
    maximum = np.maximum.reduce(input, axis=axis, keepdims=True)
    max_sub = input - maximum
    exponent = np.exp(max_sub)
    sum = np.add.reduce(exponent, axis=axis, keepdims=True)
    log_sum = np.log(sum)
    output[:] = max_sub - log_sum


@op_implementation(op="Slice", name="pure")
class PureSlice(ONNXForward):
    '''
        Slice expansion
    '''

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # check that all the inputs (even the optional ones) are present and constant

        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        if in_edge_with_name(node, state, "starts").src.data not in sdfg._parent_onnx_model.clean_weights:
            return False
        if in_edge_with_name(node, state, "ends").src.data not in sdfg._parent_onnx_model.clean_weights:
            return False

        # optional inputs
        is_axes_present = True
        try:
            if in_edge_with_name(node, state, "axes").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_axes_present = False

        is_steps_present = True
        try:
            if in_edge_with_name(node, state, "steps").src.data not in sdfg._parent_onnx_model.clean_weights:
                return False
        except ValueError:
            is_steps_present = False

        # Current constraints: axes and steps must be explict. Axes must be zero and steps must be 1
        if not is_axes_present or not is_steps_present:
            return False

        step = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "steps").src.data].numpy()[0]
        axis = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "axes").src.data].numpy()[0]

        if step != 1 or axis != 0:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        start = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "starts").src.data].numpy()[0]
        end = sdfg._parent_onnx_model.clean_weights[in_edge_with_name(node, state, "ends").src.data].numpy()[0]

        output_shape = out_desc_with_name(node, state, sdfg, "output").shape
        if end == np.iinfo(np.int64).max:
            # Pytorch exporter artifact
            end = start + output_shape[0]

        def prog(data, output):
            tmp = data[start:end:1, :]
            # We need reshape to avoid Invalid Edge errors
            output[:] = np.reshape(tmp, output.shape)

        return program_for_node(prog, sdfg, state, node)


@python_pure_op_implementation
def Softplus(X, Y):
    Y[:] = np.log(1 + np.exp(X))


@op_implementation(op="Sigmoid", name="pure")
class PureSigmoid(ONNXForward):
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
        Y_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "Y"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("X", X_desc)
        nsdfg.add_datadesc("Y", Y_desc)
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["Y"].transient = False

        # Add access nodes
        X_read = nstate.add_read("X")
        Y_write = nstate.add_write("Y")

        # Create tasklet that performs the sigmoid operation
        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs={"__X": dace.pointer(X_desc.dtype)},
            outputs={"__Y": dace.pointer(Y_desc.dtype)},
            code=f"""
            // Loop over all dimensions
            {chr(10).join([f'for (int i{i} = 0; i{i} < {X_desc.shape[i]}; i{i}++) {{' for i in range(len(X_desc.shape))])}
                // Calculate index for input and output arrays
                int x_idx = {'+'.join([f'i{i} * {X_desc.strides[i]}' for i in range(len(X_desc.shape))])};
                int y_idx = {'+'.join([f'i{i} * {Y_desc.strides[i]}' for i in range(len(Y_desc.shape))])};
                
                // Compute sigmoid: 1 / (1 + exp(-x))
                __Y[y_idx] = 1.0 / (1.0 + exp(-__X[x_idx]));
            {chr(10).join(['}' for _ in range(len(X_desc.shape))])}
            """,
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", 
                       dace.Memlet.from_array("X", X_desc))
        nstate.add_edge(tasklet, "__Y", Y_write, None,
                       dace.Memlet.from_array("Y", Y_desc))

        return nsdfg

@op_implementation(op="LayerNormalization", name="pure")
class PureLayerNormalization(ONNXForward):
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

        tasklet = nstate.add_tasklet(
            name=node.label + "_tasklet",
            inputs=tasklet_inputs,
            outputs=tasklet_outputs,
            code=code,
            language=dace.Language.CPP
        )

        # Connect the tasklet with memlets
        nstate.add_edge(X_read, None, tasklet, "__X", 
                       dace.Memlet.from_array("X", X_desc))
        nstate.add_edge(scale_read, None, tasklet, "__Scale",
                       dace.Memlet.from_array("Scale", scale_desc))
        nstate.add_edge(B_read, None, tasklet, "__B",
                       dace.Memlet.from_array("B", B_desc))
        nstate.add_edge(tasklet, "__Y", Y_write, None,
                       dace.Memlet.from_array("Y", Y_desc))
        if has_mean:
            nstate.add_edge(tasklet, "__Mean", mean_write, None,
                           dace.Memlet.from_array("Mean", mean_desc))
        if has_inv_std_dev:
            nstate.add_edge(tasklet, "__InvStdDev", inv_std_dev_write, None,
                           dace.Memlet.from_array("InvStdDev", inv_std_dev_desc))

        return nsdfg
    
@op_implementation(op="Transpose", name="einsum")
class EinsumTranspose(ONNXForward):

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        perm = node.perm
        input_desc = in_desc_with_name(node, state, sdfg, "data")
        output_desc = out_desc_with_name(node, state, sdfg, "transposed")

        letters = [chr(ord('z') - i) for i in range(26)]
        input_letters = "".join(letters[i] for i, _ in enumerate(input_desc.shape))
        output_letters = "".join(letters[i] for i in perm)
        equation_str = f"{input_letters}->{output_letters}"

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXEinsum
        einsum_node: onnx_op.ONNXOp = ONNXEinsum(node.label + "_einsum_expansion", equation=equation_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        nsdfg.add_datadesc("data", copy.deepcopy(input_desc))
        nsdfg.add_datadesc("transposed", copy.deepcopy(output_desc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["transposed"].transient = False

        nstate.add_edge(nstate.add_read("data"), None, einsum_node, "Inputs__0", nsdfg.make_array_memlet("data"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("transposed"), None,
                        nsdfg.make_array_memlet("transposed"))

        return nsdfg


@op_implementation(op="Split", name="pure")
class SplitPure(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        from daceml.transformation.replacement import onnx_constant_or_none
        
        # Check if we have either split input or num_outputs attribute
        has_split_input = len(list(state.in_edges_by_connector(node, "split"))) > 0
        has_num_outputs = hasattr(node, 'num_outputs')
        
        if not (has_split_input or has_num_outputs):
            return False
        
        # If split input is provided, it must be a constant
        if has_split_input:
            split_node = next(state.in_edges_by_connector(node, "split")).src
            if not onnx_constant_or_none(sdfg, split_node):
                return False
                
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from daceml.transformation.replacement import onnx_constant_or_none

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        split_dim = node.axis
        idesc = in_desc_with_name(node, state, sdfg, "input")
        nsdfg.add_datadesc("input", copy.deepcopy(idesc))
        nsdfg.arrays["input"].transient = False

        rnode = nstate.add_read("input")

        # Get split sizes either from input or compute from num_outputs
        if len(list(state.in_edges_by_connector(node, "split"))) > 0:
            # Get split sizes from input tensor
            split_node = next(state.in_edges_by_connector(node, "split")).src
            split_sizes = onnx_constant_or_none(sdfg, split_node)
            if split_sizes is None:
                raise ValueError("Split sizes must be constant")
                
            # Add split input as a data descriptor
            split_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "split"))
            split_desc.transient = False
            nsdfg.add_datadesc("split", split_desc)
            split_read = nstate.add_read("split")
        else:
            # Compute split sizes from num_outputs
            num_outputs = node.num_outputs
            total_size = idesc.shape[split_dim]
            base_size = total_size // num_outputs
            remainder = total_size % num_outputs
            split_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_outputs)]

        # Verify split sizes
        if sum(split_sizes) != idesc.shape[split_dim]:
            raise ValueError(f"Sum of split sizes ({sum(split_sizes)}) must equal dimension size ({idesc.shape[split_dim]})")

        offset = 0
        for i, odim in enumerate(split_sizes):
            # Set up new node shape and memlet
            new_shape = list(idesc.shape)
            new_shape[split_dim] = odim
            rng = subsets.Range([(0, s - 1, 1) if j != split_dim else
                                 (offset, offset + odim - 1, 1)
                                 for j, s in enumerate(new_shape)])
            offset += odim

            # Set up data descriptor
            oname = f"outputs__{i}"
            odesc = copy.deepcopy(out_desc_with_name(node, state, sdfg, oname))
            odesc.transient = False
            nsdfg.add_datadesc(oname, odesc)
            wnode = nstate.add_write(oname)

            # Perform copy (view)
            nstate.add_nedge(
                rnode, wnode,
                dace.Memlet(data="input",
                            subset=rng,
                            other_subset=subsets.Range.from_array(odesc)))

        return nsdfg


@op_implementation(op="Slice", name="pure")
class PureSliceAllConstant(ONNXForward):

    @staticmethod
    def _get_constant(conn: str, node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG):
        try:
            srcnode = next(state.in_edges_by_connector(node, conn)).src
        except StopIteration:
            return None
        # Scalar copied to GPU
        if 'gpu_' in srcnode.data:
            srcnode = state.predecessors(srcnode)[0]
        return onnx_constant_or_none(sdfg, srcnode)

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        for inconn in ("axes", "ends", "starts", "steps"):
            if PureSliceAllConstant._get_constant(inconn, node, state, sdfg) is None:
                return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = PureSliceAllConstant._get_constant('axes', node, state, sdfg)
        ends = PureSliceAllConstant._get_constant('ends', node, state, sdfg)
        starts = PureSliceAllConstant._get_constant('starts', node, state, sdfg)
        steps = PureSliceAllConstant._get_constant('steps', node, state, sdfg)

        constant_folding.remove_node_and_computation(sdfg, state, node, "axes")
        constant_folding.remove_node_and_computation(sdfg, state, node, "ends")
        constant_folding.remove_node_and_computation(sdfg, state, node, "starts")
        constant_folding.remove_node_and_computation(sdfg, state, node, "steps")

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        idesc = in_desc_with_name(node, state, sdfg, "data")
        odesc = out_desc_with_name(node, state, sdfg, "output")
        nsdfg.add_datadesc("data", copy.deepcopy(idesc))
        nsdfg.add_datadesc("output", copy.deepcopy(odesc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["output"].transient = False

        if not isinstance(axes, (tuple, list)):
            axes = [axes]
            ends = [ends]
            starts = [starts]
            steps = [steps]

        # Set up slicing memlet
        rng = [(0, s - 1, 1) for s in idesc.shape]
        for axis, start, end, step in zip(axes, starts, ends, steps):
            s = idesc.shape[axis]
            if end > s:
                end = s
            rng[axis] = (start, end - 1, step)

        sbs = subsets.Range(rng)
        osbs = subsets.Range.from_array(odesc)

        # Make copy / view
        rnode = nstate.add_read("data")
        wnode = nstate.add_write("output")

        nstate.add_nedge(rnode, wnode, dace.Memlet(data="data", subset=sbs, other_subset=osbs))

        return nsdfg


@op_implementation(op="Shape", name="pure")
class PureShape(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState,
                               sdfg: SDFG) -> bool:
        data_desc = in_desc_with_name(node, state, sdfg, "data")

        try:
            np.array(data_desc.shape, np.int64)
        except Exception:
            # this happens if the shape is symbolic, for example
            return False

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        data_desc = in_desc_with_name(node, state, sdfg, "data")
        shape_val = np.array(data_desc.shape, np.int64)

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        nsdfg.add_datadesc(
            "data",
            copy.deepcopy(data_desc),
        )
        nsdfg.arrays["data"].transient = False
        nsdfg.add_array("shape", shape_val.shape, dtype=dace.int64)
        s = nstate.add_write("shape")

        for i, v in enumerate(shape_val):
            tasklet = nstate.add_tasklet("write_shape", {},
                                         {'shape_scalar': dace.int64},
                                         f"shape_scalar = {v}")
            nstate.add_edge(tasklet, "shape_scalar", s, None,
                            dace.Memlet("shape[{}]".format(i)))

        return nsdfg


@op_implementation(op="Gather", name="pure")
class PureGather(ONNXForward):
    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # To understand this operator, read the docs for np.take.
        # The ONNX docs are not easy to understand (and are incorrect in opset 11)

        nsdfg, nstate, _, _ = empty_sdfg_for_node(sdfg,
                                                  state,
                                                  node,
                                                  add_access_nodes=False)
        out_desc = out_desc_with_name(node, state, sdfg, "output")
        out_shape = out_desc.shape
        idx_desc = in_desc_with_name(node, state, sdfg, "indices")
        idx_shape = idx_desc.shape
        data_shape = in_desc_with_name(node, state, sdfg, "data").shape

        # FIXME: we can sometimes generate views

        # Generate a copy kernel that loops over every element in the output
        # and read the correct element according to the indices

        axis = node.axis

        map_ranges = [(f"i{i}", f"0:{s}") for i, s in enumerate(out_shape)]
        # the map ranges can be partitioned into two parts.
        # the first part is the range over the indices, the second part is the
        # range over the data
        if isinstance(idx_desc, data.Scalar):
            # handle the edgecase here because the shape of a scalar in dace is
            # (1,) not ()
            idx_len = 0
        else:
            idx_len = len(idx_shape)
        map_ranges_indices = map_ranges[axis:axis + idx_len]
        map_ranges_data = map_ranges[:axis] + map_ranges[axis + idx_len:]

        # compute the indexing expressions
        fst = lambda x: x[0]
        output_idx_str = 'output[' + ', '.join(map(fst, map_ranges)) + ']'
        # the memlet string used to read data, which reads the whole axis
        data_memlet_elems = list(map(fst, map_ranges_data))
        data_memlet_elems.insert(axis, f'0:{data_shape[axis]}')

        data_memlet_str = 'data[' + ', '.join(data_memlet_elems) + ']'

        indices_idx_str = 'indices'
        if map_ranges_indices:
            indices_idx_str += '[' + ', '.join(map(fst,
                                                   map_ranges_indices)) + ']'
        else:
            indices_idx_str += '[0]'

        tasklet, me, mx = nstate.add_mapped_tasklet(
            node.label + "_tasklet",
            map_ranges=map_ranges,
            inputs={
                "__data": dace.Memlet(data_memlet_str),
                "idx": dace.Memlet(indices_idx_str),
            },
            code=f"__output = __data[idx]",
            outputs={"__output": dace.Memlet(output_idx_str)},
            external_edges=True)

        # required to make underlying code to see it as a pointer and enable index-based access
        # even if the data contains just a single element
        tasklet.in_connectors["__data"] = dace.pointer(out_desc.dtype)

        return nsdfg
    
@op_implementation(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Allow any axes input
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        keepdims = getattr(node, 'keepdims', 1)
        noop_with_empty_axes = getattr(node, 'noop_with_empty_axes', 0)
        uid = state.node_id(node)
        nsdfg = SDFG(f'reduce_sum_{uid}')
        nstate = nsdfg.add_state(f'reduce_sum_{uid}')
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        reduced_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "reduced"))
        data_name = f"data"
        reduced_name = f"reduced"
        nsdfg.add_datadesc(data_name, data_desc)
        nsdfg.add_datadesc(reduced_name, reduced_desc)
        nsdfg.arrays[data_name].transient = False
        nsdfg.arrays[reduced_name].transient = False
        data_read = nstate.add_read(data_name)
        reduced_write = nstate.add_write(reduced_name)
        axes_name = f"axes"
        if axes_name in node.in_connectors:
            [axes_edge] = state.in_edges_by_connector(node, 'axes')
            axes_desc = copy.deepcopy(sdfg.arrays[axes_edge.data.data])
            nsdfg.add_datadesc(axes_name, axes_desc)
            nsdfg.arrays[axes_name].transient = False
            axes_node = nstate.add_access(axes_name)
        else:
            axes = getattr(node, axes_name, None)
            if not axes:
                if noop_with_empty_axes:
                    axes_values = []
                else:
                    axes_values = list(range(len(data_desc.shape)))
            else:
                axes_values = list(axes)
            axes_arr_shape = [len(axes_values)]
            axes_arr_dtype = dace.int64
            _, axes_desc = nsdfg.add_array(axes_name, axes_arr_shape, axes_arr_dtype)
            axes_node = nstate.add_access(axes_name)
            axes_init_tasklet = nstate.add_tasklet(
                f"init_axes",
                set(),
                {"out": dace.pointer(axes_arr_dtype)},
                "\n".join([f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
                language=dace.Language.CPP
            )
            nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"{axes_name}[0:{len(axes_values)}]"))

        is_axes_scalar = len(axes_desc.shape) == 1 and axes_desc.shape[0] == 1

        # Create the reduction tasklet with embedded constants
        tasklet_code = (f"""
            constexpr long long input_dims = {len(data_desc.shape)};
            constexpr long long output_dims = {len(reduced_desc.shape)};
            constexpr long long num_reduce_dims = {len(axes_desc.shape)};
            constexpr long long num_non_reduce_dims = {len(data_desc.shape) - len(axes_desc.shape)};
            constexpr long long keepdims = {keepdims};
            long long reduce_dims [num_reduce_dims] = """ +
            ("{axes_arr};" if is_axes_scalar else f'{{{", ".join([f"axes_arr[{i}]" for i in range(len(axes_desc.shape))])}}};') +
            f""" 

            long long input_shape [input_dims] = {{{", ".join([str(data_desc.shape[i]) for i in range(len(data_desc.shape))])}}};
            long long output_shape [output_dims] = {{{", ".join([str(reduced_desc.shape[i]) for i in range(len(reduced_desc.shape))])}}};
            long long input_strides [input_dims] = {{{", ".join([str(data_desc.strides[i]) for i in range(len(data_desc.shape))])}}};
            long long output_strides [output_dims] = {{{", ".join([str(reduced_desc.strides[i]) for i in range(len(reduced_desc.shape))])}}};
            long long output_strides_input [input_dims];
            long long output_strides_input_idx = 0;
            for (long long i = 0; i < input_dims; i++) {{
                int is_reduce_dim = 0;
                for (long long j = 0; j < num_reduce_dims; j++) {{
                    if (reduce_dims[j] == i) {{
                        is_reduce_dim = 1;
                        break;
                    }}
                }}
                if (is_reduce_dim) {{
                    output_strides_input[i] = 0;
                    if (keepdims) {{
                        output_strides_input_idx++;
                    }}
                }} else {{
                    output_strides_input[i] = output_strides [output_strides_input_idx];
                    output_strides_input_idx++;
                }}
            }}
            
            // initialize output to zero
            """ +
            "\n".join([f"for (long long i{i} = 0; i{i} < output_shape[{i}]; i{i}++) {{" for i in range(len(reduced_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides [{i}] * i{i}" for i in range(len(reduced_desc.shape))]) + "] = 0.0;" + "\n" +
            "\n".join(["}" for _ in reduced_desc.shape]) + "\n" +
            """
            
            // Loop over all input elements
            """ + 
            "\n".join([f"for (long long i{i} = 0; i{i} < input_shape[{i}]; i{i}++) {{" for i in range(len(data_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))]) +"]" +
            " += inp [" + " + ".join([f"input_strides[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "];" + "\n" +
            "\n".join(["}" for _ in data_desc.shape]) + "\n" +
            """
        """)
        tasklet = nstate.add_tasklet(
            f'reduce_sum_{uid}',
            {'inp': dace.pointer(data_desc.dtype), 'axes_arr': axes_desc.dtype},
            {'out': dace.pointer(reduced_desc.dtype)},
            tasklet_code,
            language=dace.Language.CPP)
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_name))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_name))
        return nsdfg
    
@op_implementation(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        keepdims = getattr(node, 'keepdims', 1)
        noop_with_empty_axes = getattr(node, 'noop_with_empty_axes', 0)
        uid = state.node_id(node)
        nsdfg = SDFG(f'reduce_max_{uid}')
        nstate = nsdfg.add_state(f'reduce_max_{uid}')
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        reduced_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "reduced"))
        data_name = f"data"
        reduced_name = f"reduced"
        nsdfg.add_datadesc(data_name, data_desc)
        nsdfg.add_datadesc(reduced_name, reduced_desc)
        nsdfg.arrays[data_name].transient = False
        nsdfg.arrays[reduced_name].transient = False
        data_read = nstate.add_read(data_name)
        reduced_write = nstate.add_write(reduced_name)
        axes_name = f"axes"
        if axes_name in node.in_connectors:
            [axes_edge] = state.in_edges_by_connector(node, 'axes')
            axes_desc = copy.deepcopy(sdfg.arrays[axes_edge.data.data])
            nsdfg.add_datadesc(axes_name, axes_desc)
            nsdfg.arrays[axes_name].transient = False
            axes_node = nstate.add_access(axes_name)
        else:
            axes = getattr(node, axes_name, None)
            if not axes:
                if noop_with_empty_axes:
                    axes_values = []
                else:
                    axes_values = list(range(len(data_desc.shape)))
            else:
                axes_values = list(axes)
            axes_arr_shape = [len(axes_values)]
            axes_arr_dtype = dace.int64
            _, axes_desc = nsdfg.add_array(axes_name, axes_arr_shape, axes_arr_dtype)
            axes_node = nstate.add_access(axes_name)
            axes_init_tasklet = nstate.add_tasklet(
                f"init_axes",
                set(),
                {"out": dace.pointer(axes_arr_dtype)},
                "\n".join([f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
                language=dace.Language.CPP
            )
            nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"{axes_name}[0:{len(axes_values)}]"))

        is_axes_scalar = len(axes_desc.shape) == 1 and axes_desc.shape[0] == 1

        # Create the reduction tasklet with embedded constants
        tasklet_code = (f"""
            constexpr long long input_dims = {len(data_desc.shape)};
            constexpr long long output_dims = {len(reduced_desc.shape)};
            constexpr long long num_reduce_dims = {len(axes_desc.shape)};
            constexpr long long num_non_reduce_dims = {len(data_desc.shape) - len(axes_desc.shape)};
            constexpr long long keepdims = {keepdims};
            long long reduce_dims[num_reduce_dims] = """ +
            ("{axes_arr};" if is_axes_scalar else f'{{{", ".join([f"axes_arr[{i}]" for i in range(len(axes_desc.shape))])}}};') +
            f""" 

            long long input_shape [input_dims] = {{{", ".join([str(data_desc.shape[i]) for i in range(len(data_desc.shape))])}}};
            long long output_shape [output_dims] = {{{", ".join([str(reduced_desc.shape[i]) for i in range(len(reduced_desc.shape))])}}};
            long long input_strides [input_dims] = {{{", ".join([str(data_desc.strides[i]) for i in range(len(data_desc.shape))])}}};
            long long output_strides [output_dims] = {{{", ".join([str(reduced_desc.strides[i]) for i in range(len(reduced_desc.shape))])}}};
            long long output_strides_input[input_dims];
            long long output_strides_input_idx = 0;
            for (long long i = 0; i < input_dims; i++) {{
                int is_reduce_dim = 0;
                for (long long j = 0; j < num_reduce_dims; j++) {{
                    if (reduce_dims[j] == i) {{
                        is_reduce_dim = 1;
                        break;
                    }}
                }}
                if (is_reduce_dim) {{
                    output_strides_input[i] = 0;
                    if (keepdims) {{
                        output_strides_input_idx++;
                    }}
                }} else {{
                    output_strides_input[i] = output_strides [output_strides_input_idx];
                    output_strides_input_idx++;
                }}
            }}
            
            // initialize output to negative infinity
            """ +
            "\n".join([f"for (long long i{i} = 0; i{i} < output_shape[{i}]; i{i}++) {{" for i in range(len(reduced_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides [{i}] * i{i}" for i in range(len(reduced_desc.shape))]) + "] = -INFINITY;" + "\n" +
            "\n".join(["}" for _ in reduced_desc.shape]) + "\n" +
            """
            
            // Loop over all input elements
            """ + 
            "\n".join([f"for (long long i{i} = 0; i{i} < input_shape[{i}]; i{i}++) {{" for i in range(len(data_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))]) +"]" +
            " = std::max(out [" + " + ".join([f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "], " +
            "inp [" + " + ".join([f"input_strides[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "]);" + "\n" +
            "\n".join(["}" for _ in data_desc.shape]) + "\n" +
            """
        """)
        tasklet = nstate.add_tasklet(
            f'reduce_max_{uid}',
            {'inp': dace.pointer(data_desc.dtype), 'axes_arr': axes_desc.dtype},
            {'out': dace.pointer(reduced_desc.dtype)},
            tasklet_code,
            language=dace.Language.CPP)
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_name))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_name))
        return nsdfg
    
@op_implementation(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        keepdims = getattr(node, 'keepdims', 1)
        noop_with_empty_axes = getattr(node, 'noop_with_empty_axes', 0)
        uid = state.node_id(node)
        nsdfg = SDFG(f'reduce_min_{uid}')
        nstate = nsdfg.add_state(f'reduce_min_{uid}')
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        reduced_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "reduced"))
        data_name = f"data"
        reduced_name = f"reduced"
        nsdfg.add_datadesc(data_name, data_desc)
        nsdfg.add_datadesc(reduced_name, reduced_desc)
        nsdfg.arrays[data_name].transient = False
        nsdfg.arrays[reduced_name].transient = False
        data_read = nstate.add_read(data_name)
        reduced_write = nstate.add_write(reduced_name)
        axes_name = f"axes"
        if axes_name in node.in_connectors:
            [axes_edge] = state.in_edges_by_connector(node, 'axes')
            axes_desc = copy.deepcopy(sdfg.arrays[axes_edge.data.data])
            nsdfg.add_datadesc(axes_name, axes_desc)
            nsdfg.arrays[axes_name].transient = False
            axes_node = nstate.add_access(axes_name)
        else:
            axes = getattr(node, axes_name, None)
            if not axes:
                if noop_with_empty_axes:
                    axes_values = []
                else:
                    axes_values = list(range(len(data_desc.shape)))
            else:
                axes_values = list(axes)
            axes_arr_shape = [len(axes_values)]
            axes_arr_dtype = dace.int64
            _, axes_desc = nsdfg.add_array(axes_name, axes_arr_shape, axes_arr_dtype)
            axes_node = nstate.add_access(axes_name)
            axes_init_tasklet = nstate.add_tasklet(
                f"init_axes",
                set(),
                {"out": dace.pointer(axes_arr_dtype)},
                "\n".join([f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
                language=dace.Language.CPP
            )
            nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"{axes_name}[0:{len(axes_values)}]"))

        is_axes_scalar = len(axes_desc.shape) == 1 and axes_desc.shape[0] == 1

        # Create the reduction tasklet with embedded constants
        tasklet_code = (f"""
            constexpr long long input_dims = {len(data_desc.shape)};
            constexpr long long output_dims = {len(reduced_desc.shape)};
            constexpr long long num_reduce_dims = {len(axes_desc.shape)};
            constexpr long long num_non_reduce_dims = {len(data_desc.shape) - len(axes_desc.shape)};
            constexpr long long keepdims = {keepdims};
            long long reduce_dims[num_reduce_dims] = """ +
            ("{axes_arr};" if is_axes_scalar else f'{{{", ".join([f"axes_arr[{i}]" for i in range(len(axes_desc.shape))])}}};') +
            f""" 

            long long input_shape [input_dims] = {{{", ".join([str(data_desc.shape[i]) for i in range(len(data_desc.shape))])}}};
            long long output_shape [output_dims] = {{{", ".join([str(reduced_desc.shape[i]) for i in range(len(reduced_desc.shape))])}}};
            long long input_strides [input_dims] = {{{", ".join([str(data_desc.strides[i]) for i in range(len(data_desc.shape))])}}};
            long long output_strides [output_dims] = {{{", ".join([str(reduced_desc.strides[i]) for i in range(len(reduced_desc.shape))])}}};
            long long output_strides_input [input_dims];
            long long output_strides_input_idx = 0;
            for (long long i = 0; i < input_dims; i++) {{
                int is_reduce_dim = 0;
                for (long long j = 0; j < num_reduce_dims; j++) {{
                    if (reduce_dims[j] == i) {{
                        is_reduce_dim = 1;
                        break;
                    }}
                }}
                if (is_reduce_dim) {{
                    output_strides_input[i] = 0;
                    if (keepdims) {{
                        output_strides_input_idx++;
                    }}
                }} else {{
                    output_strides_input[i] = output_strides [output_strides_input_idx];
                    output_strides_input_idx++;
                }}
            }}
            
            // initialize output to positive infinity
            """ +
            "\n".join([f"for (long long i{i} = 0; i{i} < output_shape[{i}]; i{i}++) {{" for i in range(len(reduced_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides [{i}] * i{i}" for i in range(len(reduced_desc.shape))]) + "] = INFINITY;" + "\n" +
            "\n".join(["}" for _ in reduced_desc.shape]) + "\n" +
            """
            
            // Loop over all input elements
            """ + 
            "\n".join([f"for (long long i{i} = 0; i{i} < input_shape[{i}]; i{i}++) {{" for i in range(len(data_desc.shape))]) + "\n" +
            "out [" + " + ".join([f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))]) +"]" +
            " = std::min(out [" + " + ".join([f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "], " +
            "inp [" + " + ".join([f"input_strides[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "]);" + "\n" +
            "\n".join(["}" for _ in data_desc.shape]) + "\n" +
            """
        """)
        tasklet = nstate.add_tasklet(
            f'reduce_min_{uid}',
            {'inp': dace.pointer(data_desc.dtype), 'axes_arr': axes_desc.dtype},
            {'out': dace.pointer(reduced_desc.dtype)},
            tasklet_code,
            language=dace.Language.CPP)
        nstate.add_edge(data_read, None, tasklet, 'inp', nsdfg.make_array_memlet(data_name))
        nstate.add_edge(axes_node, None, tasklet, 'axes_arr', nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(tasklet, 'out', reduced_write, None, nsdfg.make_array_memlet(reduced_name))
        return nsdfg