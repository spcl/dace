# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Image and Signal Processing Operations for ONNX in DaCe.

This module provides implementations for ONNX operations related to image and signal
processing, including resizing, interpolation, and related transformations.

Operations implemented:
- Resize: Image resizing with various interpolation modes (nearest, linear, cubic)
  and coordinate transformation modes
"""

import copy
import typing

import dace
from dace import SDFG, SDFGState
from dace.sdfg.nodes import Node
from dace.util import in_desc_with_name, out_desc_with_name

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.op_implementations.utils import op_implementation


@op_implementation(op="Resize", name="pure")
class PureResize(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
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
            if nearest_mode is not None and nearest_mode not in [
                    'round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil'
            ]:
                return False

        # Check coordinate transformation mode
        coord_mode = getattr(node, 'coordinate_transformation_mode', 'half_pixel')
        if coord_mode is not None and coord_mode not in [
                'half_pixel', 'half_pixel_symmetric', 'pytorch_half_pixel', 'align_corners', 'asymmetric',
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
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

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

        tasklet = nstate.add_tasklet(f'tasklet_reshape',
                                     tasklet_inputs, {'__out': dace.pointer(out_data_desc.dtype)},
                                     "\n".join(tasklet_code),
                                     language=dace.Language.CPP)

        # Connect tasklet inputs
        nstate.add_edge(inp_read, None, tasklet, "__inp", dace.Memlet.from_array(inp_name, inp_data_desc))
        if has_scales:
            nstate.add_edge(scales_read, None, tasklet, "__scales",
                            dace.Memlet.from_array(scales_name, scales_data_desc))
        if has_sizes:
            nstate.add_edge(sizes_read, None, tasklet, "__sizes", dace.Memlet.from_array(sizes_name, sizes_data_desc))
        if has_roi:
            nstate.add_edge(roi_read, None, tasklet, "__roi", dace.Memlet.from_array(roi_name, roi_data_desc))

        # Connect tasklet output
        nstate.add_edge(tasklet, "__out", out_write, None, dace.Memlet.from_array(out_name, out_data_desc))

        return nsdfg
