# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Array and Tensor Manipulation Operations for ONNX in DaCe.

This module provides pure DaCe implementations for ONNX array/tensor manipulation
operations. These operations handle shape manipulation, slicing, and other array transformations.

The module contains:
- Shape manipulation operations (Reshape, Flatten, Squeeze, Unsqueeze, Expand)
- Slicing and indexing operations (Slice, SliceAllConstant, Gather)
- Concatenation and splitting operations (Concat, Split)
- Transposition operations (Transpose, EinsumTranspose)
- Shape query operations (Shape)

Each implementation follows the ONNX specification and is designed to be:
- Semantically correct according to ONNX standards
- Efficient when converted to DaCe SDFGs
"""

import copy
from math import prod
import typing

import dace
import numpy as np
from dace import SDFG, SDFGState, subsets
from dace.sdfg.nodes import Node
from dace.util import in_desc_with_name, in_edge_with_name, iterables_equal, out_desc_with_name

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.utils import (empty_sdfg_for_node, op_implementation, program_for_node,
                                                          python_pure_op_implementation)
from dace.libraries.onnx.op_implementations.common import broadcast_indices, create_memlet_str
from dace.transformation.onnx import constant_folding
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.libraries.onnx import converters

# ==============================================================================
# Pad Operations
# ==============================================================================


@op_implementation(op="Pad", name="pure")
class PurePad(ONNXForward):
    """
    Pure implementation of ONNX Pad operator.

    Pads a tensor with a constant value along specified dimensions.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # For now, only support constant padding with pads as an input
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # The ONNX Pad operator takes:
        # - data: input tensor
        # - pads: padding values (2 * rank of data)
        # - constant_value (optional): value to pad with (default 0)

        # Get descriptors
        data_desc = in_desc_with_name(node, state, sdfg, "data")
        output_desc = out_desc_with_name(node, state, sdfg, "output")
        pads_desc = in_desc_with_name(node, state, sdfg, "pads")

        # Check if constant_value input exists
        has_constant_value = len(list(state.in_edges_by_connector(node, "constant_value"))) > 0

        # Create a new SDFG for the expansion
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors to nested SDFG
        nsdfg.add_datadesc("data", copy.deepcopy(data_desc))
        nsdfg.add_datadesc("output", copy.deepcopy(output_desc))
        nsdfg.add_datadesc("pads", copy.deepcopy(pads_desc))
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["output"].transient = False
        nsdfg.arrays["pads"].transient = False

        if has_constant_value:
            constant_value_desc = in_desc_with_name(node, state, sdfg, "constant_value")
            nsdfg.add_datadesc("constant_value", copy.deepcopy(constant_value_desc))
            nsdfg.arrays["constant_value"].transient = False

        # Generate code to copy with padding
        ndim = len(data_desc.shape)
        output_shape = output_desc.shape
        input_shape = data_desc.shape

        # Create a map over the output tensor
        map_ranges = {f"i{d}": f"0:{output_shape[d]}" for d in range(ndim)}
        output_indices = ', '.join([f'i{d}' for d in range(ndim)])

        # Generate tasklet code
        code_lines = []
        code_lines.append(f"# Compute input indices from output indices")
        for d in range(ndim):
            code_lines.append(f"input_i{d} = i{d} - int(__pads[{d}])")

        # Check if we're in the padding region
        condition_parts = []
        for d in range(ndim):
            condition_parts.append(f"(input_i{d} >= 0 and input_i{d} < {input_shape[d]})")
        condition = " and ".join(condition_parts)

        code_lines.append(f"if {condition}:")
        input_indices = ', '.join([f'input_i{d}' for d in range(ndim)])
        code_lines.append(f"    __out = __data[{input_indices}]")
        code_lines.append("else:")
        if has_constant_value:
            code_lines.append("    __out = __constant_value[0]")
        else:
            code_lines.append("    __out = 0")

        tasklet_code = '\n'.join(code_lines)

        # Build tasklet inputs
        tasklet_inputs = {
            "__data": dace.Memlet.from_array("data", data_desc),
            "__pads": dace.Memlet.from_array("pads", pads_desc)
        }
        if has_constant_value:
            tasklet_inputs["__constant_value"] = dace.Memlet.from_array("constant_value", constant_value_desc)

        # Create mapped tasklet with external edges
        nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                  map_ranges=map_ranges,
                                  inputs=tasklet_inputs,
                                  code=tasklet_code,
                                  outputs={"__out": dace.Memlet(f"output[{output_indices}]")},
                                  external_edges=True)

        return nsdfg


# ==============================================================================
# Concatenation Operations
# ==============================================================================


@op_implementation(op="Concat", name="pure")
class PureConcat(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
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

            map_entry, map_exit = nstate.add_map(f"concat_map_{inp_idx}", {
                f"i{i}": f"0:{s}"
                for i, s in enumerate(inp_shapes[inp_idx])
            })

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

            nstate.add_memlet_path(inp_read, map_entry, tasklet, memlet=inp_memlet, dst_conn="inp")
            nstate.add_memlet_path(tasklet, map_exit, out_write, memlet=out_memlet, src_conn="out")

        return nsdfg


# ==============================================================================
# Shape Manipulation Operations - Unsqueeze
# ==============================================================================


@op_implementation(op="Unsqueeze", name="pure")
class PureUnsqueeze(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # Get input/output descriptors
        expanded_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "expanded"))

        def prog(data, expanded):
            expanded[:] = np.reshape(data, expanded_desc.shape)

        return program_for_node(prog, sdfg, state, node)


# ==============================================================================
# Shape Manipulation Operations - Squeeze
# ==============================================================================


@op_implementation(op="Squeeze", name="pure")
class PureSqueeze(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        squeezed_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "squeezed"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("data", data_desc)
        nsdfg.add_datadesc("squeezed", squeezed_desc)
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["squeezed"].transient = False

        # Add access nodes
        data_read = nstate.add_read("data")
        squeezed_write = nstate.add_write("squeezed")

        # Check if axes input is provided
        has_axes = len(list(state.in_edges_by_connector(node, "axes"))) > 0

        # Prepare tasklet inputs
        tasklet_inputs = {"__data": dace.pointer(data_desc.dtype)}
        if has_axes:
            # Get axes descriptor and add to SDFG
            axes_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "axes"))
            nsdfg.add_datadesc("axes", axes_desc)
            nsdfg.arrays["axes"].transient = False
            axes_read = nstate.add_read("axes")
            tasklet_inputs["__axes"] = dace.pointer(axes_desc.dtype)

        is_scalar_input = not isinstance(node.in_connectors['data'], dace.dtypes.pointer) and data_desc.total_size == 1
        is_scalar_input = False
        if is_scalar_input:
            data_str = "(&__data)"
        else:
            data_str = "__data"

        # Create tasklet that performs the squeeze operation
        data_size = int(np.prod(data_desc.shape))
        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs=tasklet_inputs,
                                     outputs={"__squeezed": dace.pointer(squeezed_desc.dtype)},
                                     code=f"""
            for (int i = 0; i < {data_size}; i++) {{
                __squeezed[i] = {data_str}[i];
            }}
            """,
                                     language=dace.Language.CPP)

        # Connect the tasklet with memlets
        nstate.add_edge(data_read, None, tasklet, "__data", dace.Memlet("data"))
        if has_axes:
            nstate.add_edge(axes_read, None, tasklet, "__axes", dace.Memlet("axes"))
        nstate.add_edge(tasklet, "__squeezed", squeezed_write, None, dace.Memlet("squeezed"))

        return nsdfg


# ==============================================================================
# Shape Manipulation Operations - Expand
# ==============================================================================


@op_implementation(op="Expand", name="pure")
class PureExpand(ONNXForward):
    """
    Pure implementation of ONNX Expand operator using broadcasting.

    The Expand operator broadcasts the input tensor to a new shape following NumPy's broadcasting rules.
    This implementation uses a mapped tasklet to copy elements with proper index translation.
    It handles both the general broadcasting case and the no-op case where shapes are identical.
    """

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from dace.libraries.onnx.op_implementations.common import broadcast_indices
        from dace.transformation.onnx import constant_folding

        # Get input and output descriptors
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")

        # Remove the 'shape' input since we already know the output shape
        # and don't need to propagate gradients through it
        constant_folding.remove_node_and_computation(sdfg, state, node, "shape")

        # Create nested SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors to nested SDFG
        nsdfg.add_datadesc("input", copy.deepcopy(input_desc))
        nsdfg.add_datadesc("output", copy.deepcopy(output_desc))
        nsdfg.arrays["input"].transient = False
        nsdfg.arrays["output"].transient = False

        # Generate broadcast-aware indexing using the common helper
        input_indices = broadcast_indices(input_desc.shape, output_desc.shape)
        input_memlet_str = create_memlet_str("input", input_indices, input_desc.shape)

        # Create map over output shape
        map_ranges = {f"i{i}": f"0:{output_desc.shape[i]}" for i in range(len(output_desc.shape))}
        output_index_str = ", ".join(map_ranges.keys())

        # Create mapped tasklet for broadcasting copy
        nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                  map_ranges=map_ranges,
                                  inputs={"__input": dace.Memlet(input_memlet_str)},
                                  code="__output = __input",
                                  outputs={"__output": dace.Memlet(f"output[{output_index_str}]")},
                                  external_edges=True)

        return nsdfg


# ==============================================================================
# Transposition Operations
# ==============================================================================


@python_pure_op_implementation(
    perm=lambda node, data: node.perm if node.perm is not None else list(reversed(range(len(data.shape)))))
def Transpose(data, transposed):
    transposed[:] = np.transpose(data, axes=perm)


@op_implementation(op="Transpose", name="einsum")
class EinsumTranspose(ONNXForward):

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from dace.libraries.onnx.nodes.onnx_op_registry import ONNXEinsum  # avoid import loop
        perm = node.perm
        input_desc = in_desc_with_name(node, state, sdfg, "data")
        output_desc = out_desc_with_name(node, state, sdfg, "transposed")

        letters = [chr(ord('z') - i) for i in range(26)]
        input_letters = "".join(letters[i] for i, _ in enumerate(input_desc.shape))
        output_letters = "".join(letters[i] for i in perm)
        equation_str = f"{input_letters}->{output_letters}"

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
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


# ==============================================================================
# Reshape and Flatten Operations
# ==============================================================================


@python_pure_op_implementation(shape=lambda reshaped: reshaped.shape,
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


# ==============================================================================
# Slicing Operations
# ==============================================================================


@op_implementation(op="Slice", name="pure")
class PureSlice(ONNXForward):
    """
    Unified Slice implementation supporting both constant and dynamic inputs.

    Handles ONNX Slice operator with:
    - Required inputs: data, starts, ends
    - Optional inputs: axes (default: [0, 1, ..., len(starts)-1]), steps (default: all 1s)

    For constant inputs, uses efficient memlet slicing.
    For dynamic inputs, generates runtime slicing code.
    """

    @staticmethod
    def _get_constant(conn: str, node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG):
        """Try to get constant value for an input connector."""
        try:
            srcnode = next(state.in_edges_by_connector(node, conn)).src
        except StopIteration:
            # Input not provided - return None (will use defaults)
            return None
        # Handle GPU scalar copies
        if 'gpu_' in srcnode.data:
            srcnode = state.predecessors(srcnode)[0]

        # Try to get constant value - only works if _parent_onnx_model is available
        try:
            if hasattr(sdfg, '_parent_onnx_model'):
                return onnx_constant_or_none(sdfg, srcnode)
        except:
            pass
        return None

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # Always applicable - handles both constant and dynamic inputs
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Try to get constant values for all inputs
        axes_const = PureSlice._get_constant('axes', node, state, sdfg)
        ends_const = PureSlice._get_constant('ends', node, state, sdfg)
        starts_const = PureSlice._get_constant('starts', node, state, sdfg)
        steps_const = PureSlice._get_constant('steps', node, state, sdfg)

        # Check if all required inputs are constant
        all_constant = (starts_const is not None and ends_const is not None)

        # For fully constant case, use optimized memlet slicing
        if all_constant and axes_const is not None and (steps_const is not None or steps_const == 1):
            # Remove constant inputs
            constant_folding.remove_node_and_computation(sdfg, state, node, "axes")
            constant_folding.remove_node_and_computation(sdfg, state, node, "ends")
            constant_folding.remove_node_and_computation(sdfg, state, node, "starts")
            if steps_const is not None:
                constant_folding.remove_node_and_computation(sdfg, state, node, "steps")

            nsdfg = dace.SDFG(node.label + "_expansion")
            nstate = nsdfg.add_state()

            idesc = in_desc_with_name(node, state, sdfg, "data")
            odesc = out_desc_with_name(node, state, sdfg, "output")
            nsdfg.add_datadesc("data", copy.deepcopy(idesc))
            nsdfg.add_datadesc("output", copy.deepcopy(odesc))
            nsdfg.arrays["data"].transient = False
            nsdfg.arrays["output"].transient = False

            # Normalize to lists
            if not isinstance(axes_const, (tuple, list)):
                axes_const = [int(axes_const)]
                ends_const = [int(ends_const)]
                starts_const = [int(starts_const)]
                if steps_const is not None:
                    steps_const = [int(steps_const)]
                else:
                    steps_const = [1]

            # Set up slicing memlet
            rng = [(0, int(s) - 1, 1) for s in idesc.shape]
            for i, axis in enumerate(axes_const):
                axis = int(axis)
                start = int(starts_const[i])
                end = int(ends_const[i])
                step = int(steps_const[i]) if i < len(steps_const) else 1

                s = int(idesc.shape[axis])
                # Handle negative indices
                if start < 0:
                    start = s + start
                if end < 0:
                    end = s + end

                # Handle negative steps
                if step < 0:
                    # For negative steps, we go from start down to end (exclusive)
                    # Clamp start to valid range
                    if start >= s:
                        start = s - 1
                    # For negative step, end can be < 0 (meaning go to beginning)
                    # Clamp end to be at least -1 (which means include index 0)
                    if end < -1:
                        end = -1
                    # In DaCe Range with negative step, the range is (start, end, step)
                    # where start > end
                    rng[axis] = (start, end, step)
                else:
                    # Handle out of bounds for positive steps
                    if end > s:
                        end = s
                    # For positive steps, DaCe Range uses inclusive end, so end - 1
                    rng[axis] = (start, end - 1, step)

            sbs = subsets.Range(rng)
            osbs = subsets.Range.from_array(odesc)

            # Make copy / view
            rnode = nstate.add_read("data")
            wnode = nstate.add_write("output")

            nstate.add_nedge(rnode, wnode, dace.Memlet(data="data", subset=sbs, other_subset=osbs))

            return nsdfg

        # Dynamic case: generate runtime slicing code
        else:
            nsdfg = dace.SDFG(node.label + "_expansion")
            nstate = nsdfg.add_state()

            idesc = in_desc_with_name(node, state, sdfg, "data")
            odesc = out_desc_with_name(node, state, sdfg, "output")

            nsdfg.add_datadesc("data", copy.deepcopy(idesc))
            nsdfg.add_datadesc("output", copy.deepcopy(odesc))
            nsdfg.arrays["data"].transient = False
            nsdfg.arrays["output"].transient = False

            # Add input descriptors for dynamic inputs
            starts_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "starts"))
            ends_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "ends"))
            nsdfg.add_datadesc("starts", starts_desc)
            nsdfg.add_datadesc("ends", ends_desc)
            nsdfg.arrays["starts"].transient = False
            nsdfg.arrays["ends"].transient = False

            # Check for optional axes and steps
            has_axes = False
            has_steps = False
            try:
                next(state.in_edges_by_connector(node, "axes"))
                has_axes = True
                axes_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "axes"))
                nsdfg.add_datadesc("axes", axes_desc)
                nsdfg.arrays["axes"].transient = False
            except (StopIteration, ValueError):
                pass

            try:
                next(state.in_edges_by_connector(node, "steps"))
                has_steps = True
                steps_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "steps"))
                nsdfg.add_datadesc("steps", steps_desc)
                nsdfg.arrays["steps"].transient = False
            except (StopIteration, ValueError):
                pass

            # Generate C++ code for dynamic slicing
            num_dims = len(idesc.shape)
            num_slices = int(np.prod(starts_desc.shape))

            # Build tasklet inputs
            tasklet_inputs = {
                "__data": dace.pointer(idesc.dtype),
                "__starts": dace.pointer(starts_desc.dtype),
                "__ends": dace.pointer(ends_desc.dtype)
            }
            if has_axes:
                tasklet_inputs["__axes"] = dace.pointer(axes_desc.dtype)
            if has_steps:
                tasklet_inputs["__steps"] = dace.pointer(steps_desc.dtype)

            # Pre-compute shape information for code generation
            out_shape_strs = [f'({odesc.shape[dim_i]})' for dim_i in range(num_dims)]
            in_shape_strs = [f'({idesc.shape[dim_i]})' for dim_i in range(num_dims)]

            # Generate shape arrays as comma-separated lists
            out_shapes_list = ', '.join(out_shape_strs)
            in_shapes_list = ', '.join(in_shape_strs)

            # Generate slicing code
            code = f"""
            // Define shapes
            long long out_shape[{num_dims}] = {{{out_shapes_list}}};
            long long in_shape[{num_dims}] = {{{in_shapes_list}}};

            // Compute output strides (row-major order)
            long long out_strides[{num_dims}];
            out_strides[{num_dims - 1}] = 1;
            for (int i = {num_dims} - 2; i >= 0; i--) {{
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }}

            // Compute input strides (row-major order)
            long long in_strides[{num_dims}];
            in_strides[{num_dims - 1}] = 1;
            for (int i = {num_dims} - 2; i >= 0; i--) {{
                in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
            }}

            // Create mapping from axis to slice parameters
            long long slice_start[{num_dims}];
            long long slice_step[{num_dims}];
            bool axis_is_sliced[{num_dims}];

            // Initialize: non-sliced axes have identity mapping
            for (int i = 0; i < {num_dims}; i++) {{
                slice_start[i] = 0;
                slice_step[i] = 1;
                axis_is_sliced[i] = false;
            }}

            // Fill in slice parameters for sliced axes
            {"// Single slice case - parameters may be scalars after simplification" if num_slices == 1 else ""}
            for (int slice_i = 0; slice_i < {num_slices}; slice_i++) {{
                int axis = slice_i;  // Default: axes = [0, 1, ..., num_slices-1]
                {"if (__axes) axis = (int)__axes;" if has_axes and num_slices == 1 else ""}
                {"if (__axes) axis = (int)__axes[slice_i];" if has_axes and num_slices > 1 else ""}

                long long start = (long long){("__starts" if num_slices == 1 else "__starts[slice_i]")};
                long long end = (long long){("__ends" if num_slices == 1 else "__ends[slice_i]")};
                long long step = 1;
                {"if (__steps) step = (long long)__steps;" if has_steps and num_slices == 1 else ""}
                {"if (__steps) step = (long long)__steps[slice_i];" if has_steps and num_slices > 1 else ""}

                // Handle negative indices
                if (start < 0) start += in_shape[axis];
                if (end < 0) end += in_shape[axis];

                slice_start[axis] = start;
                slice_step[axis] = step;
                axis_is_sliced[axis] = true;
            }}

            // Process each output element
            long long out_total = 1;
            for (int i = 0; i < {num_dims}; i++) {{
                out_total *= out_shape[i];
            }}

            for (long long out_idx = 0; out_idx < out_total; out_idx++) {{
                // Convert flat output index to multi-dimensional coordinates
                long long out_coords[{num_dims}];
                long long tmp = out_idx;
                for (int i = 0; i < {num_dims}; i++) {{
                    out_coords[i] = (tmp / out_strides[i]) % out_shape[i];
                }}

                // Map output coordinates to input coordinates
                long long in_idx = 0;
                for (int i = 0; i < {num_dims}; i++) {{
                    long long in_coord = slice_start[i] + out_coords[i] * slice_step[i];
                    in_idx += in_coord * in_strides[i];
                }}

                // Copy data
                __output[out_idx] = __data[in_idx];
            }}
            """

            # Create tasklet
            tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                         inputs=tasklet_inputs,
                                         outputs={"__output": dace.pointer(odesc.dtype)},
                                         code=code,
                                         language=dace.Language.CPP)

            # Connect inputs
            data_read = nstate.add_read("data")
            starts_read = nstate.add_read("starts")
            ends_read = nstate.add_read("ends")
            output_write = nstate.add_write("output")

            nstate.add_edge(data_read, None, tasklet, "__data", dace.Memlet.from_array("data", idesc))
            nstate.add_edge(starts_read, None, tasklet, "__starts", dace.Memlet.from_array("starts", starts_desc))
            nstate.add_edge(ends_read, None, tasklet, "__ends", dace.Memlet.from_array("ends", ends_desc))

            if has_axes:
                axes_read = nstate.add_read("axes")
                nstate.add_edge(axes_read, None, tasklet, "__axes", dace.Memlet.from_array("axes", axes_desc))
            if has_steps:
                steps_read = nstate.add_read("steps")
                nstate.add_edge(steps_read, None, tasklet, "__steps", dace.Memlet.from_array("steps", steps_desc))

            nstate.add_edge(tasklet, "__output", output_write, None, dace.Memlet.from_array("output", odesc))

            return nsdfg


# ==============================================================================
# Split Operations
# ==============================================================================


@op_implementation(op="Split", name="pure")
class SplitPure(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        from dace.transformation.onnx.replacement import onnx_constant_or_none

        # Check if we have either split input or num_outputs attribute
        has_split_input = len(list(state.in_edges_by_connector(node, "split"))) > 0
        has_num_outputs = hasattr(node, 'num_outputs')

        if not (has_split_input or has_num_outputs):
            return False

        # If split input is provided, it must be a constant (if we can check)
        if has_split_input:
            split_node = next(state.in_edges_by_connector(node, "split")).src
            try:
                if not onnx_constant_or_none(sdfg, split_node):
                    return False
            except AttributeError:
                # No _parent_onnx_model - can't verify if constant, but allow anyway
                pass

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        from dace.transformation.onnx.replacement import onnx_constant_or_none

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
            try:
                split_sizes = onnx_constant_or_none(sdfg, split_node)
            except AttributeError:
                # No _parent_onnx_model - try to get from array descriptor initial value
                split_desc_orig = sdfg.arrays[split_node.data]
                if hasattr(split_desc_orig, 'start_offset') and split_desc_orig.start_offset is not None:
                    # Has initial value - this is a workaround for tests
                    split_sizes = None  # Will be handled dynamically
                else:
                    split_sizes = None

            if split_sizes is None:
                # For now, we require split sizes to be constant
                # In the future, this could be made dynamic
                raise ValueError("Split sizes must be constant. Use num_outputs attribute instead.")

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
            raise ValueError(
                f"Sum of split sizes ({sum(split_sizes)}) must equal dimension size ({idesc.shape[split_dim]})")

        offset = 0
        for i, odim in enumerate(split_sizes):
            # Set up new node shape and memlet
            new_shape = list(idesc.shape)
            new_shape[split_dim] = odim
            rng = subsets.Range([(0, s - 1, 1) if j != split_dim else (offset, offset + odim - 1, 1)
                                 for j, s in enumerate(new_shape)])
            offset += odim

            # Set up data descriptor
            oname = f"outputs__{i}"
            odesc = copy.deepcopy(out_desc_with_name(node, state, sdfg, oname))
            odesc.transient = False
            nsdfg.add_datadesc(oname, odesc)
            wnode = nstate.add_write(oname)

            # Perform copy (view)
            nstate.add_nedge(rnode, wnode,
                             dace.Memlet(data="input", subset=rng, other_subset=subsets.Range.from_array(odesc)))

        return nsdfg


# ==============================================================================
# Shape Query Operations
# ==============================================================================


@op_implementation(op="Shape", name="pure")
class PureShape(ONNXForward):

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        data_desc = in_desc_with_name(node, state, sdfg, "data")

        try:
            np.array(data_desc.shape, np.int64)
        except Exception:
            # this happens if the shape is symbolic, for example
            return False

        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

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
            tasklet = nstate.add_tasklet("write_shape", {}, {'shape_scalar': dace.int64}, f"shape_scalar = {v}")
            nstate.add_edge(tasklet, "shape_scalar", s, None, dace.Memlet("shape[{}]".format(i)))

        return nsdfg


# ==============================================================================
# Gather Operations
# ==============================================================================


@op_implementation(op="Gather", name="pure")
class PureGather(ONNXForward):

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # To understand this operator, read the docs for np.take.
        # The ONNX docs are not easy to understand (and are incorrect in opset 11)

        nsdfg, nstate, _, _ = empty_sdfg_for_node(sdfg, state, node, add_access_nodes=False)
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
        if isinstance(idx_desc, dace.data.Scalar):
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
            indices_idx_str += '[' + ', '.join(map(fst, map_ranges_indices)) + ']'
        else:
            indices_idx_str += '[0]'

        tasklet, me, mx = nstate.add_mapped_tasklet(node.label + "_tasklet",
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


# ==============================================================================
# Utility Operations
# ==============================================================================


@python_pure_op_implementation
def Where(condition, X, Y, output):
    output[:] = np.where(condition, X, Y)


@python_pure_op_implementation
def Identity(input, output):
    output[:] = input


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


# ============================================================================
# Constant Generation Operations
# ============================================================================


@op_implementation(op="Constant", name="pure")
class PureConstant(ONNXForward):
    """
    Pure implementation of ONNX Constant operator.

    Generates a constant tensor with a given value.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Constant nodes should have a value attribute
        return hasattr(node, 'value') or hasattr(node, 'value_float') or hasattr(node, 'value_int')

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        output_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "output"))
        nsdfg.add_datadesc("output", output_desc)
        nsdfg.arrays["output"].transient = False

        output_write = nstate.add_write("output")

        # Get the constant value
        if hasattr(node, 'value'):
            const_value = node.value
        elif hasattr(node, 'value_float'):
            const_value = node.value_float
        elif hasattr(node, 'value_int'):
            const_value = node.value_int
        else:
            raise ValueError("Constant node must have value, value_float, or value_int attribute")

        # Convert to numpy array if needed
        if not isinstance(const_value, np.ndarray):
            const_value = np.array(const_value, dtype=output_desc.dtype.as_numpy_dtype())

        # Flatten the constant value for iteration
        flat_value = const_value.flatten()

        # Create tasklets to write each element
        for i, val in enumerate(flat_value):
            tasklet = nstate.add_tasklet(f"write_const_{i}", {}, {"out": output_desc.dtype}, f"out = {val}")
            nstate.add_edge(tasklet, "out", output_write, None, dace.Memlet(f"output[{i}]"))

        return nsdfg


@op_implementation(op="ConstantOfShape", name="pure")
class PureConstantOfShape(ONNXForward):
    """
    Pure implementation of ONNX ConstantOfShape operator.

    Generates a tensor with given shape filled with a constant value.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Always applicable - handles both constant and runtime shape inputs
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get the shape value (if constant)
        shape_node = next(state.in_edges_by_connector(node, "input")).src
        shape_val = onnx_constant_or_none(sdfg, shape_node)

        # Remove shape input if it's constant
        if shape_val is not None:
            constant_folding.remove_node_and_computation(sdfg, state, node, "input")

        # Get fill value (default is 0)
        fill_value = 0
        if hasattr(node, 'value') and node.value is not None:
            # node.value is already a numpy array (converted during ONNX import)
            fill_value = float(node.value.flat[0])  # Extract scalar from array and convert to Python type

        output_desc = out_desc_with_name(node, state, sdfg, "output")

        # Create a simple nested SDFG that fills the output with the constant value
        nsdfg = SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add output array to nested SDFG
        nsdfg.add_array('output', output_desc.shape, output_desc.dtype)
        nsdfg.arrays['output'].transient = False

        # If shape is constant, we can use static maps
        if shape_val is not None:
            # Create a map and tasklet to fill the output
            map_ranges = {f'__i{i}': f'0:{s}' for i, s in enumerate(output_desc.shape)}
            output_indices = ', '.join(f'__i{i}' for i in range(len(output_desc.shape)))

            # Use add_mapped_tasklet for simpler construction
            tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(
                name='fill',
                map_ranges=map_ranges,
                inputs={},
                code=f'__out = {output_desc.dtype.ctype}({fill_value})',
                outputs={'__out': dace.Memlet(f'output[{output_indices}]')},
                external_edges=True,
                output_nodes={'output': nstate.add_write('output')})
        else:
            # Shape is not constant, need to handle it at runtime
            # Add input shape array to nested SDFG
            input_desc = in_desc_with_name(node, state, sdfg, "input")
            nsdfg.add_datadesc('input', copy.deepcopy(input_desc))
            nsdfg.arrays['input'].transient = False

            # Since the shape is dynamic, we can't use static maps
            # Instead, we'll create a tasklet that fills the entire output tensor
            # using nested loops based on the runtime shape values

            # Get the number of dimensions from the output descriptor
            ndims = len(output_desc.shape)

            # Create a single tasklet that fills the entire output
            # Generate C++ code for nested loops based on the shape array
            code_lines = []

            # Read shape values into local variables
            # Handle both scalar input (when ndims=1) and array input
            if ndims == 1 and prod(input_desc.shape) == 1:
                # Special case: single dimension, input might be passed as scalar
                code_lines.append(f'const auto dim0 = __input;')
            else:
                for i in range(ndims):
                    code_lines.append(f'const auto dim{i} = __input[{i}];')

            # Generate nested loops
            indent = ''
            for i in range(ndims):
                code_lines.append(f'{indent}for (auto i{i} = 0; i{i} < dim{i}; ++i{i}) {{')
                indent += '    '

            # Fill the output
            if ndims == 1:
                code_lines.append(f'{indent}__output[i0] = {fill_value};')
            else:
                output_indices = ', '.join(f'i{i}' for i in range(ndims))
                code_lines.append(f'{indent}__output[{output_indices}] = {fill_value};')

            # Close loops
            for i in range(ndims):
                indent = indent[:-4]
                code_lines.append(f'{indent}}}')

            tasklet_code = '\n'.join(code_lines)

            # Create access nodes
            input_node = nstate.add_read('input')
            output_node = nstate.add_write('output')

            # Create tasklet
            tasklet = nstate.add_tasklet(name='fill',
                                         inputs={'__input'},
                                         outputs={'__output'},
                                         code=tasklet_code,
                                         language=dace.Language.CPP)

            # Connect edges
            # For scalar input (single element), use from_array which will optimize to scalar
            nstate.add_edge(input_node, None, tasklet, '__input', dace.Memlet.from_array('input', input_desc))
            nstate.add_edge(tasklet, '__output', output_node, None, dace.Memlet.from_array('output', output_desc))

        return nsdfg


# ============================================================================
# Range Operation
# ============================================================================


@op_implementation(op="Range", name="pure")
class PureRange(ONNXForward):
    """
    Pure implementation of ONNX Range operator.

    Generates a sequence of numbers from start to limit with step increment.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Check if all inputs are constants
        for conn in ["start", "limit", "delta"]:
            try:
                input_node = next(state.in_edges_by_connector(node, conn)).src
                if onnx_constant_or_none(sdfg, input_node) is None:
                    return False
            except (StopIteration, ValueError, AttributeError):
                # AttributeError occurs when _parent_onnx_model is not available
                return False
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get constant values
        start_node = next(state.in_edges_by_connector(node, "start")).src
        limit_node = next(state.in_edges_by_connector(node, "limit")).src
        delta_node = next(state.in_edges_by_connector(node, "delta")).src

        start_val = onnx_constant_or_none(sdfg, start_node)
        limit_val = onnx_constant_or_none(sdfg, limit_node)
        delta_val = onnx_constant_or_none(sdfg, delta_node)

        # Remove constant inputs
        constant_folding.remove_node_and_computation(sdfg, state, node, "start")
        constant_folding.remove_node_and_computation(sdfg, state, node, "limit")
        constant_folding.remove_node_and_computation(sdfg, state, node, "delta")

        output_desc = out_desc_with_name(node, state, sdfg, "output")

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        output_copy = copy.deepcopy(output_desc)
        output_copy.transient = False
        nsdfg.add_datadesc("output", output_copy)

        # Calculate number of elements
        num_elements = int(np.ceil((limit_val - start_val) / delta_val))

        # Create mapped tasklet to generate range
        map_ranges = {"i": f"0:{num_elements}"}

        # Cast values to the appropriate type
        dtype_str = output_desc.dtype.ctype
        tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(
            name=node.label + "_tasklet",
            map_ranges=map_ranges,
            inputs={},
            code=f"__out = {dtype_str}({start_val}) + {dtype_str}(i) * {dtype_str}({delta_val})",
            outputs={"__out": dace.Memlet("output[i]")},
            external_edges=True,
            output_nodes={"output": nstate.add_write("output")})

        return nsdfg


# ============================================================================
# GatherND Operation
# ============================================================================


@op_implementation(op="GatherND", name="pure")
class PureGatherND(ONNXForward):
    """
    Pure implementation of ONNX GatherND operator.

    Gathers values from data tensor using multi-dimensional indices.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg, nstate, input_nodes, output_nodes = empty_sdfg_for_node(sdfg, state, node, add_access_nodes=True)

        data_desc = in_desc_with_name(node, state, sdfg, "data")
        indices_desc = in_desc_with_name(node, state, sdfg, "indices")
        output_desc = out_desc_with_name(node, state, sdfg, "output")

        batch_dims = getattr(node, 'batch_dims', 0)

        # Get index depth (last dimension of indices)
        indices_shape = indices_desc.shape
        index_depth = indices_shape[-1]

        # Calculate output shape for mapping
        output_shape = output_desc.shape

        # Generate code for GatherND operation
        data_idx_expr = " + ".join(
            [f"indices_flat[i * {index_depth} + {j}] * {data_desc.strides[j]}" for j in range(index_depth)])

        num_indices = int(np.prod(indices_shape[:-1]))

        code = f"""
        for (int i = 0; i < {num_indices}; i++) {{
            long long data_idx = {data_idx_expr};
            __output[i] = __data[data_idx];
        }}
        """

        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs={
                                         "__data": dace.pointer(data_desc.dtype),
                                         "indices_flat": dace.pointer(indices_desc.dtype)
                                     },
                                     outputs={"__output": dace.pointer(output_desc.dtype)},
                                     language=dace.Language.CPP,
                                     code=code)

        nstate.add_edge(input_nodes["data"], None, tasklet, "__data", dace.Memlet.from_array("data", data_desc))
        nstate.add_edge(input_nodes["indices"], None, tasklet, "indices_flat",
                        dace.Memlet.from_array("indices", indices_desc))
        nstate.add_edge(tasklet, "__output", output_nodes["output"], None,
                        dace.Memlet.from_array("output", output_desc))

        return nsdfg


# ============================================================================
# ScatterND Operation
# ============================================================================


@op_implementation(op="ScatterND", name="pure")
class PureScatterND(ONNXForward):
    """
    Pure implementation of ONNX ScatterND operator.

    Scatters values into a tensor at locations specified by indices.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg, nstate, input_nodes, output_nodes = empty_sdfg_for_node(sdfg, state, node, add_access_nodes=True)

        data_desc = in_desc_with_name(node, state, sdfg, "data")
        indices_desc = in_desc_with_name(node, state, sdfg, "indices")
        updates_desc = in_desc_with_name(node, state, sdfg, "updates")
        output_desc = out_desc_with_name(node, state, sdfg, "output")

        # Get index depth (last dimension of indices)
        indices_shape = indices_desc.shape
        index_depth = indices_shape[-1]

        num_updates = int(np.prod(indices_shape[:-1]))
        data_size = int(np.prod(data_desc.shape))

        # Generate code for ScatterND operation
        code = f"""
        // Copy input data to output
        for (int i = 0; i < {data_size}; i++) {{
            __output[i] = __data[i];
        }}

        // Scatter updates
        for (int i = 0; i < {num_updates}; i++) {{
            long long idx = 0;
            """ + "\n            ".join(
            [f"idx += __indices[i * {index_depth} + {j}] * {data_desc.strides[j]};" for j in range(index_depth)]) + f"""
            __output[idx] = __updates[i];
        }}
        """

        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs={
                                         "__data": dace.pointer(data_desc.dtype),
                                         "__indices": dace.pointer(indices_desc.dtype),
                                         "__updates": dace.pointer(updates_desc.dtype)
                                     },
                                     outputs={"__output": dace.pointer(output_desc.dtype)},
                                     language=dace.Language.CPP,
                                     code=code)

        nstate.add_edge(input_nodes["data"], None, tasklet, "__data", dace.Memlet.from_array("data", data_desc))
        nstate.add_edge(input_nodes["indices"], None, tasklet, "__indices",
                        dace.Memlet.from_array("indices", indices_desc))
        nstate.add_edge(input_nodes["updates"], None, tasklet, "__updates",
                        dace.Memlet.from_array("updates", updates_desc))
        nstate.add_edge(tasklet, "__output", output_nodes["output"], None,
                        dace.Memlet.from_array("output", output_desc))

        return nsdfg


# ============================================================================
# TopK Operation
# ============================================================================


@op_implementation(op="TopK", name="pure")
class PureTopK(ONNXForward):
    """
    Pure implementation of ONNX TopK operator.

    Finds the top K largest (or smallest) elements along a given axis.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Check if K is constant
        try:
            k_node = next(state.in_edges_by_connector(node, "K")).src
            return onnx_constant_or_none(sdfg, k_node) is not None
        except (StopIteration, ValueError, AttributeError):
            # AttributeError occurs when _parent_onnx_model is not available
            return False

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get K value
        k_node = next(state.in_edges_by_connector(node, "K")).src
        k_val = int(onnx_constant_or_none(sdfg, k_node))

        # Remove K input since it's constant
        constant_folding.remove_node_and_computation(sdfg, state, node, "K")

        input_desc = in_desc_with_name(node, state, sdfg, "X")
        values_desc = out_desc_with_name(node, state, sdfg, "Values")
        indices_desc = out_desc_with_name(node, state, sdfg, "Indices")

        axis = getattr(node, 'axis', -1)
        largest = getattr(node, 'largest', 1)
        sorted_output = getattr(node, 'sorted', 1)

        # Normalize axis
        if axis < 0:
            axis = len(input_desc.shape) + axis

        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        nsdfg.add_datadesc("X", copy.deepcopy(input_desc))
        nsdfg.add_datadesc("Values", copy.deepcopy(values_desc))
        nsdfg.add_datadesc("Indices", copy.deepcopy(indices_desc))
        nsdfg.arrays["X"].transient = False
        nsdfg.arrays["Values"].transient = False
        nsdfg.arrays["Indices"].transient = False

        input_read = nstate.add_read("X")
        values_write = nstate.add_write("Values")
        indices_write = nstate.add_write("Indices")

        # Generate code for partial sort along axis
        input_shape = input_desc.shape
        axis_size = input_shape[axis]

        # Calculate strides for iteration
        outer_size = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        inner_size = int(np.prod(input_shape[axis + 1:])) if axis < len(input_shape) - 1 else 1

        # Simple bubble sort for top K (not optimal but works for pure SDFG)
        # For production, we'd use a more efficient algorithm
        comparison_op = ">" if largest else "<"

        code = f"""
        for (int outer = 0; outer < {outer_size}; outer++) {{
            for (int inner = 0; inner < {inner_size}; inner++) {{
                // Create temporary arrays for sorting
                {input_desc.dtype.ctype} temp_vals[{axis_size}];
                long long temp_idxs[{axis_size}];

                // Copy slice to temporary array
                for (int i = 0; i < {axis_size}; i++) {{
                    int idx = outer * {axis_size * inner_size} + i * {inner_size} + inner;
                    temp_vals[i] = __X[idx];
                    temp_idxs[i] = i;
                }}

                // Partial sort to find top K
                for (int i = 0; i < {k_val}; i++) {{
                    for (int j = i + 1; j < {axis_size}; j++) {{
                        if (temp_vals[j] {comparison_op} temp_vals[i]) {{
                            // Swap values
                            {input_desc.dtype.ctype} tmp_v = temp_vals[i];
                            temp_vals[i] = temp_vals[j];
                            temp_vals[j] = tmp_v;

                            // Swap indices
                            long long tmp_i = temp_idxs[i];
                            temp_idxs[i] = temp_idxs[j];
                            temp_idxs[j] = tmp_i;
                        }}
                    }}
                }}

                // Write top K results
                for (int i = 0; i < {k_val}; i++) {{
                    int out_idx = outer * {k_val * inner_size} + i * {inner_size} + inner;
                    __Values[out_idx] = temp_vals[i];
                    __Indices[out_idx] = temp_idxs[i];
                }}
            }}
        }}
        """

        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs={"__X": dace.pointer(input_desc.dtype)},
                                     outputs={
                                         "__Values": dace.pointer(values_desc.dtype),
                                         "__Indices": dace.pointer(indices_desc.dtype)
                                     },
                                     language=dace.Language.CPP,
                                     code=code)

        nstate.add_edge(input_read, None, tasklet, "__X", dace.Memlet.from_array("X", input_desc))
        nstate.add_edge(tasklet, "__Values", values_write, None, dace.Memlet.from_array("Values", values_desc))
        nstate.add_edge(tasklet, "__Indices", indices_write, None, dace.Memlet.from_array("Indices", indices_desc))

        return nsdfg
