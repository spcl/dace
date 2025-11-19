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
from dace.util import in_desc_with_name, out_desc_with_name

from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.op_implementations.utils import (empty_sdfg_for_node, op_implementation, program_for_node,
                                                          python_pure_op_implementation)
from dace.libraries.onnx.op_implementations.common import broadcast_indices, create_memlet_str
from dace.transformation.onnx import constant_folding
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.libraries.onnx import converters
from dace.transformation.onnx.replacement import onnx_constant_or_none
import onnx
# ==============================================================================
# Pad Operations
# ==============================================================================


@op_implementation(op="Pad", name="pure")
class PurePad(ONNXForward):
    """
    Pure implementation of ONNX Pad operator.

    Pads a tensor with a constant value along specified dimensions.
    The ONNX Pad operator takes:
        - data: input tensor
        - pads: padding values in ONNX format [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                Length is 2 * rank of data
        - constant_value (optional): value to pad with (default 0)
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:

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
        map_ranges = [(f"i{d}", f"0:{output_shape[d]}") for d in range(ndim)]
        output_indices = ', '.join([f'i{d}' for d in range(ndim)])

        # Generate tasklet code using Python syntax
        # ONNX pads format: [dim0_begin, dim1_begin, ..., dimN_begin, dim0_end, dim1_end, ..., dimN_end]
        # We only need the begin pads to compute input coordinates
        code_lines = []
        code_lines.append(f"# Compute input indices from output indices using begin pads")
        for d in range(ndim):
            # pads[d] contains the begin padding for dimension d
            code_lines.append(f"input_i{d} = i{d} - int(__pads[{d}])")

        # Check if we're in the valid input region (not in padding)
        condition_parts = []
        for d in range(ndim):
            condition_parts.append(f"(input_i{d} >= 0 and input_i{d} < {input_shape[d]})")
        condition = " and ".join(condition_parts)

        code_lines.append(f"if {condition}:")
        input_indices = ', '.join([f'input_i{d}' for d in range(ndim)])
        code_lines.append(f"    __out = __data[{input_indices}]")
        code_lines.append("else:")
        if has_constant_value:
            # constant_value is passed as a scalar, not indexed
            code_lines.append("    __out = __constant_value")
        else:
            code_lines.append("    __out = 0")

        tasklet_code = '\n'.join(code_lines)

        pads_memlet_str = f"pads[0:{pads_desc.shape[0]}]"
        data_memlet_str = "data[" + ", ".join([f"0:{input_shape[d]}" for d in range(ndim)]) + "]"

        # Build tasklet inputs with proper memlet strings
        tasklet_inputs = {"__data": dace.Memlet(data_memlet_str), "__pads": dace.Memlet(pads_memlet_str)}
        if has_constant_value:
            # constant_value is read as a scalar (single element)
            tasklet_inputs["__constant_value"] = dace.Memlet("constant_value[0]")

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

        # Generate broadcast-aware indexing
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
    Slice implementation that requires constant inputs for slicing parameters.

    Handles ONNX Slice operator with:
    - Required inputs: data, starts, ends (must be constant)
    - Optional inputs: axes (must be constant if provided), steps (must be constant if provided)

    Uses efficient memlet slicing for constant parameters.
    """

    @staticmethod
    def _get_constant(conn: str, node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG):
        """Try to get constant value for an input connector."""
        edges = list(state.in_edges_by_connector(node, conn))
        if not edges:
            return None

        srcnode = edges[0].src

        if hasattr(sdfg, '_parent_onnx_model'):
            return onnx_constant_or_none(sdfg, srcnode)
        return None

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # Check that all non-data inputs are constant
        # starts and ends are required
        starts_const = PureSlice._get_constant('starts', node, state, sdfg)
        ends_const = PureSlice._get_constant('ends', node, state, sdfg)

        if starts_const is None or ends_const is None:
            return False

        # axes and steps are optional, but if provided must be constant
        if list(state.in_edges_by_connector(node, "axes")):
            axes_const = PureSlice._get_constant('axes', node, state, sdfg)
            if axes_const is None:
                return False

        if list(state.in_edges_by_connector(node, "steps")):
            steps_const = PureSlice._get_constant('steps', node, state, sdfg)
            if steps_const is None:
                return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get constant values for all inputs (already validated in forward_can_be_applied)
        axes_const = PureSlice._get_constant('axes', node, state, sdfg)
        ends_const = PureSlice._get_constant('ends', node, state, sdfg)
        starts_const = PureSlice._get_constant('starts', node, state, sdfg)
        steps_const = PureSlice._get_constant('steps', node, state, sdfg)

        # Verify that required inputs are constant (should have been checked in forward_can_be_applied)
        if starts_const is None or ends_const is None:
            raise ValueError("Slice operation requires constant starts and ends parameters")

        # Use optimized memlet slicing for constant parameters
        # Remove constant inputs
        if axes_const is not None:
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

        # If axes_const is None, use default axes [0, 1, ..., len(starts)-1]
        if axes_const is None:
            # Default axes for when not specified
            if isinstance(starts_const, (list, tuple)):
                axes_const = list(range(len(starts_const)))
            else:
                axes_const = [0]

        # Normalize to lists
        if not isinstance(axes_const, (tuple, list)):
            axes_const = [int(axes_const)]
        if not isinstance(ends_const, (tuple, list)):
            ends_const = [int(ends_const)]
        if not isinstance(starts_const, (tuple, list)):
            starts_const = [int(starts_const)]

        if steps_const is not None:
            if not isinstance(steps_const, (tuple, list)):
                steps_const = [int(steps_const)]
        else:
            steps_const = [1] * len(starts_const)

        # Set up slicing memlet
        rng = [(0, int(s) - 1, 1) for s in idesc.shape]
        for i, axis in enumerate(axes_const):
            axis = int(axis)
            start = int(starts_const[i])
            end = int(ends_const[i])
            step = int(steps_const[i]) if i < len(steps_const) else 1

            s = int(idesc.shape[axis])

            # Handle negative steps separately as they have different semantics
            if step < 0:
                # For negative steps, normalize start
                if start < 0:
                    start = s + start
                # Clamp start to valid range
                if start >= s:
                    start = s - 1

                # For negative end: convert to positive index
                if end < 0:
                    end = s + end
                    # If end becomes negative, it means "go to/past beginning"
                    # DaCe Range with negative step needs end to be < start
                    # If we want to include index 0, end should be -1
                    if end < 0:
                        end = -1

                # For negative steps, if end >= start, the slice is empty (wrong direction)
                # This shouldn't happen
                assert end < start

                # DaCe Range with negative step: (start, end, step) where start > end
                # The range includes indices from start down to (but NOT including) end
                rng[axis] = (start, end, step)
            else:
                # Positive steps: normalize negative indices
                if start < 0:
                    start = s + start
                if end < 0:
                    end = s + end

                # Handle out of bounds for positive steps
                if end > s:
                    end = s
                # For positive steps, DaCe Range uses inclusive end, so end - 1
                rng[axis] = (start, end - 1, step)

        sbs = subsets.Range(rng)
        osbs = subsets.Range.from_array(odesc)

        rnode = nstate.add_read("data")
        wnode = nstate.add_write("output")

        nstate.add_nedge(rnode, wnode, dace.Memlet(data="data", subset=sbs, other_subset=osbs))

        return nsdfg


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

        # If split input is provided, it must be a constant
        if has_split_input:
            split_node = next(state.in_edges_by_connector(node, "split")).src
            if hasattr(sdfg, '_parent_onnx_model'):
                if not onnx_constant_or_none(sdfg, split_node):
                    return False

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

            if hasattr(sdfg, '_parent_onnx_model'):
                split_sizes = onnx_constant_or_none(sdfg, split_node)
            else:
                split_sizes = None

            if split_sizes is None:
                raise ValueError("Split sizes must be constant. Use num_outputs attribute instead.")

            # Add split input as a data descriptor
            split_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "split"))
            split_desc.transient = False
            nsdfg.add_datadesc("split", split_desc)
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

        # Check if shape contains symbolic values
        if any(dace.symbolic.issymbolic(dim) for dim in data_desc.shape):
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
        data_desc = in_desc_with_name(node, state, sdfg, "data")
        data_shape = data_desc.shape

        # Handle degenerate case: scalar data (size 1)
        # In this case, just copy the value without indexing
        is_scalar_data = (isinstance(data_desc, dace.data.Scalar)
                          or (hasattr(data_desc, 'total_size') and data_desc.total_size == 1))
        is_scalar_output = (isinstance(out_desc, dace.data.Scalar)
                            or (hasattr(out_desc, 'total_size') and out_desc.total_size == 1))

        if is_scalar_data and is_scalar_output:
            # Both data and output are scalars - just copy
            tasklet = nstate.add_tasklet(node.label + "_tasklet", {"__data": data_desc.dtype},
                                         {"__output": out_desc.dtype}, "__output = __data")
            data_read = nstate.add_read("data")
            output_write = nstate.add_write("output")
            nstate.add_edge(data_read, None, tasklet, "__data", dace.Memlet.from_array("data", data_desc))
            nstate.add_edge(tasklet, "__output", output_write, None, dace.Memlet.from_array("output", out_desc))
            return nsdfg

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
        tasklet.in_connectors["__data"] = dace.pointer(data_desc.dtype)

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
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get fill value (default is 0)
        fill_value = 0
        if hasattr(node, 'value') and node.value is not None:
            fill_value = float(node.value.flat[0])

        output_desc = out_desc_with_name(node, state, sdfg, "output")
        ndims = len(output_desc.shape)

        # Create Python program based on number of dimensions
        if ndims == 1:

            def prog(output):
                for i in range(output.shape[0]):
                    output[i] = fill_value
        elif ndims == 2:

            def prog(output):
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        output[i, j] = fill_value
        elif ndims == 3:

            def prog(output):
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        for k in range(output.shape[2]):
                            output[i, j, k] = fill_value
        elif ndims == 4:

            def prog(output):
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        for k in range(output.shape[2]):
                            for l in range(output.shape[3]):
                                output[i, j, k, l] = fill_value
        else:
            raise NotImplementedError(f"ConstantOfShape not implemented for {ndims}D arrays")

        # Remove the shape input if it's constant (handled by program_for_node)
        shape_node = next(state.in_edges_by_connector(node, "input")).src
        shape_val = onnx_constant_or_none(sdfg, shape_node)
        if shape_val is not None:
            constant_folding.remove_node_and_computation(sdfg, state, node, "input")

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Range", name="pure")
class PureRange(ONNXForward):
    """
    Pure implementation of ONNX Range operator.

    Generates a sequence of numbers from start to limit with step increment.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        # Check if all inputs are constants
        if not hasattr(sdfg, '_parent_onnx_model'):
            return False

        for conn in ["start", "limit", "delta"]:
            edges = list(state.in_edges_by_connector(node, conn))
            if not edges:
                return False
            input_node = edges[0].src
            if onnx_constant_or_none(sdfg, input_node) is None:
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
        indices_desc = in_desc_with_name(node, state, sdfg, "indices")

        # Get index depth (last dimension of indices)
        indices_shape = indices_desc.shape
        index_depth = int(indices_shape[-1])

        # Create Python program based on index_depth
        if index_depth == 1:

            def prog(data, indices, output):
                for i in range(indices.shape[0]):
                    idx0 = indices[i, 0]
                    output[i] = data[idx0]
        elif index_depth == 2:
            if len(indices_shape) == 2:

                def prog(data, indices, output):
                    for i in range(indices.shape[0]):
                        idx0 = indices[i, 0]
                        idx1 = indices[i, 1]
                        output[i] = data[idx0, idx1]
            else:  # 3D indices

                def prog(data, indices, output):
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            idx0 = indices[i, j, 0]
                            idx1 = indices[i, j, 1]
                            output[i, j] = data[idx0, idx1]
        elif index_depth == 3:
            if len(indices_shape) == 2:

                def prog(data, indices, output):
                    for i in range(indices.shape[0]):
                        idx0 = indices[i, 0]
                        idx1 = indices[i, 1]
                        idx2 = indices[i, 2]
                        output[i] = data[idx0, idx1, idx2]
            else:  # More dimensions

                def prog(data, indices, output):
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            idx0 = indices[i, j, 0]
                            idx1 = indices[i, j, 1]
                            idx2 = indices[i, j, 2]
                            output[i, j] = data[idx0, idx1, idx2]
        else:
            raise NotImplementedError(f"GatherND not implemented for index_depth={index_depth}")

        return program_for_node(prog, sdfg, state, node)


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
        indices_desc = in_desc_with_name(node, state, sdfg, "indices")

        # Get index depth (last dimension of indices)
        indices_shape = indices_desc.shape
        index_depth = int(indices_shape[-1])

        # Create Python program based on index_depth
        if index_depth == 1:

            def prog(data, indices, updates, output):
                output[:] = data[:]
                for i in range(indices.shape[0]):
                    idx0 = indices[i, 0]
                    output[idx0] = updates[i]
        elif index_depth == 2:
            if len(indices_shape) == 2:

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        idx0 = indices[i, 0]
                        idx1 = indices[i, 1]
                        output[idx0, idx1] = updates[i]
            else:  # 3D indices

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            idx0 = indices[i, j, 0]
                            idx1 = indices[i, j, 1]
                            output[idx0, idx1] = updates[i, j]
        elif index_depth == 3:
            if len(indices_shape) == 2:

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        idx0 = indices[i, 0]
                        idx1 = indices[i, 1]
                        idx2 = indices[i, 2]
                        output[idx0, idx1, idx2] = updates[i]
            else:  # More dimensions

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            idx0 = indices[i, j, 0]
                            idx1 = indices[i, j, 1]
                            idx2 = indices[i, j, 2]
                            output[idx0, idx1, idx2] = updates[i, j]
        else:
            raise NotImplementedError(f"ScatterND not implemented for index_depth={index_depth}")

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="GatherElements", name="pure")
class PureGatherElements(ONNXForward):
    """
    Pure implementation of ONNX GatherElements operator.

    GatherElements takes two inputs: data and indices. The output has the same shape as indices.
    For each element in indices at position [i,j,k,...]:
    - If axis=d, output[i,j,k,...] = data[i,...,indices[i,j,k,...], ...,k]
      (indices value replaces the d-th dimension)
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axis_val = node.axis if hasattr(node, 'axis') else 0
        data_desc = in_desc_with_name(node, state, sdfg, "data")
        indices_desc = in_desc_with_name(node, state, sdfg, "indices")
        output_desc = out_desc_with_name(node, state, sdfg, "output")

        # Normalize negative axis
        if axis_val < 0:
            axis_val += len(output_desc.shape)

        # Create numpy program that manually implements gather_elements logic
        # We need to iterate over each output element and gather from data
        ndim = len(output_desc.shape)

        if ndim == 1:

            def prog(data, indices, output):
                for i in range(output.shape[0]):
                    output[i] = data[indices[i]]
        elif ndim == 2:
            if axis_val == 0:

                def prog(data, indices, output):
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            output[i, j] = data[indices[i, j], j]
            else:  # axis == 1

                def prog(data, indices, output):
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            output[i, j] = data[i, indices[i, j]]
        elif ndim == 3:
            if axis_val == 0:

                def prog(data, indices, output):
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            for k in range(output.shape[2]):
                                output[i, j, k] = data[indices[i, j, k], j, k]
            elif axis_val == 1:

                def prog(data, indices, output):
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            for k in range(output.shape[2]):
                                output[i, j, k] = data[i, indices[i, j, k], k]
            else:  # axis == 2

                def prog(data, indices, output):
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            for k in range(output.shape[2]):
                                output[i, j, k] = data[i, j, indices[i, j, k]]
        else:
            raise NotImplementedError(f"GatherElements not implemented for {ndim}D arrays")

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ScatterElements", name="pure")
class PureScatterElements(ONNXForward):
    """
    Pure implementation of ONNX ScatterElements operator.

    ScatterElements takes three inputs: data, indices, and updates.
    The output has the same shape as data.
    For each element in indices/updates at position [i,j,k,...]:
    - If axis=d, output[i,...,indices[i,j,k,...], ...,k] = updates[i,j,k,...]
      (indices value replaces the d-th dimension)
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axis_val = node.axis if hasattr(node, 'axis') else 0
        indices_desc = in_desc_with_name(node, state, sdfg, "indices")
        output_desc = out_desc_with_name(node, state, sdfg, "output")

        # Normalize negative axis
        if axis_val < 0:
            axis_val += len(output_desc.shape)

        # Create Python program based on dimensionality
        ndim = len(indices_desc.shape)

        if ndim == 1:

            def prog(data, indices, updates, output):
                output[:] = data[:]
                for i in range(indices.shape[0]):
                    output[indices[i]] = updates[i]
        elif ndim == 2:
            if axis_val == 0:

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            output[indices[i, j], j] = updates[i, j]
            else:  # axis == 1

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            output[i, indices[i, j]] = updates[i, j]
        elif ndim == 3:
            if axis_val == 0:

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            for k in range(indices.shape[2]):
                                output[indices[i, j, k], j, k] = updates[i, j, k]
            elif axis_val == 1:

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            for k in range(indices.shape[2]):
                                output[i, indices[i, j, k], k] = updates[i, j, k]
            else:  # axis == 2

                def prog(data, indices, updates, output):
                    output[:] = data[:]
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            for k in range(indices.shape[2]):
                                output[i, j, indices[i, j, k]] = updates[i, j, k]
        else:
            raise NotImplementedError(f"ScatterElements not implemented for {ndim}D arrays")

        return program_for_node(prog, sdfg, state, node)


# ==============================================================================
# Triangular Matrix Operations
# ==============================================================================


@op_implementation(op="Trilu", name="pure")
class PureTrilu(ONNXForward):
    """
    Pure implementation of ONNX Trilu operator.

    Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower
    triangular part of the tensor(s). The other elements of the tensor are set to zero.

    Attributes:
        upper (int): Boolean. Indicates whether upper or lower part of matrix is
                     retained. Default is 1 (true - upper triangular).

    Inputs:
        input: Input tensor of rank >= 2
        k (optional): A 0-D tensor containing a single value corresponding to the
                     number diagonals above or below the main diagonal to exclude
                     or include. Default value is 0.
    """

    @staticmethod
    def forward_can_be_applied(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get the upper attribute (default is 1/True for upper triangular)
        upper = getattr(node, 'upper', 1)

        # Get descriptors
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")

        # Check if k input exists (optional diagonal offset)
        has_k = len(list(state.in_edges_by_connector(node, "k"))) > 0

        # Create a new SDFG for the expansion
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Add data descriptors to nested SDFG
        nsdfg.add_datadesc("input", copy.deepcopy(input_desc))
        nsdfg.add_datadesc("output", copy.deepcopy(output_desc))
        nsdfg.arrays["input"].transient = False
        nsdfg.arrays["output"].transient = False

        if has_k:
            k_desc = in_desc_with_name(node, state, sdfg, "k")
            nsdfg.add_datadesc("k", copy.deepcopy(k_desc))
            nsdfg.arrays["k"].transient = False

        # Get shape information
        input_shape = input_desc.shape
        ndim = len(input_shape)

        # The last two dimensions are the matrix dimensions
        # Earlier dimensions are batch dimensions
        if ndim < 2:
            raise ValueError(f"Trilu requires input rank >= 2, got {ndim}")

        # Create a map over all dimensions of the output tensor
        map_ranges = [(f"i{d}", f"0:{input_shape[d]}") for d in range(ndim)]
        output_indices = ', '.join([f'i{d}' for d in range(ndim)])

        # Generate tasklet code
        # For upper triangular (upper=1): keep elements where col >= row + k
        # For lower triangular (upper=0): keep elements where col <= row + k
        # The last two dimensions are the row and column indices
        row_idx = f"i{ndim-2}"  # Second to last dimension is row
        col_idx = f"i{ndim-1}"  # Last dimension is column

        if has_k:
            # k is a scalar input, extract first element
            if upper:
                condition = f"{col_idx} >= {row_idx} + __k"
            else:
                condition = f"{col_idx} <= {row_idx} + __k"
        else:
            # Default k = 0
            if upper:
                condition = f"{col_idx} >= {row_idx}"
            else:
                condition = f"{col_idx} <= {row_idx}"

        input_indices = ', '.join([f'i{d}' for d in range(ndim)])
        tasklet_code = f"""
if {condition}:
    __out = __input[{input_indices}]
else:
    __out = 0
"""

        # Build input memlet
        input_memlet_str = "input[" + ", ".join([f"0:{input_shape[d]}" for d in range(ndim)]) + "]"

        # Build tasklet inputs
        tasklet_inputs = {"__input": dace.Memlet(input_memlet_str)}
        if has_k:
            tasklet_inputs["__k"] = dace.Memlet("k[0]")

        # Create mapped tasklet
        nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                  map_ranges=map_ranges,
                                  inputs=tasklet_inputs,
                                  code=tasklet_code,
                                  outputs={"__out": dace.Memlet(f"output[{output_indices}]")},
                                  external_edges=True)

        return nsdfg
