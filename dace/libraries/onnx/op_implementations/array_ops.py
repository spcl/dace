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
from dace.transformation.onnx import constant_folding
from dace.transformation.onnx.replacement import onnx_constant_or_none
from dace.libraries.onnx import converters

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
        # Avoid this expansion if the backward pass will be constructed
        # TODO pass the backward flag to the functions
        return False

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Create new SDFG
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
        expanded_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "expanded"))

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("data", data_desc)
        nsdfg.add_datadesc("expanded", expanded_desc)
        nsdfg.arrays["data"].transient = False
        nsdfg.arrays["expanded"].transient = False

        # Add access nodes
        data_read = nstate.add_read("data")
        expanded_write = nstate.add_write("expanded")

        # Handle axes based on ONNX version
        if node.schema.since_version < 13:
            # axes is attribute - create transient array and initialize it
            axes_values = node.axes if hasattr(node, 'axes') else []
            axes_arr_shape = [len(axes_values)]
            axes_arr_dtype = dace.int64
            _, axes_desc = nsdfg.add_array("axes", axes_arr_shape, axes_arr_dtype, transient=True)
            axes_node = nstate.add_access("axes")

            # Add tasklet to initialize axes array
            axes_init_tasklet = nstate.add_tasklet(
                f"init_axes",
                set(), {"out": dace.pointer(axes_arr_dtype)},
                "\n".join([f"out [{idx}] = {val};" for idx, val in enumerate(axes_values)]),
                language=dace.Language.CPP)
            nstate.add_edge(axes_init_tasklet, "out", axes_node, None, dace.Memlet(f"axes[0:{len(axes_values)}]"))
        else:
            # axes is input - get from input connector
            axes_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "axes"))
            nsdfg.add_datadesc("axes", axes_desc)
            nsdfg.arrays["axes"].transient = False
            axes_node = nstate.add_read("axes")

        is_scalar_input = not isinstance(node.in_connectors['data'], dace.dtypes.pointer) and data_desc.total_size == 1
        if is_scalar_input:
            data_str = "(&__data)"
        else:
            data_str = "__data"

        # Create tasklet that performs the unsqueeze operation
        data_size = int(np.prod(data_desc.shape))
        tasklet = nstate.add_tasklet(name=node.label + "_tasklet",
                                     inputs={
                                         "__data": dace.pointer(data_desc.dtype),
                                         "__axes": dace.pointer(axes_desc.dtype),
                                     },
                                     outputs={"__unsqueezed": dace.pointer(expanded_desc.dtype)},
                                     code=f"""
            for (int i = 0; i < {data_size}; i++) {{
                __unsqueezed[i] = {data_str}[i];
            }}
            """,
                                     language=dace.Language.CPP)

        # Connect the tasklet with memlets
        nstate.add_edge(data_read, None, tasklet, "__data", dace.Memlet("data"))
        nstate.add_edge(axes_node, None, tasklet, "__axes", dace.Memlet("axes"))
        nstate.add_edge(tasklet, "__unsqueezed", expanded_write, None, dace.Memlet("expanded"))

        return nsdfg


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
        input_index_str = ", ".join(input_indices) if input_indices else "0"

        # Create map over output shape
        map_ranges = {f"i{i}": f"0:{output_desc.shape[i]}" for i in range(len(output_desc.shape))}
        output_index_str = ", ".join(map_ranges.keys())

        # Create mapped tasklet for broadcasting copy
        nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                  map_ranges=map_ranges,
                                  inputs={"__input": dace.Memlet(f"input[{input_index_str}]")},
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
    '''
        Slice expansion
    '''

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> bool:
        # Check that all the inputs (even the optional ones) are present and constant

        if not hasattr(sdfg, "_parent_onnx_model"):
            return False

        constant_starts = in_edge_with_name(node, state, "starts").src.data in sdfg._parent_onnx_model.clean_weights

        if not constant_starts:
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


@op_implementation(op="Slice", name="pure")
class PureSliceAllConstant(ONNXForward):

    @staticmethod
    def _get_constant(conn: str, node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG):
        try:
            srcnode = next(state.in_edges_by_connector(node, conn)).src
        except StopIteration:
            # Return default values
            if conn == "steps":
                return 1
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

        # If split input is provided, it must be a constant
        if has_split_input:
            split_node = next(state.in_edges_by_connector(node, "split")).src
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
        # Check if shape input is constant
        try:
            shape_node = next(state.in_edges_by_connector(node, "shape")).src
            return onnx_constant_or_none(sdfg, shape_node) is not None
        except (StopIteration, ValueError):
            return False

    @staticmethod
    def forward(node: 'ONNXOp', state: SDFGState, sdfg: SDFG) -> typing.Union[Node, SDFG]:
        # Get the shape value
        shape_node = next(state.in_edges_by_connector(node, "shape")).src
        shape_val = onnx_constant_or_none(sdfg, shape_node)

        # Remove shape input since it's constant
        constant_folding.remove_node_and_computation(sdfg, state, node, "shape")

        # Get fill value (default is 0)
        fill_value = 0
        if hasattr(node, 'value'):
            fill_value = node.value

        output_desc = out_desc_with_name(node, state, sdfg, "output")

        def prog(output):
            output[:] = fill_value

        return program_for_node(prog, sdfg, state, node)


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
            except (StopIteration, ValueError):
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

        nsdfg.add_datadesc("output", copy.deepcopy(output_desc))
        nsdfg.arrays["output"].transient = False

        output_write = nstate.add_write("output")

        # Calculate number of elements
        num_elements = int(np.ceil((limit_val - start_val) / delta_val))

        # Create mapped tasklet to generate range
        map_ranges = {"i": f"0:{num_elements}"}

        tasklet, map_entry, map_exit = nstate.add_mapped_tasklet(name=node.label + "_tasklet",
                                                                 map_ranges=map_ranges,
                                                                 inputs={},
                                                                 code=f"__out = {start_val} + i * {delta_val}",
                                                                 outputs={"__out": dace.Memlet("output[i]")},
                                                                 external_edges=True)

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
        except (StopIteration, ValueError):
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
