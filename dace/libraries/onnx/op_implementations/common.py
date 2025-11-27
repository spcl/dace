# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Common utilities and helper functions for ONNX pure implementations.

This module provides shared utilities used across different ONNX operator implementations,
including:
- Stride computation and shape utilities
- Broadcasting helpers
- Reduction operation setup and code generation
- Common imports and logging configuration
"""

import copy
import logging
import dace
from dace.sdfg.utils import in_desc_with_name, out_desc_with_name
from dace import SDFG, SDFGState

log = logging.getLogger(__name__)


def strides_from_shape(shape, *, order="C", itemsize=1, in_bytes=False):
    """Compute contiguous array strides for a given shape.

    :param shape: Shape tuple, e.g. (2, 8, 128, 1).
    :param order: Row-major ("C") or column-major ("F").
    :param itemsize: Size of one element in bytes (used if in_bytes=True).
    :param in_bytes: Return strides in bytes instead of elements.
    :return: Strides per dimension.
    """
    if not shape:
        return ()
    ndim = len(shape)
    strides = [0] * ndim

    if order.upper() == "C":
        step = 1
        for i in range(ndim - 1, -1, -1):
            strides[i] = step
            step *= shape[i]
    elif order.upper() == "F":
        step = 1
        for i in range(ndim):
            strides[i] = step
            step *= shape[i]
    else:
        raise ValueError("order must be 'C' or 'F'")

    if in_bytes:
        strides = [s * itemsize for s in strides]
    return tuple(strides)


def broadcast_indices(input_shape, output_shape):
    """
    Returns a list of index expressions for broadcasting input_shape to output_shape.
    For each output dimension, if the input has that dimension and its size is 1, append "0";
    if it has that dimension and its size > 1, append the corresponding index (e.g. f"i{i}");
    if the input does not have that dimension, skip it (do not append anything).
    """
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    indices = []
    for i in range(output_rank):
        if i >= output_rank - input_rank:
            input_idx = i - (output_rank - input_rank)
            if input_shape[input_idx] == 1:
                indices.append("0")
            else:
                indices.append(f"i{i}")
    return indices


def setup_reduction_sdfg(node: 'ONNXOp', state: SDFGState, sdfg: SDFG, operation_name: str):
    """Helper function to set up the common SDFG structure for reduction operations.

    :param node: The ONNX operation node.
    :param state: The SDFG state.
    :param sdfg: The parent SDFG.
    :param operation_name: Name of the reduction operation (e.g., 'reduce_mean', 'reduce_sum').
    :return: Tuple of (nsdfg, nstate, data_desc, reduced_desc, data_read, reduced_write,
             axes_node, axes_desc, num_reduce_axes).
    """
    # Get attributes
    keepdims = getattr(node, 'keepdims', 1)
    noop_with_empty_axes = getattr(node, 'noop_with_empty_axes', 0)

    # Create a new SDFG for the reduction with unique name
    uid = state.node_id(node)
    nsdfg = SDFG(f'{operation_name}_{uid}')
    nstate = nsdfg.add_state(f'{operation_name}_{uid}')

    # Get input and output arrays with deep copies and unique names
    data_desc = copy.deepcopy(in_desc_with_name(node, state, sdfg, "data"))
    reduced_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "reduced"))

    # Add arrays to the SDFG with unique names
    data_name = "data"
    reduced_name = "reduced"
    nsdfg.add_datadesc(data_name, data_desc)
    nsdfg.add_datadesc(reduced_name, reduced_desc)
    nsdfg.arrays[data_name].transient = False
    nsdfg.arrays[reduced_name].transient = False

    # Create access nodes
    data_read = nstate.add_read(data_name)
    reduced_write = nstate.add_write(reduced_name)

    # Determine axes and num_axes
    axes_name = "axes"
    tmp_axes_name = "tmp_axes"
    if axes_name in node.in_connectors:
        [axes_edge] = state.in_edges_by_connector(node, axes_name)
        axes_desc = sdfg.arrays[axes_edge.data.data]
        [axes_size] = axes_desc.shape
        axes_dtype = axes_desc.dtype

        # Add original axes as input
        nsdfg.add_array(axes_name, shape=[axes_size], dtype=axes_dtype, transient=False)
        axes_node = nstate.add_read(axes_name)

        tmp_axes_size = axes_size + 1  # adding extra element to ensure we never pass scalars by reference
        _, tmp_axes_desc = nsdfg.add_array(tmp_axes_name, shape=[tmp_axes_size], dtype=axes_dtype, transient=True)
        tmp_axes_node = nstate.add_access(tmp_axes_name)

        # Add a copy edge to copy values to the new array
        nstate.add_edge(axes_node, None, tmp_axes_node, None,
                        dace.Memlet(f"{axes_name}[0:{axes_size}] -> {tmp_axes_name}[0:{axes_size}]"))
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
        axes_size = len(axes_values)
        tmp_axes_size = axes_size + 1  # adding extra element to ensure we never pass scalars by reference
        axes_dtype = dace.int64
        _, axes_desc = nsdfg.add_array(tmp_axes_name, shape=[tmp_axes_size], dtype=axes_dtype, transient=True)
        tmp_axes_node = nstate.add_access(tmp_axes_name)

        # Add a tasklet to initialize the axes array in the SDFG
        axes_init_tasklet = nstate.add_tasklet(
            f"init_axes",
            set(),
            {"__tmp_axes": dace.pointer(axes_dtype)},
            "\n".join([f"__tmp_axes[{idx}] = {val};" for idx, val in enumerate(axes_values)]) +
            f"\n__tmp_axes[{axes_size}] = 0;",  # Set the extra element to 0
            language=dace.Language.CPP)
        nstate.add_edge(axes_init_tasklet, "__tmp_axes", tmp_axes_node, None,
                        dace.Memlet(f"{tmp_axes_name}[0:{tmp_axes_size}]"))

    return nsdfg, nstate, data_desc, reduced_desc, data_read, reduced_write, tmp_axes_node, axes_desc, axes_size


def generate_reduction_tasklet_code(data_desc, reduced_desc, num_reduce_axes, keepdims, reduction_type, **kwargs):
    """Helper function to generate the C++ tasklet code for reduction operations.

    :param data_desc: Input data descriptor.
    :param reduced_desc: Output data descriptor.
    :param num_reduce_axes: Number of reduction axes.
    :param keepdims: Whether to keep dimensions.
    :param reduction_type: Type of reduction ('sum', 'mean', 'max', 'min').
    :param kwargs: Additional arguments for specific reduction types.
    :return: The generated C++ tasklet code.
    """
    axes_arr_code = f"long long reduce_dims [num_reduce_dims] = {{{', '.join([f'axes_arr[{i}]' for i in range(num_reduce_axes)])}}};"

    # Common setup code
    setup_code = f"""
        constexpr long long input_dims = {len(data_desc.shape)};
        constexpr long long output_dims = {len(reduced_desc.shape)};
        constexpr long long num_reduce_dims = {num_reduce_axes};
        constexpr long long keepdims = {keepdims};
        {axes_arr_code}
        for (long long i = 0; i < num_reduce_dims; i++) {{
            if (reduce_dims[i] < 0) {{
                reduce_dims[i] = input_dims + reduce_dims[i];
            }}
        }}

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
    """

    # Add reduce_size computation for mean reduction
    if reduction_type == 'mean':
        setup_code += """
        // Compute the number of elements to reduce over for mean
        long long reduce_size = 1;
        for (long long i = 0; i < input_dims; i++) {
            int is_reduce_dim = 0;
            for (long long j = 0; j < num_reduce_dims; j++) {
                if (reduce_dims[j] == i) {
                    is_reduce_dim = 1;
                    break;
                }
            }
            if (is_reduce_dim) {
                reduce_size *= input_shape[i];
            }
        }
        """

    # Initialize output based on reduction type
    if reduction_type == 'sum':
        initial_value = "0.0"
        init_comment = "// initialize output to zero"
    elif reduction_type == 'mean':
        initial_value = "0.0"
        init_comment = "// initialize output to zero"
    elif reduction_type == 'max':
        initial_value = "-INFINITY"
        init_comment = "// initialize output to negative infinity"
    elif reduction_type == 'min':
        initial_value = "INFINITY"
        init_comment = "// initialize output to positive infinity"
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")

    init_code = init_comment + "\n" + "\n".join([
        f"for (long long i{i} = 0; i{i} < output_shape[{i}]; i{i}++) {{" for i in range(len(reduced_desc.shape))
    ]) + "\n" + "out [" + " + ".join([f"output_strides [{i}] * i{i}" for i in range(len(reduced_desc.shape))
                                      ]) + f"] = {initial_value};" + "\n" + "\n".join(["}" for _ in reduced_desc.shape])

    # Generate reduction operation code
    if reduction_type == 'sum':
        reduction_code = "out [" + " + ".join([
            f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))
        ]) + "]" + " += inp [" + " + ".join([f"input_strides[{i}] * i{i}" for i in range(len(data_desc.shape))]) + "];"
    elif reduction_type == 'mean':
        # For mean, we divide by reduce_size (computed in setup)
        reduction_code = "out [" + " + ".join([
            f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))
        ]) + "]" + " += inp [" + " + ".join([f"input_strides[{i}] * i{i}"
                                             for i in range(len(data_desc.shape))]) + "] / reduce_size;"
    elif reduction_type == 'max':
        reduction_code = "out [" + " + ".join(
            [f"output_strides_input[{i}] * i{i}"
             for i in range(len(data_desc.shape))]) + "]" + " = std::max(out [" + " + ".join([
                 f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))
             ]) + "], " + "inp [" + " + ".join([f"input_strides[{i}] * i{i}"
                                                for i in range(len(data_desc.shape))]) + "]);"
    elif reduction_type == 'min':
        reduction_code = "out [" + " + ".join(
            [f"output_strides_input[{i}] * i{i}"
             for i in range(len(data_desc.shape))]) + "]" + " = std::min(out [" + " + ".join([
                 f"output_strides_input[{i}] * i{i}" for i in range(len(data_desc.shape))
             ]) + "], " + "inp [" + " + ".join([f"input_strides[{i}] * i{i}"
                                                for i in range(len(data_desc.shape))]) + "]);"
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")

    # Generate loop structure
    loop_code = """
        // Loop over all input elements
        """ + "\n".join([
        f"for (long long i{i} = 0; i{i} < input_shape[{i}]; i{i}++) {{" for i in range(len(data_desc.shape))
    ]) + "\n" + reduction_code + "\n" + "\n".join(["}" for _ in data_desc.shape])

    # Combine all parts
    tasklet_code = setup_code + "\n" + init_code + "\n" + loop_code + "\n"

    return tasklet_code


def iterables_equal(a, b) -> bool:
    """ Return whether the two iterables ``a`` and ``b`` are equal. """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))
