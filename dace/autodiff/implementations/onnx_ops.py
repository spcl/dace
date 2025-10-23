# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
ONNX Backward Pass Implementations for Automatic Differentiation.

This module provides backward pass implementations for ONNX operations in the DaCe autodiff
system. Each class implements the BackwardImplementation interface to compute gradients
for specific ONNX operations during reverse-mode automatic differentiation.

The implementations handle various ONNX operations including:
- Mathematical operations (Einsum, Clip, Softmax, etc.)
- Neural network layers (Conv, LayerNormalization, etc.)
- Pooling operations (MaxPool, GlobalAveragePool)
- Utility operations (Transpose, Where, etc.)
"""

import copy
import ctypes
import itertools
from typing import List, Optional, Tuple, Dict, Union

import numpy as np

# DaCe core imports
import dace
from dace.frontend.common import einsum
import dace.libraries
from dace.registry import autoregister_params
from dace import nodes as nd, dtypes, subsets
import dace.transformation.transformation as xf

# ONNX-specific imports
import dace.libraries.onnx as donnx
from dace.libraries.onnx.converters import clean_onnx_name
from dace.libraries.onnx.op_implementations.linalg_ops import PureEinsum
from dace.transformation.onnx.replacement import onnx_constant_or_none

# Autodiff imports
import dace.autodiff.utils as butils
from dace.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult

# Utility imports
from dace.util import in_desc_with_name


def reverse_einsum_wrt_input(forward_node: donnx.ONNXEinsum, input_name: str) -> Tuple[List[str], str]:
    """Produce the einsum string that computes the gradient of forward_node w.r.t. input_name.

    Note:
        There is an edge case we currently don't handle (can be implemented though).
        Something like 'ii->i' would become 'i->ii'. This is invalid because 'i' is repeated in the output.

    Args:
        forward_node: The einsum node to reverse
        input_name: The connector on the forward node to produce the gradient computation for

    Returns:
        Tuple of (list of forward node connectors required as inputs, einsum string)
        The first parameter of the produced einsum string is implicitly the grad of Output
    """

    _, input_idx = donnx.parse_variadic_param(input_name)
    parser = einsum.EinsumParser(forward_node.equation)

    backward_input_expressions = [parser.output] + parser.inputs[:input_idx] + parser.inputs[input_idx + 1:]
    backward_input_arrays = [
        f"Inputs__{i}" for i in itertools.chain(range(input_idx), range(input_idx + 1, len(parser.inputs)))
    ]

    einsum_str = f"{','.join(backward_input_expressions)}->{parser.inputs[input_idx]}"
    return backward_input_arrays, einsum_str


@autoregister_params(op="Einsum", name="default")
class DefaultEinsumBackward(BackwardImplementation):
    """Backward implementation for ONNX Einsum operation.

    The symbolic autodiff can automatically derive matmuls, but the produced maps are more difficult to optimize.
    This implementation provides a more efficient ONNX-based backward pass.
    """

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return PureEinsum.forward_can_be_applied(node, state, sdfg)

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        # setup arrays
        output_desc = butils.forward_out_desc_with_name(forward_node, context, "Output")
        result = BackwardResult.empty()
        result.given_grad_names["Output"] = butils.add_backward_desc(nsdfg, context.forward_sdfg, output_desc, "Output")
        access_output_grad = nstate.add_read(result.given_grad_names["Output"])

        def create_access_node(connector: str) -> nd.AccessNode:
            nsdfg.add_datadesc(connector,
                               copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, connector)))
            return nstate.add_read(connector)

        # the forward inputs we will require
        # maps the connector name to the accessnode
        required_forward_inputs: Dict[str, nd.AccessNode] = {}

        for input_name in sorted(required_gradients):
            # we add an einsum for each required gradient
            forward_inputs, einsum_str = reverse_einsum_wrt_input(forward_node, input_name)

            einsum_node = donnx.ONNXEinsum(input_name + "_backward", equation=einsum_str)
            nstate.add_node(einsum_node)

            # the first input is always the output grad
            einsum_node.add_in_connector(f"Inputs__0")
            nstate.add_edge(access_output_grad, None, einsum_node, "Inputs__0",
                            nsdfg.make_array_memlet(result.given_grad_names["Output"]))

            # add the other inputs from forward that we need
            for i, forward_input in enumerate(sorted(forward_inputs)):
                connector = f"Inputs__{i + 1}"
                einsum_node.add_in_connector(connector)
                if forward_input not in required_forward_inputs:
                    required_forward_inputs[forward_input] = create_access_node(forward_input)

                nstate.add_edge(required_forward_inputs[forward_input], None, einsum_node, connector,
                                nsdfg.make_array_memlet(forward_input))

            # write out the gradient
            butils.forward_in_desc_with_name(forward_node, context, input_name)
            result.required_grad_names[input_name] = butils.add_backward_desc_for_connector(
                nsdfg, forward_node, context, input_name, True)
            memlet = nsdfg.make_array_memlet(result.required_grad_names[input_name])
            # Add a wcr for gradient accumulation
            memlet.wcr = "lambda x, y: x + y"
            nstate.add_edge(einsum_node, "Output", nstate.add_write(result.required_grad_names[input_name]), None,
                            memlet)

        result_node = context.backward_state.add_nested_sdfg(
            nsdfg,
            set(result.given_grad_names.values()).union(required_forward_inputs),
            set(result.required_grad_names.values()))

        return result_node, result


@autoregister_params(op="Clip", name="default")
class DefaultClipBackward(BackwardImplementation):
    """Backward implementation for ONNX Clip operation.

    Computes gradients by zeroing out regions where the input was clipped
    and passing through gradients where the input was within bounds.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        result_node, result = butils.add_empty_sdfg_for_node(forward_node, ["input_grad", "output_grad", "input"],
                                                             context)

        nstate = result_node.sdfg.add_state()

        min_node = next(context.forward_state.in_edges_by_connector(forward_node, 'min')).src
        max_node = next(context.forward_state.in_edges_by_connector(forward_node, 'max')).src
        minval = onnx_constant_or_none(context.forward_sdfg, min_node)
        maxval = onnx_constant_or_none(context.forward_sdfg, max_node)

        idesc = butils.forward_in_desc_with_name(forward_node, context, "input")
        shape = idesc.shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}

        input_dtype = idesc.dtype
        minstr = f"dace.{input_dtype.to_string()}({minval})"
        maxstr = f"dace.{input_dtype.to_string()}({maxval})"

        index_str = f"{', '.join(map_ranges.keys())}"
        code = f"""
if __input < {minstr} or __input > {maxstr}:
    __input_grad = 0
else:
    __input_grad = __output_grad
                """
        nstate.add_mapped_tasklet(forward_node.label + "_backward",
                                  map_ranges=map_ranges,
                                  inputs={
                                      f"__output_grad": dace.Memlet(f"output_grad[{index_str}]"),
                                      f"__input": dace.Memlet(f"input[{index_str}]"),
                                  },
                                  code=code,
                                  outputs={f"__input_grad": dace.Memlet(f"input_grad[{index_str}]")},
                                  external_edges=True)

        return result_node, result


@autoregister_params(op="Dropout", name="default")
class DefaultDropoutBackward(BackwardImplementation):
    """Backward implementation for ONNX Dropout operation.

    Applies the dropout mask to the output gradients and scales by the keep probability.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        result_node, result = butils.add_empty_sdfg_for_node(forward_node,
                                                             ["data_grad", "output_grad", "mask", "ratio"], context)

        nstate = result_node.sdfg.add_state()

        shape = butils.forward_in_desc_with_name(forward_node, context, "data").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        code = f"""
scale = dace.float32(1.0) / (1 - __ratio)
__data_grad = __output_grad * __mask * scale
        """
        nstate.add_mapped_tasklet(forward_node.label + "_backward",
                                  map_ranges=map_ranges,
                                  inputs={
                                      "__output_grad": dace.Memlet(f"output_grad[{index_str}]"),
                                      "__mask": dace.Memlet(f"mask[{index_str}]"),
                                      "__ratio": dace.Memlet("ratio[0]")
                                  },
                                  code=code,
                                  outputs={f"__data_grad": dace.Memlet(f"data_grad[{index_str}]")},
                                  external_edges=True)

        return result_node, result


@autoregister_params(op="Softmax", name="default")
class DefaultSoftmaxBackward(BackwardImplementation):
    """Backward implementation for ONNX Softmax operation.

    Computes gradients using the mathematical relationship:
    dX = softmax(X) * (dY - sum(dY * softmax(X)))
    where dY is the output gradient and dX is the input gradient.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        dim = forward_node.axis

        output_desc = copy.deepcopy(butils.forward_out_desc_with_name(forward_node, context, "output"))
        output_desc.transient = False

        sums_shape = list(copy.deepcopy(output_desc.shape))
        sums_shape[dim] = 1

        # Create new SDFG
        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        result = BackwardResult.empty()

        # Given gradients (from output of forward pass)
        result.given_grad_names["output"] = "output_grad"
        output_grad_desc = copy.deepcopy(output_desc)
        nsdfg.add_datadesc("output_grad", output_grad_desc)

        # Required gradient to be computed
        input_name = "input"
        if "input" not in required_gradients:
            # this can happen for example in bert, where the input to softmax is masked
            input_name = next(iter(required_gradients))

        input_grad_desc = copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, input_name))
        input_grad_desc.transient = False
        result.required_grad_names[input_name] = "input_grad"
        nsdfg.add_datadesc("input_grad", input_grad_desc)

        # We need the output of the forward op
        nsdfg.add_datadesc("output", output_desc)

        # Intermediate arrays
        prod_desc = copy.deepcopy(output_desc)
        prod_desc.transient = True
        nsdfg.add_datadesc("prod", prod_desc)

        sums_desc = dace.data.Array(dace.float32, sums_shape, transient=True)
        nsdfg.add_datadesc("sums", sums_desc)

        sub_term_desc = copy.deepcopy(output_desc)
        sub_term_desc.transient = True
        nsdfg.add_datadesc("sub_term", sub_term_desc)

        # Add nodes
        output_grad_read = nstate.add_read("output_grad")
        forward_output_read = nstate.add_read("output")
        input_grad_write = nstate.add_write("input_grad")
        prod_access = nstate.add_access("prod")
        sums_access = nstate.add_access("sums")
        sub_term_access = nstate.add_access("sub_term")

        # prod = forward_output * output_grad
        mul_node1 = donnx.ONNXMul("mul_prod")
        nstate.add_node(mul_node1)
        nstate.add_edge(forward_output_read, None, mul_node1, "A", nsdfg.make_array_memlet("output"))
        nstate.add_edge(output_grad_read, None, mul_node1, "B", nsdfg.make_array_memlet("output_grad"))
        nstate.add_edge(mul_node1, "C", prod_access, None, nsdfg.make_array_memlet("prod"))

        # sums = ReduceSum(prod, axes=[dim], keepdims=1)
        reduce_sum_node = donnx.ONNXReduceSum("reduce_sum", keepdims=1, optional={"axes"})
        reduce_sum_node.axes = dim
        nstate.add_node(reduce_sum_node)

        # Setup the axes input for the ReduceSum node
        axes_name, _ = nsdfg.add_array(name="reduce_sum_axes", shape=[1], dtype=dace.int64, transient=True)
        axes_access = nstate.add_access(axes_name)
        axes_tasklet = nstate.add_tasklet("init_axes", {}, {"out"}, f"out = {dim};", language=dace.Language.CPP)
        nstate.add_edge(axes_tasklet, "out", axes_access, None, dace.Memlet(f"{axes_name}"))

        nstate.add_edge(prod_access, None, reduce_sum_node, "data", nsdfg.make_array_memlet("prod"))
        nstate.add_edge(axes_access, None, reduce_sum_node, "axes", nsdfg.make_array_memlet(axes_name))
        nstate.add_edge(reduce_sum_node, "reduced", sums_access, None, nsdfg.make_array_memlet("sums"))

        # sub_term = forward_output * sums
        mul_node2 = donnx.ONNXMul("mul_sub_term")
        nstate.add_node(mul_node2)
        nstate.add_edge(forward_output_read, None, mul_node2, "A", nsdfg.make_array_memlet("output"))
        nstate.add_edge(sums_access, None, mul_node2, "B", nsdfg.make_array_memlet("sums"))
        nstate.add_edge(mul_node2, "C", sub_term_access, None, nsdfg.make_array_memlet("sub_term"))

        # input_grad = prod - sub_term
        sub_node = donnx.ONNXSub("sub_input_grad")
        nstate.add_node(sub_node)
        nstate.add_edge(prod_access, None, sub_node, "A", nsdfg.make_array_memlet("prod"))
        nstate.add_edge(sub_term_access, None, sub_node, "B", nsdfg.make_array_memlet("sub_term"))
        nstate.add_edge(sub_node, "C", input_grad_write, None, nsdfg.make_array_memlet("input_grad"))

        # Create nested SDFG
        result_node = context.backward_state.add_nested_sdfg(
            nsdfg,
            # Inputs to nested SDFG
            {"output", "output_grad"},
            # Outputs from nested SDFG
            {"input_grad"})

        butils.connect_output_from_forward(forward_node, result_node, context, "output")

        return result_node, result


def _find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    """Find the first map entry node by the given parameter name.

    Args:
        sdfg: The SDFG to search
        pname: The parameter name to look for

    Returns:
        The first MapEntry node containing the specified parameter

    Raises:
        StopIteration: If no MapEntry with the parameter is found
    """
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


@autoregister_params(op="MaxPool", name="default")
class DefaultMaxPoolBackward(BackwardImplementation):
    """Backward implementation for ONNX MaxPool operation.

    Implements gradient computation by routing gradients only to the locations
    that achieved the maximum value in the forward pass.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        output_shape = butils.forward_out_desc_with_name(forward_node, context, "Y").shape

        N, C, H, W = output_shape
        sty, stx = forward_node.strides
        sy, sx = forward_node.kernel_shape

        def maxpool_backward(X, Y_grad, X_grad):
            for b, c, ti, tj in dace.map[0:N, 0:C, 0:H, 0:W]:
                maxv = np.empty([1], dtype=dace.float32)
                maxi = np.empty([1], dtype=dace.int32)
                maxj = np.empty([1], dtype=dace.int32)
                with dace.tasklet:
                    v >> maxv
                    v = -9999999

                # Deterministic argmax (assuming sequential map)
                for i, j in dace.map[0:sy, 0:sx]:
                    with dace.tasklet:
                        o << X[b, c, sty * ti + i, stx * tj + j]
                        vin << maxv
                        v >> maxv(-1)
                        ind_i >> maxi(-1)
                        ind_j >> maxj(-1)
                        if o > vin:
                            v = o
                            ind_i = i
                            ind_j = j
                with dace.tasklet:
                    igrad << Y_grad[b, c, ti, tj]
                    ind_i << maxi
                    ind_j << maxj
                    ograd >> X_grad(1)[b, c, :, :]
                    ograd[ind_i, ind_j] = igrad

        result_node, result = butils.backward_program_for_node(maxpool_backward, context, forward_node)

        _find_map_by_param(result_node.sdfg, 'i').schedule = \
            dace.ScheduleType.Sequential

        return result_node, result


@autoregister_params(op="LogSoftmax", name="default")
class DefaultLogSoftmaxBackward(BackwardImplementation):
    """Backward implementation for ONNX LogSoftmax operation.

    Computes gradients using the mathematical relationship for log-softmax:
    dX = dY - exp(Y) * sum(dY)
    where Y is the forward output and dY is the output gradient.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:

        dim = forward_node.axis
        output_shape = butils.forward_out_desc_with_name(forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def logsoftmax_backward(output, output_grad, input_grad):
            exp_output = dace.define_local(output_shape, output_dtype)
            donnx.ONNXExp(input=output, output=exp_output)

            grad_output_sum = dace.define_local(sums_shape, output_dtype)
            donnx.ONNXReduceSum(data=output_grad, reduced=grad_output_sum, keepdims=1, axes=[dim])
            # let's not use ONNXMul here; not sure how this inplace op is handled by ORT...
            exp_output[:] = exp_output * grad_output_sum
            donnx.ONNXSub(A=output_grad, B=exp_output, C=input_grad)

        result_node, result = butils.backward_program_for_node(logsoftmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context, "output")
        return result_node, result


@autoregister_params(op="Conv", name="PyTorch-dwise")
class PyTorchConvBackward(BackwardImplementation):
    """Depthwise convolution backward implementation using PyTorch.

    This implementation leverages PyTorch's optimized CUDA kernels for
    depthwise convolution backward pass computation.
    """

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        X_desc = in_desc_with_name(node, state, sdfg, "X")
        return len(X_desc.shape) == 4

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        X_desc = butils.forward_in_desc_with_name(forward_node, context, "X")
        W_desc = butils.forward_in_desc_with_name(forward_node, context, "W")

        T = X_desc.dtype
        if str(T) == 'float':
            pytorch_dtype = 'kFloat'
        elif str(T) == 'double':
            pytorch_dtype = 'kDouble'
        else:
            raise ValueError(f"PyTorch backward conv expansion supports only float and double tensors, got {str(T)}. "
                             f"Supported types: float, double")

        # setup gradient arrays
        result = BackwardResult.empty()
        required_grads = set(required_gradients)
        for r in sorted(required_grads):
            result.required_grad_names[r] = butils.add_backward_desc_for_connector(nsdfg,
                                                                                   forward_node,
                                                                                   context,
                                                                                   r,
                                                                                   input=True)
        result.given_grad_names["Y"] = butils.add_backward_desc_for_connector(nsdfg,
                                                                              forward_node,
                                                                              context,
                                                                              "Y",
                                                                              input=False)

        # setup non-gradient arrays
        required_forward_inputs = ["W", "X"]
        for i in sorted(required_forward_inputs):
            new_desc = copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, i))
            new_desc.transient = False
            nsdfg.add_datadesc(i, new_desc)

        # setup state
        nstate = nsdfg.add_state()
        unique_id = "{}_{}_{}_{}_bwd".format(clean_onnx_name(forward_node.name), context.forward_sdfg.sdfg_id,
                                             context.forward_sdfg.node_id(context.forward_state),
                                             context.forward_state.node_id(forward_node))

        init_code = ""
        finalize_code = ""
        code_global = """
            #include <ATen/Tensor.h>
            #include <ATen/Functions.h>
        """
        tasklet_inputs = {f"_{i}": dace.pointer(T) for i in itertools.chain(["dY"], sorted(required_forward_inputs))}
        tasklet_outputs = {f"_d{i}": dace.pointer(T) for i in itertools.chain(sorted(required_gradients))}

        tasklet_code = f"""
            std::vector<int64_t> x_shape = {{ {", ".join(map(str, X_desc.shape))} }};
            std::vector<int64_t> x_strides = {{ {", ".join(map(str, X_desc.strides))} }};
            std::vector<int64_t> w_shape = {{ {", ".join(map(str, W_desc.shape))} }};
            std::vector<int64_t> w_strides = {{ {", ".join(map(str, W_desc.strides))} }};
            at::Tensor x = at::from_blob(_X, x_shape, x_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor w = at::from_blob(_W, w_shape, w_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor dy = at::from_blob(_dY, x_shape, x_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor dw = at::from_blob(_dW, w_shape, w_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));
            at::Tensor dx = at::from_blob(_dX, x_shape, x_strides, [](void*){{}}, at::TensorOptions().device(at::kCUDA).dtype(at::{pytorch_dtype}).requires_grad(false));

            std::vector<int64_t> kernel_shape = {{ {", ".join(map(str, forward_node.kernel_shape))} }};
            std::vector<int64_t> conv_strides = {{ {", ".join(map(str, forward_node.strides))} }};
            std::vector<int64_t> padding = {{ {", ".join(map(str, forward_node.pads[::2]))} }};
            std::vector<int64_t> dilation = {{ {", ".join(map(str, forward_node.dilations))} }};

            at::thnn_conv_depthwise2d_backward_out(dx, dw, dy, x, w, kernel_shape, conv_strides, padding, dilation);
        """

        tasklet = nstate.add_tasklet(name=unique_id,
                                     inputs=tasklet_inputs,
                                     outputs=tasklet_outputs,
                                     code=tasklet_code,
                                     language=dace.dtypes.Language.CPP,
                                     code_global=code_global,
                                     code_init=init_code,
                                     code_exit=finalize_code)
        tasklet.environments = {dace.libraries.torch.environments.PyTorch.full_class_path()}

        nstate.add_edge(nstate.add_read(result.given_grad_names["Y"]), None, tasklet, f"_dY",
                        nsdfg.make_array_memlet((result.given_grad_names["Y"])))
        for name in sorted(required_forward_inputs):
            nstate.add_edge(nstate.add_read(name), None, tasklet, f"_{name}", nsdfg.make_array_memlet(name))

        for name in sorted(required_gradients):
            arr_name = result.required_grad_names[name]
            nstate.add_edge(tasklet, f"_d{name}", nstate.add_write(arr_name), None, nsdfg.make_array_memlet(arr_name))

        inputs = {result.given_grad_names["Y"]}.union(required_forward_inputs)
        outputs = {result.required_grad_names[n] for n in sorted(required_gradients)}
        node = context.backward_state.add_nested_sdfg(nsdfg, inputs, outputs)

        return node, result


@autoregister_params(op="GlobalAveragePool", name="pure")
class PureGlobalAveragePoolingBackward(BackwardImplementation):
    """Pure implementation of GlobalAveragePool backward pass.

    Broadcasts the output gradient uniformly across the spatial dimensions
    with appropriate scaling by the pool size.
    """

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return len(in_desc_with_name(node, state, sdfg, "X").shape) == 4

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:
        desc = butils.forward_in_desc_with_name(forward_node, context, "X")
        N, C, H, W = desc.shape
        dtype = desc.dtype

        inv = 1.0 / (H * W)

        def bwd(X_grad, Y_grad):
            for n, c, h, w in dace.map[0:N, 0:C, 0:H, 0:W]:
                with dace.tasklet:
                    y_grad << Y_grad[n, c]
                    x_grad >> X_grad[n, c, h, w]
                    x_grad = y_grad * dtype(inv)

        return butils.backward_program_for_node(bwd, context, forward_node)


@autoregister_params(op="Transpose", name="default")
class DefaultTransposeBackward(BackwardImplementation):
    """Backward implementation for ONNX Transpose operation.

    The gradient of transpose is another transpose with inverted permutation.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:
        inv_perm = tuple(np.argsort(forward_node.perm))

        node = donnx.ONNXTranspose(forward_node.name + "_backward", perm=inv_perm)
        context.backward_state.add_node(node)

        result = BackwardResult.empty()
        result.given_grad_names["transposed"] = "data"
        result.required_grad_names["data"] = "transposed"

        return node, result


@autoregister_params(op="Where", name="default")
class WhereBackward(BackwardImplementation):
    """Backward implementation for ONNX Where operation.

    Routes gradients based on the condition: gradients flow to X where condition is True,
    and to Y where condition is False.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:
        # condition, X, Y -> Output
        # Get condition descriptor for shape information
        _ = butils.forward_in_desc_with_name(forward_node, context, "condition")

        # NOTE: We cannot use ONNX ops for further potential lowering
        # transformations because ONNXMul does not support boolean inputs.
        # notcondition = dace.define_local(condition_shape, condition_dtype)
        # donnx.ONNXMul(A=condition, B=output_grad, C=X_grad)
        # donnx.ONNXNot(X=condition, Y=notcondition)
        # donnx.ONNXMul(A=notcondition, B=output_grad, C=Y_grad)

        if 'X' in required_gradients and 'Y' not in required_gradients:

            def where_backward(condition, output_grad, X_grad):
                X_grad[:] = condition * output_grad
        elif 'Y' in required_gradients and 'X' not in required_gradients:

            def where_backward(condition, output_grad, Y_grad):
                Y_grad[:] = ~condition * output_grad
        elif 'X' in required_gradients and 'Y' in required_gradients:

            def where_backward(condition, output_grad, X_grad, Y_grad):
                X_grad[:] = condition * output_grad
                Y_grad[:] = ~condition * output_grad

        result_node, result = butils.backward_program_for_node(where_backward, context, forward_node)

        return result_node, result


@autoregister_params(op="LayerNormalization", name="default")
class DefaultLayerNormalizationBackward(BackwardImplementation):
    """Backward implementation for ONNX LayerNormalization operation.

    Computes gradients for input, scale, and bias parameters using the
    mathematical formulation of layer normalization backward pass.
    """

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:
        # Create new SDFG
        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        # Get input/output descriptors
        X_desc = copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, "X"))
        Scale_desc = copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, "Scale"))
        Y_grad_desc = copy.deepcopy(butils.forward_out_desc_with_name(forward_node, context, "Y"))
        X_desc.transient = False
        Y_grad_desc.transient = False
        Scale_desc.transient = False

        result = BackwardResult.empty()
        # setup gradient arrays
        result.given_grad_names["Y"] = "Y_grad"
        if "X" in required_gradients:
            result.required_grad_names["X"] = "X_grad"
        if "Scale" in required_gradients:
            result.required_grad_names["Scale"] = "Scale_grad"
        if "B" in required_gradients:
            result.required_grad_names["B"] = "B_grad"

        # Add data descriptors to SDFG
        nsdfg.add_datadesc("X", X_desc)
        nsdfg.add_datadesc("Scale", Scale_desc)
        nsdfg.add_datadesc("Y_grad", Y_grad_desc)

        if "X" in required_gradients:
            X_grad_desc = copy.deepcopy(X_desc)
            nsdfg.add_datadesc("X_grad", X_grad_desc)
        if "Scale" in required_gradients:
            Scale_grad_desc = copy.deepcopy(Scale_desc)
            nsdfg.add_datadesc("Scale_grad", Scale_grad_desc)
        if "B" in required_gradients:
            B_desc = copy.deepcopy(butils.forward_in_desc_with_name(forward_node, context, "B"))
            B_desc.transient = False
            B_grad_desc = copy.deepcopy(B_desc)
            nsdfg.add_datadesc("B_grad", B_grad_desc)
            # Add B to SDFG inputs when needed
            nsdfg.add_datadesc("B", B_desc)

        # Get axis and epsilon
        axis = forward_node.axis if hasattr(forward_node, 'axis') else -1
        epsilon = forward_node.epsilon if hasattr(forward_node, 'epsilon') else 1e-5

        rank = len(X_desc.shape)
        if axis < 0:
            axis = rank + axis
        reduction_axes = list(range(axis, rank))
        leading_non_normalized_axes = list(range(axis))
        # Calculate normalization size for reference (currently unused)
        _ = float(np.prod([X_desc.shape[i] for i in range(axis, rank)]))

        # Create axes tensor for reduction
        axes_name = "reduction_axes"
        axes_desc = dace.data.Array(dace.int64, [len(reduction_axes)])
        axes_desc.transient = True  # Make it transient since it's internal
        nsdfg.add_datadesc(axes_name, axes_desc)
        axes_access = nstate.add_access(axes_name)

        # Initialize reduction axes as a constant array
        axes_tasklet = nstate.add_tasklet(name="init_axes",
                                          inputs={},
                                          outputs={"out": dace.pointer(dace.int64)},
                                          code=f"\n".join([f"out[{i}] = {0};" for i, _ in enumerate(reduction_axes)]),
                                          language=dace.Language.CPP)
        nstate.add_edge(axes_tasklet, "out", axes_access, None, dace.Memlet(f"{axes_name}[0:{len(reduction_axes)}]"))

        # Create mean descriptor with reduced shape
        mean_shape = list(X_desc.shape)
        for i in reduction_axes:
            mean_shape[i] = 1
        mean_desc = dace.data.Array(X_desc.dtype, mean_shape)
        mean_desc.transient = True
        mean_name = "mean"
        nsdfg.add_datadesc(mean_name, mean_desc)

        mean_op = donnx.ONNXReduceMean("mean_op", keepdims=1, optional={"axes"})
        mean_op.axes = reduction_axes
        nstate.add_node(mean_op)
        nstate.add_edge(nstate.add_read("X"), None, mean_op, "data", nsdfg.make_array_memlet("X"))
        nstate.add_edge(axes_access, None, mean_op, "axes", nsdfg.make_array_memlet(axes_name))
        mean_access = nstate.add_access("mean")
        nstate.add_edge(mean_op, "reduced", mean_access, None, nsdfg.make_array_memlet("mean"))

        # Recompute variance
        diff_shape = list(X_desc.shape)
        diff_desc = dace.data.Array(X_desc.dtype, diff_shape)
        diff_desc.transient = True
        diff_name = "diff"
        nsdfg.add_datadesc(diff_name, diff_desc)

        diff_op = donnx.ONNXSub("diff_op")
        nstate.add_node(diff_op)
        nstate.add_edge(nstate.add_read("X"), None, diff_op, "A", nsdfg.make_array_memlet("X"))
        nstate.add_edge(mean_access, None, diff_op, "B", nsdfg.make_array_memlet("mean"))
        diff_access = nstate.add_access("diff")
        nstate.add_edge(diff_op, "C", diff_access, None, nsdfg.make_array_memlet("diff"))

        # Create squared difference descriptor
        sq_diff_shape = list(X_desc.shape)
        sq_diff_desc = dace.data.Array(X_desc.dtype, sq_diff_shape)
        sq_diff_desc.transient = True
        sq_diff_name = "sq_diff"
        nsdfg.add_datadesc(sq_diff_name, sq_diff_desc)

        sq_diff_op = donnx.ONNXMul("sq_diff_op")
        nstate.add_node(sq_diff_op)
        nstate.add_edge(diff_access, None, sq_diff_op, "A", nsdfg.make_array_memlet("diff"))
        nstate.add_edge(diff_access, None, sq_diff_op, "B", nsdfg.make_array_memlet("diff"))
        sq_diff_access = nstate.add_access("sq_diff")
        nstate.add_edge(sq_diff_op, "C", sq_diff_access, None, nsdfg.make_array_memlet("sq_diff"))

        # Create variance descriptor with reduced shape
        variance_shape = list(X_desc.shape)
        for i in reduction_axes:
            variance_shape[i] = 1
        variance_desc = dace.data.Array(X_desc.dtype, variance_shape)
        variance_desc.transient = True
        variance_name = "variance"
        nsdfg.add_datadesc(variance_name, variance_desc)

        variance_op = donnx.ONNXReduceMean("variance_op", keepdims=1, optional={"axes"})
        variance_op.axes = reduction_axes
        nstate.add_node(variance_op)
        nstate.add_edge(sq_diff_access, None, variance_op, "data", nsdfg.make_array_memlet("sq_diff"))
        nstate.add_edge(axes_access, None, variance_op, "axes", nsdfg.make_array_memlet(axes_name))
        variance_access = nstate.add_access("variance")
        nstate.add_edge(variance_op, "reduced", variance_access, None, nsdfg.make_array_memlet("variance"))

        # Add epsilon to variance
        epsilon_name, _ = nsdfg.add_scalar("epsilon", X_desc.dtype, transient=True)
        epsilon_tasklet = nstate.add_tasklet(
            "make_epsilon",
            {},
            {"out"},
            f"out = {epsilon};",
            language=dace.Language.CPP,
        )
        epsilon_write = nstate.add_write(epsilon_name)
        nstate.add_edge(epsilon_tasklet, "out", epsilon_write, None, dace.Memlet(f"{epsilon_name}[0]"))

        # Create variance_eps descriptor
        variance_eps_desc = dace.data.Array(X_desc.dtype, variance_shape)
        variance_eps_desc.transient = True
        variance_eps_name = "variance_eps"
        nsdfg.add_datadesc(variance_eps_name, variance_eps_desc)

        variance_eps_op = donnx.ONNXAdd("variance_eps_op")
        nstate.add_node(variance_eps_op)
        nstate.add_edge(variance_access, None, variance_eps_op, "A", nsdfg.make_array_memlet("variance"))
        nstate.add_edge(epsilon_write, None, variance_eps_op, "B", nsdfg.make_array_memlet(epsilon_name))
        variance_eps_access = nstate.add_access("variance_eps")
        nstate.add_edge(variance_eps_op, "C", variance_eps_access, None, nsdfg.make_array_memlet("variance_eps"))

        # Create std_dev descriptor
        std_dev_desc = dace.data.Array(X_desc.dtype, variance_shape)
        std_dev_desc.transient = True
        std_dev_name = "std_dev"
        nsdfg.add_datadesc(std_dev_name, std_dev_desc)

        std_dev_op = donnx.ONNXSqrt("std_dev_op")
        nstate.add_node(std_dev_op)
        nstate.add_edge(variance_eps_access, None, std_dev_op, "X", nsdfg.make_array_memlet("variance_eps"))
        std_dev_access = nstate.add_access("std_dev")
        nstate.add_edge(std_dev_op, "Y", std_dev_access, None, nsdfg.make_array_memlet("std_dev"))

        # Create inv_std_dev descriptor
        one_name, _ = nsdfg.add_scalar("one", X_desc.dtype, transient=True)
        one_tasklet = nstate.add_tasklet("make_one", {}, {"out"}, "out = 1.0;", language=dace.Language.CPP)
        one_write = nstate.add_write(one_name)
        nstate.add_edge(one_tasklet, "out", one_write, None, dace.Memlet(f"{one_name}[0]"))

        inv_std_dev_desc = dace.data.Array(X_desc.dtype, variance_shape)
        inv_std_dev_desc.transient = True
        inv_std_dev_name = "inv_std_dev"
        nsdfg.add_datadesc(inv_std_dev_name, inv_std_dev_desc)

        inv_std_dev_op = donnx.ONNXDiv("inv_std_dev_op")
        nstate.add_node(inv_std_dev_op)
        nstate.add_edge(one_write, None, inv_std_dev_op, "A", nsdfg.make_array_memlet(one_name))
        nstate.add_edge(std_dev_access, None, inv_std_dev_op, "B", nsdfg.make_array_memlet("std_dev"))
        inv_std_dev_access = nstate.add_access("inv_std_dev")
        nstate.add_edge(inv_std_dev_op, "C", inv_std_dev_access, None, nsdfg.make_array_memlet("inv_std_dev"))

        # Create x_hat descriptor (normalized input)
        x_hat_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
        x_hat_desc.transient = True
        x_hat_name = "x_hat"
        nsdfg.add_datadesc(x_hat_name, x_hat_desc)

        x_hat_op = donnx.ONNXMul("x_hat_op")
        nstate.add_node(x_hat_op)
        nstate.add_edge(diff_access, None, x_hat_op, "A", nsdfg.make_array_memlet("diff"))
        nstate.add_edge(inv_std_dev_access, None, x_hat_op, "B", nsdfg.make_array_memlet("inv_std_dev"))
        x_hat_access = nstate.add_access("x_hat")
        nstate.add_edge(x_hat_op, "C", x_hat_access, None, nsdfg.make_array_memlet("x_hat"))

        # Compute bias gradient if needed
        if "B" in required_gradients:
            b_grad_op = donnx.ONNXReduceSum("b_grad_op", keepdims=0, optional={"axes"})
            # This reduction will sum over the leading non-normalized axes
            b_grad_op.axes = leading_non_normalized_axes
            nstate.add_node(b_grad_op)
            nstate.add_edge(nstate.add_read("Y_grad"), None, b_grad_op, "data", nsdfg.make_array_memlet("Y_grad"))
            nstate.add_edge(axes_access, None, b_grad_op, "axes", nsdfg.make_array_memlet(axes_name))
            nstate.add_edge(b_grad_op, "reduced", nstate.add_write("B_grad"), None, nsdfg.make_array_memlet("B_grad"))

        # Compute scale gradient if needed
        if "Scale" in required_gradients:
            dY_x_hat_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            dY_x_hat_desc.transient = True
            dY_x_hat_name = "dY_x_hat"
            nsdfg.add_datadesc(dY_x_hat_name, dY_x_hat_desc)

            dY_x_hat_op = donnx.ONNXMul("dY_x_hat_op")
            nstate.add_node(dY_x_hat_op)
            nstate.add_edge(nstate.add_read("Y_grad"), None, dY_x_hat_op, "A", nsdfg.make_array_memlet("Y_grad"))
            nstate.add_edge(x_hat_access, None, dY_x_hat_op, "B", nsdfg.make_array_memlet("x_hat"))
            dY_x_hat_access = nstate.add_access("dY_x_hat")
            nstate.add_edge(dY_x_hat_op, "C", dY_x_hat_access, None, nsdfg.make_array_memlet("dY_x_hat"))

            scale_grad_op = donnx.ONNXReduceSum("scale_grad_op", keepdims=0, optional={"axes"})
            scale_grad_op.axes = leading_non_normalized_axes
            nstate.add_node(scale_grad_op)
            nstate.add_edge(dY_x_hat_access, None, scale_grad_op, "data", nsdfg.make_array_memlet("dY_x_hat"))
            nstate.add_edge(axes_access, None, scale_grad_op, "axes", nsdfg.make_array_memlet(axes_name))
            nstate.add_edge(scale_grad_op, "reduced", nstate.add_write("Scale_grad"), None,
                            nsdfg.make_array_memlet("Scale_grad"))

        # Compute X gradient if needed
        if "X" in required_gradients:
            # Create dX_hat descriptor (gradient with respect to normalized input)
            dX_hat_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            dX_hat_desc.transient = True
            dX_hat_name = "dX_hat"
            nsdfg.add_datadesc(dX_hat_name, dX_hat_desc)

            dX_hat_op = donnx.ONNXMul("dX_hat_op")
            nstate.add_node(dX_hat_op)
            nstate.add_edge(nstate.add_read("Y_grad"), None, dX_hat_op, "A", nsdfg.make_array_memlet("Y_grad"))
            nstate.add_edge(nstate.add_read("Scale"), None, dX_hat_op, "B", nsdfg.make_array_memlet("Scale"))
            dX_hat_access = nstate.add_access("dX_hat")
            nstate.add_edge(dX_hat_op, "C", dX_hat_access, None, nsdfg.make_array_memlet("dX_hat"))

            # Compute mean of dX_hat over reduction axes
            dX_hat_mean_desc = dace.data.Array(X_desc.dtype, variance_shape)
            dX_hat_mean_desc.transient = True
            dX_hat_mean_name = "dX_hat_mean"
            nsdfg.add_datadesc(dX_hat_mean_name, dX_hat_mean_desc)

            dX_hat_mean_op = donnx.ONNXReduceMean("dX_hat_mean_op", keepdims=1, optional={"axes"})
            dX_hat_mean_op.axes = reduction_axes
            nstate.add_node(dX_hat_mean_op)
            nstate.add_edge(dX_hat_access, None, dX_hat_mean_op, "data", nsdfg.make_array_memlet("dX_hat"))
            nstate.add_edge(axes_access, None, dX_hat_mean_op, "axes", nsdfg.make_array_memlet(axes_name))
            dX_hat_mean_access = nstate.add_access("dX_hat_mean")
            nstate.add_edge(dX_hat_mean_op, "reduced", dX_hat_mean_access, None, nsdfg.make_array_memlet("dX_hat_mean"))

            # Compute dX_hat * x_hat
            dX_hat_x_hat_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            dX_hat_x_hat_desc.transient = True
            dX_hat_x_hat_name = "dX_hat_x_hat"
            nsdfg.add_datadesc(dX_hat_x_hat_name, dX_hat_x_hat_desc)

            dX_hat_x_hat_op = donnx.ONNXMul("dX_hat_x_hat_op")
            nstate.add_node(dX_hat_x_hat_op)
            nstate.add_edge(dX_hat_access, None, dX_hat_x_hat_op, "A", nsdfg.make_array_memlet("dX_hat"))
            nstate.add_edge(x_hat_access, None, dX_hat_x_hat_op, "B", nsdfg.make_array_memlet("x_hat"))
            dX_hat_x_hat_access = nstate.add_access("dX_hat_x_hat")
            nstate.add_edge(dX_hat_x_hat_op, "C", dX_hat_x_hat_access, None, nsdfg.make_array_memlet("dX_hat_x_hat"))

            # Compute mean of dX_hat * x_hat over reduction axes
            dX_hat_x_hat_mean_desc = dace.data.Array(X_desc.dtype, variance_shape)
            dX_hat_x_hat_mean_desc.transient = True
            dX_hat_x_hat_mean_name = "dX_hat_x_hat_mean"
            nsdfg.add_datadesc(dX_hat_x_hat_mean_name, dX_hat_x_hat_mean_desc)

            dX_hat_x_hat_mean_op = donnx.ONNXReduceMean("dX_hat_x_hat_mean_op", keepdims=1, optional={"axes"})
            dX_hat_x_hat_mean_op.axes = reduction_axes
            nstate.add_node(dX_hat_x_hat_mean_op)
            nstate.add_edge(dX_hat_x_hat_access, None, dX_hat_x_hat_mean_op, "data",
                            nsdfg.make_array_memlet("dX_hat_x_hat"))
            nstate.add_edge(axes_access, None, dX_hat_x_hat_mean_op, "axes", nsdfg.make_array_memlet(axes_name))
            dX_hat_x_hat_mean_access = nstate.add_access("dX_hat_x_hat_mean")
            nstate.add_edge(dX_hat_x_hat_mean_op, "reduced", dX_hat_x_hat_mean_access, None,
                            nsdfg.make_array_memlet("dX_hat_x_hat_mean"))

            # Compute x_hat * mean(dX_hat * x_hat)
            x_hat_dX_hat_x_hat_mean_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            x_hat_dX_hat_x_hat_mean_desc.transient = True
            x_hat_dX_hat_x_hat_mean_name = "x_hat_dX_hat_x_hat_mean"
            nsdfg.add_datadesc(x_hat_dX_hat_x_hat_mean_name, x_hat_dX_hat_x_hat_mean_desc)

            x_hat_dX_hat_x_hat_mean_op = donnx.ONNXMul("x_hat_dX_hat_x_hat_mean_op")
            nstate.add_node(x_hat_dX_hat_x_hat_mean_op)
            nstate.add_edge(x_hat_access, None, x_hat_dX_hat_x_hat_mean_op, "A", nsdfg.make_array_memlet("x_hat"))
            nstate.add_edge(dX_hat_x_hat_mean_access, None, x_hat_dX_hat_x_hat_mean_op, "B",
                            nsdfg.make_array_memlet("dX_hat_x_hat_mean"))
            x_hat_dX_hat_x_hat_mean_access = nstate.add_access("x_hat_dX_hat_x_hat_mean")
            nstate.add_edge(x_hat_dX_hat_x_hat_mean_op, "C", x_hat_dX_hat_x_hat_mean_access, None,
                            nsdfg.make_array_memlet("x_hat_dX_hat_x_hat_mean"))

            # Compute dX_hat - mean(dX_hat) - x_hat * mean(dX_hat * x_hat)
            dX_hat_minus_mean_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            dX_hat_minus_mean_desc.transient = True
            dX_hat_minus_mean_name = "dX_hat_minus_mean"
            nsdfg.add_datadesc(dX_hat_minus_mean_name, dX_hat_minus_mean_desc)

            dX_hat_minus_mean_op = donnx.ONNXSub("dX_hat_minus_mean_op")
            nstate.add_node(dX_hat_minus_mean_op)
            nstate.add_edge(dX_hat_access, None, dX_hat_minus_mean_op, "A", nsdfg.make_array_memlet("dX_hat"))
            nstate.add_edge(dX_hat_mean_access, None, dX_hat_minus_mean_op, "B", nsdfg.make_array_memlet("dX_hat_mean"))
            dX_hat_minus_mean_access = nstate.add_access("dX_hat_minus_mean")
            nstate.add_edge(dX_hat_minus_mean_op, "C", dX_hat_minus_mean_access, None,
                            nsdfg.make_array_memlet("dX_hat_minus_mean"))

            # Final subtraction
            dX_hat_final_desc = dace.data.Array(X_desc.dtype, X_desc.shape)
            dX_hat_final_desc.transient = True
            dX_hat_final_name = "dX_hat_final"
            nsdfg.add_datadesc(dX_hat_final_name, dX_hat_final_desc)

            dX_hat_final_op = donnx.ONNXSub("dX_hat_final_op")
            nstate.add_node(dX_hat_final_op)
            nstate.add_edge(dX_hat_minus_mean_access, None, dX_hat_final_op, "A",
                            nsdfg.make_array_memlet("dX_hat_minus_mean"))
            nstate.add_edge(x_hat_dX_hat_x_hat_mean_access, None, dX_hat_final_op, "B",
                            nsdfg.make_array_memlet("x_hat_dX_hat_x_hat_mean"))
            dX_hat_final_access = nstate.add_access("dX_hat_final")
            nstate.add_edge(dX_hat_final_op, "C", dX_hat_final_access, None, nsdfg.make_array_memlet("dX_hat_final"))

            # Multiply by inv_std_dev to get final X gradient
            x_grad_op = donnx.ONNXMul("x_grad_op")
            nstate.add_node(x_grad_op)
            nstate.add_edge(inv_std_dev_access, None, x_grad_op, "A", nsdfg.make_array_memlet("inv_std_dev"))
            nstate.add_edge(dX_hat_final_access, None, x_grad_op, "B", nsdfg.make_array_memlet("dX_hat_final"))
            nstate.add_edge(x_grad_op, "C", nstate.add_write("X_grad"), None, nsdfg.make_array_memlet("X_grad"))

        # Set up inputs for nested SDFG
        inputs = {"X", "Scale", "Y_grad"}
        if "B" in required_gradients:
            inputs.add("B")

        outputs = set(result.required_grad_names.values())
        bwd_node = context.backward_state.add_nested_sdfg(nsdfg, inputs, outputs)
        return bwd_node, result


@autoregister_params(op="ReduceSum", name="default")
class DefaultReduceSumBackward(BackwardImplementation):
    """Backward implementation for ONNX ReduceSum operation.

    The backward pass of a reduction is a broadcast of the output gradient
    to match the input shape. Handles both keepdims=True and keepdims=False cases.
    """

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return True

    @staticmethod
    def backward(forward_node: nd.Node, context: BackwardContext, given_gradients: List[Optional[str]],
                 required_gradients: List[Optional[str]]) -> Tuple[nd.Node, BackwardResult]:

        # The backward pass of a reduction is a broadcast.
        # We use ONNXExpand to perform the broadcast.
        # If keepdims=False, we first need to unsqueeze the gradient.

        input_desc = butils.forward_in_desc_with_name(forward_node, context, "data")
        output_desc = butils.forward_out_desc_with_name(forward_node, context, "reduced")

        nsdfg = dace.SDFG(f"{forward_node.label}_backward")
        nstate = nsdfg.add_state()

        result = BackwardResult.empty()
        result.given_grad_names["reduced"] = "reduced_grad"
        result.required_grad_names["data"] = "data_grad"

        reduced_grad_desc = copy.deepcopy(output_desc)
        reduced_grad_desc.transient = False
        nsdfg.add_datadesc("reduced_grad", reduced_grad_desc)

        data_grad_desc_tmp = copy.deepcopy(input_desc)
        data_grad_desc_tmp.transient = True
        nsdfg.add_datadesc("data_grad_tmp", data_grad_desc_tmp)

        data_grad_desc = copy.deepcopy(input_desc)
        data_grad_desc.transient = False
        nsdfg.add_datadesc("data_grad", data_grad_desc)

        grad_to_expand = "reduced_grad"
        read_grad_to_expand = nstate.add_read(grad_to_expand)

        keepdims = getattr(forward_node, 'keepdims', 1)

        if not keepdims:
            # When keepdims is False, the rank of the output is reduced. We need to
            # unsqueeze the gradient to match the input rank before broadcasting.

            # Deduce reduced axes by comparing input and output shapes.
            in_shape = input_desc.shape
            out_shape = reduced_grad_desc.shape
            unsqueezed_shape = []
            axes = []
            if len(in_shape) < len(out_shape):
                raise ValueError(f"Input shape {in_shape} has fewer dimensions than output shape {out_shape}. "
                                 f"This is unexpected for a ReduceSum operation.")
            if len(in_shape) > len(out_shape):
                # This assumes that non-reduced dimensions are preserved in order.
                out_shape_idx = 0
                for i, dim in enumerate(in_shape):
                    if out_shape_idx < len(out_shape) and dim == out_shape[out_shape_idx]:
                        out_shape_idx += 1
                        unsqueezed_shape.append(dim)
                    else:
                        axes.append(i)
                        unsqueezed_shape.append(1)

            # If shapes are equal, it's a no-op reduction and axes is empty.
            if (not axes) != (len(in_shape) == len(out_shape)):
                raise ValueError(f"Inconsistent state: axes={axes}, input_shape={in_shape}, output_shape={out_shape}. "
                                 f"For equal shapes, axes should be empty.")

            if 'axes' in forward_node.in_connectors:
                # The axes are a dynamic input to the forward node. Pass them to the backward node.
                axes_desc = butils.forward_in_desc_with_name(forward_node, context, "axes")
                axes_desc_copy = copy.deepcopy(axes_desc)
                axes_desc_copy.transient = False
                nsdfg.add_datadesc("axes", axes_desc_copy)
                axes_access = nstate.add_read("axes")
            elif axes:
                # Create a constant array for the axes to be passed to Unsqueeze
                axes_name_in_bwd, axes_desc_bwd = nsdfg.add_array(f"axes_for_unsqueeze_{forward_node.name}",
                                                                  [len(axes)],
                                                                  dace.int64,
                                                                  transient=True)
                axes_tasklet = nstate.add_tasklet(
                    'init_axes',
                    {},
                    {'out'},
                    '\n'.join([f'out[{i}] = {v};' for i, v in enumerate(axes)]),
                    language=dace.Language.CPP,
                )
                axes_access = nstate.add_access(axes_name_in_bwd)
                nstate.add_edge(axes_tasklet, 'out', axes_access, None,
                                dace.Memlet.from_array(axes_name_in_bwd, axes_desc_bwd))

            unsqueezed_desc = dace.data.Array(dtype=reduced_grad_desc.dtype, shape=unsqueezed_shape, transient=True)
            nsdfg.add_datadesc("unsqueezed_grad", unsqueezed_desc)

            unsqueeze_op = donnx.ONNXUnsqueeze("unsqueeze_grad")
            nstate.add_node(unsqueeze_op)

            nstate.add_edge(read_grad_to_expand, None, unsqueeze_op, "data", nsdfg.make_array_memlet("reduced_grad"))
            nstate.add_edge(axes_access, None, unsqueeze_op, "axes",
                            dace.Memlet(data=axes_access.data, subset=f'0:{axes_access.desc(nsdfg).shape[0]}'))

            grad_to_expand = "unsqueezed_grad"
            read_grad_to_expand = nstate.add_access(grad_to_expand)
            nstate.add_edge(unsqueeze_op, "expanded", read_grad_to_expand, None,
                            nsdfg.make_array_memlet("unsqueezed_grad"))

        # Create shape tensor for ONNXExpand
        shape_name, shape_desc = nsdfg.add_array("shape_for_expand", [len(input_desc.shape)],
                                                 dace.int64,
                                                 transient=True)
        shape_tasklet = nstate.add_tasklet("init_shape", {}, {"out"},
                                           '\n'.join([f"out[{i}] = {s};" for i, s in enumerate(input_desc.shape)]))
        shape_access = nstate.add_access(shape_name)
        nstate.add_edge(shape_tasklet, "out", shape_access, None, dace.Memlet.from_array(shape_name, shape_desc))

        expand_op = donnx.ONNXExpand("expand_grad")
        nstate.add_node(expand_op)
        write_data_grad_tmp = nstate.add_write("data_grad_tmp")

        nstate.add_edge(read_grad_to_expand, None, expand_op, "input", nsdfg.make_array_memlet(grad_to_expand))
        nstate.add_edge(shape_access, None, expand_op, "shape", nsdfg.make_array_memlet(shape_name))
        nstate.add_edge(expand_op, "output", write_data_grad_tmp, None, nsdfg.make_array_memlet("data_grad_tmp"))

        # We add an additional write from data_grad_tmp to data_grad
        # This is necessary to accumulate gradients in the backward pass.
        finale_memlet = nsdfg.make_array_memlet("data_grad")
        finale_memlet.wcr = "lambda x, y: x + y"
        write_data_grad = nstate.add_write("data_grad")
        nstate.add_edge(write_data_grad_tmp, None, write_data_grad, None, finale_memlet)

        inputs = {"reduced_grad"}
        if not keepdims and 'axes' in forward_node.in_connectors:
            inputs.add("axes")

        result_node = context.backward_state.add_nested_sdfg(nsdfg, inputs, {"data_grad"})

        return result_node, result
