# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import itertools
from typing import List, Optional, Tuple

import dace
import dace.libraries.torch
from dace.registry import autoregister_params
from dace import nodes as nd

from dace.libraries.onnx.converters import clean_onnx_name

import dace.autodiff.utils as butils
from dace.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult
from dace.util import in_desc_with_name


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
