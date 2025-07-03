import pytest
import torch

from dace.library import change_default
from efficientnet_pytorch import get_model_params
from efficientnet_pytorch.model import MBConvBlock
from torch import nn

import dace.libraries.onnx as donnx
from dace.libraries.onnx.op_implementations import CudnnConvolution
from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close
from dace.transformation.onnx import ConstantDeviceCopyElimination, PadConvFusion


@pytest.mark.gpu
def test_bn_cudnn(sdfg_name):
    with change_default(donnx.ONNXBatchNormalization, "cuDNN"):
        torch_bn = nn.BatchNorm2d(3).cuda()
        dace_bn = DaceModule(nn.BatchNorm2d(3).cuda(),
                             backward=True,
                             training=True)

        with torch.no_grad():
            dace_inputs = torch.rand(8, 3, 224, 224).cuda()
            torch_inputs = torch.clone(dace_inputs)

        dace_inputs.requires_grad = True
        torch_inputs.requires_grad = True

        dace_output = dace_bn(dace_inputs)
        torch_output = torch_bn(torch_inputs)

        torch_tensors_close("output", torch_output, dace_output)


def faster_groupconv(mod):
    # Choose depthwise convolution implementation
    def _choose_impl(module):
        for state in module.sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, donnx.ONNXConv):
                    if node.group > 1:
                        print("using pytorch dwise gconv:", node.label)
                        node.backward_implementation = "PyTorch-dwise"

    mod.append_post_onnx_hook("faster_groupconv", _choose_impl)


@pytest.mark.skip("TODO high tolerance required")
@pytest.mark.gpu
def test_mbconv(sdfg_name):
    with change_default(donnx.ONNXConv,
                        "cuDNN"), change_default(donnx.ONNXBatchNormalization,
                                                 "cuDNN"):

        with torch.no_grad():
            dace_inputs = torch.rand(8, 32, 224, 224).cuda()
            torch_inputs = torch.clone(dace_inputs)

        dace_inputs.requires_grad = True
        torch_inputs.requires_grad = True

        block_params, global_params = get_model_params("efficientnet-b0", {})

        torch_model = MBConvBlock(block_params[0], global_params).cuda()
        torch_model.set_swish(memory_efficient=False)
        dace_model = MBConvBlock(block_params[0], global_params).cuda()
        dace_model.set_swish(memory_efficient=False)
        dace_model = DaceModule(dace_model, training=True, backward=True)
        dace_model.model.load_state_dict(torch_model.state_dict())

        for (dace_name,
             dace_value), (torch_name,
                           value) in zip(dace_model.model.state_dict().items(),
                                         torch_model.state_dict().items()):
            assert dace_name == torch_name
            torch_tensors_close(dace_name, value, dace_value)

        def set_impls_fuse_conv(module: DaceModule):
            assert module.sdfg.apply_transformations(
                ConstantDeviceCopyElimination) == 1
            assert module.sdfg.apply_transformations(PadConvFusion) == 1

        dace_model.prepend_post_onnx_hook("high_level", set_impls_fuse_conv)

        faster_groupconv(dace_model)
        dace_output = dace_model(dace_inputs)

        torch_output = torch_model(torch_inputs)
        torch_tensors_close("output", torch_output, dace_output)

        # check that the batch norm running means and so on are written out correctly
        for (dace_name,
             dace_value), (torch_name,
                           value) in zip(dace_model.model.state_dict().items(),
                                         torch_model.state_dict().items()):

            assert dace_name == torch_name
            if "num_batches_tracked" in dace_name:
                # we don't update this parameter
                continue
            torch_tensors_close(dace_name, value, dace_value)

        # backward pass

        dy = torch.rand(8, 16, 224, 224).cuda()

        torch_output.backward(dy)
        dace_output.backward(dy)

        torch_tensors_close("input_grad",
                            torch_inputs.grad,
                            dace_inputs.grad,
                            atol=1e-3,
                            rtol=1e-3)

        for (name,
             dace_param), (pt_name,
                           pt_param) in zip(torch_model.named_parameters(),
                                            dace_model.named_parameters()):
            assert "model." + name == pt_name
            torch_tensors_close(name,
                                pt_param.grad,
                                dace_param.grad,
                                atol=1e-3,
                                rtol=1e-3)


if __name__ == "__main__":
    test_bn_cudnn("test_bn_cudnn")
    test_mbconv("test_mbconv")
