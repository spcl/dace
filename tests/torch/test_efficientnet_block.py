# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
import numpy as np
from dace.transformation.dataflow import TrivialMapElimination
from dace.transformation.interstate import HoistState

from dace.ml import DaceModule
from tests.utils import torch_tensors_close


class SimpleMBConv(nn.Module):
    """Simplified MBConv block similar to EfficientNet architecture"""

    def __init__(self, in_channels=16, out_channels=24, expand_ratio=6, stride=2):
        super().__init__()

        expanded_channels = in_channels * expand_ratio

        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)

        self.depthwise_conv = nn.Conv2d(expanded_channels,
                                        expanded_channels,
                                        3,
                                        stride=stride,
                                        padding=1,
                                        groups=expanded_channels,
                                        bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)

        se_channels = max(1, in_channels // 4)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1)
        self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1)

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.swish = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.swish(out)

        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.swish(out)

        se = self.se_pool(out)
        se = self.se_reduce(se)
        se = self.swish(se)
        se = self.se_expand(se)
        se = self.sigmoid(se)
        out = out * se

        out = self.project_conv(out)
        out = self.project_bn(out)

        return out


@pytest.mark.torch
def test_mbconv(use_cpp_dispatcher: bool):

    with torch.no_grad():
        dace_inputs = torch.rand(1, 16, 112, 112)
        torch_inputs = torch.clone(dace_inputs)

    # Create SimpleMBConv block (similar to EfficientNet MBConv block)
    torch_model = SimpleMBConv(in_channels=16, out_channels=24, expand_ratio=6, stride=2).eval()
    dace_model_block = SimpleMBConv(in_channels=16, out_channels=24, expand_ratio=6, stride=2).eval()

    # Get the DaceModule
    sdfg_name = f"efficientnet_mbconv_{use_cpp_dispatcher}"
    dace_model = DaceModule(dace_model_block, sdfg_name=sdfg_name, compile_torch_extension=use_cpp_dispatcher)
    dace_model.model.load_state_dict(torch_model.state_dict())

    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):
        assert dace_name == torch_name, f"Parameter name mismatch: {dace_name} != {torch_name}"
        np.testing.assert_allclose(dace_value, value, err_msg=f"{dace_name} tensors do not match")

    dace_output = dace_model(dace_inputs)

    torch_output = torch_model(torch_inputs)
    np.testing.assert_allclose(dace_output.detach(),
                               torch_output.detach(),
                               rtol=1e-3,
                               atol=1e-3,
                               err_msg="output tensors do not match")

    # check that the batch norm running means and so on are written out correctly
    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):

        assert dace_name == torch_name, f"Parameter name mismatch after inference: {dace_name} != {torch_name}"
        np.testing.assert_allclose(dace_value, value, err_msg=f"{dace_name} tensors do not match")


@pytest.mark.torch
def test_fast_mb(use_cpp_dispatcher: bool):
    with torch.no_grad():
        dace_inputs = torch.rand(1, 16, 112, 112)
        torch_inputs = torch.clone(dace_inputs)

    # Create SimpleMBConv block
    torch_model = SimpleMBConv(in_channels=16, out_channels=24, expand_ratio=6, stride=2).eval()
    dace_model_block = SimpleMBConv(in_channels=16, out_channels=24, expand_ratio=6, stride=2).eval()

    # Get the DaceModule
    sdfg_name = f"efficientnet_fast_mbconv_{use_cpp_dispatcher}"
    dace_model = DaceModule(dace_model_block, sdfg_name=sdfg_name, compile_torch_extension=use_cpp_dispatcher)
    dace_model.model.load_state_dict(torch_model.state_dict())

    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):
        assert dace_name == torch_name, f"Parameter name mismatch: {dace_name} != {torch_name}"
        torch_tensors_close(dace_name, value, dace_value)

    def fuse_everything(module: DaceModule):
        sdfg = module.sdfg

        sdfg.apply_transformations_repeated(HoistState)
        sdfg.apply_transformations_repeated(TrivialMapElimination)
        sdfg.simplify()

    dace_model.append_post_onnx_hook("fuse_sg", fuse_everything)

    dace_output = dace_model(dace_inputs)

    torch_output = torch_model(torch_inputs)
    torch_tensors_close("output", dace_output, torch_output)

    # check that the batch norm running means and so on are written out correctly
    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):

        assert dace_name == torch_name, f"Parameter name mismatch after inference: {dace_name} != {torch_name}"
        torch_tensors_close(dace_name, value, dace_value)


if __name__ == "__main__":
    test_mbconv(use_cpp_dispatcher=True)
    test_mbconv(use_cpp_dispatcher=False)
    test_fast_mb(use_cpp_dispatcher=True)
    test_fast_mb(use_cpp_dispatcher=False)
