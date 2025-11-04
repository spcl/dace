# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import numpy as np
from dace.transformation.dataflow import TrivialMapElimination
from dace.transformation.interstate import HoistState
from efficientnet_pytorch import get_model_params
from efficientnet_pytorch.model import MBConvBlock

from dace.ml import DaceModule
from tests.utils import torch_tensors_close


@pytest.mark.torch
def test_mbconv(use_cpp_dispatcher: bool):

    with torch.no_grad():
        dace_inputs = torch.rand(8, 32, 224, 224)
        torch_inputs = torch.clone(dace_inputs)

    block_params, global_params = get_model_params("efficientnet-b0", {})

    torch_model = MBConvBlock(block_params[0], global_params).eval()
    torch_model.set_swish(memory_efficient=False)
    dace_model = MBConvBlock(block_params[0], global_params).eval()
    dace_model.set_swish(memory_efficient=False)

    # Get the DaceModule
    sdfg_name = f"efficientnet_mbconv_{use_cpp_dispatcher}"
    dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, compile_torch_extension=use_cpp_dispatcher)
    dace_model.model.load_state_dict(torch_model.state_dict())

    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):
        assert dace_name == torch_name, f"Parameter name mismatch: {dace_name} != {torch_name}"
        np.testing.assert_allclose(value, dace_value, err_msg=f"{dace_name} tensors do not match")

    dace_output = dace_model(dace_inputs)

    torch_output = torch_model(torch_inputs)
    np.testing.assert_allclose(torch_output.detach(),
                               dace_output.detach(),
                               rtol=1e-3,
                               atol=1e-3,
                               err_msg="output tensors do not match")

    # check that the batch norm running means and so on are written out correctly
    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):

        assert dace_name == torch_name, f"Parameter name mismatch after inference: {dace_name} != {torch_name}"
        if "num_batches_tracked" in dace_name:
            # we don't update this parameter
            continue
        np.testing.assert_allclose(value, dace_value, err_msg=f"{dace_name} tensors do not match")


@pytest.mark.torch
def test_fast_mb(use_cpp_dispatcher: bool):
    with torch.no_grad():
        dace_inputs = torch.rand(8, 32, 224, 224)
        torch_inputs = torch.clone(dace_inputs)

    block_params, global_params = get_model_params("efficientnet-b0", {})

    torch_model = MBConvBlock(block_params[0], global_params).eval()
    torch_model.set_swish(memory_efficient=False)
    dace_model = MBConvBlock(block_params[0], global_params).eval()
    dace_model.set_swish(memory_efficient=False)

    # Get the DaceModule
    sdfg_name = f"efficientnet_fast_mbconv_{use_cpp_dispatcher}"
    dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, compile_torch_extension=use_cpp_dispatcher)
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
    torch_tensors_close("output", torch_output, dace_output, rtol=1e-3, atol=1e-3)

    # check that the batch norm running means and so on are written out correctly
    for (dace_name, dace_value), (torch_name, value) in zip(dace_model.model.state_dict().items(),
                                                            torch_model.state_dict().items()):

        assert dace_name == torch_name, f"Parameter name mismatch after inference: {dace_name} != {torch_name}"
        if "num_batches_tracked" in dace_name:
            # we don't update this parameter
            continue
        torch_tensors_close(dace_name, value, dace_value)


if __name__ == "__main__":
    test_mbconv(use_cpp_dispatcher=True)
    test_mbconv(use_cpp_dispatcher=False)
    test_fast_mb(use_cpp_dispatcher=True)
    test_fast_mb(use_cpp_dispatcher=False)
