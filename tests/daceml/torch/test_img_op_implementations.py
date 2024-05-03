import pytest
import torch
from dace.library import change_default
from torch import nn

from dace.libraries import onnx as donnx

from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close, copy_to_gpu


@pytest.mark.parametrize("implementation", ["pure", "cuDNN"])
def test_bn(gpu, implementation, sdfg_name):
    if implementation == "cuDNN" and not gpu:
        pytest.skip("cuDNN is GPU only")

    with change_default(donnx.ONNXBatchNormalization, implementation):
        inputs = copy_to_gpu(gpu, torch.rand(1, 64, 60, 60))

        pt_model = copy_to_gpu(gpu, nn.BatchNorm2d(64))
        dace_model = copy_to_gpu(gpu, nn.BatchNorm2d(64))

        dace_model.load_state_dict(pt_model.state_dict())

        dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, training=True)
        dace_output = dace_model(inputs)
        pt_output = pt_model(inputs)

        torch_tensors_close("output", pt_output, dace_output)
        torch_tensors_close("mean", pt_model.running_mean,
                            dace_model.model.running_mean)
        torch_tensors_close("var", pt_model.running_var,
                            dace_model.model.running_var)


@pytest.mark.pure
def test_global_avg_pool(gpu, sdfg_name):
    inputs = copy_to_gpu(gpu, torch.rand(1, 64, 60, 60))

    pt_model = copy_to_gpu(gpu, nn.AdaptiveAvgPool2d(1))
    dace_model = copy_to_gpu(gpu, nn.AdaptiveAvgPool2d(1))

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, training=True)
    dace_output = dace_model(inputs)
    pt_output = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)
