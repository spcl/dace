import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from torch import nn
import numpy as np

from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close


class CustomBatchNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, running_mean, running_var, weight, bias, training, momentum, eps):
        output = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
        return output, running_mean, running_var

    @staticmethod
    def symbolic(g, x, running_mean, running_var, weight, bias, training, momentum, eps):
        outputs = g.op("BatchNormalization",
                       x,
                       weight,
                       bias,
                       running_mean,
                       running_var,
                       training_mode_i=int(training),
                       momentum_f=momentum,
                       epsilon_f=eps,
                       outputs=3)
        y, new_running_mean, new_running_var = outputs
        return y, new_running_mean, new_running_var


class BatchNorm2dMeanVar(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2dMeanVar, self).__init__()
        self.bn = nn.BatchNorm2d(num_features,
                                 eps=eps,
                                 momentum=momentum,
                                 affine=affine,
                                 track_running_stats=track_running_stats)

    def forward(self, x):
        return CustomBatchNorm.apply(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias,
                                     self.bn.training, self.bn.momentum, self.bn.eps)


@pytest.mark.torch
def test_bn(sdfg_name: str):

    inputs = torch.rand(1, 64, 60, 60)

    # pytorch and onnx specification differ in the way they use momentum:
    # pytorch_momentum = 1 - onnx_momentum
    # to guarantee matching behavior, we set the momentum to 0.5

    pt_model = BatchNorm2dMeanVar(64, momentum=0.5)
    dace_model = BatchNorm2dMeanVar(64, momentum=0.5)
    pt_model.train()
    dace_model.train()

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, training=True)
    dace_output, dace_mean, dace_var = dace_model(inputs)
    pt_output, pt_mean, pt_var = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)
    torch_tensors_close("mean", pt_mean, dace_mean)
    torch_tensors_close("var", pt_var, dace_var)


@pytest.mark.torch
def test_global_avg_pool(sdfg_name: str):
    inputs = torch.rand(1, 64, 60, 60)

    pt_model = nn.AdaptiveAvgPool2d(1)
    dace_model = nn.AdaptiveAvgPool2d(1)

    # Note: AdaptiveAvgPool2d has no parameters, but load_state_dict ensures compatibility
    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, training=True)
    dace_output = dace_model(inputs)
    pt_output = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)


if __name__ == "__main__":
    test_bn(sdfg_name="test_bn")
    test_global_avg_pool(sdfg_name="test_global_avg_pool")
