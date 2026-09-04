# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from torch import nn
from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


class Model(nn.Module):

    def __init__(self, new_shape):
        super(Model, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x.reshape(self.new_shape)
        return x


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_reshape_module(device):

    dev = torch_device(device)

    ptmodel = Model([5, 5]).to(dev)
    x = torch.rand([25]).to(dev)

    torch_output = ptmodel(torch.clone(x))

    # dummy_inputs triggers compilation at construction time, so build under the experimental backend.
    with experimental_cuda():
        dace_model = DaceModule(ptmodel,
                                sdfg_name=f"test_reshape_module_{device}",
                                auto_optimize=False,
                                dummy_inputs=(x, ),
                                cuda=is_gpu(device))

        dace_output = dace_model(x)

    torch_tensors_close("output", torch_output, dace_output)


if __name__ == "__main__":
    test_reshape_module(device="cpu")
