# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import dace
from dace.ml import DaceModule

from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_conv2d(use_cpp_dispatcher: bool, device):

    dev = torch_device(device)

    class Model(nn.Module):

        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3)
            self.conv2 = nn.Conv2d(4, 4, 3)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    ptmodel = Model().to(dev)
    x = torch.rand(1, 1, 8, 8).to(dev)

    dispatcher_suffix = "cpp" if use_cpp_dispatcher else "ctypes"

    @dace.ml.module(sdfg_name=f"test_conv2d_decorator_{dispatcher_suffix}_{device}", cuda=is_gpu(device))
    class TestDecorator(Model):
        pass

    dace_model = DaceModule(ptmodel,
                            sdfg_name=f"test_conv2d_{dispatcher_suffix}_{device}",
                            compile_torch_extension=use_cpp_dispatcher,
                            cuda=is_gpu(device))
    with experimental_cuda():
        dace_output = dace_model(x)

        dace_model_decorated = TestDecorator()
        dace_model_decorated(x)

    torch_output = ptmodel(x)

    np.testing.assert_allclose(torch_output.detach().cpu().numpy(),
                               dace_output.detach().cpu().numpy(),
                               atol=1e-06,
                               err_msg="Conv2d output mismatch between PyTorch and DaCe")


if __name__ == "__main__":
    test_conv2d(use_cpp_dispatcher=True, device="cpu")
    test_conv2d(use_cpp_dispatcher=False, device="cpu")
