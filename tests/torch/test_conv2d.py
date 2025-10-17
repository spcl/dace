import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import dace
from dace.frontend.python.module import DaceModule


@pytest.mark.torch
def test_conv2d(sdfg_name: str, use_cpp_dispatcher: bool):

    class Model(nn.Module):

        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3)
            self.conv2 = nn.Conv2d(4, 4, 3)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    ptmodel = Model()
    x = torch.rand(1, 1, 8, 8)

    @dace.module(sdfg_name=sdfg_name)
    class TestDecorator(Model):
        pass

    dace_model = DaceModule(ptmodel, sdfg_name=sdfg_name + "_wrapped", compile_torch_extension=use_cpp_dispatcher)
    dace_output = dace_model(x)

    dace_model_decorated = TestDecorator()
    dace_model_decorated(x)

    torch_output = ptmodel(x)

    np.testing.assert_allclose(torch_output.detach().numpy(),
                               dace_output.detach().numpy(),
                               atol=1e-06,
                               err_msg="Conv2d output mismatch between PyTorch and DaCe")


if __name__ == "__main__":
    test_conv2d(sdfg_name="test_conv2d_cpp_True", use_cpp_dispatcher=True)
    test_conv2d(sdfg_name="test_conv2d_cpp_False", use_cpp_dispatcher=False)
