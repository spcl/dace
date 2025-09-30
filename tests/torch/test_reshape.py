import pytest
import torch
from torch import nn
import numpy as np
from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close


class Model(nn.Module):

    def __init__(self, new_shape):
        super(Model, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x.reshape(self.new_shape)
        return x


@pytest.mark.torch
def test_reshape_module(sdfg_name: str):

    ptmodel = Model([5, 5])
    x = torch.rand([25])

    torch_output = ptmodel(torch.clone(x))

    dace_model = DaceModule(ptmodel, auto_optimize=False, dummy_inputs=(x, ), sdfg_name=sdfg_name)

    dace_output = dace_model(x)

    torch_tensors_close("output", torch_output, dace_output)
