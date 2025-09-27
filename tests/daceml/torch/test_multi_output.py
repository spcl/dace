import pytest
import torch
from torch import nn

from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close


class Model(nn.Module):

    def __init__(self, new_shape):
        super(Model, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x + 1, x + 2


def test_multiple_outputs(sdfg_name, use_cpp_dispatcher):

    ptmodel = Model([5, 5])
    x = torch.rand([25])

    torch_outputs = ptmodel(torch.clone(x))

    dace_model = DaceModule(ptmodel,
                            auto_optimize=False,
                            sdfg_name=sdfg_name,
                            compile_torch_extension=use_cpp_dispatcher)

    dace_outputs = dace_model(x)

    torch_tensors_close("output_0", torch_outputs[0], dace_outputs[0])
    torch_tensors_close("output_1", torch_outputs[1], dace_outputs[1])
