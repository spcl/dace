import numpy as np
import torch
import torch.nn as nn
import pytest

import dace
import dace.libraries.onnx as donnx
from dace.frontend.python.module import DaceModule
from dace.transformation.onnx import InputToConstant


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(5, 3)

    def forward(self, x):
        return x + 2


@pytest.mark.skip(
    reason=
    "this transformation will need to be rewritten: dace now supports accessing as acessnodes"
)
@pytest.mark.pure
def test_input_to_constant(sdfg_name):

    net = TestModule()
    dace_net = DaceModule(net, sdfg_name=sdfg_name)

    inp = torch.rand((10, 5))

    def ApplyInputToConst(dace_module):
        sdfg = dace_module.sdfg
        sdfg.expand_library_nodes()
        applied = sdfg.apply_transformations_repeated([InputToConstant],
                                                      print_report=True)
        assert applied == 1

    dace_net.append_post_onnx_hook("ApplyInputToConst", ApplyInputToConst)

    torch_result = net(torch.clone(inp))
    with dace.library.change_default(donnx.ONNXAdd, "pure"):
        dace_result = dace_net(torch.clone(inp))

    assert np.allclose(torch_result.detach().numpy(), dace_result)
