from dace.transformation.onnx import ReshapeElimination, expand_library_nodes_except_reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dace.frontend.python.module import DaceModule
import pytest
import dace
import dace.libraries.onnx as donnx


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.view(-1, 256)
        return F.relu(x)


@pytest.mark.skip(reason="Does not work on CI")
@pytest.mark.pure
def test_reshape_elimination(gpu, sdfg_name):

    ptmodel = Model()
    x = torch.rand((100, 6, 12, 12))
    dace_model = DaceModule(ptmodel,
                            auto_optimize=False,
                            sdfg_name=sdfg_name,
                            cuda=gpu)

    def ApplyReshapeElimination(dace_module):
        sdfg = dace_module.sdfg
        expand_library_nodes_except_reshape(sdfg)
        applied = sdfg.apply_transformations_repeated([ReshapeElimination],
                                                      print_report=True)
        assert applied == 1

    dace_model.append_post_onnx_hook("ApplyReshapeElimination",
                                     ApplyReshapeElimination)

    torch_output = ptmodel(x)
    dace_output = dace_model(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


if __name__ == "__main__":
    gpu = False
    sdfg_name = "test_reshape_elimination"
    test_reshape_elimination(gpu, sdfg_name)
