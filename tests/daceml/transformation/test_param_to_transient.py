import pytest
import torch
from torch import nn

from dace.frontend.python.module import DaceModule
from dace.transformation.onnx import parameter_to_transient


@pytest.mark.gpu
def test_torch_from_dlpack(sdfg_name):

    class Module(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc1(x)

    pt_module = Module().cuda()
    dace_module = Module().cuda()
    dace_module.load_state_dict(pt_module.state_dict())

    input = torch.rand(2, 10).cuda()

    dace_module_wrapped = DaceModule(dace_module, sdfg_name=sdfg_name)

    assert torch.allclose(dace_module_wrapped(input), pt_module(input))
    dace_module_wrapped = DaceModule(dace_module,
                                     sdfg_name=sdfg_name + "_after")

    def param_to_trans(model):
        parameter_to_transient(model, "fc1.weight")

    dace_module_wrapped.append_post_onnx_hook("param_to_transient",
                                              param_to_trans)

    assert torch.allclose(dace_module_wrapped(input), pt_module(input))


if __name__ == "__main__":
    test_torch_from_dlpack("test_sdfg")
