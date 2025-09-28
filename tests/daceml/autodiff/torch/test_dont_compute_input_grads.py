import torch
from torch import nn

from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close


def test_skip_input_grads(sdfg_name, use_cpp_dispatcher):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.rand(10, 10))

        def forward(self, x):
            return x @ self.fc1

    dace_module = Module()
    pt_module = Module()
    pt_module.load_state_dict(dace_module.state_dict())

    shape = [8, 10]
    input_value = torch.rand(*shape, dtype=torch.float32)

    pytorch_input = torch.empty(
        *shape,
        dtype=torch.float32,
        requires_grad=False,
    )
    pytorch_input.copy_(input_value)
    dace_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False)
    dace_input.copy_(input_value)

    # TODO: provide a better API for input names
    dace_module = DaceModule(dace_module,
                             backward=True,
                             inputs_to_skip=["onnx::MatMul_0"],
                             compile_torch_extension=use_cpp_dispatcher,
                             sdfg_name=sdfg_name)

    dy = torch.rand(8, 10)

    dace_output = dace_module(dace_input)
    pt_output = pt_module(pytorch_input)

    torch_tensors_close("output", pt_output, dace_output)

    # check that fc1.grad is being computed
    dace_output.backward(dy)
    pt_output.backward(dy)
    torch_tensors_close("param_grad", pt_module.fc1.grad, dace_module.model.fc1.grad)

    # make sure that input grad is not being computed
    assert len(dace_module.backward_sdfg.node(0).sink_nodes()) == 1


if __name__ == "__main__":
    test_skip_input_grads("test_skip_input_grads", use_cpp_dispatcher=False)
    test_skip_input_grads("test_skip_input_grads_cpp", use_cpp_dispatcher=True)
