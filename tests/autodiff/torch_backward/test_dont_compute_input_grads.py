# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from torch import nn

from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_skip_input_grads(use_cpp_dispatcher: bool, device):

    dev = torch_device(device)

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.rand(10, 10))

        def forward(self, x):
            return x @ self.fc1

    dace_module = Module().to(dev)
    pt_module = Module().to(dev)
    pt_module.load_state_dict(dace_module.state_dict())

    shape = [8, 10]
    input_value = torch.rand(*shape, dtype=torch.float32, device=dev)

    pytorch_input = torch.empty(
        *shape,
        dtype=torch.float32,
        requires_grad=False,
        device=dev,
    )
    pytorch_input.copy_(input_value)
    dace_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False, device=dev)
    dace_input.copy_(input_value)

    # TODO: provide a better API for input names
    dispatcher_suffix = "cpp" if use_cpp_dispatcher else "ctypes"
    dace_module = DaceModule(dace_module,
                             sdfg_name=f"test_skip_input_grads_{dispatcher_suffix}_{device}",
                             backward=True,
                             inputs_to_skip=["onnx::MatMul_0"],
                             compile_torch_extension=use_cpp_dispatcher,
                             cuda=is_gpu(device))

    dy = torch.rand(8, 10, device=dev)

    with experimental_cuda():
        dace_output = dace_module(dace_input)
    pt_output = pt_module(pytorch_input)

    torch_tensors_close("output", pt_output, dace_output)

    # check that fc1.grad is being computed
    with experimental_cuda():
        dace_output.backward(dy)
    pt_output.backward(dy)
    torch_tensors_close("param_grad", pt_module.fc1.grad, dace_module.model.fc1.grad)

    # Make sure that input grad is not being computed
    assert len(dace_module.backward_sdfg.node(0).sink_nodes()) == 1, \
        f"Expected 1 sink node (no input gradient), got {len(dace_module.backward_sdfg.node(0).sink_nodes())}"


if __name__ == "__main__":
    test_skip_input_grads(use_cpp_dispatcher=True, device="cpu")
    test_skip_input_grads(use_cpp_dispatcher=False, device="cpu")
