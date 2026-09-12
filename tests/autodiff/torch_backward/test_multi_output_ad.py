# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch

from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_multi_output(use_cpp_dispatcher: bool, device):

    dev = torch_device(device)

    class Module(torch.nn.Module):

        def forward(self, x):
            return x + 1, x * 2

    module = Module().to(dev)

    input_value = torch.rand(5, 10, dtype=torch.float32, device=dev)

    pytorch_input = torch.empty(
        5,
        10,
        dtype=torch.float32,
        requires_grad=False,
        device=dev,
    )
    pytorch_input.copy_(input_value)

    dace_input = torch.empty(5, 10, dtype=torch.float32, requires_grad=False, device=dev)
    dace_input.copy_(input_value)

    pytorch_input.requires_grad = True
    dace_input.requires_grad = True

    torch_dy = torch.randn(5, 10, dtype=torch.float32, device=dev)
    dace_dy = torch_dy.clone()

    pytorch_y1, pytorch_y2 = module(pytorch_input)

    pytorch_y1.backward(torch_dy)
    pytorch_y2.backward(torch_dy)

    dispatcher_suffix = "cpp" if use_cpp_dispatcher else "ctypes"
    dace_module = DaceModule(
        module,
        sdfg_name=f"test_multi_output_ad_{dispatcher_suffix}_{device}",
        backward=True,
        compile_torch_extension=use_cpp_dispatcher,
        cuda=is_gpu(device),
    )

    with experimental_cuda():
        dace_y1, dace_y2 = dace_module(dace_input)

        dace_y1.backward(dace_dy, retain_graph=True)
        dace_y2.backward(dace_dy)

    torch_tensors_close("grad", pytorch_input.grad, dace_input.grad)


if __name__ == "__main__":
    test_multi_output(use_cpp_dispatcher=True, device="cpu")
    test_multi_output(use_cpp_dispatcher=False, device="cpu")
