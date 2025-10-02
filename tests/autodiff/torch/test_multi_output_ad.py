import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch

from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close


@pytest.mark.torch
@pytest.mark.autodiff
def test_multi_output(sdfg_name: str, use_cpp_dispatcher: bool):

    class Module(torch.nn.Module):

        def forward(self, x):
            return x + 1, x * 2

    module = Module()

    input_value = torch.rand(5, 10, dtype=torch.float32)

    pytorch_input = torch.empty(
        5,
        10,
        dtype=torch.float32,
        requires_grad=False,
    )
    pytorch_input.copy_(input_value)

    dace_input = torch.empty(5, 10, dtype=torch.float32, requires_grad=False)
    dace_input.copy_(input_value)

    pytorch_input.requires_grad = True
    dace_input.requires_grad = True

    torch_dy = torch.randn(5, 10, dtype=torch.float32)
    dace_dy = torch_dy.clone()

    pytorch_y1, pytorch_y2 = module(pytorch_input)

    pytorch_y1.backward(torch_dy)
    pytorch_y2.backward(torch_dy)

    dace_module = DaceModule(
        module,
        backward=True,
        sdfg_name=sdfg_name,
        compile_torch_extension=use_cpp_dispatcher,
    )

    dace_y1, dace_y2 = dace_module(dace_input)

    dace_y1.backward(dace_dy, retain_graph=True)
    dace_y2.backward(dace_dy)

    torch_tensors_close("grad", pytorch_input.grad, dace_input.grad)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
