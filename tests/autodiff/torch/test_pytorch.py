# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import copy

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
import torch.nn.functional as F

from dace.ml import DaceModule
from tests.utils import torch_tensors_close


@pytest.mark.torch
@pytest.mark.autodiff
def run_pytorch_module(
    module: torch.nn.Module,
    sdfg_name: str,
    shape: tuple = None,
    use_max: bool = False,
    auto_optimize: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-3,
    post_onnx_hooks: list = None,
):
    shape = shape or (3, 5)

    pt_model_for_dace = copy.deepcopy(module)

    input_value = torch.rand(*shape, dtype=torch.float32)

    pytorch_input = torch.empty(
        *shape,
        dtype=torch.float32,
        requires_grad=False,
    )
    pytorch_input.copy_(input_value)

    dace_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False)
    dace_input.copy_(input_value)

    pytorch_input.requires_grad = True
    dace_input.requires_grad = True

    if use_max:
        pytorch_s = module(pytorch_input).max()
    else:
        pytorch_s = module(pytorch_input).sum()
    pytorch_s.backward()

    dace_module = DaceModule(
        pt_model_for_dace,
        simplify=False,
        backward=True,
        sdfg_name=sdfg_name,
        auto_optimize=auto_optimize,
        compile_torch_extension=True,
    )
    if post_onnx_hooks is not None:
        for i, h in enumerate(post_onnx_hooks):
            dace_module.append_post_onnx_hook(str(i), h)

    if use_max:
        dace_s = dace_module(dace_input).max()
    else:
        dace_s = dace_module(dace_input).sum()
    dace_s.backward()
    torch_tensors_close("grad", pytorch_input.grad, dace_input.grad, rtol=rtol, atol=atol)

    for (name, dace_param), (pt_name, pt_param) in zip(module.named_parameters(), dace_module.named_parameters()):
        assert 'model.' + name == pt_name
        torch_tensors_close(name, pt_param.grad, dace_param.grad, rtol=rtol, atol=atol)


@pytest.mark.torch
@pytest.mark.autodiff
def test_simple(sdfg_name: str):

    class Module(torch.nn.Module):

        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.log(x)
            return x

    run_pytorch_module(Module(), sdfg_name)


@pytest.mark.torch
@pytest.mark.autodiff
def test_repeated(sdfg_name):

    class Module(torch.nn.Module):

        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.sqrt(x)
            return x

    run_pytorch_module(Module(), sdfg_name)


@pytest.mark.torch
@pytest.mark.autodiff
def test_softmax(sdfg_name):

    class Module(torch.nn.Module):

        def forward(self, x):
            x = F.softmax(x, dim=1)
            return x

    run_pytorch_module(Module(), sdfg_name, use_max=True)


@pytest.mark.torch
@pytest.mark.autodiff
def test_reshape_on_memlet_path(sdfg_name):
    # required test: this function in a nn.Module, with apply simplify so that the reshape is
    # inlined and copy is removed
    class Module(torch.nn.Module):

        def forward(self, x):
            reshaped = torch.reshape(x + 1, [3, 3])
            return torch.log(reshaped) + torch.reshape(torch.tensor([[3, 2, 1]], device=reshaped.device), [3])

    run_pytorch_module(Module(), sdfg_name, shape=(9, ))


@pytest.mark.torch
@pytest.mark.autodiff
def test_weights_ln(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 32)
            self.ln = nn.LayerNorm(32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.ln(x)
            x = self.fc3(x)
            return x

    run_pytorch_module(Module(), sdfg_name, shape=(4, 784), auto_optimize=False)


@pytest.mark.torch
@pytest.mark.autodiff
def test_layernorm(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.ln = nn.LayerNorm(3)

        def forward(self, x):
            return self.ln(x)

    run_pytorch_module(Module(), sdfg_name, shape=(2, 3), use_max=True, atol=1e-2)


@pytest.mark.torch
@pytest.mark.autodiff
def test_weights(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    run_pytorch_module(Module(), sdfg_name, shape=(4, 784), use_max=False, auto_optimize=False)


@pytest.mark.torch
@pytest.mark.autodiff
def test_nested_gradient_summation(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.rand(10, 10))

        def forward(self, x):
            y = x @ self.fc1
            z = x * 2
            return z + y

    run_pytorch_module(Module(), sdfg_name, shape=(4, 10), use_max=False, auto_optimize=False)


@pytest.mark.torch
@pytest.mark.autodiff
def test_trans_add(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()

        def forward(self, x):
            x = x + 1
            x = torch.transpose(x.reshape(4, 4), 1, 0)
            return x

    run_pytorch_module(Module(), sdfg_name, shape=(16, ), use_max=False)


@pytest.mark.torch
@pytest.mark.autodiff
def test_batched_matmul(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.ones([10, 5, 3]))

        def forward(self, x):
            return self.fc1 @ x

    run_pytorch_module(Module(), sdfg_name, use_max=False, auto_optimize=False)


@pytest.mark.torch
@pytest.mark.autodiff
def test_scalar_forwarding(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.factor = nn.Parameter(torch.ones(()))

        def forward(self, x):
            return self.factor * x

    run_pytorch_module(Module(), sdfg_name, use_max=False, auto_optimize=False)


@pytest.mark.torch
@pytest.mark.autodiff
def test_scalar_buffer(sdfg_name):

    class Module(torch.nn.Module):

        def __init__(self):
            super(Module, self).__init__()
            self.register_buffer("factor", torch.tensor(2))

        def forward(self, x):
            return self.factor * x

    run_pytorch_module(Module(), sdfg_name, use_max=False)


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.skip(reason="Requires pure implementation of expand")
def test_simple_broadcasted_mul(sdfg_name):

    class Module(torch.nn.Module):

        def forward(self, x):
            y = x.sum(axis=0)
            return x * y

    run_pytorch_module(Module(), sdfg_name)


if __name__ == "__main__":
    test_simple(sdfg_name="test_simple")
    test_repeated(sdfg_name="test_repeated")
    test_softmax(sdfg_name="test_softmax")
    test_reshape_on_memlet_path(sdfg_name="test_reshape_on_memlet_path")
    test_weights_ln(sdfg_name="test_weights_ln")
    test_layernorm(sdfg_name="test_layernorm")
    test_weights(sdfg_name="test_weights")
    test_nested_gradient_summation(sdfg_name="test_nested_gradient_summation")
    test_trans_add(sdfg_name="test_trans_add")
    test_batched_matmul(sdfg_name="test_batched_matmul")
    test_scalar_forwarding(sdfg_name="test_scalar_forwarding")
    test_scalar_buffer(sdfg_name="test_scalar_buffer")
    # test_simple_broadcasted_mul is skipped
