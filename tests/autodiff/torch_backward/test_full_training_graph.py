# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import pytest

import numpy as np

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch

import dace

from dace.ml import DaceModule
from tests.utils import torch_tensors_close, tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device, run_sdfg


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_module(device):
    dev = torch_device(device)
    module = torch.nn.Sequential(torch.nn.Linear(12, 2, bias=False)).to(dev)

    torch_module = copy.deepcopy(module)
    dace_module = copy.deepcopy(module)

    dace_module = DaceModule(dace_module,
                             sdfg_name=f"test_full_training_graph_module_{device}",
                             simplify=False,
                             backward=True,
                             training=True,
                             auto_optimize=False,
                             cuda=is_gpu(device))

    x = torch.randn(8, 12, device=dev)

    expected_output = torch_module(x)
    with experimental_cuda():
        result = dace_module(x)
    torch_tensors_close('output', expected_output, result)

    with experimental_cuda():
        dc_loss = dace_module(x).sum()
        dc_loss.backward()

    pt_loss = torch_module(x).sum()
    pt_loss.backward()

    tensors_close("loss", pt_loss, dc_loss)
    assert all(hasattr(p, 'grad') and p.grad is not None for p in dace_module.parameters()), \
        "Not all parameters have gradients computed"

    for d, t in zip(dace_module.parameters(), torch_module.parameters()):
        torch_tensors_close("param", t.grad, d.grad)


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_parse_backward_simple(device):
    x = torch.randn(10, 5, dtype=torch.float64)
    dy = torch.randn(10, dtype=torch.float64)

    @dace.program
    def train_step(x: dace.float64[10, 5], dy: dace.float64[10]):
        x.requires_grad_()
        red = np.add.reduce(x, axis=1)
        torch.autograd.backward(red, dy)
        return x.grad

    sdfg = train_step.to_sdfg()
    sdfg.name = f"{sdfg.name}_parse_backward_simple_{device}"
    sdfg.expand_library_nodes()
    sdfg.validate()

    result = run_sdfg(sdfg, device, x=x.clone(), dy=dy.clone())
    tensors_close('x.grad', dy.reshape(10, 1).expand(10, 5), result)


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_parse_backward_scalar(device):
    x = torch.randn(10, 5, dtype=torch.float64)

    @dace.program
    def train_step(x: dace.float64[10, 5]):
        x.requires_grad_()
        red = np.add.reduce(x, axis=[0, 1])
        torch.autograd.backward(red)
        return x.grad

    sdfg = train_step.to_sdfg()
    sdfg.name = f"{sdfg.name}_parse_backward_scalar_{device}"
    sdfg.expand_library_nodes()
    sdfg.validate()

    result = run_sdfg(sdfg, device, x=x.clone())
    tensors_close('x.grad', 1, result)


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_parse_backward_with_forwarding(device):
    x = torch.randn(10, 5, dtype=torch.float64)
    dy = torch.randn(10, dtype=torch.float64)

    @dace.program
    def train_step(x: dace.float64[10, 5]):
        x.requires_grad_()
        y = x + 1
        red = np.add.reduce(x, axis=1, keepdims=True)
        z = red * y
        loss = np.add.reduce(z, axis=[0, 1])
        torch.autograd.backward(loss)
        return x.grad

    def torch_fn(x):
        x.requires_grad_()
        y = x + 1
        red = x.sum(axis=1, keepdims=True)
        z = red * y
        loss = z.sum()
        loss.backward()
        return x.grad

    sdfg = train_step.to_sdfg()
    sdfg.name = f"{sdfg.name}_parse_backward_with_forwarding_{device}"
    sdfg.expand_library_nodes()
    sdfg.validate()

    result = run_sdfg(sdfg, device, x=x.clone())
    expected = torch_fn(x.clone())
    tensors_close('x.grad', expected, result)


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_two_backward_passes(device):

    @dace.program
    def train_step(x1: dace.float64[10, 5], x2: dace.float64[5], dy: dace.float64[10]):
        x1.requires_grad_()
        x2.requires_grad_()

        z1 = x1 + 1
        y1 = np.log(z1)
        l1 = np.add.reduce(y1, axis=1)

        z2 = x2 * 2
        y2 = np.log(z2)
        l2 = y2.sum()

        l2.backward()
        l1.backward(dy)
        return x1.grad, x2.grad

    def torch_fn(x1, x2, dy):
        x1.requires_grad_()
        x2.requires_grad_()
        z1 = x1 + 1
        y1 = torch.log(z1).sum(axis=1)

        z2 = x2 * 2
        y2 = torch.log(z2).sum()
        y2.backward()
        y1.backward(dy)
        return x1.grad, x2.grad

    sdfg = train_step.to_sdfg()
    sdfg.name = f"{sdfg.name}_two_backward_passes_{device}"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    x1 = torch.randn(10, 5, dtype=torch.float64)
    x2 = torch.randn(5, dtype=torch.float64)
    dy = torch.randn(10, dtype=torch.float64)

    r1, r2 = run_sdfg(sdfg, device, x1=x1.clone(), x2=x2.clone(), dy=dy.clone())
    ex_1, ex_2 = torch_fn(x1.clone(), x2.clone(), dy.clone())
    tensors_close('x2.grad', ex_2, r2)
    tensors_close('x1.grad', ex_1, r1)


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_two_backward_passes_accumulate(device):

    @dace.program
    def train_step(x: dace.float64[10, 5], dy: dace.float64[10]):
        x.requires_grad_()

        z1 = x + 1
        y1 = np.log(z1)
        l1 = np.add.reduce(y1, axis=1)

        z2 = x * 2
        y2 = np.log(z2)
        l2 = y2.sum()

        l2.backward()
        l1.backward(dy)
        return x.grad

    def torch_fn(x, dy):
        x.requires_grad = True
        z1 = x + 1
        y1 = torch.log(z1).sum(axis=1)

        z2 = x * 2
        y2 = torch.log(z2).sum()
        y2.backward()
        y1.backward(dy)
        return x.grad

    sdfg = train_step.to_sdfg()
    sdfg.name = f"{sdfg.name}_two_backward_passes_accumulate_{device}"
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    x1 = torch.randn(10, 5, dtype=torch.float64)
    dy = torch.randn(10, dtype=torch.float64)

    result = run_sdfg(sdfg, device, x=x1.clone(), dy=dy.clone())
    expected = torch_fn(x1.clone(), dy.clone())

    tensors_close('x.grad', expected, result)


if __name__ == "__main__":
    test_module(device="cpu")
    test_parse_backward_simple(device="cpu")
    test_parse_backward_scalar(device="cpu")
    test_parse_backward_with_forwarding(device="cpu")
    test_two_backward_passes(device="cpu")
    test_two_backward_passes_accumulate(device="cpu")
