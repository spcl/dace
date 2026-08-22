# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import numpy as np
import torch
from torch import nn

import dace

from dace.ml import DaceModule
from tests.utils import tensors_close, torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_parse_forward_simple(device):
    dev = torch_device(device)
    torch_module = torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Linear(24, 2)).to(dev)
    dace_module = DaceModule(torch_module, sdfg_name=f'test_parse_forward_simple_{device}', cuda=is_gpu(device))
    x = torch.randn(2, 12).to(dev)
    expected = torch_module(x)
    with experimental_cuda():
        result = dace_module(x)

    torch_tensors_close('output', expected, result)

    @dace
    def train_step(y):
        # output is potentially a gpu tensor
        output = dace_module(y)
        cpu = np.empty_like(output)
        cpu[:] = output
        return cpu.sum()

    with experimental_cuda():
        result = train_step(x)
    tensors_close('parsed', expected.sum(), result)


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_parse_forward_nested(device):

    dev = torch_device(device)
    torch_module = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Linear(24, 2)),
                                       nn.Softmax(dim=1)).to(dev)
    dace_module2 = DaceModule(torch_module, sdfg_name=f'test_parse_forward_nested_{device}', cuda=is_gpu(device))
    x = torch.randn(2, 12).to(dev)
    expected = torch_module(x)
    with experimental_cuda():
        result = dace_module2(x)

    torch_tensors_close('output', expected, result)

    @dace
    def train_step(y):
        output = dace_module2(y)
        cpu = np.empty_like(output)
        cpu[:] = output
        return cpu.sum()

    with experimental_cuda():
        result = train_step(x)
    tensors_close('parsed', expected.sum(), result)


if __name__ == "__main__":
    test_parse_forward_simple(device="cpu")
    test_parse_forward_nested(device="cpu")
