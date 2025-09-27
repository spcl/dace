import pytest

import numpy as np
import torch
from torch import nn

import dace

from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close, copy_to_gpu, tensors_close


def test_parse_forward_simple(gpu):
    torch_module = copy_to_gpu(gpu, torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Linear(24, 2)))
    dace_module = DaceModule(torch_module, sdfg_name='test_parse_forward_simple')
    x = copy_to_gpu(gpu, torch.randn(2, 12))
    expected = torch_module(x)
    result = dace_module(x)

    torch_tensors_close('output', expected, result)

    @dace
    def train_step(y):
        # output is potentially a gpu tensor
        output = dace_module(y)
        cpu = np.empty_like(output)
        cpu[:] = output
        return cpu.sum()

    result = train_step(x)
    tensors_close('parsed', expected.sum(), result)


def test_parse_forward_nested(gpu):

    torch_module = copy_to_gpu(
        gpu, torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Linear(24, 2)),
                                 nn.Softmax(dim=1)))
    dace_module2 = DaceModule(torch_module, sdfg_name='test_parse_forward_nested')
    x = copy_to_gpu(gpu, torch.randn(2, 12))
    expected = torch_module(x)
    result = dace_module2(x)

    torch_tensors_close('output', expected, result)

    @dace
    def train_step(y):
        output = dace_module2(y)
        cpu = np.empty_like(output)
        cpu[:] = output
        return cpu.sum()

    result = train_step(x)
    tensors_close('parsed', expected.sum(), result)
