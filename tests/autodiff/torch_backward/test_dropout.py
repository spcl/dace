# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Literal, Union
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from torch import nn
from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_dropout_fwd_training(device):
    dev = torch_device(device)
    p = 0.5
    module = nn.Dropout(p=p).train().to(dev)
    # dummy_inputs triggers compilation in the constructor -> keep it under experimental_cuda()
    with experimental_cuda():
        dace_module = DaceModule(module,
                                 sdfg_name=f"test_dropout_fwd_training_{device}",
                                 dummy_inputs=(torch.ones(10, 10, device=dev), ),
                                 training=True,
                                 cuda=is_gpu(device))

    # dropout will set some of these to zero
    test_data = torch.randint(1, 10, (10, 10)).float().to(dev)

    with experimental_cuda():
        out = dace_module(torch.clone(test_data))
    zeroed = out == 0

    scale = 1 / (1 - p)
    torch_tensors_close("output", test_data[~zeroed] * scale, out[~zeroed])


@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p", [0, 0.99, 0.6, 0.5])
def test_dropout_bwd(p: Union[float, Literal[0]], device):
    dev = torch_device(device)
    module = nn.Dropout(p=p).train().to(dev)
    sdfg_name = f"test_dropout_{str(p).replace('.', '_')}_bwd_{device}"
    # dummy_inputs triggers compilation in the constructor -> keep it under experimental_cuda()
    with experimental_cuda():
        dace_module = DaceModule(module,
                                 sdfg_name=sdfg_name,
                                 dummy_inputs=(torch.ones(10, 10, device=dev), ),
                                 backward=True,
                                 training=True,
                                 cuda=is_gpu(device))

    test_data = torch.randint(1, 10, (10, 10)).float().to(dev)
    test_data.requires_grad = True
    dy = torch.rand_like(test_data)

    with experimental_cuda():
        out = dace_module(torch.clone(test_data))

        zeroed = out == 0
        scale = 1 / (1 - p)
        # check that fwd was correct
        torch_tensors_close("output", test_data[~zeroed] * scale, out[~zeroed])

        out.backward(dy)

    # check that the gradient is correct:
    zeros = torch.zeros_like(test_data.grad)
    # check that zeroed values are zero in the grad
    torch_tensors_close("grad_zeroed", zeros[zeroed], test_data.grad[zeroed])

    # check that non-zeroed values are correct
    torch_tensors_close("grad_zeroed", dy[~zeroed] * scale, test_data.grad[~zeroed])


if __name__ == "__main__":
    test_dropout_fwd_training(device="cpu")
    # Test with different dropout probabilities
    for p in [0, 0.99, 0.6, 0.5]:
        test_dropout_bwd(p=p, device="cpu")
