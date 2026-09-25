# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from torch import nn

from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


class Module(nn.Module):

    def forward(self, x):
        y = x + 3
        return y * 5


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_debug_transients(device):

    dev = torch_device(device)

    module = DaceModule(Module(),
                        debug_transients=True,
                        sdfg_name=f"test_debug_transients_{device}",
                        cuda=is_gpu(device))

    x = torch.rand(5, 5).to(dev)
    with experimental_cuda():
        outputs = module(x)
    output, y, y2 = outputs

    torch_tensors_close("output", (x + 3) * 5, output)
    torch_tensors_close("y2", (x + 3) * 5, y2)
    torch_tensors_close("y", x + 3, y)


if __name__ == "__main__":
    test_debug_transients(device="cpu")
