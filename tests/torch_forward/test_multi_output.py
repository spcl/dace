# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
from torch import nn

from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


class Model(nn.Module):

    def __init__(self, new_shape):
        super(Model, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x + 1, x + 2


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_multiple_outputs(use_cpp_dispatcher: bool, device):

    dev = torch_device(device)

    ptmodel = Model([5, 5]).to(dev)
    x = torch.rand([25]).to(dev)

    torch_outputs = ptmodel(torch.clone(x))

    dispatcher_suffix = "cpp" if use_cpp_dispatcher else "ctypes"
    dace_model = DaceModule(ptmodel,
                            sdfg_name=f"test_multi_output_{dispatcher_suffix}_{device}",
                            auto_optimize=False,
                            compile_torch_extension=use_cpp_dispatcher,
                            cuda=is_gpu(device))

    with experimental_cuda():
        dace_outputs = dace_model(x)

    torch_tensors_close("output_0", torch_outputs[0], dace_outputs[0])
    torch_tensors_close("output_1", torch_outputs[1], dace_outputs[1])


if __name__ == "__main__":
    test_multiple_outputs(use_cpp_dispatcher=True, device="cpu")
    test_multiple_outputs(use_cpp_dispatcher=False, device="cpu")
