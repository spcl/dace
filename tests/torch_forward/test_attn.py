# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch

from dace.ml import DaceModule

from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device


@pytest.mark.torch
@pytest.mark.parametrize("device", DEVICES)
def test_attn(use_cpp_dispatcher: bool, device):
    dev = torch_device(device)
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    K, Q, V = [torch.randn([SM, B, N]).to(dev), torch.randn([SN, B, N]).to(dev), torch.randn([SM, B, N]).to(dev)]
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False).to(dev)

    pt_outputs = ptmodel(Q, K, V)

    dispatcher_suffix = "cpp" if use_cpp_dispatcher else "ctypes"
    dace_model = DaceModule(ptmodel,
                            sdfg_name=f"test_attn_{dispatcher_suffix}_{device}",
                            compile_torch_extension=use_cpp_dispatcher,
                            auto_optimize=False,
                            cuda=is_gpu(device))

    with experimental_cuda():
        dace_outputs = dace_model(Q, K, V)

    torch_tensors_close("outputs_0", pt_outputs[0], dace_outputs[0])
    torch_tensors_close("outputs_1", pt_outputs[1], dace_outputs[1])


if __name__ == "__main__":
    test_attn(use_cpp_dispatcher=True, device="cpu")
    test_attn(use_cpp_dispatcher=False, device="cpu")
