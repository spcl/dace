# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch

from dace.ml import DaceModule

from dace.transformation.dataflow import RedundantSecondArray
from tests.utils import torch_tensors_close


@pytest.mark.torch
def test_attn(use_cpp_dispatcher: bool):
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    K, Q, V = [torch.randn([SM, B, N]), torch.randn([SN, B, N]), torch.randn([SM, B, N])]
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

    pt_outputs = ptmodel(Q, K, V)

    dispatcher_suffix = "cpp" if use_cpp_dispatcher else "ctypes"
    dace_model = DaceModule(ptmodel,
                            sdfg_name=f"test_attn_{dispatcher_suffix}",
                            compile_torch_extension=use_cpp_dispatcher,
                            auto_optimize=False)

    dace_outputs = dace_model(Q, K, V)

    torch_tensors_close("outputs_0", pt_outputs[0], dace_outputs[0])
    torch_tensors_close("outputs_1", pt_outputs[1], dace_outputs[1])


if __name__ == "__main__":
    test_attn(use_cpp_dispatcher=True)
    test_attn(use_cpp_dispatcher=False)
