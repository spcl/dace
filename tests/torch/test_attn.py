# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch

from dace.frontend.python.module import DaceModule

from dace.transformation.dataflow import RedundantSecondArray
from tests.utils import torch_tensors_close


@pytest.mark.torch
def test_attn(sdfg_name: str, use_cpp_dispatcher: bool):
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    K, Q, V = [torch.randn([SM, B, N]), torch.randn([SN, B, N]), torch.randn([SM, B, N])]
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

    pt_outputs = ptmodel(Q, K, V)

    dace_model = DaceModule(ptmodel,
                            sdfg_name=sdfg_name,
                            compile_torch_extension=use_cpp_dispatcher,
                            auto_optimize=False)

    dace_outputs = dace_model(Q, K, V)

    torch_tensors_close("outputs_0", pt_outputs[0], dace_outputs[0])
    torch_tensors_close("outputs_1", pt_outputs[1], dace_outputs[1])


if __name__ == "__main__":
    test_attn(sdfg_name="test_attn_cpp_True", use_cpp_dispatcher=True)
    test_attn(sdfg_name="test_attn_cpp_False", use_cpp_dispatcher=False)
