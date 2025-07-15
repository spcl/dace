import pytest
import numpy as np
import torch
from transformers import BertConfig, BertLayer

from dace.frontend.python.module import DaceModule
from dace.testing import copy_to_gpu, torch_tensors_close
from dace.transformation.onnx import parameter_to_transient


@pytest.mark.cpublas
def test_bert_encoder(gpu, default_implementation, sdfg_name):
    batch_size = 2
    seq_len = 32
    hidden_size = 48

    input = copy_to_gpu(gpu, torch.randn([batch_size, seq_len, hidden_size]))

    ptmodel = copy_to_gpu(gpu, BertLayer(BertConfig(hidden_size=hidden_size)).eval())
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel,
                            training=False,
                            sdfg_name=sdfg_name,
                            simplify=True)

    if gpu:

        def param_to_trans(model):
            for name, _ in model.model.named_parameters():
                parameter_to_transient(model, name)

        dace_model.append_post_onnx_hook("param_to_transient", param_to_trans)

    dace_outputs0 = dace_model(input.clone())
    torch_tensors_close("output", pt_outputs[0], dace_outputs0)

    if default_implementation == "pure":
        ort_nodes = [
            n for n, _ in dace_model.sdfg.all_nodes_recursive()
            if hasattr(n, "environments") and any("onnx" in e.lower()
                                                  for e in n.environments)
        ]
        if len(ort_nodes) > 0:
            assert False, f"expected pure graph, found ORT nodes: {ort_nodes} "

        # check that cuBLAS is being used
        if gpu:
            assert any(
                (hasattr(n, "environments") and "cuBLAS" in n.environments or
                 hasattr(n, "implementation") and n.implementation == "cuBLAS")
                for n, _ in dace_model.sdfg.all_nodes_recursive())
