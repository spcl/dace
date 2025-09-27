import pytest
import numpy as np
import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close
from dace.transformation.onnx import parameter_to_transient


@pytest.mark.cpublas
def test_bert_encoder(default_implementation, sdfg_name):
    batch_size = 2
    seq_len = 32
    hidden_size = 48

    input = torch.randn([batch_size, seq_len, hidden_size])

    ptmodel = BertLayer(BertConfig(hidden_size=hidden_size, attn_implementation="eager")).eval()
    pt_outputs = ptmodel(input.clone())

    dace_model = DaceModule(ptmodel,
                            sdfg_name=sdfg_name,
                            training=False,
                            simplify=False,
                            backward=True,
                            auto_optimize=False)

    dace_outputs0 = dace_model(input.clone())
    torch_tensors_close("output", pt_outputs[0], dace_outputs0)

    if default_implementation == "pure":
        ort_nodes = [
            n for n, _ in dace_model.sdfg.all_nodes_recursive()
            if hasattr(n, "environments") and any("onnx" in e.lower() for e in n.environments)
        ]
        if len(ort_nodes) > 0:
            assert False, f"expected pure graph, found ORT nodes: {ort_nodes} "


if __name__ == "__main__":
    torch.manual_seed(42)
    test_bert_encoder(default_implementation="pure", sdfg_name="bert_encoder")
