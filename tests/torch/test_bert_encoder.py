import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import pytest
import numpy as np
import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from dace.frontend.python.module import DaceModule
from dace.transformation.onnx import parameter_to_transient
from tests.utils import torch_tensors_close


@pytest.mark.torch
def test_bert_encoder(sdfg_name: str):
    batch_size = 2
    seq_len = 32
    hidden_size = 24

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

    ort_nodes = [
        n for n, _ in dace_model.sdfg.all_nodes_recursive()
        if hasattr(n, "environments") and any("onnx" in e.lower() for e in n.environments)
    ]
    assert len(ort_nodes) == 0, f"Expected pure graph, found {len(ort_nodes)} ORT nodes: {ort_nodes}"


if __name__ == "__main__":
    test_bert_encoder(sdfg_name="test_bert_encoder")
