# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Test a full model including indexing and input preparation. The model also includes lots of symbolic dimensions.
"""

import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
pytest.importorskip("onnxsim", reason="ONNX Simplifier not installed. Please install with: pip install dace[ml]")
pytest.importorskip("transformers",
                    reason="transformers not installed. Please install with: pip install dace[ml-testing]")
import os
import tempfile

import onnx
import onnxsim

import torch
from transformers import BertTokenizer, BertModel

import dace.libraries.onnx as donnx
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device

BERT_TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"


class _BertONNXExportWrapper(torch.nn.Module):
    """ Fixes the forward kwargs: the ONNX tracer passes BertModel's use_cache default positionally otherwise. """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            use_cache=False,
                            return_dict=False)
        return output[0], output[1]


@pytest.mark.xdist_group("large_ML_models")
@pytest.mark.onnx
@pytest.mark.parametrize("device", DEVICES)
def test_bert_full(device):
    tokenizer = BertTokenizer.from_pretrained(BERT_TINY_MODEL)
    pt_model = BertModel.from_pretrained(BERT_TINY_MODEL)
    pt_model.eval()

    text = "[CLS] how are you today [SEP] dude [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [0] * 6 + [1] * 2

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    # a 4D mask is passed through as-is; the mask factory reads traced shapes, which the ONNX tracer cannot handle
    attention_mask = torch.zeros(1, 1, 1, 8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bert_path = os.path.join(tmp_dir, "bert-tiny.onnx")
        # eval(): the exporter restores the wrapper's mode afterwards, which would turn dropout back on
        torch.onnx.export(_BertONNXExportWrapper(pt_model).eval(), (tokens_tensor, attention_mask, segments_tensors),
                          bert_path,
                          input_names=["input_ids", "attention_mask", "token_type_ids"],
                          output_names=["output_0", "output_1"],
                          opset_version=14,
                          dynamo=False)

        model = onnx.load(bert_path)
        # infer shapes
        model, _ = onnxsim.simplify(model,
                                    skip_fuse_bn=True,
                                    input_shapes=dict(input_ids=tokens_tensor.shape,
                                                      token_type_ids=segments_tensors.shape,
                                                      attention_mask=attention_mask.shape))

    dace_model = donnx.ONNXModel(f"test_bert_full_{device}", model, auto_merge=True, cuda=is_gpu(device))

    with experimental_cuda():
        dace_output = dace_model(input_ids=tokens_tensor.to(torch_device(device)),
                                 token_type_ids=segments_tensors.to(torch_device(device)),
                                 attention_mask=attention_mask.to(torch_device(device)))

    output = pt_model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)

    if is_gpu(device):
        dace_output = [t.cpu() for t in dace_output]

    torch_tensors_close("output_0", output[0], dace_output[0])
    torch_tensors_close("output_1", output[1], dace_output[1])


if __name__ == "__main__":
    test_bert_full("cpu")
