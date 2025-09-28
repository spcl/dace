"""
Test a full model including indexing and input preparation. The model also includes lots of symbolic dimensions.
"""

import os

import onnx
import onnxsim

import pytest

import torch
from transformers import BertTokenizer, BertModel

import dace.libraries.onnx as donnx
from dace.testing import torch_tensors_close, get_data_file


def test_bert_full(sdfg_name):
    bert_tiny_root = 'http://spclstorage.inf.ethz.ch/~rauscho/bert-tiny'
    get_data_file(bert_tiny_root + "/config.json", directory_name='bert-tiny')
    vocab = get_data_file(bert_tiny_root + "/vocab.txt", directory_name='bert-tiny')
    bert_path = get_data_file(bert_tiny_root + "/bert-tiny.onnx", directory_name='bert-tiny')
    get_data_file(bert_tiny_root + "/pytorch_model.bin", directory_name='bert-tiny')
    model_dir = os.path.dirname(vocab)

    tokenizer = BertTokenizer.from_pretrained(vocab)
    pt_model = BertModel.from_pretrained(model_dir)

    text = "[CLS] how are you today [SEP] dude [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [0] * 6 + [1] * 2

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    attention_mask = torch.ones(1, 8, dtype=torch.int64)

    model = onnx.load(bert_path)
    # infer shapes
    model, check = onnxsim.simplify(model,
                                    skip_fuse_bn=True,
                                    input_shapes=dict(input_ids=tokens_tensor.shape,
                                                      token_type_ids=segments_tensors.shape,
                                                      attention_mask=attention_mask.shape))

    dace_model = donnx.ONNXModel(sdfg_name, model, auto_merge=True)

    dace_output = dace_model(input_ids=tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)

    output = pt_model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)

    torch_tensors_close("output_0", output[0], dace_output[0])
    torch_tensors_close("output_1", output[1], dace_output[1])
