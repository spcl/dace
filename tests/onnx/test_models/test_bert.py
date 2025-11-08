# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Test a full model including indexing and input preparation. The model also includes lots of symbolic dimensions.
"""

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import os

import pathlib
import urllib

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from dace.frontend.ml.torch.module import DaceModule
import dace
from tests.utils import torch_tensors_close


def get_data_file(url, directory_name=None) -> str:
    """ Get a data file from ``url``, cache it locally and return the local file path to it.

        :param url: the url to download from.
        :param directory_name: an optional relative directory path where the file will be downloaded to.
        :returns: the path of the downloaded file.
    """

    data_directory = (pathlib.Path(dace.__file__).parent.parent / 'tests' / 'data')

    if directory_name is not None:
        data_directory /= directory_name

    data_directory.mkdir(exist_ok=True, parents=True)

    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    file_path = str(data_directory / file_name)

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    return file_path


class BertModelWrapper(nn.Module):

    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output


@pytest.mark.torch
def test_bert_full(sdfg_name):
    bert_tiny_root = 'http://spclstorage.inf.ethz.ch/~rauscho/bert-tiny'
    get_data_file(bert_tiny_root + "/config.json", directory_name='bert-tiny')
    vocab = get_data_file(bert_tiny_root + "/vocab.txt", directory_name='bert-tiny')
    get_data_file(bert_tiny_root + "/pytorch_model.bin", directory_name='bert-tiny')
    model_dir = os.path.dirname(vocab)

    tokenizer = BertTokenizer.from_pretrained(vocab)
    pt_model = BertModel.from_pretrained(model_dir)
    pt_model.eval()

    text = "[CLS] how are you today [SEP] dude [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [0] * 6 + [1] * 2

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    attention_mask = torch.ones(1, 8, dtype=torch.int64)

    wrapped_model = BertModelWrapper(pt_model)

    dace_model = DaceModule(
        wrapped_model,
        sdfg_name=sdfg_name,
        simplify=True,
        backward=False,
    )

    tokens_tensor_pt = torch.clone(tokens_tensor)
    segments_tensors_pt = torch.clone(segments_tensors)
    attention_mask_pt = torch.clone(attention_mask)

    tokens_tensor_dace = torch.clone(tokens_tensor)
    segments_tensors_dace = torch.clone(segments_tensors)
    attention_mask_dace = torch.clone(attention_mask)

    with torch.no_grad():
        output = wrapped_model(tokens_tensor_pt, segments_tensors_pt, attention_mask_pt)

    dace_output = dace_model(tokens_tensor_dace, segments_tensors_dace, attention_mask_dace)

    torch_tensors_close("last_hidden_state", output[0], dace_output[0])
    torch_tensors_close("pooler_output", output[1], dace_output[1])


if __name__ == "__main__":
    test_bert_full(sdfg_name="test_bert_full")
