"""
Test a full model including indexing and input preparation. The model also includes lots of symbolic dimensions.
"""

import pytest

pytest.importorskip("onnx", reason="ONNX not installed. Please install with: pip install dace[ml]")
pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
pytest.importorskip("onnxsim", reason="ONNX Simplifier not installed. Please install with: pip install dace[ml]")
import os

import onnx
import onnxsim
import pathlib
import urllib

import torch
from transformers import BertTokenizer, BertModel

import dace
import dace.libraries.onnx as donnx
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


@pytest.mark.onnx
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
    model, _ = onnxsim.simplify(model,
                                skip_fuse_bn=True,
                                input_shapes=dict(input_ids=tokens_tensor.shape,
                                                  token_type_ids=segments_tensors.shape,
                                                  attention_mask=attention_mask.shape))

    dace_model = donnx.ONNXModel(sdfg_name, model, auto_merge=True)

    dace_output = dace_model(input_ids=tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)

    output = pt_model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)

    torch_tensors_close("output_0", output[0], dace_output[0])
    torch_tensors_close("output_1", output[1], dace_output[1])


if __name__ == "__main__":
    test_bert_full(sdfg_name="test_bert_full")
