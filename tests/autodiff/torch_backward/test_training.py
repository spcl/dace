# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import os
import copy
import pytest

import numpy as np

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
pytest.importorskip("transformers",
                    reason="transformers not installed. Please install with: pip install dace[ml-testing]")
import torch
from torch import nn, optim
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from dace.ml import DaceModule
from tests.utils import torch_tensors_close


def training_step(
    dace_model: torch.nn.Module,
    pt_model: torch.nn.Module,
    train_batch: tuple,
    sdfg_name: str,
    train_criterion: torch.nn.Module = None,
):

    # Copy over the weights
    dace_model.load_state_dict(pt_model.state_dict())
    for dace_value, value in zip(pt_model.state_dict().values(), dace_model.state_dict().values()):
        assert torch.allclose(dace_value, value), "State dict copy verification failed"

    dace_model = DaceModule(dace_model, sdfg_name=sdfg_name, backward=True, simplify=True, training=True)

    x, y = train_batch

    train_criterion = train_criterion or nn.NLLLoss()

    pt_loss = train_criterion(pt_model(x), y)

    dace_output = dace_model(x)
    dace_loss = train_criterion(dace_output, y)

    diff = abs(pt_loss.item() - dace_loss.item()) / pt_loss.item()
    assert diff < 1e-5, f"Loss mismatch: relative difference {diff:.2e} exceeds tolerance 1e-5"

    pt_loss.backward()
    dace_loss.backward()

    for (name, dace_param), (pt_name, pt_param) in zip(pt_model.named_parameters(), dace_model.named_parameters()):
        assert 'model.' + name == pt_name, f"Parameter name mismatch: expected 'model.{name}', got '{pt_name}'"
        torch_tensors_close(name, pt_param.grad, dace_param.grad)

    optimizer = optim.SGD(pt_model.parameters(), lr=0.001)
    dace_optimizer = optim.SGD(dace_model.parameters(), lr=0.001)
    optimizer.step()
    dace_optimizer.step()

    for (name, dace_param), (pt_name, pt_param) in zip(pt_model.named_parameters(), dace_model.named_parameters()):
        assert 'model.' + name == pt_name, f"Parameter name mismatch after optimizer step: expected 'model.{name}', got '{pt_name}'"
        torch_tensors_close(name, pt_param.detach(), dace_param.detach())


@pytest.mark.torch
@pytest.mark.autodiff
def test_mnist():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # initialize modules
    # yapf: disable
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LayerNorm(output_size),
                          nn.LogSoftmax(dim=1))

    dace_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                               nn.ReLU(),
                               nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                               nn.ReLU(),
                               nn.Linear(hidden_sizes[1], output_size),
                               nn.LayerNorm(output_size),
                               nn.LogSoftmax(dim=1))

    # check forward pass using loss
    images = torch.randn(64, 784)
    labels = torch.randint(0, 10, [64], dtype=torch.long)

    training_step(dace_model, model, (images, labels), sdfg_name="test_mnist_training")

@pytest.mark.xdist_group("large_ML_models")
@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.skip(reason="Requires pure implementation of expand")
def test_bert():
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    class BertTokenSoftmaxClf(nn.Module):

        def __init__(self):
            super(BertTokenSoftmaxClf, self).__init__()
            self.bert = BertLayer(BertConfig(hidden_act="relu")).eval()
            self.sm = nn.LogSoftmax(dim=-1)

        def forward(self, x):
            embs = self.bert(x)[0]
            return self.sm(embs.sum(dim=-1))

    # check forward pass using loss
    input = torch.randn([batch_size, seq_len, hidden_size])
    labels = torch.tensor([0, 123], dtype=torch.long)

    training_step(BertTokenSoftmaxClf(), BertTokenSoftmaxClf(), (input, labels), sdfg_name="test_bert_training")


if __name__ == "__main__":
    test_mnist()
    # test_bert is skipped
