# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
pytest.importorskip("transformers",
                    reason="transformers not installed. Please install with: pip install dace[ml-testing]")
import torch
from torch import nn, optim
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from dace.ml import DaceModule
from tests.utils import torch_tensors_close

# DaCe and PyTorch round a float32 matmul differently, so a ReLU input closer to zero than the rounding noise (~1e-6)
# gets a different gradient mask in each and the weight gradients of that unit disagree by far more than the test
# tolerance. Inputs are drawn so that no ReLU input is that close to the kink.
RELU_KINK_MARGIN = 1e-4


def min_relu_input_magnitude(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Smallest absolute value that any ``nn.ReLU`` of ``model`` receives on input ``x``."""
    magnitudes = []
    hooks = [
        module.register_forward_pre_hook(lambda _, args: magnitudes.append(args[0].abs().min().item()))
        for module in model.modules() if isinstance(module, nn.ReLU)
    ]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return min(magnitudes)


def randn_away_from_relu_kink(model: torch.nn.Module, *shape: int) -> torch.Tensor:
    x = torch.randn(*shape)
    while min_relu_input_magnitude(model, x) < RELU_KINK_MARGIN:
        x = torch.randn(*shape)
    return x


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


def identity_then_relu(*biases: float) -> torch.nn.Module:
    layers = []
    for bias in biases:
        layer = nn.Linear(1, 1)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.fill_(bias)
        layers += [layer, nn.ReLU()]
    return nn.Sequential(*layers)


@pytest.mark.torch
@pytest.mark.parametrize("values, expected", [([0.5, -3e-7, 2.0], 3e-7), ([0.5, -2.0], 0.5)])
def test_the_relu_input_closest_to_zero_is_reported(values, expected):
    x = torch.tensor(values).reshape(-1, 1)
    assert min_relu_input_magnitude(identity_then_relu(0.0), x) == pytest.approx(expected, rel=1e-6)


@pytest.mark.torch
def test_every_relu_of_the_model_is_checked():
    """Only the second ReLU sees an input at the kink: ``relu(1.0) - 1.0 == 0``."""
    x = torch.tensor([[1.0]])
    assert min_relu_input_magnitude(identity_then_relu(0.0, -1.0), x) == 0.0


@pytest.mark.torch
def test_a_batch_with_a_relu_input_at_the_kink_is_redrawn(monkeypatch):
    batches = iter([torch.tensor([[3e-7], [0.5]]), torch.tensor([[0.5], [-2.0]])])
    monkeypatch.setattr(torch, "randn", lambda *shape: next(batches))
    x = randn_away_from_relu_kink(identity_then_relu(0.0), 2, 1)
    assert x.flatten().tolist() == [0.5, -2.0]


@pytest.mark.torch
@pytest.mark.autodiff
def test_mnist():
    # Seed 607 draws a batch with a ReLU input 3e-7 away from zero, which flips the gradient mask in DaCe.
    torch.manual_seed(607)
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
    images = randn_away_from_relu_kink(model, 64, 784)
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
    test_the_relu_input_closest_to_zero_is_reported([0.5, -3e-7, 2.0], 3e-7)
    test_the_relu_input_closest_to_zero_is_reported([0.5, -2.0], 0.5)
    test_every_relu_of_the_model_is_checked()
    with pytest.MonkeyPatch.context() as patcher:
        test_a_batch_with_a_relu_input_at_the_kink_is_redrawn(patcher)
    test_mnist()
    # test_bert is skipped
