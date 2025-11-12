# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import torch
from torch import nn
import numpy as np

pytest.importorskip("torch", reason="PyTorch not installed")

from dace.frontend.ml.torch.module import DaceModule
from tests.utils import torch_tensors_close


def compare_gradients(pt_model, dace_model, rtol=1e-5, atol=1e-4):
    """Compare gradients between PyTorch and DaCe models."""

    for (pt_name, pt_param), (dace_name, dace_param) in zip(pt_model.named_parameters(), dace_model.named_parameters()):
        if pt_param.grad is None or dace_param.grad is None:
            continue

        torch_tensors_close(pt_name, pt_param.grad, dace_param.grad, rtol=rtol, atol=atol)


@pytest.mark.torch
def test_single_layernorm(sdfg_name):
    """Test a single LayerNorm layer."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16

    class SimpleLayerNorm(nn.Module):

        def __init__(self):
            super(SimpleLayerNorm, self).__init__()
            self.ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            return self.ln(x)

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = SimpleLayerNorm()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = SimpleLayerNorm()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_two_sequential_layernorms(sdfg_name):
    """Test two LayerNorm layers applied sequentially."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16

    class TwoLayerNorms(nn.Module):

        def __init__(self):
            super(TwoLayerNorms, self).__init__()
            self.ln1 = nn.LayerNorm(hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)

        def forward(self, x):
            x = self.ln1(x)
            x = self.ln2(x)
            return x

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = TwoLayerNorms()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = TwoLayerNorms()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_layernorm_with_residual(sdfg_name):
    """Test LayerNorm with residual connection."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16

    class LayerNormWithResidual(nn.Module):

        def __init__(self):
            super(LayerNormWithResidual, self).__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            residual = x
            x = self.linear(x)
            x = x + residual
            x = self.ln(x)
            return x

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = LayerNormWithResidual()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = LayerNormWithResidual()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_two_layernorms_with_residuals(sdfg_name):
    """Test two LayerNorms with residual connections (similar to BERT structure)."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16

    class TwoLayerNormsWithResiduals(nn.Module):

        def __init__(self):
            super(TwoLayerNormsWithResiduals, self).__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)

        def forward(self, x):
            # First block
            residual = x
            x = self.linear1(x)
            x = x + residual
            x = self.ln1(x)

            # Second block
            residual = x
            x = self.linear2(x)
            x = x + residual
            x = self.ln2(x)

            return x

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = TwoLayerNormsWithResiduals()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = TwoLayerNormsWithResiduals()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_ffn_layernorm(sdfg_name):
    """Test LayerNorm after feed-forward network (FFN)."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16
    intermediate_size = hidden_size * 4

    class FFNLayerNorm(nn.Module):

        def __init__(self):
            super(FFNLayerNorm, self).__init__()
            self.dense1 = nn.Linear(hidden_size, intermediate_size)
            self.relu = nn.ReLU()
            self.dense2 = nn.Linear(intermediate_size, hidden_size)
            self.ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            residual = x
            x = self.dense1(x)
            x = self.relu(x)
            x = self.dense2(x)
            x = x + residual
            x = self.ln(x)
            return x

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = FFNLayerNorm()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = FFNLayerNorm()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_attention_then_layernorm(sdfg_name):
    """Test LayerNorm after a simplified attention mechanism."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16
    num_heads = 2
    head_dim = hidden_size // num_heads

    class AttentionLayerNorm(nn.Module):

        def __init__(self):
            super(AttentionLayerNorm, self).__init__()
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)
            self.ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            batch_size, seq_len, hidden_size = x.shape

            # Multi-head attention (simplified)
            Q = self.query(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)

            # Apply attention to values
            context = torch.matmul(attn, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

            # Output projection
            output = self.output(context)

            # Residual connection
            output = output + x

            # LayerNorm
            output = self.ln(output)

            return output

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = AttentionLayerNorm()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = AttentionLayerNorm()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_simplified_bert_block(sdfg_name):
    """Test a simplified BERT block with attention and FFN, each followed by LayerNorm."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    hidden_size = 16
    num_heads = 2
    head_dim = hidden_size // num_heads
    intermediate_size = hidden_size * 4

    class SimplifiedBertBlock(nn.Module):

        def __init__(self):
            super(SimplifiedBertBlock, self).__init__()
            # Attention
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.attn_output = nn.Linear(hidden_size, hidden_size)
            self.attn_ln = nn.LayerNorm(hidden_size)

            # FFN
            self.ffn_dense1 = nn.Linear(hidden_size, intermediate_size)
            self.relu = nn.ReLU()
            self.ffn_dense2 = nn.Linear(intermediate_size, hidden_size)
            self.ffn_ln = nn.LayerNorm(hidden_size)

        def forward(self, x):
            batch_size, seq_len, hidden_size = x.shape

            # === Attention Block ===
            residual = x

            # Multi-head attention
            Q = self.query(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

            # Output projection + residual + LayerNorm
            x = self.attn_output(context)
            x = x + residual
            x = self.attn_ln(x)  # First LayerNorm

            # === FFN Block ===
            residual = x
            x = self.ffn_dense1(x)
            x = self.relu(x)
            x = self.ffn_dense2(x)
            x = x + residual
            x = self.ffn_ln(x)  # Second LayerNorm

            return x

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = SimplifiedBertBlock()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = SimplifiedBertBlock()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, training=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_bert_block_with_dropout_in_eval(sdfg_name):
    """Test BERT block with Dropout layers in eval mode"""
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 16
    hidden_size = 32
    num_heads = 2
    head_dim = hidden_size // num_heads
    intermediate_size = hidden_size * 4

    class BertBlockWithDropout(nn.Module):

        def __init__(self):
            super(BertBlockWithDropout, self).__init__()
            # Attention
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.attn_output = nn.Linear(hidden_size, hidden_size)
            self.attn_dropout = nn.Dropout(p=0.1)
            self.attn_ln = nn.LayerNorm(hidden_size, eps=1e-12)

            # FFN
            self.ffn_dense1 = nn.Linear(hidden_size, intermediate_size)
            self.relu = nn.ReLU()
            self.ffn_dense2 = nn.Linear(intermediate_size, hidden_size)
            self.ffn_dropout = nn.Dropout(p=0.1)
            self.ffn_ln = nn.LayerNorm(hidden_size, eps=1e-12)

        def forward(self, x):
            batch_size, seq_len, hidden_size = x.shape

            # === Attention Block ===
            residual = x

            # Multi-head attention
            Q = self.query(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

            # Output projection + dropout + residual + LayerNorm
            x = self.attn_output(context)
            x = self.attn_dropout(x)
            x = self.attn_ln(x + residual)  # LayerNorm after addition

            # === FFN Block ===
            residual = x
            x = self.ffn_dense1(x)
            x = self.relu(x)
            x = self.ffn_dense2(x)
            x = self.ffn_dropout(x)
            x = self.ffn_ln(x + residual)  # LayerNorm after addition

            return x

    # Put model in eval mode
    pt_model = BertBlockWithDropout().eval()
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = BertBlockWithDropout().eval()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


@pytest.mark.torch
def test_real_bert_layer_small(sdfg_name):
    """Test with actual BertLayer from transformers, but with small dimensions."""
    pytest.importorskip("transformers", reason="Transformers not installed")
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertLayer

    torch.manual_seed(42)

    batch_size = 1
    seq_len = 16
    hidden_size = 32

    config = BertConfig(hidden_size=hidden_size,
                        num_attention_heads=2,
                        intermediate_size=hidden_size * 4,
                        hidden_act="relu",
                        attn_implementation="eager")

    class BertLayerWrapper(nn.Module):

        def __init__(self):
            super(BertLayerWrapper, self).__init__()
            self.bert_layer = BertLayer(config).eval()

        def forward(self, x):
            return self.bert_layer(x)[0]

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x_dace = x.clone().detach().requires_grad_(True)

    # PyTorch
    pt_model = BertLayerWrapper()
    pt_output = pt_model(x)
    pt_loss = pt_output.sum()
    pt_loss.backward()

    # DaCe
    dace_model = BertLayerWrapper()
    dace_model.load_state_dict(pt_model.state_dict())
    dace_model = DaceModule(dace_model, backward=True, simplify=True, sdfg_name=sdfg_name)

    dace_output = dace_model(x_dace)
    dace_loss = dace_output.sum()
    dace_loss.backward()

    # Compare
    compare_gradients(pt_model, dace_model)


if __name__ == "__main__":
    test_real_bert_layer_small(sdfg_name="test_real_bert_layer_small")
    test_bert_block_with_dropout_in_eval(sdfg_name="test_bert_block_with_dropout_in_eval")
