# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig
from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close


# Create a wrapper module that handles the position embeddings internally
class LlamaDecoderLayerWrapper(nn.Module):

    def __init__(self, decoder_layer, config):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.config = config

        # Create rotary embeddings as part of the wrapper
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask, position_ids):
        # Generate position embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Call the decoder layer
        outputs = self.decoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

        # Return only the hidden states (first element of the tuple)
        return outputs[0]


@pytest.mark.torch
def test_llama_decoder(sdfg_name: str):
    # Create configuration
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_act="silu",
        attn_implementation="eager",
    )

    # Create decoder layer
    decoder_layer = LlamaDecoderLayer(config, layer_idx=0)
    decoder_layer.eval()

    # Prepare dummy inputs
    batch_size = 2
    seq_length = 128

    # Create input tensors
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_length, seq_length)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)

    # Create wrapped model
    wrapped_model = LlamaDecoderLayerWrapper(decoder_layer, config)
    wrapped_model.eval()

    dace_model = DaceModule(wrapped_model, sdfg_name=sdfg_name)

    # Test inference with dummy inputs
    with torch.no_grad():
        output_1 = dace_model(hidden_states.clone(), attention_mask.clone(), position_ids.clone())
        output_2 = wrapped_model(hidden_states.clone(), attention_mask.clone(), position_ids.clone())
        torch_tensors_close("llama_inference_output", output_1, output_2)


if __name__ == "__main__":
    test_llama_decoder(sdfg_name="test_llama_decoder")
