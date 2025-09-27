import pytest
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig
from dace.testing import torch_tensors_close
from dace.frontend.python.module import DaceModule


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


@pytest.mark.cpublas
def test_llama_decoder_backward(sdfg_name):
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

    # Prepare dummy inputs
    batch_size = 2
    seq_length = 128

    # Create input tensors
    hidden_states = torch.ones(batch_size, seq_length, config.hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_length, seq_length)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)

    # Create wrapped model
    wrapped_model = LlamaDecoderLayerWrapper(decoder_layer, config)
    wrapped_model = wrapped_model
    dace_model = DaceModule(
        wrapped_model,
        sdfg_name=sdfg_name,
        onnx_simplify=True,
        simplify=False,
        backward=True,
    )

    hidden_states_pt, attention_mask_pt, position_ids_pt = (torch.clone(hidden_states), torch.clone(attention_mask),
                                                            torch.clone(position_ids))
    hidden_states_pt.requires_grad = True

    hidden_states_dace, attention_mask_dace, position_ids_dace = (torch.clone(hidden_states),
                                                                  torch.clone(attention_mask),
                                                                  torch.clone(position_ids))
    hidden_states_dace.requires_grad = True

    wrapped_model(hidden_states_pt, attention_mask_pt, position_ids_pt).sum().backward()
    dace_model(hidden_states_dace, attention_mask_dace, position_ids_dace).sum().backward()

    # Check gradients of the parameters
    for i, (name, param) in enumerate(wrapped_model.named_parameters()):
        if param.requires_grad:
            torch_tensors_close(f"grad_{name}",
                                list(wrapped_model.parameters())[i].grad,
                                list(dace_model.model.parameters())[i].grad)

    # Check the gradients of the input tensor
    torch_tensors_close("hidden_states_pt_grad", hidden_states_pt.grad, hidden_states_dace.grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_llama_decoder_backward(sdfg_name="llama_decoder_backward_test")
