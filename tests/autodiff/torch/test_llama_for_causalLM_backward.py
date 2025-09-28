import pytest
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close


class LlamaWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids):
        # Get the embeddings
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        # Create position ids
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Process through decoder layers
        hidden_states = inputs_embeds

        # Create causal mask for attention
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Forward through each layer
        for decoder_layer in self.model.model.layers:
            # Get rotary embeddings
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            hidden_states = layer_outputs[0]

        # Final layer norm
        hidden_states = self.model.model.norm(hidden_states)

        # Get logits
        logits = self.model.lm_head(hidden_states)

        return logits


@pytest.mark.long
@pytest.mark.torch
@pytest.mark.autodiff
def test_llama_model_backward(sdfg_name):
    # Create a small LLaMA configuration
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
    )

    # Create the full model
    model = LlamaForCausalLM(config)
    model.eval()
    export_seq_length = 32
    export_batch_size = 2
    export_input = torch.randint(3, config.vocab_size, (export_batch_size, export_seq_length), dtype=torch.long)

    wrapped_model = LlamaWrapper(model)
    dace_model = DaceModule(wrapped_model, backward=True, onnx_simplify=True, simplify=True, sdfg_name=sdfg_name)

    wrapped_model(export_input.clone()).sum().backward()
    dace_model(export_input.clone()).sum().backward()

    # Check gradients of the parameters
    for i, (name, param) in enumerate(wrapped_model.named_parameters()):
        if param.requires_grad:
            torch_tensors_close(f"grad_{name}",
                                list(wrapped_model.parameters())[i].grad,
                                list(dace_model.model.parameters())[i].grad)


if __name__ == "__main__":
    test_llama_model_backward(sdfg_name="llama_model_test")
