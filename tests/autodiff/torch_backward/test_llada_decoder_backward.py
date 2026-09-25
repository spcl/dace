# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
pytest.importorskip("transformers",
                    reason="transformers not installed. Please install with: pip install dace[ml-testing]")
import torch
import torch.nn as nn
from dace.ml import DaceModule
from tests.utils import torch_tensors_close
from tests.ml_gpu_utils import DEVICES, experimental_cuda, is_gpu, torch_device

# LLaDA is not a built-in transformers model, so we fetch its custom modeling code
# from the HuggingFace hub via trust_remote_code. Only the python modeling code and
# config are downloaded (no safetensors weights), and the config is shrunk to a
# unit-scale single block with random init.
LLADA_REPO = "GSAI-ML/LLaDA-8B-Base"


# Create a wrapper module that returns only the hidden states (the block returns a
# (hidden_states, cache) tuple). LLaDA applies RoPE internally and uses non-causal
# (bidirectional) attention, so unlike the LLaMA decoder test no position_ids or
# attention mask need to be passed in.
class LLaDABlockWrapper(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states):
        outputs = self.block(hidden_states, attention_bias=None, layer_past=None, use_cache=False)

        # Return only the hidden states (first element of the tuple)
        return outputs[0]


@pytest.mark.xdist_group("large_ML_models")
@pytest.mark.torch
@pytest.mark.autodiff
@pytest.mark.parametrize("device", DEVICES)
def test_llada_decoder_backward(device):
    dev = torch_device(device)
    from transformers import AutoConfig, AutoModelForCausalLM

    # Build a small LLaDA block via trust_remote_code (random init, 1 layer). Skip
    # (rather than fail) if the hub/network or remote code is unavailable.
    try:
        config = AutoConfig.from_pretrained(LLADA_REPO, trust_remote_code=True)
        config.d_model = 512
        config.n_heads = 8
        config.n_kv_heads = 8
        config.n_layers = 1
        config.mlp_hidden_size = 1024
        config.max_sequence_length = 128
        config.block_type = "llama"  # ensure LLaDALlamaBlock
        # dropouts are already 0.0 in the pretrained config

        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.model.reset_parameters()  # ensure real (non-uninitialized) weights
    except Exception as e:
        pytest.skip(f"Could not load LLaDA model from '{LLADA_REPO}' "
                    f"(network / trust_remote_code unavailable): {e}")

    decoder_block = model.model.transformer.blocks[0]

    # Prepare dummy inputs
    batch_size = 2
    seq_length = 128

    hidden_states = torch.randn(batch_size, seq_length, config.d_model, device=dev)

    # Create wrapped model
    wrapped_model = LLaDABlockWrapper(decoder_block).to(dev)

    dace_model = DaceModule(
        wrapped_model,
        sdfg_name=f"test_llada_decoder_backward_{device}",
        onnx_simplify=True,
        backward=True,
        cuda=is_gpu(device),
    )

    hidden_states_pt = torch.clone(hidden_states)
    hidden_states_pt.requires_grad = True

    hidden_states_dace = torch.clone(hidden_states)
    hidden_states_dace.requires_grad = True

    wrapped_model(hidden_states_pt).sum().backward()
    with experimental_cuda():
        dace_model(hidden_states_dace).sum().backward()

    # Check gradients of the parameters
    for (name, dace_param), (pt_name, pt_param) in zip(wrapped_model.named_parameters(), dace_model.named_parameters()):
        assert 'model.' + name == pt_name, f"Parameter name mismatch: expected 'model.{name}', got '{pt_name}'"
        torch_tensors_close(name, dace_param.grad, pt_param.grad)

    # Check the gradients of the input tensor
    torch_tensors_close("hidden_states_pt_grad", hidden_states_pt.grad, hidden_states_dace.grad)


if __name__ == "__main__":
    test_llada_decoder_backward(device="cpu")
