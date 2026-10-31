# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Gemma 4 mixture-of-experts feed-forward block (see ``gemma4_moe.py``) as vLLM runs it, for comparison.

Builds the block from vLLM's own Gemma 4 modules (merged gate/up MLP, router with its routing kernel, and the fused
MoE experts with their Triton kernels), loads the weights of ``gemma4_moe.random_weights``, and times it. Requires a
Python environment with vLLM, which pins its own PyTorch version.
"""

import argparse
import types

import gemma4_moe
import torch
from torch import nn
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.gemma4 import Gemma4MLP, Gemma4Router
from vllm.v1.worker.workspace import init_workspace_manager


class FeedForward(nn.Module):
    """The feed-forward part of vLLM's ``Gemma4DecoderLayer`` (its modules, and its forward code)."""

    def __init__(self, config: gemma4_moe.Config, prefix: str = "layers.0"):
        super().__init__()
        model_config = types.SimpleNamespace(
            hidden_size=config.hidden, rms_norm_eps=config.eps, num_experts=config.num_experts
        )
        self.mlp = Gemma4MLP(config.hidden, config.dense_intermediate, "gelu_pytorch_tanh", prefix=f"{prefix}.mlp")
        self.router = Gemma4Router(model_config, prefix=f"{prefix}.router")
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.top_k,
            hidden_size=config.hidden,
            intermediate_size=config.expert_intermediate,
            renormalize=True,
            prefix=f"{prefix}.experts",
            custom_routing_function=self.router.routing_function,
            activation="gelu_tanh",
        )
        norm = lambda: RMSNorm(config.hidden, eps=config.eps)  # noqa: E731
        self.pre_feedforward_layernorm = norm()
        self.post_feedforward_layernorm = norm()
        self.post_feedforward_layernorm_1 = norm()
        self.pre_feedforward_layernorm_2 = norm()
        self.post_feedforward_layernorm_2 = norm()

    def load(self, w: gemma4_moe.Weights):
        with torch.no_grad():
            self.mlp.gate_up_proj.weight.copy_(torch.cat([w.gate_proj, w.up_proj]))
            self.mlp.down_proj.weight.copy_(w.down_proj)
            self.experts.routed_experts.w13_weight.copy_(w.expert_gate_up)
            self.experts.routed_experts.w2_weight.copy_(w.expert_down)
            self.router.scale.copy_(w.router_scale)
            self.router.per_expert_scale.copy_(w.per_expert_scale)
            self.router.proj.weight.copy_(w.router_proj)
            self.pre_feedforward_layernorm.weight.copy_(w.pre_norm)
            self.post_feedforward_layernorm_1.weight.copy_(w.post_norm_1)
            self.pre_feedforward_layernorm_2.weight.copy_(w.pre_norm_2)
            self.post_feedforward_layernorm_2.weight.copy_(w.post_norm_2)
            self.post_feedforward_layernorm.weight.copy_(w.post_norm)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # As in Gemma4DecoderLayer.forward, after attention
        hidden_states = self.pre_feedforward_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
        hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
        router_logits = self.router(residual)
        hidden_states_2 = self.experts(hidden_states_2, router_logits)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = self.post_feedforward_layernorm(hidden_states_1 + hidden_states_2)
        return hidden_states + residual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    args = parser.parse_args()

    config = gemma4_moe.Config()
    # Eager mode (as with ``enforce_eager``), with vLLM's own kernels (RMSNorm, GELU-and-multiply) rather than
    # PyTorch fallbacks; decoding steps are then replayed from CUDA graphs below
    vllm_config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29512", backend="gloo"
        )
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device("cuda"))  # As vLLM's GPU worker does
        vllm_config.kernel_config.ir_op_priority.set_default()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            block = FeedForward(config)
        torch.set_default_dtype(torch.float32)
        w = gemma4_moe.random_weights(config)
        block.load(w)
        # The post-loading step of vLLM's model loader (builds the fused MoE kernel)
        for module in (block.mlp.gate_up_proj, block.mlp.down_proj, block.router.proj, block.experts.routed_experts):
            module.quant_method.process_weights_after_loading(module)

        with set_forward_context(None, vllm_config), torch.inference_mode():
            for tokens in args.tokens:
                x = torch.randn(tokens, config.hidden, device="cuda").to(torch.bfloat16)
                # vLLM normalizes the router input in bfloat16 (the reference in float32), which can flip near-ties
                # between the k-th and the next expert: compare the tokens that both route alike
                _, chosen = block.router.routing_function(x, block.router(x), config.top_k, True)
                _, expected_chosen = gemma4_moe.route(x, w, config)
                alike = (chosen.sort().values == expected_chosen.sort().values).all(dim=1)
                expected = gemma4_moe.reference(x, w, config).float()[alike]
                error = (block(x).float()[alike] - expected).norm() / expected.norm()
                assert error < 2e-2, f"vLLM differs from the reference (relative error {error:.2e})"
                eager, graph = (
                    gemma4_moe.time_ms(lambda: block(x)),
                    gemma4_moe.time_ms(gemma4_moe.graphed(lambda: block(x))),
                )
                print(
                    f"{tokens:5d} tokens: vLLM {eager:8.3f} ms, vLLM (CUDA graph) {graph:8.3f} ms "
                    f"(error {error:.1e}, {int((~alike).sum())} tokens routed differently)"
                )
