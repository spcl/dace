# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The feed-forward block of a Gemma 4 mixture-of-experts layer.

In Gemma 4 26B-A4B, the feed-forward part of every decoder layer runs two branches on the same input (the residual
stream after attention) and adds them:

* a dense gated MLP (GeGLU), and
* a mixture of 128 small gated MLPs (experts), of which a router picks 8 per token and weighs their outputs.

This file holds the block's configuration, random weights at the model's shapes, and a PyTorch reference that follows
the Hugging Face ``transformers`` implementation operation by operation.
"""

import argparse
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class Config:
    """Shapes of the block (defaults: ``google/gemma-4-26B-A4B``)."""

    hidden: int = 2816  # Width of the residual stream
    dense_intermediate: int = 2112  # Width of the dense MLP
    num_experts: int = 128
    top_k: int = 8  # Experts per token
    expert_intermediate: int = 704  # Width of each expert
    eps: float = 1e-6  # RMSNorm epsilon


@dataclasses.dataclass
class Weights:
    """The block's parameters, as bfloat16 (norm and router scales in float32) PyTorch tensors."""

    # Dense branch
    pre_norm: torch.Tensor  # [hidden]
    gate_proj: torch.Tensor  # [dense_intermediate, hidden]
    up_proj: torch.Tensor  # [dense_intermediate, hidden]
    down_proj: torch.Tensor  # [hidden, dense_intermediate]
    post_norm_1: torch.Tensor  # [hidden]
    # Router
    router_scale: torch.Tensor  # [hidden]
    router_proj: torch.Tensor  # [num_experts, hidden]
    per_expert_scale: torch.Tensor  # [num_experts]
    # Experts
    pre_norm_2: torch.Tensor  # [hidden]
    expert_gate_up: torch.Tensor  # [num_experts, 2 * expert_intermediate, hidden]
    expert_down: torch.Tensor  # [num_experts, hidden, expert_intermediate]
    post_norm_2: torch.Tensor  # [hidden]
    # Sum of the branches
    post_norm: torch.Tensor  # [hidden]


def random_weights(config: Config, device: str = "cuda", seed: int = 0) -> Weights:
    """Weights with the model's shapes and plausible magnitudes (the computation does not depend on their values)."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def matrix(*shape: int) -> torch.Tensor:
        fan_in = shape[-1]
        return (torch.randn(*shape, generator=generator, device=device) / fan_in**0.5).to(torch.bfloat16)

    def scale(size: int) -> torch.Tensor:
        return 1.0 + 0.1 * torch.randn(size, generator=generator, device=device)

    c = config
    return Weights(
        pre_norm=scale(c.hidden),
        gate_proj=matrix(c.dense_intermediate, c.hidden),
        up_proj=matrix(c.dense_intermediate, c.hidden),
        down_proj=matrix(c.hidden, c.dense_intermediate),
        post_norm_1=scale(c.hidden),
        router_scale=scale(c.hidden),
        router_proj=matrix(c.num_experts, c.hidden),
        per_expert_scale=scale(c.num_experts),
        pre_norm_2=scale(c.hidden),
        expert_gate_up=matrix(c.num_experts, 2 * c.expert_intermediate, c.hidden),
        expert_down=matrix(c.num_experts, c.hidden, c.expert_intermediate),
        post_norm_2=scale(c.hidden),
        post_norm=scale(c.hidden),
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Gemma 4's RMSNorm: normalizes in float32, then scales by ``weight`` (if any)."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.type_as(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def route(x: torch.Tensor, w: Weights, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Picks ``top_k`` experts per token and their weights: softmax over all experts, top-k, renormalize, and scale
    by a learned per-expert factor."""
    h = rms_norm(x, None, config.eps) * w.router_scale * config.hidden**-0.5
    scores = F.linear(h.to(torch.bfloat16), w.router_proj)
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
    weights, experts = torch.topk(probabilities, config.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * w.per_expert_scale[experts], experts


def reference(x: torch.Tensor, w: Weights, config: Config) -> torch.Tensor:
    """The block as in ``transformers`` (eager mode): ``x`` is the residual stream, ``[tokens, hidden]``."""
    # Dense branch
    h = rms_norm(x, w.pre_norm, config.eps)
    h = F.linear(gelu(F.linear(h, w.gate_proj)) * F.linear(h, w.up_proj), w.down_proj)
    dense = rms_norm(h, w.post_norm_1, config.eps)

    # Experts: each expert processes the tokens that picked it
    weights, experts = route(x, w, config)
    h = rms_norm(x, w.pre_norm_2, config.eps)
    moe = torch.zeros_like(h)
    hit = torch.nn.functional.one_hot(experts, config.num_experts).permute(2, 1, 0)  # [expert, slot, token]
    for expert in torch.nonzero(hit.sum(dim=(1, 2))).flatten().tolist():
        slot, token = torch.where(hit[expert])
        gate, up = F.linear(h[token], w.expert_gate_up[expert]).chunk(2, dim=-1)
        y = F.linear(gelu(gate) * up, w.expert_down[expert])
        moe.index_add_(0, token, (y * weights[token, slot, None]).to(moe.dtype))
    moe = rms_norm(moe, w.post_norm_2, config.eps)

    return x + rms_norm(dense + moe, w.post_norm, config.eps)


def _dense_branch(x: torch.Tensor, w: Weights, config: Config) -> torch.Tensor:
    h = rms_norm(x, w.pre_norm, config.eps)
    h = F.linear(gelu(F.linear(h, w.gate_proj)) * F.linear(h, w.up_proj), w.down_proj)
    return rms_norm(h, w.post_norm_1, config.eps)


def _sort_pairs(x: torch.Tensor, w: Weights, config: Config):
    """Routes the tokens and sorts the (token, expert) pairs by expert. Returns the expert input, the pairs' tokens,
    experts and weights in sorted order, each pair's position among the pairs of its expert, and where each expert's
    pairs end."""
    weights, experts = route(x, w, config)
    pairs = experts.flatten()
    order = torch.argsort(pairs, stable=True)
    counts = torch.zeros(config.num_experts, dtype=torch.int32, device=x.device)
    counts.scatter_add_(0, pairs, torch.ones_like(pairs, dtype=torch.int32))
    ends = torch.cumsum(counts, 0, dtype=torch.int32)
    expert = pairs[order]
    position = torch.arange(len(order), device=x.device) - (ends - counts)[expert]
    h = rms_norm(x, w.pre_norm_2, config.eps)
    return h, order // config.top_k, expert, weights.flatten()[order], position, ends


def _combine(x, dense, h, token, weight, y, w: Weights, config: Config) -> torch.Tensor:
    moe = torch.zeros_like(h).index_add_(0, token, (y * weight[:, None]).to(h.dtype))
    moe = rms_norm(moe, w.post_norm_2, config.eps)
    return x + rms_norm(dense + moe, w.post_norm, config.eps)


def grouped(x: torch.Tensor, w: Weights, config: Config) -> torch.Tensor:
    """The block with no Python loop: every expert projection is one grouped GEMM over the experts' consecutive
    pairs. (Before Hopper GPUs, the grouped GEMM loops over the experts on the host, which synchronizes.)"""
    dense = _dense_branch(x, w, config)
    h, token, _, weight, _, ends = _sort_pairs(x, w, config)
    gate, up = torch._grouped_mm(h[token], w.expert_gate_up.transpose(1, 2), offs=ends).chunk(2, dim=-1)
    y = torch._grouped_mm(gelu(gate) * up, w.expert_down.transpose(1, 2), offs=ends)
    return _combine(x, dense, h, token, weight, y, w, config)


def padded(x: torch.Tensor, w: Weights, config: Config) -> torch.Tensor:
    """The block without host synchronization: every expert's pairs are padded to the number of tokens (the most an
    expert can get), and every expert projection is one batched GEMM. Exact, but computes ``num_experts / top_k``
    times the expert FLOPs."""
    dense = _dense_branch(x, w, config)
    h, token, expert, weight, position, _ = _sort_pairs(x, w, config)
    rows = h.new_zeros(config.num_experts, len(x), config.hidden)
    rows[expert, position] = h[token]
    gate, up = torch.bmm(rows, w.expert_gate_up.transpose(1, 2)).chunk(2, dim=-1)
    y = torch.bmm(gelu(gate) * up, w.expert_down.transpose(1, 2))[expert, position]
    return _combine(x, dense, h, token, weight, y, w, config)


def time_ms(function, repetitions: int = 20) -> float:
    """Median time of ``function()`` on the GPU, in milliseconds."""
    function()  # Warm up
    times = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def graphed(function):
    """``function`` captured in a CUDA graph (as inference servers run decoding steps): returns its replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()  # Warm up outside the graph
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph.replay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    args = parser.parse_args()

    config = Config()
    w = random_weights(config)
    for tokens in args.tokens:
        x = torch.randn(tokens, config.hidden, device="cuda").to(torch.bfloat16)
        expected = reference(x, w, config).float()
        for variant in (grouped, padded):
            if variant is padded:
                torch.cuda.set_sync_debug_mode("error")  # Fails on any host synchronization
            result = variant(x, w, config)
            torch.cuda.set_sync_debug_mode("default")
            error = (result.float() - expected).norm() / expected.norm()
            assert error < 1e-2, f"{variant.__name__} differs from the reference (relative error {error:.2e})"
        times = {f.__name__: time_ms(lambda: f(x, w, config)) for f in (reference, grouped, padded)}
        times["padded (CUDA graph)"] = time_ms(graphed(lambda: padded(x, w, config)))
        print(f"{tokens:5d} tokens: " + ", ".join(f"{name} {t:8.3f} ms" for name, t in times.items()))
