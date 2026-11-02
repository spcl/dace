# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A fused mixture-of-experts (MoE) block as a megakernel: one persistent GPU kernel that runs the whole block.

A megakernel runs, as a single kernel, work that would otherwise take a sequence of kernel launches: its thread blocks
schedule the work among themselves (the term goes back to GPU ray tracers; see Laine et al., "Megakernels Considered
Harmful: Wavefront Path Tracing on GPUs", HPG 2013). For a mixture of experts, which experts run, and on how many
tokens, is only known on the GPU: here no stage of the block waits for the previous one as a whole, and the host never
waits for the GPU.

The prototype layer is the feed-forward block of a Gemma 4 decoder layer (``google/gemma-4-26B-A4B``), which runs two
branches on the same input (the residual stream after attention) and adds them:

* a dense gated MLP (GeGLU), and
* a mixture of 128 small gated MLPs (experts), of which a router picks 8 per token and weighs their outputs.

After a kernel that normalizes the tokens (one thread block per token), a single consume scope runs the rest of the
block as tasks on a queue. Every processing element is a GPU thread block that pops a task, mostly a 64 x 128 tile of
one matrix product with tensor cores (the ``tile_matmul`` library node) followed by an epilogue on the tile. The tasks
are:

* router: the logits of 64 tokens for a slice of the input (the router's inner dimension is split, so that routing is
  done quickly while the dense tasks start);
* top-k: once all slices of 64 tokens are done, the softmax and top-k of each of them (a task per token, so that
  tokens are routed in parallel), claiming a row of every chosen expert;
* scan: once all tokens are routed, the tiles of every expert's rows (pairs of a token and an expert); it pushes the
  expert gate/up tasks;
* expert gate/up: 64 rows of an expert (reading the rows of their tokens) times 128 columns of its gate and up
  projections, interleaved in chunks of 64, followed by GELU(gate) * up;
* expert down: the same rows times 128 output columns of the expert's down projection, scaled by each pair's routing
  weight and added to the token's output (an atomic write-conflict resolution);
* dense gate/up and dense down: the same for 64 tokens in the dense MLP that runs in parallel;
* output: once all down tiles of a token are added, both branches normalized, added, and added to the residual.

A task that depends on others is pushed by the last of them to finish: each decrements a counter, reading its old
value (an acquire-release update). So the queue never waits for a whole stage to finish, and dense and expert tasks
fill each other's gaps.

Every task is a DaCe program of its own, which DaCe inlines into the consume scope (see ``build``). Other blocks of the
same kind (another router, no dense branch, another activation) are built from the same tasks, combined differently.

Besides the megakernel, this file holds the block's configuration, random weights at the model's shapes, and a PyTorch
reference that follows the Hugging Face ``transformers`` implementation operation by operation. If vLLM is installed
(it pins its own PyTorch version), the block as vLLM runs it is timed as well. Requires PyTorch and a CUDA GPU with
bfloat16 tensor cores (Ampere or newer).
"""

import argparse
import contextlib
import dataclasses
import math
import types

import numpy as np
import tile_matmul as tm
import torch
import torch.nn.functional as F
from torch import nn

import dace

try:  # Optional: the block as vLLM runs it, for comparison
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.gemma4 import Gemma4MLP, Gemma4Router
    from vllm.v1.worker.workspace import init_workspace_manager

    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False

# The block and its PyTorch reference


@dataclasses.dataclass(frozen=True)
class Config:
    """Shapes of the block (defaults: the Gemma 4 26B-A4B model)."""

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


def random_input(config: Config, tokens: int, device: str = "cuda") -> torch.Tensor:
    """The residual stream of ``tokens`` tokens, the same for every run (which experts the tokens pick, and so how
    much of the weights a block reads, depends on it)."""
    generator = torch.Generator(device).manual_seed(tokens)
    return torch.randn(tokens, config.hidden, device=device, generator=generator).to(torch.bfloat16)


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


# The megakernel

TB = dace.ScheduleType.GPU_ThreadBlock
SHARED = dace.StorageType.GPU_Shared
WARPS = 4  # Warps per thread block
THREADS = 32 * WARPS
BM, BN = 64, 128  # Rows and columns of a tile
K_STEP, STAGES = 32, 2  # Slices of the inner dimension that tile_matmul stages in shared memory
WORKSPACE = tm.workspace_shape(BM, BN, K_STEP, STAGES)
STAGING = math.prod(WORKSPACE)  # Elements of the workspace
# The kinds of tasks, multiples of ``1 << KIND`` (64-bit constants): a task is its kind, a tile and a chunk (see
# ``append_to_stream``)
KIND, FIELD = 40, 20
MASK = (1 << (KIND - FIELD)) - 1  # Of a tile or chunk
EXPERT_GATE_UP, EXPERT_DOWN, DENSE_GATE_UP, DENSE_DOWN, ROUTE, TOP_K, SCAN, OUTPUT = (k << KIND for k in range(8))
ROUTE_STEP = 256  # Columns of the input that a router task multiplies (the router's inner dimension is split)

T = dace.symbol("T")  # Tokens
TP = dace.symbol("TP")  # Tokens, padded to a multiple of BM
NT = dace.symbol("NT")  # The most expert tiles there can be: T * top_k / BM + num_experts
# The tiles of tokens whose dense gate/up tasks the prologue pushes: all of them, unless an expert has fewer rows than a
# tile on average. The expert tasks are then bound by memory and the dense ones by compute (from 64 tokens on): those
# of all tokens but the first 64 run among the expert tasks (the scan pushes them).
DT = dace.symbol("DT")


def packed(*shape) -> dace.data.Array:
    """A weight matrix ``W[N, K]`` (or a batch of them), packed for ``tile_matmul``: as ``W[N / BN, K / K_STEP, BN,
    K_STEP]``, so that the slices of the weight that a tile reads are contiguous (see ``pack``)."""
    *batch, n, k = shape
    return dace.data.Array(dace.bfloat16, [*batch, n // BN, k // K_STEP, BN, K_STEP])


def pack(weight: torch.Tensor) -> torch.Tensor:
    """Packs a weight matrix ``[..., N, K]`` (see ``packed``): a copy."""
    *batch, n, k = weight.shape
    return weight.reshape(*batch, n // BN, BN, k // K_STEP, K_STEP).transpose(-2, -3).contiguous()


def build(config: Config, num_pes: int, queue_size: int = 1 << 18) -> dace.SDFG:
    """The block for ``config`` as a GPU SDFG. Its arguments are the input, the weights (see ``arguments``), the
    output, and scratch buffers (see ``workspace``)."""
    D, DI, I, E, K = (
        config.hidden,
        config.dense_intermediate,
        config.expert_intermediate,
        config.num_experts,
        config.top_k,
    )
    assert D % BN == 0 and (2 * DI) % BN == 0 and (2 * I) % BN == 0 and E == BN == THREADS and D % ROUTE_STEP == 0
    assert BM * BN <= STAGING  # The product of a tile fits where its operands are staged
    RP = D // ROUTE_STEP  # Router tasks per tile of tokens
    RS = ROUTE_STEP // K_STEP  # Packed slices of the router weight per router task
    HALF = BN // 2  # Gate (and up) columns of a gate/up task
    # Constants in single precision (a Python float is a double on the GPU, which computes in double precision slowly)
    EPS = np.float32(config.eps)
    ROOT = np.float32(D**-0.5)
    GELU_C = np.float32(math.sqrt(2 / math.pi))
    GELU_A = np.float32(0.044715)
    ONE_HALF = np.float32(0.5)

    # Helpers

    @dace.program
    def append_to_stream(S, kind: dace.int64, tile: dace.int64, chunk: dace.int64):
        """Queues a task: ``kind | tile << FIELD | chunk``."""
        S.append(kind | tile << FIELD | chunk)

    @dace.program
    def is_last(counter, index):
        """Counts down ``counter[index]`` once for the thread block: whether this was its last count (an
        acquire-release update, so that the last block sees what the others wrote)."""
        last = dace.ndarray([1], dtype=dace.int32, storage=SHARED)
        for _ in dace.map[0:1] @ TB:
            last[0] = dace.atomic.add(counter, index, -1) == 1
        return last[0] == 1

    @dace.program
    def bf16(v: dace.float32):
        """``v`` rounded to bfloat16, as the model rounds it."""
        return dace.float32(dace.bfloat16(v))

    @dace.program
    def inverse_rms(sum_of_squares: dace.float32):
        """The inverse root mean square of a token, from the sum of the squares of its values."""
        return 1 / math.sqrt(sum_of_squares / D + EPS)

    # The prologue, a thread block per token

    @dace.program
    def prologue(
        t, x, pre_norm, pre_norm_2, router_scale, inv_rms, x_dense, x_expert, x_router, expert_acc, row_token, token_left
    ):  # fmt: skip
        """The RMS of token ``t`` (all the pre-norms normalize the same residual) and the inputs of the dense branch,
        the experts and the router (the normalized input, rounded to bfloat16 as the model does, and scaled). Also
        resets the token's scratch memory."""
        sum_of_squares = dace.ndarray([1], dtype=dace.float32, storage=SHARED)
        sum_of_squares[0] = 0
        for d in dace.map[0:D] @ TB:
            v = dace.float32(x[t, d])
            sum_of_squares[0] += v * v
            expert_acc[t, d] = 0
        for r in dace.map[t : NT * BM : T] @ TB:  # The blocks reset the rows of the expert tiles together
            row_token[r] = -1
        token_left[t] = (K + 1) * (D // BN)  # The down tiles that add to the token: its experts', and the dense
        inv = inverse_rms(sum_of_squares[0])
        inv_rms[t] = inv
        for d in dace.map[0:D] @ TB:
            n = dace.float32(x[t, d]) * inv
            x_dense[t, d] = n * pre_norm[d]
            x_router[t, d] = bf16(n) * router_scale[d] * ROOT
            x_expert[t, d] = n * pre_norm_2[d]

    @dace.program
    def first_tasks(S, count, dense_rows, route_left, dense_gate_up_left, top_k_left):
        """Resets the counters and queues the first tasks: the router's (tiles of tokens times slices of the input),
        whose last one pushes the top-k tasks of the tile, whose last one pushes the scan that pushes the expert tasks;
        then the dense gate/up tasks of the first DT tiles of tokens (see ``DT``), the tiles of a chunk of the weight
        one after the other: they read it from L2."""
        for ex in dace.map[0:E] @ TB:
            count[ex] = 0
        for g in dace.map[0 : TP // BM] @ TB:
            dense_rows[g] = min(BM, T - g * BM)
            route_left[g] = RP
            dense_gate_up_left[g] = DI // HALF
        top_k_left[0] = T  # A top-k task per token
        for i in dace.map[0 : TP // BM * RP + DT * (DI // HALF)] @ TB:
            if i < TP // BM * RP:
                append_to_stream(S, ROUTE, i // RP, i % RP)
            else:
                j = i - TP // BM * RP
                append_to_stream(S, DENSE_GATE_UP, j % DT, j // DT)

    # The tasks. ``g`` is a tile of rows (BM tokens or rows of experts), ``ch`` a chunk of columns.

    @dace.program
    def gelu_times_up(tile, nr, h, r0, ch):
        """GELU(gate) * up of ``nr`` rows, where the columns of the tile are HALF of the gate projection, then HALF of
        the up one: into rows ``r0`` of ``h``, chunk ``ch``."""
        for i in dace.map[0 : nr * HALF] @ TB:
            gate = dace.float32(tile[i // HALF, i % HALF])
            gelu = ONE_HALF * gate * (1 + math.tanh(GELU_C * (gate + GELU_A * gate * gate * gate)))
            h[r0 + i // HALF, ch * HALF + i % HALF] = bf16(gelu) * dace.float32(tile[i // HALF, HALF + i % HALF])

    @dace.program
    def expert_gate_up(
        S, g, ch, staging, x_expert, expert_gate_up_p, tile_expert, tile_rows, tile_start, row_token, count,
        gate_up_left, h_expert
    ):  # fmt: skip
        """A tile of an expert's rows (read from the rows of their tokens) times a chunk of its gate and up
        projections. The last gate/up task of the expert (of all its tiles) pushes its down tasks, the tiles of a chunk
        one after the other: they read it from L2."""
        work = staging.reshape(WORKSPACE)
        tile = staging[0 : BM * BN].reshape([BM, BN])
        e = tile_expert[g]
        nr = tile_rows[g]
        r0 = g * BM
        tm.tile_matmul(x_expert, expert_gate_up_p[e, ch], tile, work, nr, row_token[r0 : r0 + BM])
        gelu_times_up(tile, nr, h_expert, r0, ch)
        if is_last(gate_up_left, e):
            n = (count[e] + BM - 1) // BM  # Tiles of the expert
            for i in dace.map[0 : n * (D // BN)] @ TB:
                append_to_stream(S, EXPERT_DOWN, tile_start[e] + i % n, i // n)

    @dace.program
    def dense_gate_up(S, g, ch, staging, x_dense, dense_gate_up_p, dense_rows, dense_gate_up_left, h_dense):
        """BM tokens times a chunk of the dense gate and up projections. The last of the tokens pushes their down
        tasks."""
        work = staging.reshape(WORKSPACE)
        tile = staging[0 : BM * BN].reshape([BM, BN])
        nr = dense_rows[g]
        r0 = g * BM
        tm.tile_matmul(x_dense[r0 : r0 + BM, :], dense_gate_up_p[ch], tile, work, nr)
        gelu_times_up(tile, nr, h_dense, r0, ch)
        if is_last(dense_gate_up_left, g):
            for c in dace.map[0 : D // BN] @ TB:
                append_to_stream(S, DENSE_DOWN, g, c)

    @dace.program
    def expert_down(
        S, g, ch, staging, h_expert, expert_down_p, tile_expert, tile_rows, row_token, row_weight, expert_acc,
        token_left
    ):  # fmt: skip
        """A tile of an expert's rows times a chunk of its down projection, scaled by each row's routing weight and
        added to the output of the row's token. The last tile that adds to a token pushes its output task."""
        work = staging.reshape(WORKSPACE)
        tile = staging[0 : BM * BN].reshape([BM, BN])
        e = tile_expert[g]
        nr = tile_rows[g]
        r0 = g * BM
        tm.tile_matmul(h_expert[r0 : r0 + BM, :], expert_down_p[e, ch], tile, work, nr)
        for r, col in dace.map[0:nr, 0:BN] @ TB:
            expert_acc[row_token[r0 + r], ch * BN + col] += dace.float32(tile[r, col]) * row_weight[r0 + r]
        for r in dace.map[0:nr] @ TB:
            token = row_token[r0 + r]
            if dace.atomic.add(token_left, token, -1) == 1:
                append_to_stream(S, OUTPUT, token, 0)

    @dace.program
    def dense_down(S, g, ch, staging, h_dense, down_p, dense_rows, dense_out, token_left):
        """BM tokens times a chunk of the dense down projection."""
        work = staging.reshape(WORKSPACE)
        tile = staging[0 : BM * BN].reshape([BM, BN])
        nr = dense_rows[g]
        r0 = g * BM
        tm.tile_matmul(h_dense[r0 : r0 + BM, :], down_p[ch], tile, work, nr)
        for r, col in dace.map[0:nr, 0:BN] @ TB:
            dense_out[r0 + r, ch * BN + col] = tile[r, col]
        for r in dace.map[0:nr] @ TB:
            if dace.atomic.add(token_left, r0 + r, -1) == 1:
                append_to_stream(S, OUTPUT, r0 + r, 0)

    @dace.program
    def route(S, g, ch, staging, x_router, router_p, route_partial, dense_rows, route_left):
        """Part of the router logits of BM tokens: a slice of the input times the router weight. The last part of
        these tokens pushes their top-k tasks."""
        work = staging.reshape(WORKSPACE)
        nr = dense_rows[g]
        r0 = g * BM
        tm.tile_matmul(
            x_router[r0 : r0 + BM, ch * ROUTE_STEP : ch * ROUTE_STEP + ROUTE_STEP],
            router_p[0, ch * RS : ch * RS + RS],
            route_partial[ch * TP + r0 : ch * TP + r0 + BM, :],
            work,
            nr,
        )
        if is_last(route_left, g):
            for r in dace.map[0:nr] @ TB:
                append_to_stream(S, TOP_K, r0 + r, 0)

    @dace.program
    def top_k(S, g, staging, route_partial, per_expert_scale, count, pair_expert, pair_slot, pair_weight, top_k_left):
        """The experts of token ``g``: a thread per expert, then per chosen expert. The last token pushes the scan."""
        tile = staging[0 : BM * BN].reshape([BM, BN])
        chosen = dace.ndarray([K], dtype=dace.int32, storage=SHARED)
        # The token's router logits (the projection rounds them to bfloat16)
        for expert in dace.map[0:E] @ TB:
            total = dace.float32(0)
            for part in range(RP):
                total += route_partial[part * TP + g, expert]
            tile[0, expert] = total
        # The rank of every logit (ties: lowest index first): the top k are chosen
        for expert in dace.map[0:E] @ TB:
            v = dace.float32(tile[0, expert])
            rank = 0
            for other in range(E):
                v2 = dace.float32(tile[0, other])
                if v2 > v or (v2 == v and other < expert):
                    rank += 1
            if rank < K:
                chosen[rank] = expert
        # The softmax probabilities of the chosen experts, renormalized (the softmax denominator cancels) and scaled;
        # claim a row of every chosen expert
        for j in dace.map[0:K] @ TB:
            top = dace.float32(tile[0, chosen[0]])
            total = dace.float32(0)
            for j2 in range(K):
                total += math.exp(dace.float32(tile[0, chosen[j2]]) - top)
            c = chosen[j]
            pair_expert[g, j] = c
            pair_weight[g, j] = math.exp(dace.float32(tile[0, c]) - top) / total * per_expert_scale[c]
            pair_slot[g, j] = dace.atomic.add(count, c, 1)
        if is_last(top_k_left, 0):
            for _ in dace.map[0:1] @ TB:
                append_to_stream(S, SCAN, 0, 0)

    @dace.program
    def output(g, staging, x, dense_out, expert_acc, post_norm_1, post_norm_2, post_norm, out):
        """Normalizes both branches of token ``g``, adds them, normalizes, and adds the residual (in bfloat16, as the
        model). The sum of the branches stays in the staging memory."""
        sums = dace.ndarray([3], dtype=dace.float32, storage=SHARED)
        sums[:] = 0
        for d in dace.map[0:D] @ TB:
            v1 = dace.float32(dense_out[g, d])
            v2 = bf16(expert_acc[g, d])
            sums[0] += v1 * v1
            sums[1] += v2 * v2
        inv1 = inverse_rms(sums[0])
        inv2 = inverse_rms(sums[1])
        for d in dace.map[0:D] @ TB:
            b1 = bf16(dace.float32(dense_out[g, d]) * inv1 * post_norm_1[d])
            b2 = bf16(bf16(expert_acc[g, d]) * inv2 * post_norm_2[d])
            v = dace.bfloat16(b1 + b2)
            staging[d] = v
            sums[2] += dace.float32(v) * dace.float32(v)
        inv = inverse_rms(sums[2])
        for d in dace.map[0:D] @ TB:
            out[g, d] = dace.float32(x[g, d]) + bf16(dace.float32(staging[d]) * inv * post_norm[d])

    @dace.program
    def scan(
        S, count, pair_expert, pair_slot, pair_weight, tile_start, tile_expert, tile_rows, num_tiles, row_token,
        row_weight, gate_up_left
    ):  # fmt: skip
        """The tiles of every expert's rows (pairs of a token and an expert), and the expert gate/up tasks."""
        # Every expert's rows start at a whole tile. A thread per expert.
        tiles_of = dace.ndarray([E], dtype=dace.int32, storage=SHARED)
        for expert in dace.map[0:E] @ TB:
            tiles_of[expert] = (count[expert] + BM - 1) // BM
        for expert in dace.map[0:E] @ TB:
            start = 0
            for other in range(expert):
                start += tiles_of[other]
            tile_start[expert] = start
            for g2 in range(tiles_of[expert]):
                tile_expert[start + g2] = expert
                tile_rows[start + g2] = min(BM, count[expert] - g2 * BM)
            gate_up_left[expert] = tiles_of[expert] * (I // HALF)
            if expert == E - 1:
                num_tiles[0] = start + tiles_of[expert]
        # The rows of every expert, sorted by expert: their tokens and routing weights
        for j, tok in dace.map[0:K, 0:T] @ TB:  # A thread per token (the last parameter strides)
            row = tile_start[pair_expert[tok, j]] * BM + pair_slot[tok, j]
            row_token[row] = tok
            row_weight[row] = pair_weight[tok, j]
        # The expert gate/up tasks, alternating with the dense gate/up tasks that the prologue left (as many of each
        # as there are, then the rest), so that the dense products overlap the expert ones. The tasks of an expert are
        # those of its tiles (j // (I // HALF) is one of them).
        num_expert = num_tiles[0] * (I // HALF)
        num_dense = (TP // BM - DT) * (DI // HALF)
        both = 2 * min(num_expert, num_dense)
        for i in dace.map[0 : num_expert + num_dense] @ TB:
            dense = False
            j = i - num_dense
            if i < both:
                dense = i % 2 == 1
                j = i // 2
            elif num_dense > num_expert:
                dense = True
                j = i - num_expert
            if dense:
                append_to_stream(S, DENSE_GATE_UP, DT + j % (TP // BM - DT), j // (TP // BM - DT))
            else:  # The tiles of an expert's chunk one after the other: they read it from L2
                te = tile_expert[j // (I // HALF)]
                local = j - tile_start[te] * (I // HALF)
                n = (count[te] + BM - 1) // BM
                append_to_stream(S, EXPERT_GATE_UP, tile_start[te] + local % n, local // n)

    @dace.program
    def moe_block(
        x: dace.bfloat16[T, D],
        pre_norm: dace.float32[D],
        post_norm_1: dace.float32[D],
        router_scale: dace.float32[D],
        router_p: packed(E, D),
        per_expert_scale: dace.float32[E],
        pre_norm_2: dace.float32[D],
        post_norm_2: dace.float32[D],
        post_norm: dace.float32[D],
        dense_gate_up_p: packed(2 * DI, D),
        down_p: packed(D, DI),
        expert_gate_up_p: packed(E, 2 * I, D),
        expert_down_p: packed(E, D, I),
        out: dace.bfloat16[T, D],
        # Scratch
        inv_rms: dace.float32[T],
        x_router: dace.bfloat16[TP, D],
        route_partial: dace.float32[RP * TP, E],  # The router parts of every token, one after the other
        route_left: dace.int32[TP // BM],
        top_k_left: dace.int32[1],
        count: dace.int32[E],
        pair_expert: dace.int32[T, K],
        pair_slot: dace.int32[T, K],
        pair_weight: dace.float32[T, K],
        tile_start: dace.int32[E],
        tile_expert: dace.int32[NT],
        tile_rows: dace.int32[NT],
        dense_rows: dace.int32[TP // BM],
        num_tiles: dace.int32[1],
        row_token: dace.int32[NT * BM],
        row_weight: dace.float32[NT * BM],
        x_dense: dace.bfloat16[TP, D],
        x_expert: dace.bfloat16[T, D],
        h_dense: dace.bfloat16[TP, DI],
        h_expert: dace.bfloat16[NT * BM, I],
        gate_up_left: dace.int32[E],  # Of every expert's gate/up tasks
        dense_gate_up_left: dace.int32[TP // BM],
        dense_out: dace.bfloat16[TP, D],
        expert_acc: dace.float32[T, D],
        token_left: dace.int32[T],
    ):
        S = dace.define_stream(dace.int64, queue_size)

        for t in dace.map[0:T]:
            prologue(
                t, x, pre_norm, pre_norm_2, router_scale, inv_rms, x_dense, x_expert, x_router, expert_acc, row_token,
                token_left
            )  # fmt: skip
            if t == 0:
                first_tasks(S, count, dense_rows, route_left, dense_gate_up_left, top_k_left)

        for task, p in dace.consume(S, num_pes):
            # The fields of the task (see ``append_to_stream``), as symbols: they offset the operands of the task
            kind = task >> KIND << KIND
            g = (task >> FIELD) & MASK
            ch = task & MASK
            # The product of a tile overwrites the operands that tile_matmul staged: sharing their memory leaves room for
            # a third thread block on each multiprocessor (whose loads cover the gaps between the tasks of the others)
            staging = dace.ndarray([STAGING], dtype=dace.bfloat16, storage=SHARED)
            if kind == EXPERT_GATE_UP:
                expert_gate_up(
                    S, g, ch, staging, x_expert, expert_gate_up_p, tile_expert, tile_rows, tile_start, row_token,
                    count, gate_up_left, h_expert
                )  # fmt: skip
            elif kind == DENSE_GATE_UP:
                dense_gate_up(S, g, ch, staging, x_dense, dense_gate_up_p, dense_rows, dense_gate_up_left, h_dense)
            elif kind == EXPERT_DOWN:
                expert_down(
                    S, g, ch, staging, h_expert, expert_down_p, tile_expert, tile_rows, row_token, row_weight,
                    expert_acc, token_left
                )  # fmt: skip
            elif kind == DENSE_DOWN:
                dense_down(S, g, ch, staging, h_dense, down_p, dense_rows, dense_out, token_left)
            elif kind == ROUTE:
                route(S, g, ch, staging, x_router, router_p, route_partial, dense_rows, route_left)
            elif kind == TOP_K:
                top_k(
                    S, g, staging, route_partial, per_expert_scale, count, pair_expert, pair_slot, pair_weight,
                    top_k_left
                )  # fmt: skip
            elif kind == OUTPUT:
                output(g, staging, x, dense_out, expert_acc, post_norm_1, post_norm_2, post_norm, out)
            else:
                scan(
                    S, count, pair_expert, pair_slot, pair_weight, tile_start, tile_expert, tile_rows, num_tiles,
                    row_token, row_weight, gate_up_left
                )  # fmt: skip

    sdfg = moe_block.to_sdfg(simplify=True)
    to_gpu(sdfg)
    return sdfg


def to_gpu(sdfg: dace.SDFG) -> None:
    """Runs the program on the GPU: arguments in GPU memory, top-level maps as kernels of THREADS threads, the consume
    scope as a persistent kernel with a thread block per processing element. A thread-block map with more iterations
    than threads, or as many as data says, runs in strides of the thread block (each thread loops over its iterations)."""
    for desc in sdfg.arrays.values():
        if not isinstance(desc, dace.data.Scalar) and not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
        elif isinstance(desc, dace.data.Stream):
            # The task queue: allocated once (it is empty again after every call)
            desc.storage = dace.StorageType.GPU_Global
            desc.lifetime = dace.AllocationLifetime.Persistent
    # Without synchronizing the host at the end: on PyTorch's (default) stream, the caller synchronizes when it needs to
    for state in sdfg.states():
        state.nosync = True
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if parent.parent is sdfg and parent.entry_node(node) is None:
                node.map.schedule = dace.ScheduleType.GPU_Device
                node.map.gpu_block_size = [THREADS, 1, 1]
            elif node.map.schedule == TB:
                iterations = node.map.range.num_elements()
                node.map.gpu_strided = bool(dace.symbolic.issymbolic(iterations) or iterations > THREADS)
        elif isinstance(node, dace.nodes.ConsumeEntry):
            node.consume.schedule = dace.ScheduleType.GPU_Device


def arguments(w: Weights) -> dict[str, torch.Tensor]:
    """The weights as the program takes them (copies, made once): the gate and up projections interleaved in chunks of
    64 rows, so that a tile of 128 output columns holds matching gate and up columns, and all projections packed."""
    E, I2, D = w.expert_gate_up.shape
    DI = w.gate_proj.shape[0]
    half = BN // 2
    expert_gate_up = w.expert_gate_up.view(E, 2, I2 // 2 // half, half, D).transpose(1, 2).reshape(E, I2, D)
    dense_gate_up = torch.stack([w.gate_proj, w.up_proj]).view(2, DI // half, half, D).transpose(0, 1)
    dense_gate_up = dense_gate_up.reshape(2 * DI, D)
    return dict(
        pre_norm=w.pre_norm,
        post_norm_1=w.post_norm_1,
        router_scale=w.router_scale,
        router_p=pack(w.router_proj),
        per_expert_scale=w.per_expert_scale,
        pre_norm_2=w.pre_norm_2,
        post_norm_2=w.post_norm_2,
        post_norm=w.post_norm,
        dense_gate_up_p=pack(dense_gate_up),
        down_p=pack(w.down_proj),
        expert_gate_up_p=pack(expert_gate_up),
        expert_down_p=pack(w.expert_down),
    )


def workspace(config: Config, tokens: int) -> dict:
    """Scratch buffers and symbol values for ``tokens`` tokens."""
    padded = (tokens + BM - 1) // BM * BM
    tiles = tokens * config.top_k // BM + config.num_experts
    E, K, D = config.num_experts, config.top_k, config.hidden

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    return dict(
        inv_rms=zeros(tokens),
        x_router=zeros(padded, D, dtype=torch.bfloat16),
        route_partial=zeros(config.hidden // ROUTE_STEP * padded, E),
        route_left=zeros(padded // BM, dtype=torch.int32),
        top_k_left=zeros(1, dtype=torch.int32),
        count=zeros(E, dtype=torch.int32),
        pair_expert=zeros(tokens, K, dtype=torch.int32),
        pair_slot=zeros(tokens, K, dtype=torch.int32),
        pair_weight=zeros(tokens, K),
        tile_start=zeros(E, dtype=torch.int32),
        tile_expert=zeros(tiles, dtype=torch.int32),
        tile_rows=zeros(tiles, dtype=torch.int32),
        dense_rows=zeros(padded // BM, dtype=torch.int32),
        num_tiles=zeros(1, dtype=torch.int32),
        row_token=zeros(tiles * BM, dtype=torch.int32),
        row_weight=zeros(tiles * BM),
        x_dense=zeros(padded, D, dtype=torch.bfloat16),
        x_expert=zeros(tokens, D, dtype=torch.bfloat16),
        h_dense=zeros(padded, config.dense_intermediate, dtype=torch.bfloat16),
        h_expert=zeros(tiles * BM, config.expert_intermediate, dtype=torch.bfloat16),
        gate_up_left=zeros(config.num_experts, dtype=torch.int32),
        dense_gate_up_left=zeros(padded // BM, dtype=torch.int32),
        dense_out=zeros(padded, D, dtype=torch.bfloat16),
        expert_acc=zeros(tokens, D),
        token_left=zeros(tokens, dtype=torch.int32),
        T=tokens,
        TP=padded,
        NT=tiles,
        DT=1 if tokens * config.top_k < config.num_experts * BM else padded // BM,
    )


# The block as vLLM runs it (if vLLM is installed)


class VllmFeedForward(nn.Module):
    """The feed-forward part of vLLM's ``Gemma4DecoderLayer``: its modules (merged gate/up MLP, router with its
    routing kernel, and the fused MoE experts with their Triton kernels), and its forward code."""

    def __init__(self, config: Config, prefix: str = "layers.0"):
        super().__init__()
        self.top_k = config.top_k
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

    def load(self, w: Weights):
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
        # The post-loading step of vLLM's model loader (builds the fused MoE kernel)
        for module in (self.mlp.gate_up_proj, self.mlp.down_proj, self.router.proj, self.experts.routed_experts):
            module.quant_method.process_weights_after_loading(module)

    def experts_of(self, x: torch.Tensor) -> torch.Tensor:
        """The experts that the router picks for every token."""
        return self.router.routing_function(x, self.router(x), self.top_k, True)[1]

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


@contextlib.contextmanager
def vllm_block(config: Config, w: Weights):
    """vLLM's block with the weights ``w``, ready to run inside the context: eager mode (as with ``enforce_eager``),
    with vLLM's own kernels (RMSNorm, GELU-and-multiply) rather than PyTorch fallbacks, on a single GPU."""
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
            block = VllmFeedForward(config)
        torch.set_default_dtype(torch.float32)
        block.load(w)
        with set_forward_context(None, vllm_config), torch.inference_mode():
            yield block


# Comparison


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


def check(name: str, result: torch.Tensor, experts: torch.Tensor, expected: torch.Tensor, expected_experts) -> str:
    """Compares ``result`` with the reference. The router's logits are summed in another order than the reference's,
    which can flip near-ties between the k-th and the next expert: only the tokens that both route alike count."""
    alike = (experts.long().sort().values == expected_experts.sort().values).all(dim=1)
    error = ((result.float()[alike] - expected[alike]).norm() / expected[alike].norm()).item()
    assert error < 2e-2, f"{name} differs from the reference (relative error {error:.2e})"
    return f"{name} error {error:.1e}, {int((~alike).sum())} routed differently"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    parser.add_argument("--small", action="store_true", help="a small configuration, for testing")
    # Three thread blocks per multiprocessor of an RTX 3070 (as many as the registers of the workers allow)
    parser.add_argument("--pes", type=int, default=46 * 3, help="processing elements (thread blocks)")
    args = parser.parse_args()

    config = (
        Config(hidden=256, dense_intermediate=128, num_experts=128, top_k=4, expert_intermediate=128)
        if args.small
        else Config()
    )
    w = random_weights(config)
    program = build(config, args.pes)
    # The default stream, as PyTorch's (see ``to_gpu``)
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
        program = program.compile()
    weights = arguments(w)

    with contextlib.ExitStack() as stack:
        vllm = stack.enter_context(vllm_block(config, w)) if HAVE_VLLM else None
        for tokens in args.tokens:
            x = random_input(config, tokens)
            expected = reference(x, w, config).float()
            _, expected_experts = route(x, w, config)
            times = {"reference": time_ms(lambda: reference(x, w, config))}

            out = torch.empty_like(x)
            scratch = workspace(config, tokens)
            program(x=x, out=out, **weights, **scratch)
            errors = [check("DaCe", out, scratch["pair_expert"], expected, expected_experts)]
            # The time of the kernels, by events on the stream they run on (without Python's argument conversion)
            callargs, initargs = program.construct_arguments(x=x, out=out, **weights, **scratch)
            times["DaCe"] = time_ms(lambda: program.fast_call(callargs, initargs))

            if vllm is not None:
                errors.append(check("vLLM", vllm(x), vllm.experts_of(x), expected, expected_experts))
                times["vLLM"] = time_ms(lambda: vllm(x))
                times["vLLM (CUDA graph)"] = time_ms(graphed(lambda: vllm(x)))

            print(f"{tokens:5d} tokens: " + ", ".join(f"{name} {t:8.3f} ms" for name, t in times.items()))
            print("              (" + "; ".join(errors) + ")")
