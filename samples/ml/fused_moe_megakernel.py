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
one matrix product followed by an epilogue on the tile. The tasks are:

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

The products are NumPy matrix products of views (``x[r0:r0 + nr] @ w[c0:c0 + BN].T``, with a data-dependent number of
rows ``nr``), in which the rows of an expert's tokens are gathered (``x[row_token[r0:r0 + nr]]``). DaCe fuses the gather
and the transposition into the product (``GatherIntoMatrixProduct``, ``TransposeIntoMatrixProduct``), and computes a
product that the threads of a block run together with the block (the ``ThreadBlockMMA`` implementation of ``Gemm``:
tensor cores on GPUs that have them, staging the operands in a shared memory workspace of its own). The weights are
stored in blocks of 32 columns (``BlockMatrixOperands``, see ``dace.libraries.blas.blocked``), as inference engines pack
their weights once: the 32 columns of the 128 rows of a tile that a product stages at a time are then contiguous.

With ``--profile``, every task records its time on the GPU instead (the ``GPU_Events`` instrumentation of a node in a
kernel: an event per thread block that runs it, on the GPU's global timer). The sample then prints the time of every
kind of task, and the instrumentation report holds a timeline with a track per processing element.

The reference is the block in Hugging Face ``transformers``, which is also timed with each of its implementations of
the experts. The kernel stores and rounds values in bfloat16 where ``transformers`` does. If vLLM is installed (it pins
its own PyTorch version), the block as vLLM runs it is timed as well. Requires PyTorch, ``transformers`` and a CUDA GPU
with bfloat16 tensor cores (Ampere or newer).
"""

import argparse
import collections
import contextlib
import math

import numpy as np
import torch
from torch import nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextExperts,
    Gemma4TextMLP,
    Gemma4TextRouter,
)

import dace
from dace.codegen.instrumentation.report import InstrumentationReport
from dace.transformation.dataflow import GatherIntoMatrixProduct, TransposeIntoMatrixProduct
from dace.transformation.passes import BlockMatrixOperands

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

# The block in transformers: the reference

GEMMA4_26B_A4B = dict(
    hidden_size=2816, intermediate_size=2112, num_experts=128, top_k_experts=8, moe_intermediate_size=704
)
SMALL = dict(hidden_size=256, intermediate_size=128, num_experts=128, top_k_experts=4, moe_intermediate_size=128)


def block_config(**shapes) -> Gemma4TextConfig:
    """The configuration of a decoder layer with a mixture of experts, of the given shapes."""
    return Gemma4TextConfig(**shapes, enable_moe_block=True, num_hidden_layers=1, layer_types=["full_attention"])


# The block's RMSNorms (dense branch, experts, and their sum), as the decoder layer names them
NORMS = (
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm_1",
    "pre_feedforward_layernorm_2",
    "post_feedforward_layernorm_2",
    "post_feedforward_layernorm",
)


class FeedForward(nn.Module):
    """The feed-forward part of the ``transformers`` ``Gemma4TextDecoderLayer`` (its modules, and its forward code),
    with random weights of plausible magnitudes (the computation does not depend on their values)."""

    def __init__(self, config: Gemma4TextConfig, seed: int = 0):
        super().__init__()
        self.mlp = Gemma4TextMLP(config, layer_idx=0)
        self.router = Gemma4TextRouter(config)
        self.experts = Gemma4TextExperts(config)
        for norm in NORMS:
            self.add_module(norm, Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() == 1:  # Scales
                    p.copy_(1 + 0.1 * torch.randn(p.shape, generator=generator))
                else:
                    p.copy_(torch.randn(p.shape, generator=generator) / p.shape[-1] ** 0.5)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # As in Gemma4TextDecoderLayer.forward, after attention
        hidden_states = self.pre_feedforward_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
        _, top_k_weights, top_k_index = self.router(residual)
        hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
        hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = self.post_feedforward_layernorm(hidden_states_1 + hidden_states_2)
        return residual + hidden_states


def random_input(config: Gemma4TextConfig, tokens: int) -> torch.Tensor:
    """The residual stream of ``tokens`` tokens, the same for every run (which experts the tokens pick, and so how
    much of the weights a block reads, depends on it)."""
    generator = torch.Generator("cuda").manual_seed(tokens)
    return torch.randn(tokens, config.hidden_size, device="cuda", generator=generator).to(torch.bfloat16)


# The megakernel

TB = dace.ScheduleType.GPU_ThreadBlock
SHARED = dace.StorageType.GPU_Shared
WARPS = 4  # Warps per thread block
THREADS = 32 * WARPS
BM, BN = 64, 128  # Rows and columns of a tile
BLOCK = 32  # The weights are stored in blocks of BLOCK columns (see ``in_blocks``)
WEIGHTS = ("router_w", "dense_gate_up_w", "down_w", "expert_gate_up_w", "expert_down_w")
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


def build(config: Gemma4TextConfig, num_pes: int, queue_size: int = 1 << 18, profile: bool = False) -> dace.SDFG:
    """The block for ``config`` as a GPU SDFG. Its arguments are the input, the weights (see ``arguments``), the
    output, and scratch buffers (see ``workspace``). With ``profile``, every task records an event (see
    ``print_profile``)."""
    D, DI, I, E, K = (
        config.hidden_size,
        config.intermediate_size,
        config.moe_intermediate_size,
        config.num_experts,
        config.top_k_experts,
    )
    assert D % BN == 0 and (2 * DI) % BN == 0 and (2 * I) % BN == 0 and E == BN == THREADS and D % ROUTE_STEP == 0
    RP = D // ROUTE_STEP  # Router tasks per tile of tokens
    HALF = BN // 2  # Gate (and up) columns of a gate/up task
    # Constants in single precision (a Python float is a double on the GPU, which computes in double precision slowly)
    EPS = np.float32(config.rms_norm_eps)
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
        the experts and the router (the normalized input, rounded to bfloat16 as the model does, and scaled, rounding
        after each scale). Also resets the token's scratch memory."""
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
            x_dense[t, d] = n * dace.float32(pre_norm[d])
            x_router[t, d] = bf16(bf16(n) * dace.float32(router_scale[d])) * ROOT
            x_expert[t, d] = n * dace.float32(pre_norm_2[d])

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
        S, g, ch, tile, x_expert, expert_gate_up_w, tile_expert, tile_rows, tile_start, row_token, count,
        gate_up_left, h_expert
    ):  # fmt: skip
        """A tile of an expert's rows (the rows of their tokens) times a chunk of its gate and up projections. The last
        gate/up task of the expert (of all its tiles) pushes its down tasks, the tiles of a chunk one after the other:
        they read it from L2."""
        e = tile_expert[g]
        nr = tile_rows[g]
        r0 = g * BM
        tile[0:nr] = x_expert[row_token[r0 : r0 + nr]] @ expert_gate_up_w[e, ch * BN : ch * BN + BN, :].T
        gelu_times_up(tile, nr, h_expert, r0, ch)
        if is_last(gate_up_left, e):
            n = (count[e] + BM - 1) // BM  # Tiles of the expert
            for i in dace.map[0 : n * (D // BN)] @ TB:
                append_to_stream(S, EXPERT_DOWN, tile_start[e] + i % n, i // n)

    @dace.program
    def dense_gate_up(S, g, ch, tile, x_dense, dense_gate_up_w, dense_rows, dense_gate_up_left, h_dense):
        """BM tokens times a chunk of the dense gate and up projections. The last of the tokens pushes their down
        tasks."""
        nr = dense_rows[g]
        r0 = g * BM
        tile[0:nr] = x_dense[r0 : r0 + nr, :] @ dense_gate_up_w[ch * BN : ch * BN + BN, :].T
        gelu_times_up(tile, nr, h_dense, r0, ch)
        if is_last(dense_gate_up_left, g):
            for c in dace.map[0 : D // BN] @ TB:
                append_to_stream(S, DENSE_DOWN, g, c)

    @dace.program
    def expert_down(
        S, g, ch, tile, h_expert, expert_down_w, tile_expert, tile_rows, row_token, row_weight, expert_acc,
        token_left
    ):  # fmt: skip
        """A tile of an expert's rows times a chunk of its down projection, scaled by each row's routing weight and
        added to the output of the row's token. The model rounds every addition to bfloat16; the sum stays in float32
        here, whose atomic additions are native (before Hopper GPUs, those of bfloat16 are compare-and-swap loops, slow
        when many tiles add to a token). The last tile that adds to a token pushes its output task."""
        e = tile_expert[g]
        nr = tile_rows[g]
        r0 = g * BM
        tile[0:nr] = h_expert[r0 : r0 + nr, :] @ expert_down_w[e, ch * BN : ch * BN + BN, :].T
        for r, col in dace.map[0:nr, 0:BN] @ TB:
            expert_acc[row_token[r0 + r], ch * BN + col] += bf16(dace.float32(tile[r, col]) * row_weight[r0 + r])
        for r in dace.map[0:nr] @ TB:
            token = row_token[r0 + r]
            if dace.atomic.add(token_left, token, -1) == 1:
                append_to_stream(S, OUTPUT, token, 0)

    @dace.program
    def dense_down(S, g, ch, h_dense, down_w, dense_rows, dense_out, token_left):
        """BM tokens times a chunk of the dense down projection."""
        nr = dense_rows[g]
        r0 = g * BM
        dense_out[r0 : r0 + nr, ch * BN : ch * BN + BN] = h_dense[r0 : r0 + nr, :] @ down_w[ch * BN : ch * BN + BN, :].T
        for r in dace.map[0:nr] @ TB:
            if dace.atomic.add(token_left, r0 + r, -1) == 1:
                append_to_stream(S, OUTPUT, r0 + r, 0)

    @dace.program
    def route(S, g, ch, x_router, router_w, route_partial, dense_rows, route_left):
        """Part of the router logits of BM tokens: a slice of the input times the router weight (summed in float32,
        the type of the parts). The last part of these tokens pushes their top-k tasks."""
        nr = dense_rows[g]
        r0 = g * BM
        k0 = ch * ROUTE_STEP
        route_partial[ch * TP + r0 : ch * TP + r0 + nr, :] = (
            x_router[r0 : r0 + nr, k0 : k0 + ROUTE_STEP] @ router_w[:, k0 : k0 + ROUTE_STEP].T
        )
        if is_last(route_left, g):
            for r in dace.map[0:nr] @ TB:
                append_to_stream(S, TOP_K, r0 + r, 0)

    @dace.program
    def top_k(S, g, route_partial, per_expert_scale, count, pair_expert, pair_slot, pair_weight, top_k_left):
        """The experts of token ``g``: a thread per expert, then per chosen expert. The last token pushes the scan."""
        logits = dace.ndarray([E], dtype=dace.bfloat16, storage=SHARED)
        chosen = dace.ndarray([K], dtype=dace.int32, storage=SHARED)
        # The token's router logits (the projection rounds them to bfloat16)
        for expert in dace.map[0:E] @ TB:
            total = dace.float32(0)
            for part in range(RP):
                total += route_partial[part * TP + g, expert]
            logits[expert] = total
        # The rank of every logit (ties: lowest index first): the top k are chosen
        for expert in dace.map[0:E] @ TB:
            v = dace.float32(logits[expert])
            rank = 0
            for other in range(E):
                v2 = dace.float32(logits[other])
                if v2 > v or (v2 == v and other < expert):
                    rank += 1
            if rank < K:
                chosen[rank] = expert
        # The softmax probabilities of the chosen experts, renormalized (the softmax denominator cancels) and scaled;
        # claim a row of every chosen expert
        for j in dace.map[0:K] @ TB:
            top = dace.float32(logits[chosen[0]])
            total = dace.float32(0)
            for j2 in range(K):
                total += math.exp(dace.float32(logits[chosen[j2]]) - top)
            c = chosen[j]
            pair_expert[g, j] = c
            pair_weight[g, j] = math.exp(dace.float32(logits[c]) - top) / total * dace.float32(per_expert_scale[c])
            pair_slot[g, j] = dace.atomic.add(count, c, 1)
        if is_last(top_k_left, 0):
            for _ in dace.map[0:1] @ TB:
                append_to_stream(S, SCAN, 0, 0)

    @dace.program
    def branches(g, d, inv1, inv2, dense_out, expert_acc, post_norm_1, post_norm_2):
        """Element ``d`` of the sum of both branches of token ``g``, normalized (in bfloat16, as the model)."""
        b1 = bf16(dace.float32(dense_out[g, d]) * inv1 * dace.float32(post_norm_1[d]))
        b2 = bf16(bf16(expert_acc[g, d]) * inv2 * dace.float32(post_norm_2[d]))
        return bf16(b1 + b2)

    @dace.program
    def output(g, x, dense_out, expert_acc, post_norm_1, post_norm_2, post_norm, out):
        """Normalizes both branches of token ``g``, adds them, normalizes, and adds the residual (in bfloat16, as the
        model). The sum of the branches is computed again rather than kept, as it is read from L2."""
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
            v = branches(g, d, inv1, inv2, dense_out, expert_acc, post_norm_1, post_norm_2)
            sums[2] += v * v
        inv = inverse_rms(sums[2])
        for d in dace.map[0:D] @ TB:
            v = branches(g, d, inv1, inv2, dense_out, expert_acc, post_norm_1, post_norm_2)
            out[g, d] = dace.float32(x[g, d]) + bf16(v * inv * dace.float32(post_norm[d]))

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
        pre_norm: dace.bfloat16[D],
        post_norm_1: dace.bfloat16[D],
        router_scale: dace.bfloat16[D],
        router_w: dace.bfloat16[E, D],
        per_expert_scale: dace.bfloat16[E],
        pre_norm_2: dace.bfloat16[D],
        post_norm_2: dace.bfloat16[D],
        post_norm: dace.bfloat16[D],
        dense_gate_up_w: dace.bfloat16[2 * DI, D],
        down_w: dace.bfloat16[D, DI],
        expert_gate_up_w: dace.bfloat16[E, 2 * I, D],
        expert_down_w: dace.bfloat16[E, D, I],
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
            tile = dace.ndarray([BM, BN], dtype=dace.bfloat16, storage=SHARED)  # The product of a tile
            if kind == EXPERT_GATE_UP:
                expert_gate_up(
                    S, g, ch, tile, x_expert, expert_gate_up_w, tile_expert, tile_rows, tile_start, row_token,
                    count, gate_up_left, h_expert
                )  # fmt: skip
            elif kind == DENSE_GATE_UP:
                dense_gate_up(S, g, ch, tile, x_dense, dense_gate_up_w, dense_rows, dense_gate_up_left, h_dense)
            elif kind == EXPERT_DOWN:
                expert_down(
                    S, g, ch, tile, h_expert, expert_down_w, tile_expert, tile_rows, row_token, row_weight,
                    expert_acc, token_left
                )  # fmt: skip
            elif kind == DENSE_DOWN:
                dense_down(S, g, ch, h_dense, down_w, dense_rows, dense_out, token_left)
            elif kind == ROUTE:
                route(S, g, ch, x_router, router_w, route_partial, dense_rows, route_left)
            elif kind == TOP_K:
                top_k(
                    S, g, route_partial, per_expert_scale, count, pair_expert, pair_slot, pair_weight,
                    top_k_left
                )  # fmt: skip
            elif kind == OUTPUT:
                output(g, x, dense_out, expert_acc, post_norm_1, post_norm_2, post_norm, out)
            else:
                scan(
                    S, count, pair_expert, pair_slot, pair_weight, tile_start, tile_expert, tile_rows, num_tiles,
                    row_token, row_weight, gate_up_left
                )  # fmt: skip

    sdfg = moe_block.to_sdfg(simplify=False)
    if profile:  # The nested SDFG of every task call (which stays a nested SDFG, rather than being inlined)
        tasks = (expert_gate_up, dense_gate_up, expert_down, dense_down, route, top_k, output, scan)
        names = {task.name: task.f.__name__ for task in tasks}
        for node, state in sdfg.all_nodes_recursive():  # (The calls, not the nested SDFGs of the tasks' own maps)
            if isinstance(node, dace.nodes.NestedSDFG) and node.label in names and state.entry_node(node) is None:
                node.label = names[node.label]
                node.instrument = dace.InstrumentationType.GPU_Events
    sdfg.simplify(validate=False)  # As the frontend does: shared memory is valid once the program is on the GPU
    # The gathers of rows and the transposed weights, read by the products themselves
    sdfg.apply_transformations_repeated([GatherIntoMatrixProduct, TransposeIntoMatrixProduct], validate=False)
    BlockMatrixOperands(WEIGHTS, BLOCK).apply_pass(sdfg, {})  # The program then takes the weights in blocks
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


def in_blocks(weight: torch.Tensor) -> torch.Tensor:
    """A weight ``[..., N, K]`` stored in blocks of BLOCK columns, ``[..., K / BLOCK, N, BLOCK]`` (see
    ``dace.libraries.blas.blocked``): a copy."""
    *batch, n, k = weight.shape
    return weight.reshape(*batch, n, k // BLOCK, BLOCK).transpose(-3, -2).contiguous()


def arguments(block: FeedForward) -> dict[str, torch.Tensor]:
    """The weights of ``block`` as the program takes them (copies, made once): the gate and up projections interleaved
    in chunks of 64 rows, so that a tile of 128 output columns holds matching gate and up columns, and the weights of
    the products in blocks of columns (see ``in_blocks``)."""
    w = {name: p.detach() for name, p in block.named_parameters()}
    E, I2, D = w["experts.gate_up_proj"].shape
    DI = w["mlp.gate_proj.weight"].shape[0]
    half = BN // 2
    expert_gate_up = w["experts.gate_up_proj"].view(E, 2, I2 // 2 // half, half, D).transpose(1, 2)
    dense_gate_up = torch.stack([w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"]]).view(2, DI // half, half, D)
    result = dict(
        pre_norm=w["pre_feedforward_layernorm.weight"],
        post_norm_1=w["post_feedforward_layernorm_1.weight"],
        router_scale=w["router.scale"],
        router_w=w["router.proj.weight"],
        per_expert_scale=w["router.per_expert_scale"],
        pre_norm_2=w["pre_feedforward_layernorm_2.weight"],
        post_norm_2=w["post_feedforward_layernorm_2.weight"],
        post_norm=w["post_feedforward_layernorm.weight"],
        dense_gate_up_w=dense_gate_up.transpose(0, 1).reshape(2 * DI, D),
        down_w=w["mlp.down_proj.weight"],
        expert_gate_up_w=expert_gate_up.reshape(E, I2, D),
        expert_down_w=w["experts.down_proj"],
    )
    return {name: in_blocks(t) if name in WEIGHTS else t for name, t in result.items()}


def workspace(config: Gemma4TextConfig, tokens: int) -> dict:
    """Scratch buffers and symbol values for ``tokens`` tokens."""
    padded = (tokens + BM - 1) // BM * BM
    E, K, D = config.num_experts, config.top_k_experts, config.hidden_size
    tiles = tokens * K // BM + E

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    return dict(
        inv_rms=zeros(tokens),
        x_router=zeros(padded, D, dtype=torch.bfloat16),
        route_partial=zeros(D // ROUTE_STEP * padded, E),
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
        h_dense=zeros(padded, config.intermediate_size, dtype=torch.bfloat16),
        h_expert=zeros(tiles * BM, config.moe_intermediate_size, dtype=torch.bfloat16),
        gate_up_left=zeros(E, dtype=torch.int32),
        dense_gate_up_left=zeros(padded // BM, dtype=torch.int32),
        dense_out=zeros(padded, D, dtype=torch.bfloat16),
        expert_acc=zeros(tokens, D),
        token_left=zeros(tokens, dtype=torch.int32),
        T=tokens,
        TP=padded,
        NT=tiles,
        DT=1 if tokens * K < E * BM else padded // BM,
    )


# The block as vLLM runs it (if vLLM is installed)


class VllmFeedForward(nn.Module):
    """The feed-forward part of vLLM's ``Gemma4DecoderLayer``: its modules (merged gate/up MLP, router with its
    routing kernel, and the fused MoE experts with their Triton kernels), and its forward code."""

    def __init__(self, config: Gemma4TextConfig, prefix: str = "layers.0"):
        super().__init__()
        self.top_k = config.top_k_experts
        self.mlp = Gemma4MLP(
            config.hidden_size, config.intermediate_size, config.hidden_activation, prefix=f"{prefix}.mlp"
        )
        self.router = Gemma4Router(config, prefix=f"{prefix}.router")
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            prefix=f"{prefix}.experts",
            custom_routing_function=self.router.routing_function,
            activation="gelu_tanh",
        )
        for norm in NORMS:
            self.add_module(norm, RMSNorm(config.hidden_size, eps=config.rms_norm_eps))

    def load(self, block: FeedForward):
        """The weights of ``block``, then the post-loading step of vLLM's model loader (builds the fused MoE kernel)."""
        with torch.no_grad():
            self.mlp.gate_up_proj.weight.copy_(torch.cat([block.mlp.gate_proj.weight, block.mlp.up_proj.weight]))
            self.mlp.down_proj.weight.copy_(block.mlp.down_proj.weight)
            self.experts.routed_experts.w13_weight.copy_(block.experts.gate_up_proj)
            self.experts.routed_experts.w2_weight.copy_(block.experts.down_proj)
            self.router.proj.weight.copy_(block.router.proj.weight)
            self.router.scale.copy_(block.router.scale)
            self.router.per_expert_scale.copy_(block.router.per_expert_scale)
            for norm in NORMS:
                self.get_submodule(norm).weight.copy_(block.get_submodule(norm).weight)
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
def vllm_block(config: Gemma4TextConfig, block: FeedForward):
    """vLLM's block with the weights of ``block``, ready to run inside the context: eager mode (as with
    ``enforce_eager``), with vLLM's own kernels (RMSNorm, GELU-and-multiply) rather than PyTorch fallbacks, on a single
    GPU."""
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
            vllm = VllmFeedForward(config)
        torch.set_default_dtype(torch.float32)
        vllm.load(block)
        with set_forward_context(None, vllm_config):
            yield vllm


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


def print_profile(report: InstrumentationReport, num_pes: int):
    """The time of every kind of task on the GPU, from the events of the tasks."""
    events = [e for e in report.events if e.additional_info and "block" in e.additional_info]
    span = max(e.timestamp + e.duration for e in events) - min(e.timestamp for e in events)  # Microseconds
    tasks = collections.defaultdict(list)
    for e in events:
        tasks[e.name].append(e)
    for name, of_kind in sorted(tasks.items(), key=lambda t: -sum(e.duration for e in t[1])):
        total = sum(e.duration for e in of_kind)
        cycles = np.mean([e.additional_info["cycles"] for e in of_kind])
        print(
            f"  {name:>14}: {len(of_kind):6d} tasks of {total / len(of_kind):7.2f} us ({cycles:7.0f} cycles), "
            f"{total / (num_pes * span):6.1%} of the processing elements' time"
        )
    busy = sum(e.duration for e in events) / (num_pes * span)
    print(f"  {len(events)} tasks in {span / 1000:.3f} ms, {busy:.1%} busy; timeline in {report.filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    parser.add_argument("--small", action="store_true", help="a small configuration, for testing")
    # Three thread blocks per multiprocessor (as many as the registers of the workers allow, 150 per thread)
    pes = 3 * torch.cuda.get_device_properties().multi_processor_count
    parser.add_argument("--pes", type=int, default=pes, help="processing elements (thread blocks)")
    parser.add_argument("--profile", action="store_true", help="the time of every task instead of the comparison")
    args = parser.parse_args()

    config = block_config(**(SMALL if args.small else GEMMA4_26B_A4B))
    block = FeedForward(config).to("cuda", torch.bfloat16)
    program = build(config, args.pes, profile=args.profile)
    # The default stream, as PyTorch's (see ``to_gpu``), and a report of every call (for --profile)
    with (
        dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1),
        dace.config.set_temporary("instrumentation", "report_each_invocation", value=True),
    ):
        program = program.compile()
    weights = arguments(block)

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        vllm = stack.enter_context(vllm_block(config, block)) if HAVE_VLLM else None
        for tokens in args.tokens:
            x = random_input(config, tokens)
            config._experts_implementation = "eager"  # A loop over the experts: the reference
            expected = block(x).float()
            expected_experts = block.router(x)[2]

            out = torch.empty_like(x)
            scratch = workspace(config, tokens)
            program(x=x, out=out, **weights, **scratch)
            errors = [check("DaCe", out, scratch["pair_expert"], expected, expected_experts)]
            if args.profile:
                print(f"{tokens:5d} tokens ({errors[0]}):")
                print_profile(program.sdfg.get_latest_report(), args.pes)
                continue

            times = {"transformers": time_ms(lambda: block(x))}
            # Without a Python loop: a grouped GEMM over the experts' rows (which loops over the experts on the host
            # before Hopper GPUs), and a batched GEMM over the token's experts for every token (no host
            # synchronization, but a copy of the weights of every token's experts: while they fit, for decoding)
            config._experts_implementation = "grouped_mm"
            errors.append(check("grouped_mm", block(x), expected_experts, expected, expected_experts))
            times["grouped_mm"] = time_ms(lambda: block(x))
            if tokens * config.top_k_experts <= config.num_experts:
                config._experts_implementation = "batched_mm"
                errors.append(check("batched_mm", block(x), expected_experts, expected, expected_experts))
                times["batched_mm (CUDA graph)"] = time_ms(graphed(lambda: block(x)))

            # The time of the kernels, by events on the stream they run on (without Python's argument conversion)
            callargs, initargs = program.construct_arguments(x=x, out=out, **weights, **scratch)
            times["DaCe"] = time_ms(lambda: program.fast_call(callargs, initargs))

            if vllm is not None:
                errors.append(check("vLLM", vllm(x), vllm.experts_of(x), expected, expected_experts))
                times["vLLM"] = time_ms(lambda: vllm(x))
                times["vLLM (CUDA graph)"] = time_ms(graphed(lambda: vllm(x)))

            print(f"{tokens:5d} tokens: " + ", ".join(f"{name} {t:.3f} ms" for name, t in times.items()))
            print("             (" + "; ".join(errors) + ")")
