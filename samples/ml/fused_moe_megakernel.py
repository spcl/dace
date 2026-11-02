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

The products are NumPy matrix products of views (``x[first_row : first_row + rows] @ w[columns].T``, with a
data-dependent number of ``rows``), in which the rows of an expert's tokens are gathered
(``x[row_token[first_row : first_row + rows]]``). DaCe fuses the gather and the transposition into the product
(``GatherIntoMatrixProduct``, ``TransposeIntoMatrixProduct``), and computes a product that the threads of a block run
together with the block (the ``ThreadBlockMMA`` implementation of ``Gemm``:
tensor cores on GPUs that have them, staging the operands in a shared memory workspace of its own). The weights are
stored in blocks of 32 columns (``BlockMatrixOperands``, see ``dace.libraries.blas.blocked``), as inference engines pack
their weights once: the 32 columns of the 128 rows of a tile that a product stages at a time are then contiguous.

With ``--profile``, every task records its time on the GPU instead (the ``GPU_Events`` instrumentation of the task's
call, which inlining keeps as a region of the kernel's code: an event per thread block that runs it, on the GPU's
global timer). The kernel is otherwise the same. The sample then prints the time of every kind of task, and the
instrumentation report holds a timeline with a track per processing element.

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
            for parameter in self.parameters():
                if parameter.dim() == 1:  # Scales
                    parameter.copy_(1 + 0.1 * torch.randn(parameter.shape, generator=generator))
                else:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) / parameter.shape[-1] ** 0.5)

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

WARPS = 4  # Warps per thread block
THREADS = 32 * WARPS
TILE_ROWS, TILE_COLUMNS = 64, 128  # Rows and columns of a tile
BLOCK = 32  # The weights are stored in blocks of BLOCK columns (see ``in_blocks``)
WEIGHTS = ("router_w", "dense_gate_up_w", "down_w", "expert_gate_up_w", "expert_down_w")
# The kinds of tasks, multiples of ``1 << KIND_SHIFT`` (64-bit constants): a task is its kind, a tile and a chunk (see
# ``append_to_stream``)
KIND_SHIFT, INDEX_SHIFT = 40, 20
FIELD_MASK = (1 << (KIND_SHIFT - INDEX_SHIFT)) - 1  # Of a tile or chunk
EXPERT_GATE_UP, EXPERT_DOWN, DENSE_GATE_UP, DENSE_DOWN, ROUTE, TOP_K, SCAN, OUTPUT = (k << KIND_SHIFT for k in range(8))
ROUTE_STEP = 256  # Columns of the input that a router task multiplies (the router's inner dimension is split)

TOKENS = dace.symbol("TOKENS")  # Tokens
PADDED_TOKENS = dace.symbol("PADDED_TOKENS")  # Tokens, padded to a multiple of TILE_ROWS
# The most expert tiles there can be: TOKENS * top_k / TILE_ROWS + num_experts
EXPERT_TILES = dace.symbol("EXPERT_TILES")
# The tiles of tokens whose dense gate/up tasks the prologue pushes: all of them, unless an expert has fewer rows than a
# tile on average. The expert tasks are then bound by memory and the dense ones by compute (from 64 tokens on): those
# of all tokens but the first 64 run among the expert tasks (the scan pushes them).
FIRST_DENSE_TILES = dace.symbol("FIRST_DENSE_TILES")


def build(config: Gemma4TextConfig, num_pes: int, queue_size: int = 1 << 18, profile: bool = False) -> dace.SDFG:
    """The block for ``config`` as a GPU SDFG. Its arguments are the input, the weights (see ``arguments``), the
    output, and scratch buffers (see ``workspace``). With ``profile``, every task records an event (see
    ``print_profile``)."""
    HIDDEN, DENSE_INTERMEDIATE, EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_PER_TOKEN = (
        config.hidden_size,
        config.intermediate_size,
        config.moe_intermediate_size,
        config.num_experts,
        config.top_k_experts,
    )
    assert (
        HIDDEN % TILE_COLUMNS == 0
        and (2 * DENSE_INTERMEDIATE) % TILE_COLUMNS == 0
        and (2 * EXPERT_INTERMEDIATE) % TILE_COLUMNS == 0
        and EXPERTS == TILE_COLUMNS == THREADS
        and HIDDEN % ROUTE_STEP == 0
    )
    ROUTER_PARTS = HIDDEN // ROUTE_STEP  # Router tasks per tile of tokens
    GATE_COLUMNS = TILE_COLUMNS // 2  # Gate (and up) columns of a gate/up task
    # Constants in single precision (a Python float is a double on the GPU, which computes in double precision slowly)
    RMS_EPSILON = np.float32(config.rms_norm_eps)
    ROUTER_INPUT_SCALE = np.float32(HIDDEN**-0.5)
    GELU_SQRT_2_OVER_PI = np.float32(math.sqrt(2 / math.pi))
    GELU_CUBIC = np.float32(0.044715)
    ONE_HALF = np.float32(0.5)

    # Helpers

    @dace.program
    def append_to_stream(queue, kind: dace.int64, index: dace.int64, chunk: dace.int64):
        """Queues a task: ``kind | index << INDEX_SHIFT | chunk``."""
        queue.append(kind | index << INDEX_SHIFT | chunk)

    @dace.program
    def is_last(counter, index):
        """Counts down ``counter[index]`` once for the thread block: whether this was its last count (an
        acquire-release update, so that the last block sees what the others wrote)."""
        last = dace.ndarray([1], dtype=dace.int32, storage=dace.StorageType.GPU_Shared)
        for _ in dace.map[0:1] @ dace.ScheduleType.GPU_ThreadBlock:
            last[0] = dace.atomic.add(counter, index, -1) == 1
        return last[0] == 1

    @dace.program
    def bf16(value: dace.float32):
        """``value`` rounded to bfloat16, as the model rounds it."""
        return dace.float32(dace.bfloat16(value))

    @dace.program
    def inverse_rms(sum_of_squares: dace.float32):
        """The inverse root mean square of a token, from the sum of the squares of its values."""
        return 1 / math.sqrt(sum_of_squares / HIDDEN + RMS_EPSILON)

    # The prologue, a thread block per token

    @dace.program
    def prologue(
        token,
        x,
        pre_norm,
        pre_norm_2,
        router_scale,
        inv_rms,
        x_dense,
        x_expert,
        x_router,
        expert_acc,
        row_token,
        token_left,
    ):
        """The RMS of ``token`` (all the pre-norms normalize the same residual) and the inputs of the dense branch, the
        experts and the router (the normalized input, rounded to bfloat16 as the model does, and scaled, rounding after
        each scale). Also resets the token's scratch memory."""
        sum_of_squares = dace.ndarray([1], dtype=dace.float32, storage=dace.StorageType.GPU_Shared)
        sum_of_squares[0] = 0
        for column in dace.map[0:HIDDEN] @ dace.ScheduleType.GPU_ThreadBlock:
            value = dace.float32(x[token, column])
            sum_of_squares[0] += value * value
            expert_acc[token, column] = 0
        # The blocks reset the rows of the expert tiles together
        for row in dace.map[token : EXPERT_TILES * TILE_ROWS : TOKENS] @ dace.ScheduleType.GPU_ThreadBlock:
            row_token[row] = -1
        # The down tiles that add to the token: its experts', and the dense ones
        token_left[token] = (EXPERTS_PER_TOKEN + 1) * (HIDDEN // TILE_COLUMNS)
        inverse = inverse_rms(sum_of_squares[0])
        inv_rms[token] = inverse
        for column in dace.map[0:HIDDEN] @ dace.ScheduleType.GPU_ThreadBlock:
            normalized = dace.float32(x[token, column]) * inverse
            x_dense[token, column] = normalized * dace.float32(pre_norm[column])
            x_router[token, column] = bf16(bf16(normalized) * dace.float32(router_scale[column])) * ROUTER_INPUT_SCALE
            x_expert[token, column] = normalized * dace.float32(pre_norm_2[column])

    @dace.program
    def first_tasks(queue, count, dense_rows, route_left, dense_gate_up_left, top_k_left):
        """Resets the counters and queues the first tasks: the router's (tiles of tokens times slices of the input),
        whose last one pushes the top-k tasks of the tile, whose last one pushes the scan that pushes the expert tasks;
        then the dense gate/up tasks of the first tiles of tokens (see ``FIRST_DENSE_TILES``), the tiles of a chunk of
        the weight one after the other: they read it from L2."""
        for expert in dace.map[0:EXPERTS] @ dace.ScheduleType.GPU_ThreadBlock:
            count[expert] = 0
        for tile_index in dace.map[0 : PADDED_TOKENS // TILE_ROWS] @ dace.ScheduleType.GPU_ThreadBlock:
            dense_rows[tile_index] = min(TILE_ROWS, TOKENS - tile_index * TILE_ROWS)
            route_left[tile_index] = ROUTER_PARTS
            dense_gate_up_left[tile_index] = DENSE_INTERMEDIATE // GATE_COLUMNS
        top_k_left[0] = TOKENS  # A top-k task per token
        for i in (
            dace.map[
                0 : PADDED_TOKENS // TILE_ROWS * ROUTER_PARTS + FIRST_DENSE_TILES * (DENSE_INTERMEDIATE // GATE_COLUMNS)
            ]
            @ dace.ScheduleType.GPU_ThreadBlock
        ):
            if i < PADDED_TOKENS // TILE_ROWS * ROUTER_PARTS:
                append_to_stream(queue, ROUTE, i // ROUTER_PARTS, i % ROUTER_PARTS)
            else:
                j = i - PADDED_TOKENS // TILE_ROWS * ROUTER_PARTS
                append_to_stream(queue, DENSE_GATE_UP, j % FIRST_DENSE_TILES, j // FIRST_DENSE_TILES)

    # The tasks. ``tile_index`` is a tile of rows (TILE_ROWS tokens or rows of experts), ``chunk`` a chunk of columns.

    @dace.program
    def gelu_times_up(tile, rows, h, first_row, chunk):
        """GELU(gate) * up of ``rows`` rows, where the columns of the tile are GATE_COLUMNS of the gate projection, then
        GATE_COLUMNS of the up one: into rows ``first_row`` of ``h``, chunk ``chunk``."""
        for i in dace.map[0 : rows * GATE_COLUMNS] @ dace.ScheduleType.GPU_ThreadBlock:
            gate = dace.float32(tile[i // GATE_COLUMNS, i % GATE_COLUMNS])
            gelu = ONE_HALF * gate * (1 + math.tanh(GELU_SQRT_2_OVER_PI * (gate + GELU_CUBIC * gate * gate * gate)))
            h[first_row + i // GATE_COLUMNS, chunk * GATE_COLUMNS + i % GATE_COLUMNS] = bf16(gelu) * dace.float32(
                tile[i // GATE_COLUMNS, GATE_COLUMNS + i % GATE_COLUMNS]
            )

    @dace.program
    def expert_gate_up(
        queue,
        tile_index,
        chunk,
        tile,
        x_expert,
        expert_gate_up_w,
        tile_expert,
        tile_rows,
        tile_start,
        row_token,
        count,
        gate_up_left,
        h_expert,
    ):
        """A tile of an expert's rows (the rows of their tokens) times a chunk of its gate and up projections. The last
        gate/up task of the expert (of all its tiles) pushes its down tasks, the tiles of a chunk one after the other:
        they read it from L2."""
        expert = tile_expert[tile_index]
        rows = tile_rows[tile_index]
        first_row = tile_index * TILE_ROWS
        first_column = chunk * TILE_COLUMNS
        tile[0:rows] = (
            x_expert[row_token[first_row : first_row + rows]]
            @ expert_gate_up_w[expert, first_column : first_column + TILE_COLUMNS, :].T
        )
        gelu_times_up(tile, rows, h_expert, first_row, chunk)
        if is_last(gate_up_left, expert):
            expert_tiles = (count[expert] + TILE_ROWS - 1) // TILE_ROWS  # Tiles of the expert
            for i in dace.map[0 : expert_tiles * (HIDDEN // TILE_COLUMNS)] @ dace.ScheduleType.GPU_ThreadBlock:
                append_to_stream(queue, EXPERT_DOWN, tile_start[expert] + i % expert_tiles, i // expert_tiles)

    @dace.program
    def dense_gate_up(
        queue, tile_index, chunk, tile, x_dense, dense_gate_up_w, dense_rows, dense_gate_up_left, h_dense
    ):
        """TILE_ROWS tokens times a chunk of the dense gate and up projections. The last of the tokens pushes their down
        tasks."""
        rows = dense_rows[tile_index]
        first_row = tile_index * TILE_ROWS
        first_column = chunk * TILE_COLUMNS
        tile[0:rows] = (
            x_dense[first_row : first_row + rows, :] @ dense_gate_up_w[first_column : first_column + TILE_COLUMNS, :].T
        )
        gelu_times_up(tile, rows, h_dense, first_row, chunk)
        if is_last(dense_gate_up_left, tile_index):
            for down_chunk in dace.map[0 : HIDDEN // TILE_COLUMNS] @ dace.ScheduleType.GPU_ThreadBlock:
                append_to_stream(queue, DENSE_DOWN, tile_index, down_chunk)

    @dace.program
    def expert_down(
        queue,
        tile_index,
        chunk,
        tile,
        h_expert,
        expert_down_w,
        tile_expert,
        tile_rows,
        row_token,
        row_weight,
        expert_acc,
        token_left,
    ):
        """A tile of an expert's rows times a chunk of its down projection, scaled by each row's routing weight and
        added to the output of the row's token. The model rounds every addition to bfloat16; the sum stays in float32
        here, whose atomic additions are native (before Hopper GPUs, those of bfloat16 are compare-and-swap loops, slow
        when many tiles add to a token). The last tile that adds to a token pushes its output task."""
        expert = tile_expert[tile_index]
        rows = tile_rows[tile_index]
        first_row = tile_index * TILE_ROWS
        first_column = chunk * TILE_COLUMNS
        tile[0:rows] = (
            h_expert[first_row : first_row + rows, :]
            @ expert_down_w[expert, first_column : first_column + TILE_COLUMNS, :].T
        )
        for row, column in dace.map[0:rows, 0:TILE_COLUMNS] @ dace.ScheduleType.GPU_ThreadBlock:
            expert_acc[row_token[first_row + row], first_column + column] += bf16(
                dace.float32(tile[row, column]) * row_weight[first_row + row]
            )
        for row in dace.map[0:rows] @ dace.ScheduleType.GPU_ThreadBlock:
            token_of_row = row_token[first_row + row]
            if dace.atomic.add(token_left, token_of_row, -1) == 1:
                append_to_stream(queue, OUTPUT, token_of_row, 0)

    @dace.program
    def dense_down(queue, tile_index, chunk, tile, h_dense, down_w, dense_rows, dense_out, token_left):
        """TILE_ROWS tokens times a chunk of the dense down projection. The product goes through the tile in shared
        memory: the rows it then copies to the output are whole lines, where the accumulators of the threads are
        scattered."""
        rows = dense_rows[tile_index]
        first_row = tile_index * TILE_ROWS
        first_column = chunk * TILE_COLUMNS
        tile[0:rows] = (
            h_dense[first_row : first_row + rows, :] @ down_w[first_column : first_column + TILE_COLUMNS, :].T
        )
        for row, column in dace.map[0:rows, 0:TILE_COLUMNS] @ dace.ScheduleType.GPU_ThreadBlock:
            dense_out[first_row + row, first_column + column] = tile[row, column]
        for row in dace.map[0:rows] @ dace.ScheduleType.GPU_ThreadBlock:
            if dace.atomic.add(token_left, first_row + row, -1) == 1:
                append_to_stream(queue, OUTPUT, first_row + row, 0)

    @dace.program
    def route(queue, tile_index, chunk, x_router, router_w, route_partial, dense_rows, route_left):
        """Part of the router logits of TILE_ROWS tokens: a slice of the input times the router weight (summed in
        float32, the type of the parts). The last part of these tokens pushes their top-k tasks."""
        rows = dense_rows[tile_index]
        first_row = tile_index * TILE_ROWS
        first_column = chunk * ROUTE_STEP
        route_partial[chunk * PADDED_TOKENS + first_row : chunk * PADDED_TOKENS + first_row + rows, :] = (
            x_router[first_row : first_row + rows, first_column : first_column + ROUTE_STEP]
            @ router_w[:, first_column : first_column + ROUTE_STEP].T
        )
        if is_last(route_left, tile_index):
            for row in dace.map[0:rows] @ dace.ScheduleType.GPU_ThreadBlock:
                append_to_stream(queue, TOP_K, first_row + row, 0)

    @dace.program
    def top_k(queue, token, route_partial, per_expert_scale, count, pair_expert, pair_slot, pair_weight, top_k_left):
        """The experts of ``token``: a thread per expert, then per chosen expert. The last token pushes the scan."""
        logits = dace.ndarray([EXPERTS], dtype=dace.bfloat16, storage=dace.StorageType.GPU_Shared)
        chosen = dace.ndarray([EXPERTS_PER_TOKEN], dtype=dace.int32, storage=dace.StorageType.GPU_Shared)
        # The token's router logits (the projection rounds them to bfloat16)
        for expert in dace.map[0:EXPERTS] @ dace.ScheduleType.GPU_ThreadBlock:
            total = dace.float32(0)
            for part in range(ROUTER_PARTS):
                total += route_partial[part * PADDED_TOKENS + token, expert]
            logits[expert] = total
        # The rank of every logit (ties: lowest index first): the top k are chosen
        for expert in dace.map[0:EXPERTS] @ dace.ScheduleType.GPU_ThreadBlock:
            value = dace.float32(logits[expert])
            rank = 0
            for other in range(EXPERTS):
                other_value = dace.float32(logits[other])
                if other_value > value or (other_value == value and other < expert):
                    rank += 1
            if rank < EXPERTS_PER_TOKEN:
                chosen[rank] = expert
        # The softmax probabilities of the chosen experts, renormalized (the softmax denominator cancels) and scaled;
        # claim a row of every chosen expert
        for j in dace.map[0:EXPERTS_PER_TOKEN] @ dace.ScheduleType.GPU_ThreadBlock:
            top = dace.float32(logits[chosen[0]])
            total = dace.float32(0)
            for other in range(EXPERTS_PER_TOKEN):
                total += math.exp(dace.float32(logits[chosen[other]]) - top)
            chosen_expert = chosen[j]
            pair_expert[token, j] = chosen_expert
            pair_weight[token, j] = (
                math.exp(dace.float32(logits[chosen_expert]) - top)
                / total
                * dace.float32(per_expert_scale[chosen_expert])
            )
            pair_slot[token, j] = dace.atomic.add(count, chosen_expert, 1)
        if is_last(top_k_left, 0):
            for _ in dace.map[0:1] @ dace.ScheduleType.GPU_ThreadBlock:
                append_to_stream(queue, SCAN, 0, 0)

    @dace.program
    def branches(token, column, dense_inverse_rms, expert_inverse_rms, dense_out, expert_acc, post_norm_1, post_norm_2):
        """Element ``column`` of the sum of both branches of ``token``, normalized (in bfloat16, as the model)."""
        dense_branch = bf16(
            dace.float32(dense_out[token, column]) * dense_inverse_rms * dace.float32(post_norm_1[column])
        )
        expert_branch = bf16(bf16(expert_acc[token, column]) * expert_inverse_rms * dace.float32(post_norm_2[column]))
        return bf16(dense_branch + expert_branch)

    @dace.program
    def output(token, x, dense_out, expert_acc, post_norm_1, post_norm_2, post_norm, out):
        """Normalizes both branches of ``token``, adds them, normalizes, and adds the residual (in bfloat16, as the
        model). The sum of the branches is kept in shared memory, which shares its bytes with the other tasks'
        containers."""
        sums = dace.ndarray([3], dtype=dace.float32, storage=dace.StorageType.GPU_Shared)
        both = dace.ndarray([HIDDEN], dtype=dace.bfloat16, storage=dace.StorageType.GPU_Shared)
        sums[:] = 0
        for column in dace.map[0:HIDDEN] @ dace.ScheduleType.GPU_ThreadBlock:
            dense_value = dace.float32(dense_out[token, column])
            expert_value = bf16(expert_acc[token, column])
            sums[0] += dense_value * dense_value
            sums[1] += expert_value * expert_value
        dense_inverse_rms = inverse_rms(sums[0])
        expert_inverse_rms = inverse_rms(sums[1])
        for column in dace.map[0:HIDDEN] @ dace.ScheduleType.GPU_ThreadBlock:
            value = branches(
                token, column, dense_inverse_rms, expert_inverse_rms, dense_out, expert_acc, post_norm_1, post_norm_2
            )
            both[column] = value
            sums[2] += value * value
        inverse = inverse_rms(sums[2])
        for column in dace.map[0:HIDDEN] @ dace.ScheduleType.GPU_ThreadBlock:
            value = dace.float32(both[column])
            out[token, column] = dace.float32(x[token, column]) + bf16(
                value * inverse * dace.float32(post_norm[column])
            )

    @dace.program
    def scan(
        queue,
        count,
        pair_expert,
        pair_slot,
        pair_weight,
        tile_start,
        tile_expert,
        tile_rows,
        num_tiles,
        row_token,
        row_weight,
        gate_up_left,
    ):
        """The tiles of every expert's rows (pairs of a token and an expert), and the expert gate/up tasks."""
        # Every expert's rows start at a whole tile. A thread per expert.
        tiles_of = dace.ndarray([EXPERTS], dtype=dace.int32, storage=dace.StorageType.GPU_Shared)
        for expert in dace.map[0:EXPERTS] @ dace.ScheduleType.GPU_ThreadBlock:
            tiles_of[expert] = (count[expert] + TILE_ROWS - 1) // TILE_ROWS
        for expert in dace.map[0:EXPERTS] @ dace.ScheduleType.GPU_ThreadBlock:
            start = 0
            for other in range(expert):
                start += tiles_of[other]
            tile_start[expert] = start
            for local_tile in range(tiles_of[expert]):
                tile_expert[start + local_tile] = expert
                tile_rows[start + local_tile] = min(TILE_ROWS, count[expert] - local_tile * TILE_ROWS)
            gate_up_left[expert] = tiles_of[expert] * (EXPERT_INTERMEDIATE // GATE_COLUMNS)
            if expert == EXPERTS - 1:
                num_tiles[0] = start + tiles_of[expert]
        # The rows of every expert, sorted by expert: their tokens and routing weights
        # A thread per token (the last parameter strides)
        for j, routed_token in dace.map[0:EXPERTS_PER_TOKEN, 0:TOKENS] @ dace.ScheduleType.GPU_ThreadBlock:
            row = tile_start[pair_expert[routed_token, j]] * TILE_ROWS + pair_slot[routed_token, j]
            row_token[row] = routed_token
            row_weight[row] = pair_weight[routed_token, j]
        # The expert gate/up tasks, alternating with the dense gate/up tasks that the prologue left (as many of each
        # as there are, then the rest), so that the dense products overlap the expert ones. The tasks of an expert are
        # those of its tiles (j // (EXPERT_INTERMEDIATE // GATE_COLUMNS) is one of them).
        num_expert = num_tiles[0] * (EXPERT_INTERMEDIATE // GATE_COLUMNS)
        num_dense = (PADDED_TOKENS // TILE_ROWS - FIRST_DENSE_TILES) * (DENSE_INTERMEDIATE // GATE_COLUMNS)
        alternating = 2 * min(num_expert, num_dense)
        for i in dace.map[0 : num_expert + num_dense] @ dace.ScheduleType.GPU_ThreadBlock:
            dense = False
            j = i - num_dense
            if i < alternating:
                dense = i % 2 == 1
                j = i // 2
            elif num_dense > num_expert:
                dense = True
                j = i - num_expert
            if dense:
                append_to_stream(
                    queue,
                    DENSE_GATE_UP,
                    FIRST_DENSE_TILES + j % (PADDED_TOKENS // TILE_ROWS - FIRST_DENSE_TILES),
                    j // (PADDED_TOKENS // TILE_ROWS - FIRST_DENSE_TILES),
                )
            else:  # The tiles of an expert's chunk one after the other: they read it from L2
                tile_owner = tile_expert[j // (EXPERT_INTERMEDIATE // GATE_COLUMNS)]
                local = j - tile_start[tile_owner] * (EXPERT_INTERMEDIATE // GATE_COLUMNS)
                expert_tiles = (count[tile_owner] + TILE_ROWS - 1) // TILE_ROWS
                append_to_stream(
                    queue, EXPERT_GATE_UP, tile_start[tile_owner] + local % expert_tiles, local // expert_tiles
                )

    @dace.program
    def moe_block(
        x: dace.bfloat16[TOKENS, HIDDEN],
        pre_norm: dace.bfloat16[HIDDEN],
        post_norm_1: dace.bfloat16[HIDDEN],
        router_scale: dace.bfloat16[HIDDEN],
        router_w: dace.bfloat16[EXPERTS, HIDDEN],
        per_expert_scale: dace.bfloat16[EXPERTS],
        pre_norm_2: dace.bfloat16[HIDDEN],
        post_norm_2: dace.bfloat16[HIDDEN],
        post_norm: dace.bfloat16[HIDDEN],
        dense_gate_up_w: dace.bfloat16[2 * DENSE_INTERMEDIATE, HIDDEN],
        down_w: dace.bfloat16[HIDDEN, DENSE_INTERMEDIATE],
        expert_gate_up_w: dace.bfloat16[EXPERTS, 2 * EXPERT_INTERMEDIATE, HIDDEN],
        expert_down_w: dace.bfloat16[EXPERTS, HIDDEN, EXPERT_INTERMEDIATE],
        out: dace.bfloat16[TOKENS, HIDDEN],
        # Scratch
        inv_rms: dace.float32[TOKENS],
        x_router: dace.bfloat16[PADDED_TOKENS, HIDDEN],
        # The router parts of every token, one after the other
        route_partial: dace.float32[ROUTER_PARTS * PADDED_TOKENS, EXPERTS],
        route_left: dace.int32[PADDED_TOKENS // TILE_ROWS],
        top_k_left: dace.int32[1],
        count: dace.int32[EXPERTS],
        pair_expert: dace.int32[TOKENS, EXPERTS_PER_TOKEN],
        pair_slot: dace.int32[TOKENS, EXPERTS_PER_TOKEN],
        pair_weight: dace.float32[TOKENS, EXPERTS_PER_TOKEN],
        tile_start: dace.int32[EXPERTS],
        tile_expert: dace.int32[EXPERT_TILES],
        tile_rows: dace.int32[EXPERT_TILES],
        dense_rows: dace.int32[PADDED_TOKENS // TILE_ROWS],
        num_tiles: dace.int32[1],
        row_token: dace.int32[EXPERT_TILES * TILE_ROWS],
        row_weight: dace.float32[EXPERT_TILES * TILE_ROWS],
        x_dense: dace.bfloat16[PADDED_TOKENS, HIDDEN],
        x_expert: dace.bfloat16[TOKENS, HIDDEN],
        h_dense: dace.bfloat16[PADDED_TOKENS, DENSE_INTERMEDIATE],
        h_expert: dace.bfloat16[EXPERT_TILES * TILE_ROWS, EXPERT_INTERMEDIATE],
        gate_up_left: dace.int32[EXPERTS],  # Of every expert's gate/up tasks
        dense_gate_up_left: dace.int32[PADDED_TOKENS // TILE_ROWS],
        dense_out: dace.bfloat16[PADDED_TOKENS, HIDDEN],
        expert_acc: dace.float32[TOKENS, HIDDEN],
        token_left: dace.int32[TOKENS],
    ):
        queue = dace.define_stream(dace.int64, queue_size)

        for token in dace.map[0:TOKENS]:
            prologue(
                token,
                x,
                pre_norm,
                pre_norm_2,
                router_scale,
                inv_rms,
                x_dense,
                x_expert,
                x_router,
                expert_acc,
                row_token,
                token_left,
            )
            if token == 0:
                first_tasks(queue, count, dense_rows, route_left, dense_gate_up_left, top_k_left)

        for task, pe in dace.consume(queue, num_pes):
            # The fields of the task (see ``append_to_stream``), as symbols: they offset the operands of the task
            kind = task >> KIND_SHIFT << KIND_SHIFT
            index = (task >> INDEX_SHIFT) & FIELD_MASK
            chunk = task & FIELD_MASK
            tile = dace.ndarray(
                [TILE_ROWS, TILE_COLUMNS], dtype=dace.bfloat16, storage=dace.StorageType.GPU_Shared
            )  # The product of a tile
            if kind == EXPERT_GATE_UP:
                expert_gate_up(
                    queue,
                    index,
                    chunk,
                    tile,
                    x_expert,
                    expert_gate_up_w,
                    tile_expert,
                    tile_rows,
                    tile_start,
                    row_token,
                    count,
                    gate_up_left,
                    h_expert,
                )
            elif kind == DENSE_GATE_UP:
                dense_gate_up(
                    queue, index, chunk, tile, x_dense, dense_gate_up_w, dense_rows, dense_gate_up_left, h_dense
                )
            elif kind == EXPERT_DOWN:
                expert_down(
                    queue,
                    index,
                    chunk,
                    tile,
                    h_expert,
                    expert_down_w,
                    tile_expert,
                    tile_rows,
                    row_token,
                    row_weight,
                    expert_acc,
                    token_left,
                )
            elif kind == DENSE_DOWN:
                dense_down(queue, index, chunk, tile, h_dense, down_w, dense_rows, dense_out, token_left)
            elif kind == ROUTE:
                route(queue, index, chunk, x_router, router_w, route_partial, dense_rows, route_left)
            elif kind == TOP_K:
                top_k(
                    queue,
                    index,
                    route_partial,
                    per_expert_scale,
                    count,
                    pair_expert,
                    pair_slot,
                    pair_weight,
                    top_k_left,
                )
            elif kind == OUTPUT:
                output(index, x, dense_out, expert_acc, post_norm_1, post_norm_2, post_norm, out)
            else:
                scan(
                    queue,
                    count,
                    pair_expert,
                    pair_slot,
                    pair_weight,
                    tile_start,
                    tile_expert,
                    tile_rows,
                    num_tiles,
                    row_token,
                    row_weight,
                    gate_up_left,
                )

    sdfg = moe_block.to_sdfg(simplify=False)
    tasks = (expert_gate_up, dense_gate_up, expert_down, dense_down, route, top_k, output, scan)
    names = {task.name: task.f.__name__ for task in tasks}
    if profile:  # The nested SDFG of every task call, which inlining turns into a region with its instrumentation
        for node, state in sdfg.all_nodes_recursive():  # (The calls, not the nested SDFGs of the tasks' own maps)
            if isinstance(node, dace.nodes.NestedSDFG) and node.label in names and state.entry_node(node) is None:
                node.label = names[node.label]
                node.instrument = dace.InstrumentationType.GPU_Events
    sdfg.simplify(validate=False)  # As the frontend does: shared memory is valid once the program is on the GPU
    for region in sdfg.all_control_flow_regions(recursive=True):  # Named after the task (inlining prefixes labels)
        if region.instrument == dace.InstrumentationType.GPU_Events:
            region.label = next(name for name in names.values() if region.label.endswith(name))
    # The gathers of rows and the transposed weights, read by the products themselves
    sdfg.apply_transformations_repeated([GatherIntoMatrixProduct, TransposeIntoMatrixProduct], validate=False)
    BlockMatrixOperands(WEIGHTS, BLOCK).apply_pass(sdfg, {})  # The program then takes the weights in blocks
    to_gpu(sdfg)
    return sdfg


def to_gpu(sdfg: dace.SDFG) -> None:
    """Runs the program on the GPU: arguments in GPU memory, top-level maps as kernels of THREADS threads, the consume
    scope as a persistent kernel with a thread block per processing element. A thread-block map with more iterations
    than threads, or as many as data says, runs in strides of the thread block (each thread loops over its
    iterations)."""
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
            elif node.map.schedule == dace.ScheduleType.GPU_ThreadBlock:
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
    parameters = {name: parameter.detach() for name, parameter in block.named_parameters()}
    experts, expert_gate_up_rows, hidden = parameters["experts.gate_up_proj"].shape
    dense_intermediate = parameters["mlp.gate_proj.weight"].shape[0]
    gate_columns = TILE_COLUMNS // 2
    expert_gate_up = (
        parameters["experts.gate_up_proj"]
        .view(experts, 2, expert_gate_up_rows // 2 // gate_columns, gate_columns, hidden)
        .transpose(1, 2)
    )
    dense_gate_up = torch.stack([parameters["mlp.gate_proj.weight"], parameters["mlp.up_proj.weight"]]).view(
        2, dense_intermediate // gate_columns, gate_columns, hidden
    )
    result = dict(
        pre_norm=parameters["pre_feedforward_layernorm.weight"],
        post_norm_1=parameters["post_feedforward_layernorm_1.weight"],
        router_scale=parameters["router.scale"],
        router_w=parameters["router.proj.weight"],
        per_expert_scale=parameters["router.per_expert_scale"],
        pre_norm_2=parameters["pre_feedforward_layernorm_2.weight"],
        post_norm_2=parameters["post_feedforward_layernorm_2.weight"],
        post_norm=parameters["post_feedforward_layernorm.weight"],
        dense_gate_up_w=dense_gate_up.transpose(0, 1).reshape(2 * dense_intermediate, hidden),
        down_w=parameters["mlp.down_proj.weight"],
        expert_gate_up_w=expert_gate_up.reshape(experts, expert_gate_up_rows, hidden),
        expert_down_w=parameters["experts.down_proj"],
    )
    return {name: in_blocks(tensor) if name in WEIGHTS else tensor for name, tensor in result.items()}


def workspace(config: Gemma4TextConfig, tokens: int) -> dict:
    """Scratch buffers and symbol values for ``tokens`` tokens."""
    padded = (tokens + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
    experts, per_token, hidden = config.num_experts, config.top_k_experts, config.hidden_size
    tiles = tokens * per_token // TILE_ROWS + experts

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    return dict(
        inv_rms=zeros(tokens),
        x_router=zeros(padded, hidden, dtype=torch.bfloat16),
        route_partial=zeros(hidden // ROUTE_STEP * padded, experts),
        route_left=zeros(padded // TILE_ROWS, dtype=torch.int32),
        top_k_left=zeros(1, dtype=torch.int32),
        count=zeros(experts, dtype=torch.int32),
        pair_expert=zeros(tokens, per_token, dtype=torch.int32),
        pair_slot=zeros(tokens, per_token, dtype=torch.int32),
        pair_weight=zeros(tokens, per_token),
        tile_start=zeros(experts, dtype=torch.int32),
        tile_expert=zeros(tiles, dtype=torch.int32),
        tile_rows=zeros(tiles, dtype=torch.int32),
        dense_rows=zeros(padded // TILE_ROWS, dtype=torch.int32),
        num_tiles=zeros(1, dtype=torch.int32),
        row_token=zeros(tiles * TILE_ROWS, dtype=torch.int32),
        row_weight=zeros(tiles * TILE_ROWS),
        x_dense=zeros(padded, hidden, dtype=torch.bfloat16),
        x_expert=zeros(tokens, hidden, dtype=torch.bfloat16),
        h_dense=zeros(padded, config.intermediate_size, dtype=torch.bfloat16),
        h_expert=zeros(tiles * TILE_ROWS, config.moe_intermediate_size, dtype=torch.bfloat16),
        gate_up_left=zeros(experts, dtype=torch.int32),
        dense_gate_up_left=zeros(padded // TILE_ROWS, dtype=torch.int32),
        dense_out=zeros(padded, hidden, dtype=torch.bfloat16),
        expert_acc=zeros(tokens, hidden),
        token_left=zeros(tokens, dtype=torch.int32),
        TOKENS=tokens,
        PADDED_TOKENS=padded,
        EXPERT_TILES=tiles,
        FIRST_DENSE_TILES=1 if tokens * per_token < experts * TILE_ROWS else padded // TILE_ROWS,
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
    events = [event for event in report.events if event.additional_info and "block" in event.additional_info]
    start = min(event.timestamp for event in events)  # Microseconds
    span = max(event.timestamp + event.duration for event in events) - start
    tasks = collections.defaultdict(list)
    for event in events:
        tasks[event.name].append(event)
    for name, of_kind in sorted(tasks.items(), key=lambda item: -sum(event.duration for event in item[1])):
        total = sum(event.duration for event in of_kind)
        cycles = np.mean([event.additional_info["cycles"] for event in of_kind])
        print(
            f"  {name:>14}: {len(of_kind):6d} tasks of {total / len(of_kind):7.2f} us ({cycles:7.0f} cycles), "
            f"{total / (num_pes * span):6.1%} of the processing elements' time"
        )
    busy = sum(event.duration for event in events) / (num_pes * span)
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
