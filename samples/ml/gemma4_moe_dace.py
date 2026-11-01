# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Gemma 4 mixture-of-experts feed-forward block (see ``gemma4_moe.py``) as one persistent DaCe GPU kernel.

After a kernel that normalizes the tokens (one thread block per token), a single consume scope runs the rest of the
block as tasks on a queue. Every processing element is a GPU thread
block that pops a task, mostly a 64 x 128 tile of one matrix product with tensor cores (the ``tile_matmul`` library
node) followed by an epilogue on the tile. The tasks are:

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
"""

import math

import gemma4_moe
import tile_matmul as tm
import torch

import dace

TB = dace.ScheduleType.GPU_ThreadBlock
SHARED = dace.StorageType.GPU_Shared
WARPS = 4  # Warps per thread block
THREADS = 32 * WARPS
BM, BN = 64, 128  # Rows and columns of a tile
K_STEP, STAGES = 32, 2  # Slices of the inner dimension that tile_matmul stages in shared memory
WORKSPACE = tm.workspace_shape(BM, BN, K_STEP, STAGES)
STAGING = math.prod(WORKSPACE)  # Elements of the workspace
# A task is ``kind | tile << FIELD | chunk``, where the kinds are multiples of ``1 << KIND`` (64-bit constants)
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


def build(config: gemma4_moe.Config, num_pes: int, queue_size: int = 1 << 18) -> dace.SDFG:
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
    eps = config.eps
    root = D**-0.5
    gelu_c = math.sqrt(2 / math.pi)
    HALF = BN // 2  # Gate (and up) columns of a gate/up task

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

        # Python float literals are double precision on the GPU, which computes them slowly: constants are float32.
        #
        # The prologue, a thread block per token: the RMS of the token (all the pre-norms normalize the same residual)
        # and the inputs of the dense branch, the experts and the router (the normalized input, rounded to bfloat16 as
        # the model does, and scaled). The blocks also reset the scratch memory, and the first block queues the first
        # tasks: the router's (tiles of tokens times slices of the input), whose last one pushes the top-k tasks of the
        # tile, whose last one pushes the scan that pushes the expert tasks; then the dense gate/up tasks.
        for t in dace.map[0:T]:
            sum_of_squares = dace.ndarray([1], dtype=dace.float32, storage=SHARED)
            sum_of_squares[0] = 0
            for d in dace.map[0:D] @ TB:
                v = dace.float32(x[t, d])
                sum_of_squares[0] += v * v
                expert_acc[t, d] = 0
            for r in dace.map[t : NT * BM : T] @ TB:  # The blocks reset the rows of the expert tiles together
                row_token[r] = -1
            token_left[t] = (K + 1) * (D // BN)  # The down tiles that add to the token: its experts', and the dense
            if t == 0:
                for ex in dace.map[0:E] @ TB:
                    count[ex] = 0
                for g in dace.map[0 : TP // BM] @ TB:
                    dense_rows[g] = min(BM, T - g * BM)
                    route_left[g] = RP
                    dense_gate_up_left[g] = DI // HALF
                top_k_left[0] = T  # A top-k task per token
            inv = dace.float32(1) / math.sqrt(sum_of_squares[0] / D + dace.float32(eps))
            inv_rms[t] = inv
            for d in dace.map[0:D] @ TB:
                n = dace.float32(x[t, d]) * inv
                x_dense[t, d] = dace.bfloat16(n * pre_norm[d])
                x_router[t, d] = dace.bfloat16(dace.float32(dace.bfloat16(n)) * router_scale[d] * dace.float32(root))
                x_expert[t, d] = dace.bfloat16(n * pre_norm_2[d])
            if t == 0:
                # The router tasks, and the dense gate/up tasks of the first DT tiles of tokens (see ``DT``), the tiles
                # of a chunk of the weight one after the other: they read it from L2
                for i in dace.map[0 : TP // BM * RP + DT * (DI // HALF)] @ TB:
                    if i < TP // BM * RP:
                        S.append(ROUTE | (i // RP) << FIELD | i % RP)
                    else:
                        S.append(DENSE_GATE_UP | ((i - TP // BM * RP) % DT) << FIELD | (i - TP // BM * RP) // DT)

        for task, p in dace.consume(S, num_pes):
            kind = task >> KIND << KIND
            g = (task >> FIELD) & MASK  # Tile of rows
            ch = task & MASK  # Chunk of columns
            r0 = g * BM
            # The product of a tile overwrites the operands that tile_matmul staged: sharing their memory leaves room for
            # a third thread block on each multiprocessor (whose loads cover the gaps between the tasks of the others)
            staging = dace.ndarray([STAGING], dtype=dace.bfloat16, storage=SHARED)
            work = staging.reshape(WORKSPACE)
            tile = staging[0 : BM * BN].reshape([BM, BN])
            last = dace.ndarray([1], dtype=dace.int32, storage=SHARED)  # Whether a task is the last of a group
            if kind == EXPERT_GATE_UP or kind == DENSE_GATE_UP:
                if kind == EXPERT_GATE_UP:
                    e = tile_expert[g]
                    nr = tile_rows[g]
                    tm.tile_matmul(x_expert, expert_gate_up_p[e, ch], tile, work, nr, row_token[r0 : r0 + BM])
                else:
                    nr = dense_rows[g]
                    tm.tile_matmul(x_dense[r0 : r0 + BM, :], dense_gate_up_p[ch], tile, work, nr)
                # GELU(gate) * up, where the columns of the tile are 64 of the gate projection, then 64 of the up one (of
                # the rows that the down tasks read)
                for i in dace.map[0 : nr * HALF] @ TB:
                    gate = dace.float32(tile[i // HALF, i % HALF])
                    inner = dace.float32(gelu_c) * (gate + dace.float32(0.044715) * gate * gate * gate)
                    gelu = dace.float32(0.5) * gate * (dace.float32(1) + math.tanh(inner))
                    h = dace.bfloat16(
                        dace.float32(dace.bfloat16(gelu)) * dace.float32(tile[i // HALF, HALF + i % HALF])
                    )
                    if kind == EXPERT_GATE_UP:
                        h_expert[r0 + i // HALF, ch * HALF + i % HALF] = h
                    else:
                        h_dense[r0 + i // HALF, ch * HALF + i % HALF] = h
                # The last gate/up task of an expert (all its tiles), or of 64 dense rows, pushes their down tasks (the
                # tiles of an expert's chunk one after the other: they read it from L2)
                for _ in dace.map[0:1] @ TB:
                    if kind == EXPERT_GATE_UP:
                        left = dace.atomic.add(gate_up_left, e, -1)
                    else:
                        left = dace.atomic.add(dense_gate_up_left, g, -1)
                    last[0] = left == 1
                if last[0] == 1:
                    if kind == EXPERT_GATE_UP:
                        n = (count[e] + BM - 1) // BM  # Tiles of the expert
                        for i in dace.map[0 : n * (D // BN)] @ TB:
                            S.append(EXPERT_DOWN | (tile_start[e] + i % n) << FIELD | i // n)
                    else:
                        for c in dace.map[0 : D // BN] @ TB:
                            S.append(DENSE_DOWN | g << FIELD | c)
            elif kind == EXPERT_DOWN:
                e = tile_expert[g]
                nr = tile_rows[g]
                tm.tile_matmul(h_expert[r0 : r0 + BM, :], expert_down_p[e, ch], tile, work, nr)
                # Scaled by the routing weight, added to the output of the row's token
                for r, col in dace.map[0:nr, 0:BN] @ TB:
                    expert_acc[row_token[r0 + r], ch * BN + col] += dace.float32(tile[r, col]) * row_weight[r0 + r]
                for r in dace.map[0:nr] @ TB:
                    left = dace.atomic.add(token_left, row_token[r0 + r], -1)
                    if left == 1:  # The last tile that adds to the token
                        S.append(OUTPUT | row_token[r0 + r] << FIELD)
            elif kind == DENSE_DOWN:
                nr = dense_rows[g]
                tm.tile_matmul(h_dense[r0 : r0 + BM, :], down_p[ch], tile, work, nr)
                for r, col in dace.map[0:nr, 0:BN] @ TB:
                    dense_out[r0 + r, ch * BN + col] = tile[r, col]
                for r in dace.map[0:nr] @ TB:
                    left = dace.atomic.add(token_left, r0 + r, -1)
                    if left == 1:
                        S.append(OUTPUT | (r0 + r) << FIELD)
            elif kind == ROUTE:
                # Part of the router logits of 64 tokens: a slice of the input times the router weight
                nr = dense_rows[g]
                tm.tile_matmul(
                    x_router[r0 : r0 + BM, ch * ROUTE_STEP : ch * ROUTE_STEP + ROUTE_STEP],
                    router_p[0, ch * RS : ch * RS + RS],
                    route_partial[ch * TP + r0 : ch * TP + r0 + BM, :],
                    work,
                    nr,
                )
                # The last part of these tokens pushes their top-k tasks
                for _ in dace.map[0:1] @ TB:
                    left = dace.atomic.add(route_left, g, -1)
                    last[0] = left == 1
                if last[0] == 1:
                    for r in dace.map[0:nr] @ TB:
                        S.append(TOP_K | (r0 + r) << FIELD)
            elif kind == TOP_K:
                # The experts of token g. A thread per expert, then per chosen expert.
                chosen = dace.ndarray([K], dtype=dace.int32, storage=SHARED)
                # The token's router logits (the projection rounds them to bfloat16)
                for expert in dace.map[0:E] @ TB:
                    total = dace.float32(0)
                    for part in range(RP):
                        total += route_partial[part * TP + g, expert]
                    tile[0, expert] = dace.bfloat16(total)
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
                # The softmax probabilities of the chosen experts, renormalized (the softmax denominator cancels) and
                # scaled; claim a row of every chosen expert
                for j in dace.map[0:K] @ TB:
                    top = dace.float32(tile[0, chosen[0]])
                    total = dace.float32(0)
                    for j2 in range(K):
                        total += math.exp(dace.float32(tile[0, chosen[j2]]) - top)
                    c = chosen[j]
                    pair_expert[g, j] = c
                    pair_weight[g, j] = math.exp(dace.float32(tile[0, c]) - top) / total * per_expert_scale[c]
                    pair_slot[g, j] = dace.atomic.add(count, c, 1)
                for _ in dace.map[0:1] @ TB:
                    left = dace.atomic.add(top_k_left, 0, -1)
                    if left == 1:  # The last token
                        S.append(SCAN)
            elif kind == OUTPUT:
                # Normalize both branches of token g, add, normalize, add the residual (in bfloat16, as the model).
                # The sum of the branches stays in the staging memory.
                sums = dace.ndarray([3], dtype=dace.float32, storage=SHARED)
                sums[:] = 0
                for d in dace.map[0:D] @ TB:
                    v1 = dace.float32(dense_out[g, d])
                    v2 = dace.float32(dace.bfloat16(expert_acc[g, d]))
                    sums[0] += v1 * v1
                    sums[1] += v2 * v2
                inv1 = dace.float32(1) / math.sqrt(sums[0] / D + dace.float32(eps))
                inv2 = dace.float32(1) / math.sqrt(sums[1] / D + dace.float32(eps))
                for d in dace.map[0:D] @ TB:
                    b1 = dace.float32(dace.bfloat16(dace.float32(dense_out[g, d]) * inv1 * post_norm_1[d]))
                    b2 = dace.float32(
                        dace.bfloat16(dace.float32(dace.bfloat16(expert_acc[g, d])) * inv2 * post_norm_2[d])
                    )
                    v = dace.bfloat16(b1 + b2)
                    staging[d] = v
                    sums[2] += dace.float32(v) * dace.float32(v)
                inv = dace.float32(1) / math.sqrt(sums[2] / D + dace.float32(eps))
                for d in dace.map[0:D] @ TB:
                    normalized = dace.float32(dace.bfloat16(dace.float32(staging[d]) * inv * post_norm[d]))
                    out[g, d] = dace.bfloat16(dace.float32(x[g, d]) + normalized)
            else:  # SCAN
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
                # The expert gate/up tasks, alternating with the dense gate/up tasks that the prologue left (as many of
                # each as there are, then the rest), so that the dense products overlap the expert ones. The tasks of
                # an expert are those of its tiles (j // (I // HALF) is one of them).
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
                        S.append(DENSE_GATE_UP | (DT + j % (TP // BM - DT)) << FIELD | j // (TP // BM - DT))
                    else:  # The tiles of an expert's chunk one after the other: they read it from L2
                        te = tile_expert[j // (I // HALF)]
                        local = j - tile_start[te] * (I // HALF)
                        n = (count[te] + BM - 1) // BM
                        S.append(EXPERT_GATE_UP | (tile_start[te] + local % n) << FIELD | local // n)

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


def arguments(w: gemma4_moe.Weights) -> dict[str, torch.Tensor]:
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


def workspace(config: gemma4_moe.Config, tokens: int) -> dict:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    parser.add_argument("--small", action="store_true", help="a small configuration, for testing")
    # Three thread blocks per multiprocessor of an RTX 3070 (as many as the registers of the workers allow)
    parser.add_argument("--pes", type=int, default=46 * 3, help="processing elements (thread blocks)")
    args = parser.parse_args()

    config = (
        gemma4_moe.Config(hidden=256, dense_intermediate=128, num_experts=128, top_k=4, expert_intermediate=128)
        if args.small
        else gemma4_moe.Config()
    )
    w = gemma4_moe.random_weights(config)
    program = build(config, args.pes)
    # The default stream, as PyTorch's (see ``to_gpu``)
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
        program = program.compile()
    weights = arguments(w)
    for tokens in args.tokens:
        x = gemma4_moe.random_input(config, tokens)
        out = torch.empty_like(x)
        scratch = workspace(config, tokens)
        program(x=x, out=out, **weights, **scratch)
        # The router's logits are summed in another order than the reference's, which can flip near-ties between
        # the k-th and the next expert: compare the tokens that both route alike
        _, expected_experts = gemma4_moe.route(x, w, config)
        alike = (scratch["pair_expert"].long().sort().values == expected_experts.sort().values).all(dim=1)
        expected = gemma4_moe.reference(x, w, config).float()[alike]
        error = ((out.float()[alike] - expected).norm() / expected.norm()).item()
        assert error < 2e-2, f"DaCe differs from the reference (relative error {error:.2e})"
        # The time of the kernels, by events on the stream they run on (without Python's argument conversion)
        callargs, initargs = program.construct_arguments(x=x, out=out, **weights, **scratch)
        time_ms = gemma4_moe.time_ms(lambda: program.fast_call(callargs, initargs))
        print(
            f"{tokens:5d} tokens: DaCe {time_ms:8.3f} ms (error {error:.1e}, {int((~alike).sum())} routed differently)"
        )
