# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Gemma 4 mixture-of-experts feed-forward block (see ``gemma4_moe.py``) as one persistent DaCe GPU kernel.

After a few small kernels that normalize and route the tokens (one thread block per token), a single consume scope
runs all the matrix products of the block as tasks on a queue. Every processing element is a GPU thread block that
pops a task and computes a 64 x 128 tile of one product with tensor cores (the ``tile_matmul`` library node), followed
by an epilogue on the tile. There are four kinds of tasks:

* expert gate/up: 64 rows of an expert (pairs of tokens routed to it) times 128 columns of its gate and up projections,
  interleaved in chunks of 64, followed by GELU(gate) * up;
* expert down: the same rows times 128 output columns of the expert's down projection, scaled by each pair's routing
  weight and added to the token's output (an atomic write-conflict resolution);
* dense gate/up and dense down: the same for 64 tokens in the dense MLP that runs in parallel.

Down tasks depend on all the gate/up tasks of their rows. The gate/up task that finishes last (it reads the old
value of a counter it decrements, an acquire-release update) pushes the down tasks of its rows, so the queue never
waits for a whole projection to finish, and dense and expert tasks fill each other's gaps.
"""

import math
import time

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
KIND = 2**40  # A task is ``kind * KIND + tile * FIELD + chunk``
FIELD = 2**20
EXPERT_GATE_UP, EXPERT_DOWN, DENSE_GATE_UP, DENSE_DOWN = range(4)

T = dace.symbol("T")  # Tokens
TP = dace.symbol("TP")  # Tokens, padded to a multiple of BM
NT = dace.symbol("NT")  # The most expert tiles there can be: T * top_k / BM + num_experts


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
    assert D % BN == 0 and (2 * DI) % BN == 0 and (2 * I) % BN == 0 and E == BN
    eps = config.eps
    root = D**-0.5
    gelu_c = math.sqrt(2 / math.pi)
    # Python float literals are double precision in tasklets, which GPUs compute slowly: constants are float32
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
        logits: dace.bfloat16[TP, E],
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
        x_expert: dace.bfloat16[NT * BM, D],
        h_dense: dace.bfloat16[TP, DI],
        h_expert: dace.bfloat16[NT * BM, I],
        gate_up_left: dace.int32[NT],
        dense_gate_up_left: dace.int32[TP // BM],
        dense_out: dace.bfloat16[TP, D],
        expert_acc: dace.float32[T, D],
    ):
        S = dace.define_stream(dace.int64, queue_size)

        # Reset the counters and accumulators
        for e in dace.map[0:E]:
            with dace.tasklet:
                c >> count[e]
                c = 0
        for r in dace.map[0 : NT * BM]:
            with dace.tasklet:
                rt >> row_token[r]
                rt = -1
        for t, d in dace.map[0:T, 0:D]:
            with dace.tasklet:
                a >> expert_acc[t, d]
                a = 0
        for g in dace.map[0 : TP // BM]:
            with dace.tasklet:
                dr >> dense_rows[g]
                dr = min(BM, T - g * BM)

        # RMS of every token (all the pre-norms normalize the same residual), and the inputs of the dense branch and
        # of the router: the normalized input (bfloat16, as the model rounds it), scaled
        for t in dace.map[0:T]:
            ss = dace.ndarray([1], dtype=dace.float32, storage=SHARED)
            for _ in dace.map[0:1] @ TB:
                with dace.tasklet:
                    s >> ss[0]
                    s = 0
            for th in dace.map[0:THREADS] @ TB:
                with dace.tasklet:
                    xs << x[t, :]
                    s >> ss(1, lambda a, b: a + b)[0]
                    part = dace.float32(0)
                    for d in range(th, D, THREADS):
                        v = dace.float32(xs[d])
                        part = part + v * v
                    s = part
            for th in dace.map[0:THREADS] @ TB:
                with dace.tasklet:
                    s << ss[0]
                    xs << x[t, :]
                    w1 << pre_norm[:]
                    rs << router_scale[:]
                    inv >> inv_rms(-1)[t]
                    xd >> x_dense[t, :]
                    xr >> x_router[t, :]
                    v = dace.float32(1) / math.sqrt(s / D + dace.float32(eps))
                    if th == 0:
                        inv = v
                    for d in range(th, D, THREADS):
                        n = dace.float32(xs[d]) * v
                        xd[d] = dace.bfloat16(n * w1[d])
                        xr[d] = dace.bfloat16(dace.float32(dace.bfloat16(n)) * rs[d] * dace.float32(root))

        # Router logits, a tile of 64 tokens per thread block (the projection rounds them to bfloat16)
        for g in dace.map[0 : TP // BM]:
            work = dace.ndarray(WORKSPACE, dtype=dace.bfloat16, storage=SHARED)
            tile = dace.ndarray([BM, BN], dtype=dace.bfloat16, storage=SHARED)
            nr = dense_rows[g]
            tm.tile_matmul(x_router[g * BM : g * BM + BM, :], router_p[0], tile, work, nr)
            for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                for q in range(BM * BN // THREADS):
                    with dace.tasklet:
                        lv << tile[(q * THREADS + w * 32 + lane) // BN, (q * THREADS + w * 32 + lane) % BN]
                        lg >> logits[g * BM + (q * THREADS + w * 32 + lane) // BN, (q * THREADS + w * 32 + lane) % BN]
                        lg = lv

        # Softmax over all experts, top-k (ties: lowest index first), renormalize, scale; claim a row of every chosen
        # expert. A thread per expert.
        for t in dace.map[0:T]:
            red = dace.ndarray([3], dtype=dace.float32, storage=SHARED)  # Largest logit, softmax sum, chosen sum
            chosen = dace.ndarray([K], dtype=dace.int32, storage=SHARED)
            chosen_p = dace.ndarray([K], dtype=dace.float32, storage=SHARED)
            for _ in dace.map[0:1] @ TB:
                with dace.tasklet:
                    r >> red[:]
                    r[0] = -1e30
                    r[1] = 0
                    r[2] = 0
            for e in dace.map[0:E] @ TB:
                with dace.tasklet:
                    lg << logits[t, e]
                    m >> red(1, lambda a, b: max(a, b))[0]
                    m = dace.float32(lg)
            for e in dace.map[0:E] @ TB:
                with dace.tasklet:
                    lgs << logits[t, :]
                    top << red[0]
                    tot >> red(1, lambda a, b: a + b)[1]
                    v = dace.float32(lgs[e])
                    tot = math.exp(v - top)
            for e in dace.map[0:E] @ TB:
                with dace.tasklet:
                    lgs << logits[t, :]
                    r << red[0:2]
                    cs >> red(-1, lambda a, b: a + b)[2]
                    ch >> chosen(-1)[:]
                    cp >> chosen_p(-1)[:]
                    v = dace.float32(lgs[e])
                    rank = 0
                    for e2 in range(E):
                        v2 = dace.float32(lgs[e2])
                        if v2 > v or (v2 == v and e2 < e):
                            rank = rank + 1
                    if rank < K:
                        p = math.exp(v - r[0]) / r[1]
                        ch[rank] = e
                        cp[rank] = p
                        cs = p
            for j in dace.map[0:K] @ TB:
                with dace.tasklet:
                    ce << chosen[j]
                    p << chosen_p[j]
                    total << red[2]
                    pes << per_expert_scale[:]
                    c >> count(-1, lambda a, b: a + b)[:]
                    pe >> pair_expert[t, j]
                    ps >> pair_slot[t, j]
                    pw >> pair_weight[t, j]
                    pe = ce
                    pw = p / total * pes[ce]
                    ps = c[ce] = 1

        # Every expert's rows start at a whole tile
        for _ in dace.map[0:1]:
            with dace.tasklet:
                cnt << count[:]
                ts >> tile_start[:]
                te >> tile_expert(-1)[:]
                tr >> tile_rows(-1)[:]
                nt >> num_tiles[0]
                start = 0
                for e in range(E):
                    ts[e] = start
                    tiles = (cnt[e] + BM - 1) // BM
                    for g in range(tiles):
                        te[start + g] = e
                        tr[start + g] = min(BM, cnt[e] - g * BM)
                    start = start + tiles
                nt = start

        # Gather the expert input rows, sorted by expert
        for t, j in dace.map[0:T, 0:K]:
            with dace.tasklet:
                pe << pair_expert[t, j]
                ps << pair_slot[t, j]
                pw << pair_weight[t, j]
                ts << tile_start[:]
                rt >> row_token(-1)[:]
                rw >> row_weight(-1)[:]
                row = ts[pe] * BM + ps
                rt[row] = t
                rw[row] = pw
        for t, j, d in dace.map[0:T, 0:K, 0:D]:
            with dace.tasklet:
                pe << pair_expert[t, j]
                ps << pair_slot[t, j]
                ts << tile_start[:]
                xv << x[t, d]
                inv << inv_rms[t]
                w2 << pre_norm_2[d]
                xe >> x_expert(-1)[:, :]
                xe[ts[pe] * BM + ps, d] = dace.bfloat16(dace.float32(xv) * inv * w2)

        # Seed the gate/up tasks (the down tasks follow them); the dense ones first, as their chain is the longest
        for g, ch in dace.map[0 : TP // BM, 0 : DI // HALF]:
            with dace.tasklet:
                left >> dense_gate_up_left[g]
                task >> S(-1)
                left = DI // HALF
                task = DENSE_GATE_UP * KIND + g * FIELD + ch
        for g, ch in dace.map[0:NT, 0 : I // HALF]:
            with dace.tasklet:
                nt << num_tiles[0]
                left >> gate_up_left(-1)[g]
                task >> S(-1)
                if g < nt:
                    left = I // HALF
                    task = EXPERT_GATE_UP * KIND + g * FIELD + ch

        @dace.consumescope(S, num_pes)
        def worker(task, p):
            kind = task // KIND
            g = (task // FIELD) % FIELD  # Tile of rows
            ch = task % FIELD  # Chunk of columns
            r0 = g * BM
            work = dace.ndarray(WORKSPACE, dtype=dace.bfloat16, storage=SHARED)
            tile = dace.ndarray([BM, BN], dtype=dace.bfloat16, storage=SHARED)
            if kind == EXPERT_GATE_UP or kind == DENSE_GATE_UP:
                if kind == EXPERT_GATE_UP:
                    e = tile_expert[g]
                    nr = tile_rows[g]
                    tm.tile_matmul(x_expert[r0 : r0 + BM, :], expert_gate_up_p[e, ch], tile, work, nr)
                else:
                    nr = dense_rows[g]
                    tm.tile_matmul(x_dense[r0 : r0 + BM, :], dense_gate_up_p[ch], tile, work, nr)
                # GELU(gate) * up, where the columns of the tile are 64 of the gate projection, then 64 of the up one
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    for q in range(BM * HALF // THREADS):
                        with dace.tasklet:
                            gv << tile[(q * THREADS + w * 32 + lane) // HALF, (q * THREADS + w * 32 + lane) % HALF]
                            (
                                uv
                                << tile[
                                    (q * THREADS + w * 32 + lane) // HALF, HALF + (q * THREADS + w * 32 + lane) % HALF
                                ]
                            )
                            k << kind
                            row0 << r0
                            col0 << ch
                            he >> h_expert(-1)[:, :]
                            hd >> h_dense(-1)[:, :]
                            gb = dace.float32(gv)
                            inner = dace.float32(gelu_c) * (gb + dace.float32(0.044715) * gb * gb * gb)
                            gl = dace.float32(0.5) * gb * (dace.float32(1) + math.tanh(inner))
                            hv = dace.bfloat16(dace.float32(dace.bfloat16(gl)) * dace.float32(uv))
                            i = q * THREADS + w * 32 + lane
                            if k == EXPERT_GATE_UP:
                                he[row0 + i // HALF, col0 * HALF + i % HALF] = hv
                            else:
                                hd[row0 + i // HALF, col0 * HALF + i % HALF] = hv
                # The last gate/up task of these rows pushes their down tasks
                for _ in dace.map[0:1] @ TB:
                    with dace.tasklet:
                        k << kind
                        tile_index << g
                        _published_e << h_expert(-1)[:, :]  # After the block wrote its part (release)
                        _published_d << h_dense(-1)[:, :]
                        gl >> gate_up_left(-1, lambda a, b: a + b)[:]
                        dl >> dense_gate_up_left(-1, lambda a, b: a + b)[:]
                        dn >> S(-1)
                        if k == EXPERT_GATE_UP:
                            left = gl[tile_index] = -1
                            if left == 1:
                                for c in range(D // BN):
                                    dn = EXPERT_DOWN * KIND + tile_index * FIELD + c
                        else:
                            left = dl[tile_index] = -1
                            if left == 1:
                                for c in range(D // BN):
                                    dn = DENSE_DOWN * KIND + tile_index * FIELD + c
            elif kind == EXPERT_DOWN:
                e = tile_expert[g]
                nr = tile_rows[g]
                tm.tile_matmul(h_expert[r0 : r0 + BM, :], expert_down_p[e, ch], tile, work, nr)
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    for q in range(BM * BN // THREADS):
                        with dace.tasklet:
                            yv << tile[(q * THREADS + w * 32 + lane) // BN, (q * THREADS + w * 32 + lane) % BN]
                            row0 << r0
                            col0 << ch
                            tok << row_token(-1)[:]
                            rw << row_weight(-1)[:]
                            o >> expert_acc(-1, lambda a, b: a + b)[:, :]
                            row = row0 + (q * THREADS + w * 32 + lane) // BN
                            if tok[row] >= 0:
                                o[tok[row], col0 * BN + (q * THREADS + w * 32 + lane) % BN] = dace.float32(yv) * rw[row]
            else:  # DENSE_DOWN
                nr = dense_rows[g]
                tm.tile_matmul(h_dense[r0 : r0 + BM, :], down_p[ch], tile, work, nr)
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    for q in range(BM * BN // THREADS):
                        with dace.tasklet:
                            yv << tile[(q * THREADS + w * 32 + lane) // BN, (q * THREADS + w * 32 + lane) % BN]
                            row0 << r0
                            col0 << ch
                            o >> dense_out(-1)[:, :]
                            o[
                                row0 + (q * THREADS + w * 32 + lane) // BN,
                                col0 * BN + (q * THREADS + w * 32 + lane) % BN,
                            ] = yv

        # Output: normalize both branches, add, normalize, add the residual (in bfloat16, as the model)
        for t in dace.map[0:T]:
            sums = dace.ndarray([3], dtype=dace.float32, storage=SHARED)
            mixed = dace.ndarray([D], dtype=dace.float32, storage=SHARED)
            for _ in dace.map[0:1] @ TB:
                with dace.tasklet:
                    s >> sums[:]
                    s[0] = 0
                    s[1] = 0
                    s[2] = 0
            for th in dace.map[0:THREADS] @ TB:
                with dace.tasklet:
                    dn << dense_out[t, :]
                    ex << expert_acc[t, :]
                    s1 >> sums(1, lambda a, b: a + b)[0]
                    s2 >> sums(1, lambda a, b: a + b)[1]
                    a1 = dace.float32(0)
                    a2 = dace.float32(0)
                    for d in range(th, D, THREADS):
                        v1 = dace.float32(dn[d])
                        v2 = dace.float32(dace.bfloat16(ex[d]))
                        a1 = a1 + v1 * v1
                        a2 = a2 + v2 * v2
                    s1 = a1
                    s2 = a2
            for th in dace.map[0:THREADS] @ TB:
                with dace.tasklet:
                    sm << sums[0:2]
                    dn << dense_out[t, :]
                    ex << expert_acc[t, :]
                    p1 << post_norm_1[:]
                    p2 << post_norm_2[:]
                    mx >> mixed(-1)[:]
                    s3 >> sums(1, lambda a, b: a + b)[2]
                    inv1 = dace.float32(1) / math.sqrt(sm[0] / D + dace.float32(eps))
                    inv2 = dace.float32(1) / math.sqrt(sm[1] / D + dace.float32(eps))
                    a3 = dace.float32(0)
                    for d in range(th, D, THREADS):
                        b1 = dace.float32(dace.bfloat16(dace.float32(dn[d]) * inv1 * p1[d]))
                        b2 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(ex[d])) * inv2 * p2[d]))
                        v = dace.float32(dace.bfloat16(b1 + b2))
                        mx[d] = v
                        a3 = a3 + v * v
                    s3 = a3
            for th in dace.map[0:THREADS] @ TB:
                with dace.tasklet:
                    s << sums[2]
                    mxs << mixed[:]
                    xs << x[t, :]
                    po << post_norm[:]
                    o >> out[t, :]
                    inv = dace.float32(1) / math.sqrt(s / D + dace.float32(eps))
                    for d in range(th, D, THREADS):
                        o[d] = dace.bfloat16(dace.float32(xs[d]) + dace.float32(dace.bfloat16(mxs[d] * inv * po[d])))

    sdfg = moe_block.to_sdfg(simplify=True)
    to_gpu(sdfg)
    return sdfg


def to_gpu(sdfg: dace.SDFG) -> None:
    """Runs the program on the GPU: arguments in GPU memory, top-level maps as kernels, the consume scope as a
    persistent kernel with a thread block per processing element."""
    for desc in sdfg.arrays.values():
        if not isinstance(desc, dace.data.Scalar) and not desc.transient:
            desc.storage = dace.StorageType.GPU_Global
        elif isinstance(desc, dace.data.Stream):
            # The task queue: allocated once (it is empty again after every call)
            desc.storage = dace.StorageType.GPU_Global
            desc.lifetime = dace.AllocationLifetime.Persistent
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if parent.parent is sdfg and parent.entry_node(node) is None:
                node.map.schedule = dace.ScheduleType.GPU_Device
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
        logits=zeros(padded, E, dtype=torch.bfloat16),
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
        x_expert=zeros(tiles * BM, D, dtype=torch.bfloat16),
        h_dense=zeros(padded, config.dense_intermediate, dtype=torch.bfloat16),
        h_expert=zeros(tiles * BM, config.expert_intermediate, dtype=torch.bfloat16),
        gate_up_left=zeros(tiles, dtype=torch.int32),
        dense_gate_up_left=zeros(padded // BM, dtype=torch.int32),
        dense_out=zeros(padded, D, dtype=torch.bfloat16),
        expert_acc=zeros(tokens, D),
        T=tokens,
        TP=padded,
        NT=tiles,
    )


def wall_ms(function, repetitions: int = 20) -> float:
    """Median wall-clock time of ``function()`` until the GPU is idle, in milliseconds. (DaCe runs on its own
    streams, which events on PyTorch's stream do not wait for.)"""
    function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sorted(times)[repetitions // 2] * 1e3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    parser.add_argument("--small", action="store_true", help="a small configuration, for testing")
    parser.add_argument("--pes", type=int, default=46 * 2, help="processing elements (thread blocks)")
    args = parser.parse_args()

    config = (
        gemma4_moe.Config(hidden=256, dense_intermediate=128, num_experts=128, top_k=4, expert_intermediate=128)
        if args.small
        else gemma4_moe.Config()
    )
    w = gemma4_moe.random_weights(config)
    program = build(config, args.pes).compile()
    weights = arguments(w)
    for tokens in args.tokens:
        x = torch.randn(tokens, config.hidden, device="cuda").to(torch.bfloat16)
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
        # Time the generated code alone (as a replayed CUDA graph would run), not Python's argument conversion
        callargs, initargs = program.construct_arguments(x=x, out=out, **weights, **scratch)
        time_ms = wall_ms(lambda: program.fast_call(callargs, initargs))
        print(
            f"{tokens:5d} tokens: DaCe {time_ms:8.3f} ms (error {error:.1e}, {int((~alike).sum())} routed differently)"
        )
