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
        gate_up_left: dace.int32[NT],
        dense_gate_up_left: dace.int32[TP // BM],
        dense_out: dace.bfloat16[TP, D],
        expert_acc: dace.float32[T, D],
        token_left: dace.int32[T],
    ):
        S = dace.define_stream(dace.int64, queue_size)

        # Python float literals are double precision in tasklets, which GPUs compute slowly: constants are float32.
        #
        # The prologue, a thread block per token: the RMS of the token (all the pre-norms normalize the same residual)
        # and the inputs of the dense branch, the experts and the router (the normalized input, rounded to bfloat16 as
        # the model does, and scaled). The blocks also reset the scratch memory, and the first block queues the first
        # tasks: the router's (tiles of tokens times slices of the input), whose last one pushes the top-k task of the
        # tile, whose last one pushes the scan that pushes the expert tasks; then the dense gate/up tasks.
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
                    rt >> row_token(-1)[:]
                    ea >> expert_acc(-1)[t, :]
                    tkl >> token_left(-1)[t]
                    c >> count(-1)[:]
                    dr >> dense_rows(-1)[:]
                    rl >> route_left(-1)[:]
                    dl >> dense_gate_up_left(-1)[:]
                    tl >> top_k_left(-1)[0]
                    part = dace.float32(0)
                    for d in range(th, D, THREADS):
                        v = dace.float32(xs[d])
                        part = part + v * v
                        ea[d] = 0
                    s = part
                    if th == 0:  # The down tiles that add to this token: those of its experts, and the dense ones
                        tkl = (K + 1) * (D // BN)
                    for r in range(t * THREADS + th, NT * BM, T * THREADS):
                        rt[r] = -1
                    if t == 0:
                        for ex in range(th, E, THREADS):
                            c[ex] = 0
                        for g in range(th, TP // BM, THREADS):
                            dr[g] = min(BM, T - g * BM)
                            rl[g] = RP
                            dl[g] = DI // HALF
                        if th == 0:
                            tl = T  # A top-k task per token
            for th in dace.map[0:THREADS] @ TB:
                with dace.tasklet:
                    s << ss[0]
                    xs << x[t, :]
                    w1 << pre_norm[:]
                    rs << router_scale[:]
                    w2 << pre_norm_2[:]
                    _published_r << route_left(-1)[:]  # Counters before tasks
                    _published_d << dense_gate_up_left(-1)[:]
                    inv >> inv_rms(-1)[t]
                    xd >> x_dense[t, :]
                    xr >> x_router[t, :]
                    xe >> x_expert[t, :]
                    task >> S(-1)
                    v = dace.float32(1) / math.sqrt(s / D + dace.float32(eps))
                    if th == 0:
                        inv = v
                    for d in range(th, D, THREADS):
                        n = dace.float32(xs[d]) * v
                        xd[d] = dace.bfloat16(n * w1[d])
                        xr[d] = dace.bfloat16(dace.float32(dace.bfloat16(n)) * rs[d] * dace.float32(root))
                        xe[d] = dace.bfloat16(n * w2[d])
                    if t == 0:
                        num_router = TP // BM * RP
                        for i in range(th, num_router + TP // BM * (DI // HALF), THREADS):
                            if i < num_router:
                                task = ROUTE | (i // RP) << FIELD | i % RP
                            else:  # The tiles of a chunk of the weight one after the other: they read it from L2
                                task = (
                                    DENSE_GATE_UP
                                    | ((i - num_router) % (TP // BM)) << FIELD
                                    | (i - num_router) // (TP // BM)
                                )

        @dace.consumescope(S, num_pes)
        def worker(task, p):
            kind = task >> KIND << KIND
            g = (task >> FIELD) & MASK  # Tile of rows
            ch = task & MASK  # Chunk of columns
            r0 = g * BM
            # The product of a tile overwrites the operands that tile_matmul staged: sharing their memory leaves room for
            # a third thread block on each multiprocessor (whose loads cover the gaps between the tasks of the others)
            staging = dace.ndarray([STAGING], dtype=dace.bfloat16, storage=SHARED)
            work = staging.reshape(WORKSPACE)
            tile = staging[0 : BM * BN].reshape([BM, BN])
            if kind == EXPERT_GATE_UP or kind == DENSE_GATE_UP:
                if kind == EXPERT_GATE_UP:
                    e = tile_expert[g]
                    nr = tile_rows[g]
                    tm.tile_matmul(x_expert, expert_gate_up_p[e, ch], tile, work, nr, row_token[r0 : r0 + BM])
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
                # The last gate/up task of these rows pushes their down tasks (a thread each)
                last = dace.ndarray([1], dtype=dace.int32, storage=SHARED)
                for _ in dace.map[0:1] @ TB:
                    with dace.tasklet:
                        k << kind
                        tile_index << g
                        _published_e << h_expert(-1)[:, :]  # After the block wrote its part (release)
                        _published_d << h_dense(-1)[:, :]
                        gl >> gate_up_left(-1, lambda a, b: a + b)[:]
                        dl >> dense_gate_up_left(-1, lambda a, b: a + b)[:]
                        is_last >> last[0]
                        left = 0
                        if k == EXPERT_GATE_UP:
                            left = gl[tile_index] = -1
                        else:
                            left = dl[tile_index] = -1
                        is_last = left == 1
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        k << kind
                        tile_index << g
                        is_last << last[0]
                        dn >> S(-1)
                        if is_last and w * 32 + lane < D // BN:
                            if k == EXPERT_GATE_UP:
                                dn = EXPERT_DOWN | tile_index << FIELD | w * 32 + lane
                            else:
                                dn = DENSE_DOWN | tile_index << FIELD | w * 32 + lane
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
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        row0 << r0
                        nrows << nr
                        tok << row_token(-1)[:]
                        _published << expert_acc(-1)[:, :]
                        tl >> token_left(-1, lambda a, b: a + b)[:]
                        nxt >> S(-1)
                        if w * 32 + lane < nrows:
                            token = tok[row0 + w * 32 + lane]
                            left = tl[token] = -1
                            if left == 1:  # The last tile that adds to the token
                                nxt = OUTPUT | token << FIELD
            elif kind == DENSE_DOWN:
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
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        row0 << r0
                        nrows << nr
                        _published << dense_out(-1)[:, :]
                        tl >> token_left(-1, lambda a, b: a + b)[:]
                        nxt >> S(-1)
                        if w * 32 + lane < nrows:
                            left = tl[row0 + w * 32 + lane] = -1
                            if left == 1:
                                nxt = OUTPUT | (row0 + w * 32 + lane) << FIELD

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
                # The last part of these tokens pushes their top-k tasks (a thread each)
                last = dace.ndarray([1], dtype=dace.int32, storage=SHARED)
                for _ in dace.map[0:1] @ TB:
                    with dace.tasklet:
                        tile_index << g
                        _published << route_partial(-1)[:, :]
                        rl >> route_left(-1, lambda a, b: a + b)[:]
                        is_last >> last[0]
                        left = rl[tile_index] = -1
                        is_last = left == 1
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        is_last << last[0]
                        nrows << nr
                        row0 << r0
                        nxt >> S(-1)
                        if is_last and w * 32 + lane < nrows:
                            nxt = TOP_K | (row0 + w * 32 + lane) << FIELD
            elif kind == TOP_K:
                # The experts of token g. A thread per expert, then per chosen expert.
                chosen = dace.ndarray([K], dtype=dace.int32, storage=SHARED)
                # The token's router logits (the projection rounds them to bfloat16), in the tile
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        parts << route_partial(-1)[:, :]
                        tok << g
                        lg >> tile[0, w * 32 + lane]
                        total = dace.float32(0)
                        for part in range(RP):
                            total = total + parts[part * TP + tok, w * 32 + lane]
                        lg = dace.bfloat16(total)
                # The rank of every logit (ties: lowest index first): the top k are chosen
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        lgs << tile[0, :]
                        cho >> chosen(-1)[:]
                        ex = w * 32 + lane
                        v = dace.float32(lgs[ex])
                        rank = 0
                        for e2 in range(E):
                            v2 = dace.float32(lgs[e2])
                            if v2 > v or (v2 == v and e2 < ex):
                                rank = rank + 1
                        if rank < K:
                            cho[rank] = ex
                # The softmax probabilities of the chosen experts, renormalized (the softmax denominator cancels) and
                # scaled; claim a row of every chosen expert
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        lgs << tile[0, :]
                        chs << chosen[:]
                        tok << g
                        pes << per_expert_scale[:]
                        c >> count(-1, lambda a, b: a + b)[:]
                        pe >> pair_expert(-1)[:, :]
                        ps >> pair_slot(-1)[:, :]
                        pw >> pair_weight(-1)[:, :]
                        j = w * 32 + lane
                        if j < K:
                            top = dace.float32(lgs[chs[0]])
                            total = dace.float32(0)
                            for j2 in range(K):
                                total = total + math.exp(dace.float32(lgs[chs[j2]]) - top)
                            ex = chs[j]
                            pe[tok, j] = ex
                            pw[tok, j] = math.exp(dace.float32(lgs[ex]) - top) / total * pes[ex]
                            ps[tok, j] = c[ex] = 1
                for _ in dace.map[0:1] @ TB:
                    with dace.tasklet:
                        _published_e << pair_expert(-1)[:, :]
                        _published_s << pair_slot(-1)[:, :]
                        _published_w << pair_weight(-1)[:, :]
                        tl >> top_k_left(-1, lambda a, b: a + b)[0]
                        nxt >> S(-1)
                        left = tl = -1
                        if left == 1:  # The last token
                            nxt = SCAN
            elif kind == OUTPUT:
                # Normalize both branches of token g, add, normalize, add the residual (in bfloat16, as the model)
                sums = dace.ndarray([3], dtype=dace.float32, storage=SHARED)
                for _ in dace.map[0:1] @ TB:
                    with dace.tasklet:
                        sm >> sums[:]
                        sm[0] = 0
                        sm[1] = 0
                        sm[2] = 0
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        tk << g
                        dn << dense_out(-1)[:, :]
                        ex << expert_acc(-1)[:, :]
                        s1 >> sums(1, lambda a, b: a + b)[0]
                        s2 >> sums(1, lambda a, b: a + b)[1]
                        a1 = dace.float32(0)
                        a2 = dace.float32(0)
                        for d in range(w * 32 + lane, D, THREADS):
                            v1 = dace.float32(dn[tk, d])
                            v2 = dace.float32(dace.bfloat16(ex[tk, d]))
                            a1 = a1 + v1 * v1
                            a2 = a2 + v2 * v2
                        s1 = a1
                        s2 = a2
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        sm << sums[0:2]
                        tk << g
                        dn << dense_out(-1)[:, :]
                        ex << expert_acc(-1)[:, :]
                        p1 << post_norm_1[:]
                        p2 << post_norm_2[:]
                        s3 >> sums(1, lambda a, b: a + b)[2]
                        inv1 = dace.float32(1) / math.sqrt(sm[0] / D + dace.float32(eps))
                        inv2 = dace.float32(1) / math.sqrt(sm[1] / D + dace.float32(eps))
                        a3 = dace.float32(0)
                        for d in range(w * 32 + lane, D, THREADS):
                            b1 = dace.float32(dace.bfloat16(dace.float32(dn[tk, d]) * inv1 * p1[d]))
                            b2 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(ex[tk, d])) * inv2 * p2[d]))
                            v = dace.float32(dace.bfloat16(b1 + b2))
                            a3 = a3 + v * v
                        s3 = a3
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        sm << sums[:]
                        tk << g
                        dn << dense_out(-1)[:, :]
                        ex << expert_acc(-1)[:, :]
                        p1 << post_norm_1[:]
                        p2 << post_norm_2[:]
                        xs << x(-1)[:, :]
                        po << post_norm[:]
                        o >> out(-1)[:, :]
                        inv1 = dace.float32(1) / math.sqrt(sm[0] / D + dace.float32(eps))
                        inv2 = dace.float32(1) / math.sqrt(sm[1] / D + dace.float32(eps))
                        inv = dace.float32(1) / math.sqrt(sm[2] / D + dace.float32(eps))
                        for d in range(w * 32 + lane, D, THREADS):
                            b1 = dace.float32(dace.bfloat16(dace.float32(dn[tk, d]) * inv1 * p1[d]))
                            b2 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(ex[tk, d])) * inv2 * p2[d]))
                            v = dace.float32(dace.bfloat16(b1 + b2))
                            o[tk, d] = dace.bfloat16(
                                dace.float32(xs[tk, d]) + dace.float32(dace.bfloat16(v * inv * po[d]))
                            )
            else:  # SCAN
                # Every expert's rows start at a whole tile. A thread per expert.
                tiles_of = dace.ndarray([E], dtype=dace.int32, storage=SHARED)
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        cnt << count(-1)[:]
                        to >> tiles_of[w * 32 + lane]
                        to = (cnt[w * 32 + lane] + BM - 1) // BM
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        tos << tiles_of[:]
                        cnt << count(-1)[:]
                        ts >> tile_start(-1)[:]
                        te >> tile_expert(-1)[:]
                        tr >> tile_rows(-1)[:]
                        nt >> num_tiles(-1)[0]
                        ex = w * 32 + lane
                        start = 0
                        for e2 in range(ex):
                            start = start + tos[e2]
                        ts[ex] = start
                        for g2 in range(tos[ex]):
                            te[start + g2] = ex
                            tr[start + g2] = min(BM, cnt[ex] - g2 * BM)
                        if ex == E - 1:
                            nt = start + tos[ex]
                # The rows of every expert, sorted by expert: their tokens and routing weights
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        pe << pair_expert(-1)[:, :]
                        ps << pair_slot(-1)[:, :]
                        pw << pair_weight(-1)[:, :]
                        ts << tile_start[:]
                        ntl << num_tiles[0]
                        rt >> row_token(-1)[:]
                        rw >> row_weight(-1)[:]
                        gl >> gate_up_left(-1)[:]
                        for i in range(w * 32 + lane, T * K, THREADS):
                            tok = i // K
                            j = i % K
                            ex = pe[tok, j]
                            row = ts[ex] * BM + ps[tok, j]
                            rt[row] = tok
                            rw[row] = pw[tok, j]
                        for g2 in range(w * 32 + lane, ntl, THREADS):
                            gl[g2] = I // HALF
                # The expert gate/up tasks
                for w, lane in dace.map[0:WARPS, 0:32] @ TB:
                    with dace.tasklet:
                        ntl << num_tiles[0]
                        _published_t << row_token(-1)[:]
                        _published_w << row_weight(-1)[:]
                        _published_l << gate_up_left(-1)[:]
                        _published_e << tile_expert(-1)[:]
                        _published_r << tile_rows(-1)[:]
                        nxt >> S(-1)
                        for i in range(w * 32 + lane, ntl * (I // HALF), THREADS):
                            nxt = EXPERT_GATE_UP | (i // (I // HALF)) << FIELD | i % (I // HALF)

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
    # Without synchronizing the host at the end: on PyTorch's (default) stream, the caller synchronizes when it needs to
    for state in sdfg.states():
        state.nosync = True
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
        gate_up_left=zeros(tiles, dtype=torch.int32),
        dense_gate_up_left=zeros(padded // BM, dtype=torch.int32),
        dense_out=zeros(padded, D, dtype=torch.bfloat16),
        expert_acc=zeros(tokens, D),
        token_left=zeros(tokens, dtype=torch.int32),
        T=tokens,
        TP=padded,
        NT=tiles,
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
