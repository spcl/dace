# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Gemma 4 mixture-of-experts feed-forward block (see ``gemma4_moe.py``) as one persistent DaCe GPU kernel.

After a few small kernels that route the tokens, a single consume scope runs all the matrix products of the block as
tasks on a queue. Every processing element is a GPU thread block that pops a task and computes a 16 x 64 tile of
one product with tensor cores (four warps, one 16 x 16 tile each). There are four kinds of tasks:

* expert gate/up: 16 rows of an expert (tokens routed to it) times 64 columns of its gate and up projections,
  followed by GELU(gate) * up;
* expert down: the same 16 rows times 64 output columns of the expert's down projection, scaled by each token's
  routing weight and added to the token's output (an atomic write-conflict resolution);
* dense gate/up and dense down: the same for 16 tokens in the dense MLP that runs in parallel.

Down tasks depend on all the gate/up tasks of their rows. The gate/up task that finishes last (it reads the old
value of a counter it decrements, an acquire-release update) pushes the down tasks of its rows, so the queue never
waits for a whole projection to finish, and dense and expert tasks fill each other's gaps.
"""

import math
import sys

import gemma4_moe
import torch

import dace

sys.path.insert(0, f"{__file__.rsplit('/', 2)[0]}/codegen")
import tensor_cores as tc  # noqa: E402  (Tensor core fragments: storage types and code generation)

tc.extend_dace()

TC_A, TC_B, TC_C = (
    dace.StorageType.TensorCore_A,
    dace.StorageType.TensorCore_B,
    dace.StorageType.TensorCore_Accumulator,
)
WARPS = 4  # Warps per thread block (task), one 16 x 16 tile each
COLUMNS = 16 * WARPS  # Columns of a task
KIND = 2**40  # A task is ``kind * KIND + a * FIELD + b``
FIELD = 2**20
EXPERT_GATE_UP, EXPERT_DOWN, DENSE_GATE_UP, DENSE_DOWN = range(4)

T = dace.symbol("T")  # Tokens
TP = dace.symbol("TP")  # Tokens, padded to a multiple of 16
NT = dace.symbol("NT")  # The most expert tiles there can be: T * top_k / 16 + num_experts


def col_major(*shape) -> dace.data.Array:
    """A bfloat16 matrix (or batch of them) stored column-major: the transpose of a row-major weight."""
    *batch, rows, cols = shape
    strides = [rows * cols] * len(batch) + [1, rows]
    return dace.data.Array(dace.bfloat16, list(shape), strides=strides)


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
    assert D % COLUMNS == 0 and DI % COLUMNS == 0 and I % COLUMNS == 0 and E % COLUMNS == 0
    eps = config.eps
    root = D**-0.5
    gelu_c = math.sqrt(2 / math.pi)

    @dace.program
    def moe_block(
        x: dace.bfloat16[T, D],
        pre_norm: dace.float32[D],
        post_norm_1: dace.float32[D],
        router_scale: dace.float32[D],
        router_t: col_major(D, E),
        per_expert_scale: dace.float32[E],
        pre_norm_2: dace.float32[D],
        post_norm_2: dace.float32[D],
        post_norm: dace.float32[D],
        gate_t: col_major(D, DI),
        up_t: col_major(D, DI),
        down_t: col_major(DI, D),
        expert_gate_up_t: col_major(E, D, 2 * I),
        expert_down_t: col_major(E, I, D),
        out: dace.bfloat16[T, D],
        # Scratch
        inv_rms: dace.float32[T],
        x_router: dace.bfloat16[TP, D],
        logits: dace.float32[TP, E],
        count: dace.int32[E],
        pair_expert: dace.int32[T, K],
        pair_slot: dace.int32[T, K],
        pair_weight: dace.float32[T, K],
        tile_start: dace.int32[E],
        tile_expert: dace.int32[NT],
        num_tiles: dace.int32[1],
        row_token: dace.int32[NT * 16],
        row_weight: dace.float32[NT * 16],
        x_dense: dace.bfloat16[TP, D],
        x_expert: dace.bfloat16[NT * 16, D],
        h_dense: dace.bfloat16[TP, DI],
        h_expert: dace.bfloat16[NT * 16, I],
        gate_up_left: dace.int32[NT],
        dense_gate_up_left: dace.int32[TP // 16],
        dense_acc: dace.float32[TP, D],
        expert_acc: dace.float32[T, D],
    ):
        S = dace.define_stream(dace.int64, queue_size)

        # Reset the counters and accumulators
        for e in dace.map[0:E]:
            with dace.tasklet:
                c >> count[e]
                c = 0
        for r in dace.map[0 : NT * 16]:
            with dace.tasklet:
                rt >> row_token[r]
                rt = -1
        for t, d in dace.map[0:T, 0:D]:
            with dace.tasklet:
                a >> expert_acc[t, d]
                a = 0

        # RMS of every token (all the pre-norms normalize the same residual)
        for t in dace.map[0:T]:
            with dace.tasklet:
                xs << x[t, :]
                inv >> inv_rms[t]
                ss = dace.float32(0)
                for d in range(D):
                    v = dace.float32(xs[d])
                    ss = ss + v * v
                inv = 1.0 / math.sqrt(ss / D + eps)

        # Dense branch input
        for t, d in dace.map[0:T, 0:D]:
            with dace.tasklet:
                xv << x[t, d]
                inv << inv_rms[t]
                w << pre_norm[d]
                xd >> x_dense[t, d]
                xd = dace.bfloat16(dace.float32(xv) * inv * w)

        # Router logits: the RMS-normalized input (bfloat16, as the model rounds it), scaled, then projected (with
        # tensor cores, a 16 x 64 tile per thread block)
        for t, d in dace.map[0:T, 0:D]:
            with dace.tasklet:
                xv << x[t, d]
                inv << inv_rms[t]
                rs << router_scale[d]
                xr >> x_router[t, d]
                xr = dace.bfloat16(dace.float32(dace.bfloat16(dace.float32(xv) * inv)) * rs * root)
        for g, ch in dace.map[0 : TP // 16, 0 : E // COLUMNS]:
            for w, lane in dace.map[0:WARPS, 0:32]:
                acc = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                tc.frag_fill(acc, 0.0)
                for k in range(0, D, 16):
                    a = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_A)
                    b = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                    a[:] = x_router[g * 16 : g * 16 + 16, k : k + 16]
                    b[:] = router_t[k : k + 16, ch * COLUMNS + w * 16 : ch * COLUMNS + w * 16 + 16]
                    tc.wmma(a, b, acc)
                logits[g * 16 : g * 16 + 16, ch * COLUMNS + w * 16 : ch * COLUMNS + w * 16 + 16] = acc

        # Softmax over all experts, top-k, renormalize, scale; claim a row of every chosen expert
        for t in dace.map[0:T]:
            with dace.tasklet:
                lgs << logits[t, :]
                pes << per_expert_scale[:]
                c >> count(-1, lambda a, b: a + b)[:]
                pe >> pair_expert[t, :]
                ps >> pair_slot[t, :]
                pw >> pair_weight[t, :]
                # The model's projection rounds the logits to bfloat16
                top = dace.float32(-1e30)
                for e in range(E):
                    top = max(top, dace.float32(dace.bfloat16(lgs[e])))
                total = dace.float32(0)
                for e in range(E):
                    total = total + math.exp(dace.float32(dace.bfloat16(lgs[e])) - top)
                chosen_sum = dace.float32(0)
                previous = dace.float32(1e30)
                previous_e = -1
                for j in range(K):
                    best = dace.float32(-1e30)
                    best_e = 0
                    for e in range(E):
                        # The largest logit below the previous pick (ties: lowest index first)
                        v = dace.float32(dace.bfloat16(lgs[e]))
                        below = v < previous or (v == previous and e > previous_e)
                        if below and (v > best or (v == best and e < best_e)):
                            best = v
                            best_e = e
                    previous = best
                    previous_e = best_e
                    p = math.exp(best - top) / total
                    pe[j] = best_e
                    pw[j] = p
                    chosen_sum = chosen_sum + p
                for j in range(K):
                    pw[j] = pw[j] / chosen_sum * pes[pe[j]]
                    ps[j] = c[pe[j]] = 1

        # Every expert's rows start at a whole tile
        for _ in dace.map[0:1]:
            with dace.tasklet:
                cnt << count[:]
                ts >> tile_start[:]
                te >> tile_expert(-1)[:]
                nt >> num_tiles[0]
                start = 0
                for e in range(E):
                    ts[e] = start
                    tiles = (cnt[e] + 15) // 16
                    for g in range(tiles):
                        te[start + g] = e
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
                row = ts[pe] * 16 + ps
                rt[row] = t
                rw[row] = pw
        for t, j, d in dace.map[0:T, 0:K, 0:D]:
            with dace.tasklet:
                pe << pair_expert[t, j]
                ps << pair_slot[t, j]
                ts << tile_start[:]
                xv << x[t, d]
                inv << inv_rms[t]
                w << pre_norm_2[d]
                xe >> x_expert(-1)[:, :]
                xe[ts[pe] * 16 + ps, d] = dace.bfloat16(dace.float32(xv) * inv * w)

        # Seed the gate/up tasks (the down tasks follow them)
        for g, ch in dace.map[0:NT, 0 : I // COLUMNS]:
            with dace.tasklet:
                nt << num_tiles[0]
                left >> gate_up_left(-1)[g]
                task >> S(-1)
                if g < nt:
                    left = I // COLUMNS
                    task = EXPERT_GATE_UP * KIND + g * FIELD + ch
        for g, ch in dace.map[0 : TP // 16, 0 : DI // COLUMNS]:
            with dace.tasklet:
                left >> dense_gate_up_left[g]
                task >> S(-1)
                left = DI // COLUMNS
                task = DENSE_GATE_UP * KIND + g * FIELD + ch

        @dace.consumescope(S, num_pes)
        def worker(task, p):
            kind = task // KIND
            r0 = ((task // FIELD) % FIELD) * 16  # First row of the tile
            c0 = (task % FIELD) * COLUMNS  # First column of the task
            scratch = dace.ndarray([WARPS, 2, 16, 16], dtype=dace.float32, storage=dace.StorageType.GPU_Shared)
            if kind == EXPERT_GATE_UP:
                e = tile_expert[r0 // 16]
                for w, lane in dace.map[0:WARPS, 0:32]:
                    gate = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                    up = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                    tc.frag_fill(gate, 0.0)
                    tc.frag_fill(up, 0.0)
                    for k in range(0, D, 16):
                        a = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_A)
                        bg = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                        bu = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                        a[:] = x_expert[r0 : r0 + 16, k : k + 16]
                        bg[:] = expert_gate_up_t[e, k : k + 16, c0 + w * 16 : c0 + w * 16 + 16]
                        bu[:] = expert_gate_up_t[e, k : k + 16, I + c0 + w * 16 : I + c0 + w * 16 + 16]
                        tc.wmma(a, bg, gate)
                        tc.wmma(a, bu, up)
                    scratch[w, 0] = gate
                    scratch[w, 1] = up
                    tc.warp_sync(scratch[w])
                    for i in range(8):
                        with dace.tasklet:
                            gv << scratch[w, 0, (lane * 8 + i) // 16, (lane * 8 + i) % 16]
                            uv << scratch[w, 1, (lane * 8 + i) // 16, (lane * 8 + i) % 16]
                            row0 << r0
                            col0 << c0
                            hv >> h_expert(-1)[:, :]
                            gb = dace.float32(dace.bfloat16(gv))
                            gl = 0.5 * gb * (1.0 + math.tanh(gelu_c * (gb + 0.044715 * gb * gb * gb)))
                            hv[row0 + (lane * 8 + i) // 16, col0 + w * 16 + (lane * 8 + i) % 16] = dace.bfloat16(
                                dace.float32(dace.bfloat16(gl)) * dace.float32(dace.bfloat16(uv))
                            )
                for _ in dace.map[0:1]:  # One thread
                    with dace.tasklet:
                        row0 << r0
                        _published << h_expert(-1)[:, :]  # After the block wrote its part (release)
                        gl >> gate_up_left(-1, lambda a, b: a + b)[:]
                        dn >> S(-1)
                        left = gl[row0 // 16] = -1
                        if left == 1:  # The last gate/up task of these rows
                            for ch in range(D // COLUMNS):
                                dn = EXPERT_DOWN * KIND + (row0 // 16) * FIELD + ch
            elif kind == EXPERT_DOWN:
                e = tile_expert[r0 // 16]
                for w, lane in dace.map[0:WARPS, 0:32]:
                    acc = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                    tc.frag_fill(acc, 0.0)
                    for k in range(0, I, 16):
                        a = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_A)
                        b = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                        a[:] = h_expert[r0 : r0 + 16, k : k + 16]
                        b[:] = expert_down_t[e, k : k + 16, c0 + w * 16 : c0 + w * 16 + 16]
                        tc.wmma(a, b, acc)
                    scratch[w, 0] = acc
                    tc.warp_sync(scratch[w])
                    for i in range(8):
                        with dace.tasklet:
                            yv << scratch[w, 0, (lane * 8 + i) // 16, (lane * 8 + i) % 16]
                            row0 << r0
                            col0 << c0
                            tok << row_token(-1)[:]
                            rw << row_weight(-1)[:]
                            o >> expert_acc(-1, lambda a, b: a + b)[:, :]
                            row = row0 + (lane * 8 + i) // 16
                            if tok[row] >= 0:
                                o[tok[row], col0 + w * 16 + (lane * 8 + i) % 16] = (
                                    dace.float32(dace.bfloat16(yv)) * rw[row]
                                )
            elif kind == DENSE_GATE_UP:
                for w, lane in dace.map[0:WARPS, 0:32]:
                    gate = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                    up = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                    tc.frag_fill(gate, 0.0)
                    tc.frag_fill(up, 0.0)
                    for k in range(0, D, 16):
                        a = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_A)
                        bg = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                        bu = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                        a[:] = x_dense[r0 : r0 + 16, k : k + 16]
                        bg[:] = gate_t[k : k + 16, c0 + w * 16 : c0 + w * 16 + 16]
                        bu[:] = up_t[k : k + 16, c0 + w * 16 : c0 + w * 16 + 16]
                        tc.wmma(a, bg, gate)
                        tc.wmma(a, bu, up)
                    scratch[w, 0] = gate
                    scratch[w, 1] = up
                    tc.warp_sync(scratch[w])
                    for i in range(8):
                        with dace.tasklet:
                            gv << scratch[w, 0, (lane * 8 + i) // 16, (lane * 8 + i) % 16]
                            uv << scratch[w, 1, (lane * 8 + i) // 16, (lane * 8 + i) % 16]
                            row0 << r0
                            col0 << c0
                            hv >> h_dense(-1)[:, :]
                            gb = dace.float32(dace.bfloat16(gv))
                            gl = 0.5 * gb * (1.0 + math.tanh(gelu_c * (gb + 0.044715 * gb * gb * gb)))
                            hv[row0 + (lane * 8 + i) // 16, col0 + w * 16 + (lane * 8 + i) % 16] = dace.bfloat16(
                                dace.float32(dace.bfloat16(gl)) * dace.float32(dace.bfloat16(uv))
                            )
                for _ in dace.map[0:1]:
                    with dace.tasklet:
                        row0 << r0
                        _published << h_dense(-1)[:, :]  # After the block wrote its part (release)
                        gl >> dense_gate_up_left(-1, lambda a, b: a + b)[:]
                        dn >> S(-1)
                        left = gl[row0 // 16] = -1
                        if left == 1:
                            for ch in range(D // COLUMNS):
                                dn = DENSE_DOWN * KIND + (row0 // 16) * FIELD + ch
            else:  # DENSE_DOWN
                for w, lane in dace.map[0:WARPS, 0:32]:
                    acc = dace.ndarray([16, 16], dtype=dace.float32, storage=TC_C)
                    tc.frag_fill(acc, 0.0)
                    for k in range(0, DI, 16):
                        a = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_A)
                        b = dace.ndarray([16, 16], dtype=dace.bfloat16, storage=TC_B)
                        a[:] = h_dense[r0 : r0 + 16, k : k + 16]
                        b[:] = down_t[k : k + 16, c0 + w * 16 : c0 + w * 16 + 16]
                        tc.wmma(a, b, acc)
                    dense_acc[r0 : r0 + 16, c0 + w * 16 : c0 + w * 16 + 16] = acc

        # Output: normalize both branches, add, normalize, add the residual (in bfloat16, as the model)
        for t in dace.map[0:T]:
            with dace.tasklet:
                xs << x[t, :]
                dn << dense_acc[t, :]
                ex << expert_acc[t, :]
                p1 << post_norm_1[:]
                p2 << post_norm_2[:]
                po << post_norm[:]
                o >> out[t, :]
                ss1 = dace.float32(0)
                ss2 = dace.float32(0)
                for d in range(D):
                    v1 = dace.float32(dace.bfloat16(dn[d]))
                    v2 = dace.float32(dace.bfloat16(ex[d]))
                    ss1 = ss1 + v1 * v1
                    ss2 = ss2 + v2 * v2
                inv1 = 1.0 / math.sqrt(ss1 / D + eps)
                inv2 = 1.0 / math.sqrt(ss2 / D + eps)
                ss = dace.float32(0)
                for d in range(D):
                    a1 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(dn[d])) * inv1 * p1[d]))
                    a2 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(ex[d])) * inv2 * p2[d]))
                    s = dace.float32(dace.bfloat16(a1 + a2))
                    ss = ss + s * s
                inv = 1.0 / math.sqrt(ss / D + eps)
                for d in range(D):
                    a1 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(dn[d])) * inv1 * p1[d]))
                    a2 = dace.float32(dace.bfloat16(dace.float32(dace.bfloat16(ex[d])) * inv2 * p2[d]))
                    s = dace.float32(dace.bfloat16(a1 + a2))
                    o[d] = dace.bfloat16(dace.float32(xs[d]) + dace.float32(dace.bfloat16(s * inv * po[d])))

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
            elif node.map.params in (["w", "lane"], ["_"]):
                node.map.schedule = dace.ScheduleType.GPU_ThreadBlock
        elif isinstance(node, dace.nodes.ConsumeEntry):
            node.consume.schedule = dace.ScheduleType.GPU_Device
    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            if desc.storage == TC_B:
                desc.strides = (1, 16)  # Column-major fragments


def arguments(w: gemma4_moe.Weights) -> dict[str, torch.Tensor]:
    """The weights as the program takes them: projections transposed (column-major views, no copies)."""
    return dict(
        pre_norm=w.pre_norm,
        post_norm_1=w.post_norm_1,
        router_scale=w.router_scale,
        router_t=w.router_proj.T,
        per_expert_scale=w.per_expert_scale,
        pre_norm_2=w.pre_norm_2,
        post_norm_2=w.post_norm_2,
        post_norm=w.post_norm,
        gate_t=w.gate_proj.T,
        up_t=w.up_proj.T,
        down_t=w.down_proj.T,
        expert_gate_up_t=w.expert_gate_up.transpose(1, 2),
        expert_down_t=w.expert_down.transpose(1, 2),
    )


def workspace(config: gemma4_moe.Config, tokens: int) -> dict:
    """Scratch buffers and symbol values for ``tokens`` tokens."""
    padded = (tokens + 15) // 16 * 16
    tiles = tokens * config.top_k // 16 + config.num_experts
    E, K, D = config.num_experts, config.top_k, config.hidden

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    return dict(
        inv_rms=zeros(tokens),
        x_router=zeros(padded, D, dtype=torch.bfloat16),
        logits=zeros(padded, E),
        count=zeros(E, dtype=torch.int32),
        pair_expert=zeros(tokens, K, dtype=torch.int32),
        pair_slot=zeros(tokens, K, dtype=torch.int32),
        pair_weight=zeros(tokens, K),
        tile_start=zeros(E, dtype=torch.int32),
        tile_expert=zeros(tiles, dtype=torch.int32),
        num_tiles=zeros(1, dtype=torch.int32),
        row_token=zeros(tiles * 16, dtype=torch.int32),
        row_weight=zeros(tiles * 16),
        x_dense=zeros(padded, D, dtype=torch.bfloat16),
        x_expert=zeros(tiles * 16, D, dtype=torch.bfloat16),
        h_dense=zeros(padded, config.dense_intermediate, dtype=torch.bfloat16),
        h_expert=zeros(tiles * 16, config.expert_intermediate, dtype=torch.bfloat16),
        gate_up_left=zeros(tiles, dtype=torch.int32),
        dense_gate_up_left=zeros(padded // 16, dtype=torch.int32),
        dense_acc=zeros(padded, D),
        expert_acc=zeros(tokens, D),
        T=tokens,
        TP=padded,
        NT=tiles,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 2048])
    parser.add_argument("--small", action="store_true", help="a small configuration, for testing")
    parser.add_argument("--pes", type=int, default=46 * 4, help="processing elements (thread blocks)")
    args = parser.parse_args()

    config = (
        gemma4_moe.Config(hidden=256, dense_intermediate=128, num_experts=64, top_k=4, expert_intermediate=128)
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
        time = gemma4_moe.time_ms(lambda: program.fast_call(callargs, initargs))
        print(f"{tokens:5d} tokens: DaCe {time:8.3f} ms (error {error:.1e}, {int((~alike).sum())} routed differently)")
