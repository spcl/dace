# GPU Warp-Level Tile Codegen — Design

Goal: GPU kernel not thread-level but **warp-level**. A tile op = collective op of ONE warp
(32 lanes NVIDIA / 64 AMD). Grid spawns blocks, block spawns warps, codegen inside the block is
per-warp. Same dataflow IR, same tile lib nodes, new ISA backend + one scheduling transform.

## 1. Answer to the 8-warps-per-tile question

Old thread-level block: `Map[GPU_ThreadBlock, 16x16]` = 256 threads, 1 element/thread.
Warp mode, same 256-element block domain, 8 warps: a tile op is a **warp primitive** — cross-warp
cooperation needs smem + block barriers, a different (block-level) primitive class. So never assign
8 warps to one tile. **Split the block domain instead**:

```
256 elements = Map[GPU_Warp, w=0:8] x per-warp tile of W = 32
             = Map[GPU_Warp, w=0:4] x per-warp tile of W = 64   (R=2 elems/lane)
             = Map[GPU_Warp, w=0:2] x per-warp tile of W = 128  (R=4)
```

### The warp as a SIMD unit (native tile width)

Model the warp as one SIMD unit whose natural 1-D tile is `warpSize x dtype`, packed like a CPU
vector register:

| dtype  | pack (elems/lane) | native tile W = warpSize*pack | NVIDIA (32) | AMD (64) |
|--------|-------------------|-------------------------------|-------------|----------|
| fp64   | 1                 | warpSize                      | 64          | 128      |
| fp32   | 1                 | warpSize                      | 32          | 64       |
| fp16   | 2 (`half2`)       | 2*warpSize                    | 64          | 128      |
| fp8    | 4                 | 4*warpSize                    | 128         | 256      |

`warpSize` is a compile-time constant per backend (32 cuda / 64 hip, from config — never hardcode).
`pack` reuses the existing per-dtype packing the CUDA half2 path already has. So the 1-D default is
`W = warpSize * pack` with `R = 1` fragment slot per lane — the direct analogue of one AVX-512
register. `R > 1` (multiple tiles-worth per lane) is the register-blocking knob, default 1.

Rule: `W = R x pack x warpSize`. Warp count `num_warps = block_elems / W`, capped at 32
(1024-thread HW limit); if the block domain holds more tiles than warps, each warp loops over its
tiles (block-stride loop over tile slots — same shape as today's W-strided interior map).

Multi-dim block domains map through the EXISTING K-dim tile model: widths tuple, e.g. 16x16 fp32 on
NVIDIA -> per-warp widths (2, 16) (2 rows x 16 cols = 32 lanes); the same domain on AMD (64) takes
(4, 16). The warp backend flattens the widths (C-order) to `e in [0, prod(widths))` and applies the
fixed lane mapping below — but the SDFG never sees the flattening.

## 2. Schedule-tree representation (dataflow-compatible)

Three explicit scopes, each meaning exactly one hardware level:

```
Map[GPU_Device: bx, by]           # block tiles of the global domain (unchanged)
  Map[GPU_ThreadBlock: tb=0:T]    # T = num_warps*warpSize PHYSICAL threads; tb feeds NO element index
    Map[GPU_Warp: w=0:num_warps]  # warp slot = flat_tid / warpSize (WarpScopeGenerator, EXISTS)
      TileLoad / TileBinop / TileFMA / TileITE / TileReduce / TileStore   # W = R*warpSize, cooperative
```

- The ThreadBlock map degrades to a **physical spawn anchor**: it sets blockDim and nothing else.
  Its param is not read by the body (threadIdx is still what the warp map derives `w` from). This
  keeps `KernelSpec.block_dims`, smem allocation, and block-level sync on their existing scope.
- The GPU_Warp map + `WarpScopeGenerator` already exist
  (`experimental_cuda_helpers/scope_strategies.py:241`); all 32 lanes of warp `w` enter the body
  with the same `w` — exactly the all-lanes-present precondition the cooperative backend needs.
- Element addresses inside the body = device-map coords + `w`*W + in-tile offsets (internal to the
  tile ops). No lane index ever appears in the IR — lanes are a lowering detail of the backend.

Divergence invariant: a tile op must be reached by ALL lanes of its warp (shuffle/sync illegal
otherwise). Enforced structurally: tile-tagged warp bodies allow only the merge branch mode
(TileITE blend), never lane-divergent control flow. Same invariant machinery as the CPU path
(`pass_invariants.py`).

## 3. Warp-distributed register tiles (storage)

**Fragments are NOT an IR concept — only the backend ABI.** 1 warp = 1 tile (all inputs and
outputs) makes ownership total: producer and consumer of every tile buffer are cooperative tile ops
of the SAME warp, so no other party can ever observe the physical layout. Therefore the dataflow
model carries NO fragment type, no distributed-storage annotation, no layout attribute — a tile is
a plain `(W,)` Register transient, local to the warp scope. Memlets, subsets, and validation are
untouched.

**The SDFG keeps the multi-dim tile shape.** A (16,16) tile transient stays shape `(16, 16)` in the
IR — memlets and subsets address it multi-dimensionally, exactly like every K-dim CPU tile. The
INTRINSIC is what saves it into fragment arrays: the backend flattens C-order (innermost dim
contiguous) to `e in [0, prod(widths))` and distributes:

```
allocation:  (w1..wK) Register transient in warp scope -> T frag[R] per lane, R = prod(widths)/(warpSize*pack)
addressing:  flat e  <->  (r, lane) = (e / warpSize, e % warpSize)            # lane-major, fixed
K-dim:       e -> (i1..iK) row-major -> address = sum(i_d * stride_d)          # strides passed to the op
```

- Holding full W per lane would waste 32x registers; smem staging would cost latency + syncwarps.
  Per-lane fragments are the only sane physical form — but they are DESCRIBED by the backend, not
  represented in the SDFG.
- Lane-major makes `tile_load`/`tile_store` of a unit-stride tile perfectly coalesced: at step `r`
  the 32 lanes touch 32 consecutive addresses.
- Every CUDA_WARP expansion uses this one mapping, so tile-to-tile dataflow needs no shuffles for
  elementwise chains (each lane owns its elements end-to-end; no syncwarp between elementwise ops).
- The contract is closed by two invariants: (a) no plain tasklet touches a tile — already enforced
  (ConvertTaskletsToTileOps + pass_invariants); (b) NEW: a tile transient never escapes its warp
  scope — any value leaving goes through `TileStore` to global/shared. Break either and the layout
  leaks; keep both and the IR stays layout-free.
- Codegen change: allocating such a transient emits `T name[R]`, not `T name[W]`. Keyed on
  (Register array) x (inside GPU_Warp scope) x (warp mode); no new StorageType (KISS).
- This is the same shape as `nvcuda::wmma::fragment` / cuTile: IR at warp granularity, lanes as an
  opaque ABI. Tensor-core tiles later = a different fragment layout inside the backend, zero IR
  change; lane-major is just the trivial layout instance.

## 4. New ISA backend: `CUDA_WARP`

`_ISA_TO_IMPL["CUDA_WARP"] = "cuda_warp"`, header `dace/tile_ops/cuda_warp.h`, all `__device__`,
dtype-generic (fp64 first-class — unlike the per-thread `CUDA` backend which is the fp16 half2 path).
`warpSize` compile-time from config (32 cuda / 64 hip). Contract per op (R = VLEN/warpSize,
`lane = threadIdx-flat % warpSize`):

- `tile_load` (unit stride): `frag[r] = src[r*warpSize + lane]` — coalesced. Strided: same with
  `*stride`. Masked: per-element predicate, ZERO-FILL inactive, guarded read (OOB tail safe) —
  identical semantics to scalar.h.
- `tile_binop / tile_fma / tile_unop / tile_ite`: per-lane loop over R. No communication, no sync.
  Broadcast operand: splat once per lane.
- `tile_reduce` (K=1 full): per-lane fold of R elements, then `__shfl_xor_sync(FULL_MASK, ...)`
  butterfly (log2(warpSize) steps) — all lanes end holding the result. Op order fixed so verificaton
  vs the scalar reference stays a pure reassociation (documented atol, as today).
- `tile_gather / tile_scatter`: per-lane loop, distributed idx tile; masked lanes never dereference.
- `tile_mask_gen`: `mask[r] = (base + r*warpSize + lane) < ub`.
- Remainder: **masked_tail only**. A scalar postamble inside a warp is lane divergence — refused,
  same NotImplementedError shape as scalar_postamble at K>=2 today.

`select_tile_implementation`: CUDA_WARP joins CUDA/CUTILE as a schedule-gated device ISA (never
host-executed; arch-native check does not apply).

## 5. Transform: `VectorizeGPUWarp` (thin wrapper, reuse everything)

Mode of the existing multi-dim pipeline (like `VectorizeGPUMultiDim`), applied to already-offloaded
SDFGs (canonicalize-GPU offloads first, as today):

1. Pick innermost map(s) under `GPU_Device`; choose `widths` with `prod(widths) = R*warpSize`
   (innermost dim first for coalescing; K>=2 widths when innermost extent < warpSize).
2. Run the standard tile pipeline (MarkTileDims -> StrideMapByTileWidths -> stage/insert tile
   load/store -> ConvertTaskletsToTileOps -> mask gen -> remainder split) with
   `target_isa=CUDA_WARP`, `remainder_strategy=masked_tail`, `branch_mode=merge`.
3. **WarpSchedule step** (the only new pass): rewrite the W-strided tile-slot map to
   `Map[GPU_Warp, w=0:num_warps]` (+ per-warp tile-slot loop if slots > num_warps), insert/retarget
   the physical `Map[GPU_ThreadBlock, tb=0:num_warps*warpSize]` spawn anchor, stamp
   `gpu_block_size`.

Constraints enforced: `W % warpSize == 0`; `num_warps*warpSize <= 1024`; masked-tail only;
merge-branch only.

## 6. Minimal extension to the experimental GPU codegen

The whole feature is ~1 header + 1 transform + 4 small codegen deltas. Nothing in the existing
thread-level path changes; warp mode is reached only when a `GPU_Warp` map carries tile nodes.

**Already present, reused as-is (no edits):**
- `ScheduleType.GPU_Warp` + `WarpScopeGenerator` (`experimental_cuda_helpers/scope_strategies.py:241`)
  — derives `w = flat_tid / warpSize`, emits map params, guards partial warps. Its
  "parent must be GPU_ThreadBlock" check is satisfied by the spawn anchor.
- `ScopeManager`, `KernelSpec` (already carries `warpSize` from config, line ~1098), stream manager.
- The entire K-dim tile pipeline + tile lib nodes + `_isa_codegen.make_isa_expansions`.

**Deltas (all additive):**

| # | file | change |
|---|---|---|
| D1 | `dace/runtime/include/dace/tile_ops/cuda_warp.h` | NEW backend header (§4) |
| D2 | `libraries/tileops/_dispatch.py` | one row `"CUDA_WARP": "cuda_warp"`; add to the schedule-gated device-ISA set beside CUDA/CUTILE |
| D3 | `libraries/tileops/environments/` | one env class (headers = the new .h), mirroring the existing per-ISA envs |
| D4 | `experimental_cuda.py` `KernelSpec` | when the ThreadBlock map is a spawn anchor, block_dims = `num_warps * warpSize` |
| D5 | `experimental_cuda.py` allocation | warp-scope Register array of tile shape -> emit `T name[R]` (§3). One branch in the array-allocation path |
| D6 | `passes/vectorization/vectorize_gpu_warp.py` | NEW thin orchestrator + the one WarpSchedule step (§5) |

No new StorageType, no new node type, no ScheduleType addition, no change to memlet propagation,
validation, or the thread-level generators.

## 7. Verification (no local GPU)

- Expansion string tests per op (mirror `test_cpu_tile_reduce_lowering.py`).
- **Syntax-only device compile gate**: extend `test_tile_ops_isa_syntax.py` with a
  `cuda_warp` case via `clang++ -x cuda --cuda-device-only -nocudalib -fsyntax-only` (works without
  a GPU, same all-ops driver; `-D__CUDACC__` path). hip via `-x hip` if clang supports it on the box.
- E2E numeric: daint GPU allocation (fp64, tsvc/npbench subset + cloudsc snippets), compare vs the
  unvectorized reference — the standing corpus convention.
- Invariant tests: refusal of scalar_postamble, divergent-branch bodies, W not multiple of warpSize.

## 8. Later (explicitly out of scope now)

- Cross-warp (block-level) tile ops via smem staging + `__syncthreads` (block reduce).
- Tensor-core tiles (wmma fragments) — same schedule tree, different fragment layout; the
  lane-major contract is the thing to generalize, not the IR.
- AMD wave64 tuning beyond the warpSize constant.
