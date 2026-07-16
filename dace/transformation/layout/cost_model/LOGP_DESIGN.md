# LogP cost analysis of a loop nest

## Purpose

Give an SDFG loop nest (a `Map` scope, or a lowered loop) a memory cost that is sensitive to data
layout, so a layout optimizer can rank `Permute` / `Pad` / `Block` / `Shuffle` choices by predicted
time instead of by trial compilation. The model is LogP/LogGP, applied to memory rather than a
network: each contiguous block request is a message, and the nest's cost is the messages it issues,
priced by measured hardware latency and bandwidth.

The model targets GPU global memory first. On the GPU, a warp's coalesced access maps cleanly onto
"one contiguous request is one message, a scattered request is many"; on the CPU, hardware
prefetching blurs that boundary, so the same numbers are an approximation there. The parametrization
(L, G) is measured on both, but the message abstraction is exact for GPU coalescing and heuristic for
CPU cache behaviour.

## From classic LogP to a loop nest

Classic LogP models a parallel machine as processors that exchange fixed-size messages over a
network. The mapping onto a loop nest reading memory is direct — a memory request is a message to
"another rank", where a rank is a memory partition rather than a processor:

| LogP / LogGP | network meaning | this analysis (memory) |
|---|---|---|
| message | a fixed-size packet | a block request — one cache line (CPU) or coalesced sector (GPU) |
| "another rank" | a remote processor's memory | global memory; a local (register/shared) access is not a message and is free |
| `L` | wire latency of a message | round-trip latency of one block (request + reply), measured by pointer chase |
| `o` | CPU send/receive occupancy | local issue cost — 0 for now (deferred with the SRAM task) |
| `g` | minimum gap between messages | minimum gap between a core's requests (1 / its request rate) |
| `G` (LogGP) | per-byte cost of a long message | per-byte channel gap `1/BW`, measured by the triad |
| `P` | number of processors | the parallel iterations / cores / warps the nest exposes |

The loop nest is the "program" each rank runs: every iteration issues the block requests its memlets
imply, and the nest's cost is the sum of those messages priced by `L` and `G`. A layout transform
does not change what the nest computes or how many bytes it needs — it changes how those bytes pack
into blocks, i.e. how many messages the same computation sends. That is the entire lever, and it is
why a message-based model is the right abstraction for layout.

The two LogP capacity ideas carry over unchanged. The gap `g` and the parameter `P` say how many
messages can be outstanding at once; the ratio `L/g` (LogP's `⌈L/g⌉`) is how many a single core keeps
in flight. Whether that is enough to hide the latency — the classic overlap question — is exactly the
latency-bound / bandwidth-bound test below.

## The framing

- **Local memory is free.** Registers and GPU shared memory are "my memory"; their access cost is a
  later, separate task (SRAM). For now a local access contributes nothing.
- **A global access is a message.** Reaching global memory is a request plus its reply; the whole
  round trip is the latency `L`. The bytes that come back cross the channels at the per-byte gap `G`.
- **A contiguous request is one message; an unstructured (scattered) request is several.** The number
  of messages an access issues per iteration is the number of distinct memory blocks it touches.

## The parameters (LogGP, for one memory level)

| symbol | meaning | unit | source |
|---|---|---|---|
| `L` | unloaded round-trip latency of one block | s | pointer chase (`membench.c`) |
| `o` | per-message overhead | s | 0 for now (it is the local-issue cost, deferred with SRAM) |
| `g` | minimum interval between a core's requests | s | concurrency sweep |
| `G` | per-byte gap = `1 / BW_saturated` | s/byte | STREAM triad, **all cores** |
| `line_bytes` | transfer granularity | bytes | 64 (CPU line) / 32 (GPU sector) |

`G` is `1 / BW_saturated`, not `1 / BW_core`: bandwidth is a property of the memory channels, and a
single core cannot saturate them (its outstanding-miss budget runs out first). The per-core bandwidth
is kept only as a diagnostic. See `loggp.LogGP` and `README`-level detail in `loggp.py`.

`L` and `G` come from `membench.py` (a small C pointer-chase and triad, self-timed); the theoretical
peak `BW` that `G` is judged against comes from `hardware.py` (parsed from `dmidecode` for DRAM,
computed from the CUDA device properties for the GPU). None of these run unless the machine is quiet
— `membench.check_environment` refuses on a loaded box, because a plausible-but-wrong number is worse
for a cost model than no number.

## How the analysis reads a loop nest

`logp_analysis.analyze_loop_nest(state, map_entry, p, block_bytes, local_arrays)` does five steps.

```
  SDFG map scope
        |
  (1) extract the nest         -> loop_ranges: [{param: (begin, end, step)}], outer-to-inner
        |
  (2) collect accesses         -> {array: unioned subset}          (get_access_subsets)
        |
  (3) classify each array      -> local (free) | global (message)  (storage type / override)
        |
  (4) per global array:
        messages_per_iter       = average new blocks touched        (blocks_touched)
        bytes_moved_per_iter    = messages_per_iter * block_bytes    (whole blocks move)
        |
  (5) combine into per-iteration latency + bandwidth, and totals
```

**(1) Extract the nest.** Walk the map scope subtree, order the maps by nesting depth, and read each
map's `params` and `range`. A collapsed multi-dimensional map is one level with several params; a
chain of nested maps is several levels. The result is `loop_ranges`, outer-to-inner.

**(2) Collect accesses.** `get_access_subsets` unions, per array, every memlet subset the innermost
tasklets read or write. For a pointwise `A[i, j]` the subset is `[i:i, j:j]` — the per-iteration
access, in terms of the (symbolic) loop parameters.

**(3) Classify local vs global.** An array in `StorageType.Register` or `StorageType.GPU_Shared`, or
named in `local_arrays`, is local and free. Everything else (global, heap, default) is a message
source.

**(4) Messages and bytes per iteration.** For a global array, the number of messages issued per
iteration is the average number of *new* blocks it touches — `blocks_touched.average_blocks_touched`,
computed for that array's subset at that array's block granularity (`block_bytes / dtype_bytes`
elements). This is the term the layout moves: a contiguous access touches `1/8` of a new 64-byte line
per fp64 element (8 elements share a line), a strided access touches a whole new block every step.
Because memory moves whole blocks, the bytes that actually cross the channels are
`messages_per_iter * block_bytes`, not the requested bytes.

**(5) Combine.** Per iteration the nest pays, summed over its global arrays:

```
  latency_per_iter   = sum( messages_per_iter    * L )     # one round trip per new block message
  bandwidth_per_iter = sum( bytes_moved_per_iter * G )     # whole blocks at the channel gap
```

Both are returned symbolically (they depend on the loop bounds `N`, ...). The caller combines them by
regime:

- `total_time_serialized = (latency_per_iter + bandwidth_per_iter) * total_iters` — every message pays
  its full latency in series. This is the latency-exposed **upper bound**.
- `total_time_overlapped = total_bytes / achievable_rate(p)` — the realistic time once requests
  overlap. `achievable_rate = min(1/G, concurrency * line / L)`: bandwidth-bound once the channels
  saturate, latency-bound while few requests are outstanding. Charging `L` per message and summing
  (the serialized form) is ~40x too slow on real hardware, because outstanding requests overlap their
  latency — that gap is exactly why both `L` and `G` are measured.

## Why this captures layout performance

The layout-sensitive term is `messages_per_iter` (the block count). For one kernel on one device,
`total_iters`, `line_bytes` and `achievable_rate` are the same for every layout, so the predicted time
is **proportional to the block count**. Two consequences:

1. A layout change is visible precisely because it changes the number of block messages. Transposing a
   read operand of a 3-array elementwise add takes its per-iteration blocks from `1/8` to `1`, so
   predicted `latency/iter` goes 35.7 -> 118.8 ns and the overlapped total 25.2 -> 83.9 ms.
2. The **ranking of layouts is invariant to the regime**: because both the latency-bound and the
   bandwidth-bound cost scale the block count by the same per-block constant, the same layout wins
   whether the kernel is latency- or bandwidth-bound. A latency model can rank layouts because a
   layout change is a change in message count. (`loggp.memory_time` and the ranking-invariance test
   pin this.)

## Latency-bound or bandwidth-bound?

A nest is **bandwidth-bound** when it keeps enough block requests in flight to fill the channels, and
**latency-bound** when it cannot and each access waits out the full `L`. The threshold is the
**bandwidth-delay product** (Little's Law):

```
  BDP = L / (line_bytes * G) = L * BW_saturated / line_bytes        # requests needed in flight
```

Compare it to the concurrency the nest actually exposes:

```
  regime = "bandwidth" if exposed_concurrency >= BDP else "latency"
```

which is the same as asking which term of `achievable_rate = min(1/G, concurrency*line/L)` binds.

For the example DRAM (L=95 ns, 100 GB/s, 64 B line) the BDP is ~148 requests. Consequences the model
then makes concrete:

- A **single CPU core** tracks ~24 outstanding misses — below 148, so one core is **latency-bound** and
  cannot saturate DRAM. This is why the model assumes many cores, and why `G` is measured with all of
  them.
- A **GPU** keeps thousands of accesses in flight — far above 148, so a parallel kernel is
  **bandwidth-bound**.
- A **dependency-serialized loop** (a pointer chase, a scan) has one request outstanding at a time —
  **latency-bound** regardless of the device.

`analyze_loop_nest` estimates the exposed concurrency from the schedule: any parallel map saturates
(the multi-core / many-warp assumption, returned as `inf` → bandwidth-bound), a fully `Sequential`
nest is serialized to 1 → latency-bound. A caller who knows the true MLP passes `concurrency=`
explicitly — `p.concurrency` for the single-core view, `1` for a dependency chain, a measured device
value otherwise. `LoopNestLogP.regime()` reports the verdict, `bandwidth_delay_product()` the
threshold, and `total_time()` returns the matching total (overlapped when bandwidth-bound, serialized
when latency-bound). The layout ranking is the same either way; the regime only sets the absolute
time.

## API

```python
cost = analyze_loop_nest(state, map_entry, p, block_bytes=32, local_arrays=frozenset())

cost.total_iters                 # symbolic iteration count of the nest
cost.arrays[name]                # ArrayLogP: is_local, messages_per_iter, bytes_moved_per_iter
cost.latency_per_iter()          # sum messages * L   (seconds, symbolic)
cost.bandwidth_per_iter()        # sum bytes   * G
cost.time_per_iter()             # latency + bandwidth (serialized)
cost.total_time_serialized()     # per-iter * iters   (upper bound)
cost.total_time_overlapped()     # total_bytes / achievable_rate  (realistic, saturated)
```

`p` is a `loggp.LogGP` from measurement (or example values for what-if analysis). `block_bytes` picks
the granularity — 64 for a CPU line, 32 for a GPU sector — so one function serves both devices.

## Module map

| file | role |
|---|---|
| `hardware.py` | theoretical peak BW (DRAM from a `dmidecode` dump, GPU from CUDA props) — the denominator |
| `membench.c` / `membench.py` | pointer-chase (L) and triad (G) microbenchmarks; environment gate |
| `loggp.py` | `LogGP` parameters, the `T(n)=α+βn` fit, `achievable_rate`, `memory_time`, `validate` |
| `blocks_touched.py` | average new blocks per iteration — the Δ term, an INPUT to the analysis |
| `access_subsets.py` | per-array access subsets from the map scope |
| `logp_analysis.py` | **this analysis**: the loop-nest LogP cost |

## Accuracy and limitations

- **`blocks_touched` is asymptotically exact, not exact.** It uses a continuous fraction
  `(extent-1)*stride/block` where the truth applies `ceil` at block granularity. It converges as
  extents grow but overcounts small tiles (a contiguous 4x4 fp64 tile is 16 contiguous elements = 2
  blocks; the per-dimension fractions report ~4, because they cannot see the inner runs combine into
  one contiguous span). The **ranking is preserved**, so layout selection is sound; absolute block
  counts for sub-block tiles are not. A brute-force traversal oracle is checked in the tests, and the
  exact streaming/coalescing count over the innermost contiguous run is the next refinement.
- **Latency under load is not modelled.** `L` is the *unloaded* round trip; at saturation it inflates
  2-4x (Little's law). A single-`L` model therefore under-predicts a bandwidth-saturated nest. Adding
  a loaded-latency curve is the largest known follow-up.
- **`o = 0`.** It is the local-issue occupancy, deferred with the SRAM cost. It is ~0.25 ns against
  `L ≈ 95 ns`, below the measurement's own noise band.
- **One `achievable_rate` for all layouts.** A badly scattered access that cannot sustain concurrency
  drops to the latency-bound branch and should be penalised a second time; the layout-dependent rate
  is a noted refinement.
- **Read/write symmetry.** Each array's accesses are counted once from the unioned subset; a
  read-modify-write of the same array is not double-counted, and a partial-line write's
  read-for-ownership traffic (the 1.33-1.5x STREAM undercount) is a hardware-accounting factor folded
  into `G`, not modelled per access.
- **Split-index accesses are not scored.** `SplitDimensions` (Block) rewrites an access into
  split-index form, `A[i, int_floor(j, 8), j % 8]`; the `int_floor`/`%` subscripts are not affine
  range begins, so `blocks_touched` cannot reduce them and the analysis raises on such an SDFG. A
  blocked layout is scored today by giving the analysis the blocked STRIDES on the original access
  (the physical layout the transform materialises), not by running it on the post-`SplitDimensions`
  graph. `PermuteDimensions` and `PadDimensions`, which keep affine accesses, are scored directly.
- **Indirect (gather/scatter) accesses are not yet scored.** An `A[idx[i]]` access is data-dependent,
  not affine; the "unstructured access is many messages" case needs the indirection-aware count and
  is future work.

## Worked example

A three-array elementwise add `C[i,j] = A[i,j] + B[i,j]`, loops `(i, j)`, block 64 B, fp64, example
DRAM `L = 95 ns`, `G = 1/100 GB/s`:

| A layout | A blocks/iter | latency/iter | total (overlapped) |
|---|---|---|---|
| row-major `(N,1)` | 1/8 | 35.7 ns | 25.2 ms |
| transposed `(1,N)` | 1 | 118.8 ns | 83.9 ms |

B and C stay contiguous in both; only A's layout changes, and the analysis attributes the whole 3.3x
difference to A's messages — which is the layout signal a `Permute` search consumes.

## Scoring an actual transform

The same signal comes out when a real transform runs, not just when strides are set by hand. Take
`C[i,j] = A[i,j] + B[j,i]` — `B` is read transposed, so its inner access is strided and it costs ~1
block per iteration. `PermuteDimensions(permute_map={"B": [1, 0]}, add_permute_maps=True)` relays `B`
into the transposed physical order; the compute map then reads a contiguous `permuted_B`, and the
analysis reports its message count dropping from ~1.0 to ~0.125 — the 8x the relayout was for.
`PadDimensions` grows a dimension but leaves the innermost stride 1, so a contiguous access stays
~0.125 (pad buys alignment, not a lower message count) — the model does not credit a pad that moved
nothing relevant. Both are pinned in `cost_model_transformations_test.py`.
