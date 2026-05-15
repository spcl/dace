# GPU Specialization Pipeline

`GPUCodegenPreprocessPipeline` transforms a DaCe SDFG with GPU storage
annotations into a form ready for `ExperimentalCUDACodeGen`. It runs as
part of the codegen target's `preprocess` step.

## Pipeline order

```
GPUCodegenPreprocessPipeline:
  1. InferDefaultSchedulesAndStorages
  2. PromoteGPUScalarsToArrays
  3. AssignmentAndCopyKernelToMemsetAndMemcpy
  4. InsertExplicitGPUGlobalMemoryCopies
  5. ExpandLibraryNodes
  6. NaiveGPUStreamScheduler
  7. LiftSharedOutOfNestedSDFG
  8. AddThreadBlockMaps
  9. ReinferConnectorTypes
```

Each step depends on the invariants its predecessors establish. Stream
scheduling sees the post-expansion SDFG (real kernels + runtime
tasklets, not opaque libnodes). The orphan-pass rewrite of trivial
in-kernel copies/zero-fills (#3) must run before any pass that adds
dynamic `__stream` connectors, because it would otherwise propagate
them onto the libnodes it creates and clash with the stream scheduler.

## What each pass does and why

### 1. `InferDefaultSchedulesAndStorages`
Resolves every `ScheduleType.Default` / `StorageType.Default` to a
concrete value based on enclosing scopes. The rest of the pipeline
assumes every descriptor and map has a determined storage/schedule.

### 2. `PromoteGPUScalarsToArrays`
Widens `Scalar` descriptors that can't live on the GPU as Scalars
into length-1 `Array` descriptors (e.g. a kernel-written Scalar
becomes `Array((1,), GPU_Global)`). After this pass every "GPU
scalar" is an `Array((1,), …)`.

### 3. `AssignmentAndCopyKernelToMemsetAndMemcpy`
Recognises trivial in-kernel patterns — `B[i, j] = A[i, j]` and
`B[i, j] = 0` — and lifts them to `CopyLibraryNode` /
`MemsetLibraryNode`. The libnodes lower to `cudaMemcpyAsync` /
`cudaMemsetAsync` rather than launching a no-op kernel. Carries a
clash guard: skips when the surrounding SDFG has arrays named like
the libnode's connectors (avoids re-triggering the libnode-connector
rename clash inside expansion-wrapper SDFGs).

### 4. `InsertExplicitGPUGlobalMemoryCopies`
Hoists transient GPU_Global arrays out of kernel scopes (the codegen
has no in-kernel allocator path) via `MoveArrayOutOfKernel`. Demotes
small literal-shape kernel-internal transients to per-thread
`Register` storage instead of lifting, gated on three conditions: no
external consumers, no incoming WCR memlet (atomic accumulator), and
`prod(shape)` ≤ `register_demotion_max_elements` (default 64).
Finally lifts every implicit `AccessNode → AccessNode` (and
map-staging) edge into an explicit `CopyLibraryNode`.

Fails loudly if any `GPU_Global → GPU_Global` direct copy still sits
inside a kernel scope after the hoist — those need manual
restructuring.

### 5. `ExpandLibraryNodes`
Recursively expands every remaining `LibraryNode`. Re-runs
`set_default_schedule_and_storage_types` after expansion so NSDFGs
spawned by the expansion don't ship with `ScheduleType.Default` Maps
inside (the codegen dispatcher rejects those).

### 6. `NaiveGPUStreamScheduler`
Computes a WCC partition over GPU-relevant nodes, assigns each
component a stream id, allocates the `gpu_streams` transient on the
top SDFG, wires `__stream` connectors on every GPU consumer
(kernels, libnodes, runtime tasklets), and emits
`cudaStreamSynchronize` tasklets at cross-stream / host boundaries.
Runs on the post-expansion SDFG.

The stream-scheduling strategy is included directly (not via the
single-pass `GPUStreamPipeline` wrapper). Reason: `Pipeline` is
decorated as a `@dataclass` and is therefore unhashable, so it can't
be a child of another `Pipeline`. Strategies extend `Pass` and are
hashable.

### 7. `LiftSharedOutOfNestedSDFG`
Promotes every `transient GPU_Shared` array that lives inside a
NestedSDFG up into the SDFG that owns the enclosing `GPU_Device`
map. The lifted descriptor lives at the kernel scope, accessed from
inside the NestedSDFG via a connector. This makes the framecode
allocation walker emit `__shared__ T name[N]` directly into the
kernel function body (the only place `__shared__` is valid) —
without it, the walker mis-routes the declaration to a stream that
never reaches any kernel.

### 8. `AddThreadBlockMaps`
Tiles every `GPU_Device` map that doesn't already have an inner
`GPU_ThreadBlock` map. Computes the `(grid, block)` dimensions for
codegen and stashes them in `pipeline_results['AddThreadBlockMaps']`
under `kernel_dimensions_map` / `tb_inserted_kernels`. The codegen
target reads them back. Runs late so the kernel-internal transient
hoist (#4) sees user-authored kernel shapes — tiling earlier would
introduce inner-map ranges like `Min(N - 1, b_i + 31) - b_i + 1`
whose `b_i` outer-loop symbol then leaks into host-side `cudaMalloc`
size expressions for any transient lifted out of the kernel.

### 9. `ReinferConnectorTypes`
Re-derives NestedSDFG connector types from their (now-mutated) inner
descriptors. Earlier passes — especially #2 widening Scalar →
length-1 Array — invalidate connector type annotations that were
correct at construction time. Without this fixup the codegen emits
the wrong pointer-vs-value signatures.

## Idempotency

`GPUStreamPipeline` checks `is_gpu_lowering_applied(sdfg)` (i.e.
`gpu_streams ∈ sdfg.arrays`) and rejects re-application. The WCC
partition is graph-shape dependent; re-running the scheduler on an
already-wired SDFG would corrupt the chains. Nested SDFGs share the
root's decisions, so calling the pipeline on a non-root SDFG raises.

## Reserved names

* `gpu_streams` — the stream array on the top SDFG. Allocated by the
  stream-scheduling strategy.
* `__stream_<id>` — per-stream connector on a fused sync tasklet,
  one in-edge per stream id touched in the state.
* `__stream` — single-stream connector on `CopyLibraryNode`,
  `MemsetLibraryNode`, kernel `MapEntry`, and pre-expanded runtime
  tasklets.

## Host vs. device-level rule

A NestedSDFG inside a `Sequential` / CPU map runs on the host and gets
streams threaded in. A NestedSDFG inside a `GPU_Device` map runs as
device code (`__device__` / `DACE_DFI`) — `cudaMemcpyAsync` /
`cudaLaunchKernel` etc. are host-only runtime entry points and cannot
be issued from a `__device__` function, so streams are never threaded
into kernel-nested NestedSDFGs.

The check (`helpers/gpu_helpers.py:is_inside_gpu_device_kernel`)
walks `parent_nsdfg_node` / `parent_sdfg` directly via
`innermost_enclosing_map`. It does not walk data-flow predecessors —
a downstream consumer of a kernel's output is at sibling scope, not
"inside" it.

## Failure modes the pipeline catches

`InsertExplicitGPUGlobalMemoryCopies` raises if it finds a transient
`GPU_Global → GPU_Global` copy whose endpoints sit inside a kernel
scope after its hoist phase. Such patterns mean a transient could
not be lifted (typically because of cross-kernel reuse) — the error
names the offenders so the caller can diagnose which transients need
manual restructuring.

## Adding a new pass

1. Decide where it goes in the pipeline order. Each pass establishes
   invariants the next one assumes; insert with care.

2. If the new pass touches connector types, dynamic inputs, or
   schedule, decide whether it must run before #5 (post-expansion
   passes see a different graph) and #6 (after stream scheduling,
   adding any `__stream` connector is fragile).

3. If the pass adds a reserved name, document it in the "Reserved
   names" section above.

4. If the pass needs scope membership, use
   `helpers/gpu_helpers.py` (`enclosing_map_chain`,
   `innermost_enclosing_map`, `is_inside_gpu_device_kernel`).
