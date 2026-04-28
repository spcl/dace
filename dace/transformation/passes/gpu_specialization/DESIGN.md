# GPU Specialization Pipeline

The `gpu_specialization` pipeline transforms a DaCe SDFG with GPU storage
annotations into a form ready for `ExperimentalCUDACodeGen`. It runs as
part of the codegen's `preprocess` step, after `InferDefaultSchedulesAndStorages`
and `PromoteGPUScalarsToArrays` have settled storage / schedule defaults.

## Pipeline order

```
GPUSpecializationPipeline:
  1. InsertExplicitGPUGlobalMemoryCopies
  2. NaiveGPUStreamScheduler
  3. InsertGPUStreams
  4. ConnectGPUStreamsToNodes
  5. InsertGPUStreamSyncTasklets

(then expand_library_nodes(recursive=True) and:)
  6. ReconnectWithinExpandedSDFGs
  7. LiftSharedOutOfNestedSDFG
```

## What each pass does and why

### 1. `InsertExplicitGPUGlobalMemoryCopies`
Lifts every implicit `AccessNode → AccessNode` (and map-staging) edge that
touches GPU memory into an explicit `CopyLibraryNode`. The codegen
otherwise infers copies from edge topology, which mixes data-movement
semantics into edge endpoints; making copies first-class library nodes
keeps the rest of the pipeline simple — every copy has a place to attach
a stream, and codegen has one expansion path per copy class. Also fails
loudly if any `GPU_Global → GPU_Global` copy is found inside a kernel
scope (those need `MoveTransientOutOfKernel` first).

### 2. `NaiveGPUStreamScheduler`
Analyzes the SDFG and assigns a stream id to every node that needs one
(GPU kernel `MapEntry`, copy/memset library nodes targeting GPU memory).
Pure analysis: returns a `Dict[Node, int]` for downstream passes. Naive
because it doesn't try to overlap independent components — that's a
follow-up optimization.

### 3. `InsertGPUStreams`
Materializes the `gpu_streams` transient on the top SDFG and threads it
through every nested SDFG that needs it, adding an in-connector and
data-flow path to each NestedSDFG node. Skips NestedSDFGs that execute
as device code (transitively inside a `GPU_Device` map) — those run
inside a kernel and cannot host-issue stream-bound calls.

`gpu_streams`'s presence on the top SDFG is also the **idempotency
signal** for the whole pipeline (`is_gpu_lowering_applied`).

### 4. `ConnectGPUStreamsToNodes`
For each scheduled node and each stream id, builds the per-stream chain
of `gpu_streams[i]` AccessNodes that wires kernels and library nodes
together. Same device-level skip as `InsertGPUStreams`. Sequential map
chains use `IN_stream` / `OUT_stream` pass-through connectors so stream
state crosses scope boundaries cleanly.

### 5. `InsertGPUStreamSyncTasklets`
Identifies cross-stream and cross-device synchronization points (GPU →
host, host → GPU, kernel → memcpy on a different stream, etc.) and
inserts `cudaStreamSynchronize` tasklets there. The pass picks the
narrowest sync point that preserves correctness, so independent streams
keep their concurrency.

### 6. `ReconnectWithinExpandedSDFGs`
Runs after `expand_library_nodes(recursive=True)`. Library-node
expansions (`CopyLibraryNode`, `MemsetLibraryNode`, cuBLAS, etc.) often
spawn a NestedSDFG that inherits a single `stream` connector. This pass
walks the inherited body and wires every internal GPU consumer to that
one bound stream — no fresh `gpu_streams` array threaded into already-bound
bodies.

### 7. `LiftSharedOutOfNestedSDFG`
Promotes every `transient GPU_Shared` array that lives inside a
NestedSDFG up into the SDFG that owns the enclosing `GPU_Device` map.
The lifted descriptor lives at the kernel scope, accessed from inside
the NestedSDFG via a connector. This makes the framecode allocation
walker emit `__shared__ T name[N]` directly into the kernel function
body (the only place `__shared__` is valid) — without it, the walker
mis-routes the declaration to a stream that never reaches any kernel.

## Idempotency

`GPUSpecializationPipeline` checks `is_gpu_lowering_applied(sdfg)` (i.e.
`gpu_streams` ∈ `sdfg.arrays`) and short-circuits on a re-run. The
preprocess always runs end-to-end so codegen-side state (kernel
dimensions, stream manager, per-kernel arglists) is set up regardless,
but the SDFG-modifying passes don't double-apply — re-application would
double-add stream connectors and corrupt the per-stream chains, causing
runtime memory faults.

## Names reserved by this pipeline

* `gpu_streams` — the stream array. **Enforced**: `SDFG.add_datadesc`
  rejects user-driven additions with this name. The pipeline itself adds
  it via `add_datadesc(..., _internal_pipeline_use=True)`.
* `__stream_<id>` — per-kernel stream connector prefix on `MapEntry` nodes
  (added by `ConnectGPUStreamsToNodes`).
* `stream` — single-stream connector on `CopyLibraryNode` /
  `MemsetLibraryNode`.

The canonical list lives in
`helpers/gpu_helpers.py:RESERVED_GPU_PIPELINE_NAMES` and is mirrored
inline in `dace/sdfg/sdfg.py:SDFG._RESERVED_PIPELINE_NAMES` to avoid a
circular import.

## Host vs. device-level rule

A NestedSDFG inside a `Sequential` / CPU map runs on the host and gets
streams threaded in. A NestedSDFG inside a `GPU_Device` map runs as
device code (`__device__` / `DACE_DFI`) — `cudaMemcpyAsync` /
`cudaLaunchKernel` etc. are host-only runtime entry points and cannot
be issued from a `__device__` function, so streams are never threaded
into kernel-nested NestedSDFGs.

The check (`helpers/gpu_helpers.py:is_inside_gpu_device_kernel`) walks
`parent_nsdfg_node` / `parent_sdfg` directly via
`innermost_enclosing_map`. It does **not** walk data-flow predecessors —
a downstream consumer of a kernel's output is at sibling scope, not
"inside" it.

## Scope-membership lookups

The shared helpers in `gpu_helpers.py` invalidate `state.scope_dict()`'s
cache before every traversal. Reason: pipeline passes that add nodes
(stream AccessNodes, sync tasklets) can leave the cache stale relative
to the new topology. The walkers downstream that key on scope
membership (allocation routing in framecode, the stream-skip rule
above) need fresh data.

## Idempotency

`GPUSpecializationPipeline.apply_pass` checks
`is_gpu_lowering_applied(sdfg)` and short-circuits on a re-run. The
signal is the presence of `gpu_streams` on the top SDFG — created by
`InsertGPUStreams` (the only pass that introduces it) and stable across
the rest of the pipeline.

`ExperimentalCUDACodeGen.preprocess` always runs end-to-end because
codegen-side state (kernel dimensions, stream manager, per-kernel
arglists, statestruct entry) must be set up regardless of whether the
SDFG was already lowered. The SDFG-modifying part — the
`GPUSpecializationPipeline` invocation — is the part guarded by the
idempotency check.

When this matters: callers that pre-apply the pipeline before
`compile()` (e.g. test scaffolds that compose their own pipeline). A
naive re-application would double-add `gpu_streams`, double-thread
NestedSDFG connectors, and corrupt the per-stream chains; the runtime
manifestation is a segfault during kernel launch sequencing.

## Failure modes the pipeline catches

`InsertExplicitGPUGlobalMemoryCopies` raises if it finds a
`GPU_Global → GPU_Global` direct copy whose endpoints sit inside a
kernel scope. Such patterns mean a transient leaked into the kernel
body and need `MoveTransientOutOfKernel` (in
`dace/transformation/passes/move_array_out_of_kernel.py`) to hoist
them out before this pipeline runs.

The error message names the offenders so the caller can diagnose which
transients need hoisting.

## Adding a new pass

1. Decide whether it touches the SDFG state. If yes, place it in the
   pipeline order above (the existing passes assume the invariants
   established by their predecessors); if no, it can run independently.

2. Make it idempotent. The pipeline-level guard handles re-runs of the
   *whole* pipeline, but a pass added later in `preprocess` (after
   `expand_library_nodes` / `ReconnectWithinExpandedSDFGs`) needs its
   own re-application story — typically by checking a structural
   invariant the pass itself establishes (e.g.
   `LiftSharedOutOfNestedSDFG` checks whether the inner SDFG's Shared
   transients are still `transient=True`).

3. If the pass needs scope membership, use
   `helpers/gpu_helpers.py`'s `enclosing_map_chain` /
   `innermost_enclosing_map` / `is_inside_gpu_device_kernel`. They
   invalidate `scope_dict` first.

4. If the pass introduces a reserved name, add it to
   `RESERVED_GPU_PIPELINE_NAMES` and the inline mirror in
   `SDFG._RESERVED_PIPELINE_NAMES`, and use `_internal_pipeline_use=True`
   when calling `add_datadesc`.
