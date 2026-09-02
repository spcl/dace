# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-canonicalization target finalization for performance.

:func:`~dace.transformation.passes.canonicalize.pipeline.canonicalize` leaves
an SDFG in a clean, vectorizer-friendly form: ``Reduce`` / ``MatMul`` library
nodes are kept un-expanded (``implementation=None``) so later passes see one
shape per computation. That form codegens to the naive library expansion,
which is materially slower than the tiled/fast implementations ``auto_optimize``
selects.

This module supplies the optimization tail to run *after* canonicalization when
the goal is raw runtime (not vectorization or equivalence checking): it picks
the fast library implementation per node and moves small/independent transients
to registers and persistent storage. The library nodes themselves stay
un-expanded -- ``compile()`` expands each one exactly once at codegen, using the
implementation selected here -- so one shape per computation survives all the way
to the backend. It mirrors ``auto_optimize``'s library-and-storage finalization
(everything fusion-related is already done by the canonicalize pipeline), so
``canonicalize(s); finalize_for_target(s)`` is the perf-path counterpart to
``auto_optimize(s)``.
"""
import os

from dace import SDFG, dtypes, symbolic
from dace.config import Config
from dace.sdfg import infer_types, nodes
from dace.sdfg.state import SDFGState
from dace.libraries.blas.environments import openblas
from dace.transformation.auto.auto_optimize import (apply_cpu_library_parallelism, apply_gpu_storage, find_fast_library,
                                                    libnode_is_sequential, make_transients_persistent,
                                                    move_small_arrays_to_stack, set_fast_implementations)
from dace.transformation.passes.canonicalize.hoist_loop_range_calls import HoistLoopRangeCalls
from dace.transformation.passes.canonicalize.pipeline import run_structural_cleanup
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import HoistParallelRegion
from dace.transformation.passes.cpu_specialization.pipeline import cpu_specialize
from dace.transformation.passes.gpu_block_size_selection import select_gpu_device_block_size
from dace.transformation.passes.gpu_specialization.promote_warp_tiles import PromoteWarpTiles
from dace.transformation.passes.gpu_specialization.sequentialize_nested_device_scopes import (
    SequentializeNestedDeviceScopes)
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.libraries.standard.nodes.scan import Scan
from dace.libraries.standard.nodes.symmetrize import Symmetrize
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation import helpers as xfh

#: Map the canonicalize target string to the codegen device type.
_TARGET_DEVICE = {'cpu': dtypes.DeviceType.CPU, 'gpu': dtypes.DeviceType.GPU}

#: Per-dimension matmul extent at or below which canonicalization picks an inlined expansion over
#: a BLAS call. MEASURED against OpenBLAS at 64/128/256 cubed: OpenBLAS wins at every one of them
#: (0.1/13/25 ms against 36/48/75 ms for plain ``'pure'``), because the plain nest emits one
#: ``reduce_atomic`` per multiply-add. So the override is worth taking only where the BLAS call
#: overhead really dominates, and only in the ``'rowwise'`` form, which carries no atomic.
_SMALL_MATMUL_DIM = 32


def _all_matmul_extents_small(state, node, limit: int) -> bool:
    """True iff every operand/output extent of a matmul library ``node`` is a constant
    at most ``limit`` (the matmul is known-small). A symbolic extent -> not known-small."""
    saw = False
    for e in list(state.in_edges(node)) + list(state.out_edges(node)):
        if e.data is None or e.data.subset is None:
            continue
        for ext in e.data.subset.size():
            saw = True
            try:
                if int(ext) > limit:
                    return False
            except (TypeError, ValueError):
                return False  # symbolic extent -> unknown size, treat as not-small
    return saw


def canonicalize_fast_library_priority(device: dtypes.DeviceType):
    """Availability-aware fast-implementation priority for the canonicalize perf tail.

    Prefer OpenBLAS (BLAS + LAPACKE, i.e. LAPACK) over MKL -- MKL is blocklisted by the caller, per
    the directive to always use OpenBLAS/LAPACK, cuBLAS/CUB, or an OpenMP-based parallel expansion --
    and add the non-BLAS fast expansions that :func:`find_fast_library` omits so a lifted library node
    never falls to the serial ``pure`` loop:

    * CPU: ``OpenBLAS`` (if installed), ``HPTT`` (tensor transpose, if ``HPTT_ROOT`` is set),
      ``TTGT`` (tensor contraction via transpose+GEMM, no external dependency), ``OpenMP`` (``Reduce``),
      ``CPU`` (OpenMP-5 ``Scan``, radix ``IntegerSort``, ``ska_sort`` ``ScatterConflictCheck``).
    * GPU: taken straight from ``auto_optimize``, so the per-backend rows cannot drift -- cuBLAS /
      cuSolverDn / cuTENSOR on CUDA, rocBLAS / rocSOLVER / hipTENSOR on HIP, plus ``GPUAuto``,
      ``CUB`` and ``CUDA`` (``gpucub::DeviceScan`` / device sort / ``DeviceReduce::ArgMax`` / the
      bounding-box ``Symmetrize``) on both.

    Both lists are the SAME ORDER as ``auto_optimize``'s :func:`~dace.transformation.auto.
    auto_optimize.find_fast_library`, deliberately and for the reason stated there: the two pipelines
    are compared column against column, so a node that lowers to a tuned expansion under one and to
    the serial ``pure`` loop under the other measures the priority list rather than the pipeline.
    Naming the GPU row here a second time is what let it keep the CUDA vendors on a ROCm host, where
    every rocBLAS-lowered node fell back to ``pure``.

    Only impls whose environment is available on this host are listed, so forcing a pick never selects
    an unbuilt library. ``MatMul``/``Gemm`` still get the tiny-matmul ``rowwise`` override in
    :func:`canonicalize_set_fast_implementations`.
    """
    if device == dtypes.DeviceType.GPU:
        # Each node's own environment gates the actual build. ``pure`` is auto_optimize's terminal
        # fallback rather than a forced pick, so it is the one entry dropped here.
        return [impl for impl in find_fast_library(device) if impl != 'pure']
    prio = []
    if openblas.OpenBLAS.is_installed():
        prio.append('OpenBLAS')
    if 'HPTT_ROOT' in os.environ:
        prio.append('HPTT')
    prio.append('TTGT')
    prio += ['OpenMP', 'CPU']
    return prio


def libnode_is_device_code(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> bool:
    """``node`` sits inside a GPU kernel, so whatever it lowers to has to be device code.

    Reuses the walk :func:`~dace.transformation.auto.auto_optimize.libnode_is_sequential` relies on:
    it crosses every nested-SDFG boundary out to the root, so a node several nsdfg levels under a
    kernel is still seen to be in one. A node whose only enclosing scopes are host loops and
    ``Sequential`` taskloops is host code, and free to issue a device library call.
    """
    return any(
        isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES
        for scope in xfh.get_parent_map_and_loop_scopes(sdfg, node, state))


def canonicalize_set_fast_implementations(sdfg: SDFG, device: dtypes.DeviceType, small_dim: int = _SMALL_MATMUL_DIM):
    """Select library-node implementations for the canonicalize perf tail.

    Delegates to :func:`~dace.transformation.auto.auto_optimize.set_fast_implementations` with the
    canonicalize priority (:func:`canonicalize_fast_library_priority`) -- OpenBLAS/LAPACK, HPTT/TTGT,
    OpenMP, cuBLAS/cuSolverDn/cuTENSOR/CUB, never MKL -- so EVERY library node the pipeline introduces
    (Reduce, Scan, Transpose, TensorTranspose, Symm, Cholesky, Solve, ...) lowers to its fast expansion
    rather than the serial ``pure`` loop. Then OVERRIDES any GEMM/MatMul whose every dimension is a
    known constant at most ``small_dim`` to the inlined ``'rowwise'`` expansion. A tiny matmul's
    BLAS/cuBLAS call is pure overhead, and -- unlike an opaque library call -- a loop nest is
    fusible/vectorizable and keeps a loop of small matmuls sequential instead of issuing serialized
    library calls. Symbolic- or large-dimensioned matmuls keep the fast BLAS implementation, and so
    does a node with no ``'rowwise'`` expansion.
    """
    set_fast_implementations(sdfg, device, blocklist=['MKL'], find_fast_library_fn=canonicalize_fast_library_priority)
    gpu_priority = canonicalize_fast_library_priority(device) if device == dtypes.DeviceType.GPU else []
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        impls = type(node).implementations
        # A node re-entered inside an outer parallel/repeated scope (a parallel parent map, an
        # enclosing loop, or a ``Sequential`` nested SDFG) must NOT open its own (nested) parallel
        # region -- that fork/join per outer iteration is the "constant parallel reductions"
        # catastrophe. ``node.schedule`` is storage-derived and unreliable for this, so decide from
        # SCOPE (:func:`libnode_is_sequential`). Only a genuinely top-level node opens a parallel region.
        sequential = libnode_is_sequential(node, state, sdfg)
        # Pin the SCHEDULE of a re-entered node to ``Sequential`` too, not just its implementation:
        # a nested node left on the device parallel schedule (``CPU_Multicore`` / ``GPU_Device``,
        # storage-derived) would have its own expansion (e.g. the ``pure`` reduction map) scheduled
        # parallel and emit a nested parallel region -- exactly what ``assert_no_nested_parallel_maps``
        # forbids. Pinning it single-core keeps the whole expanded subtree serial.
        if sequential and node.schedule != dtypes.ScheduleType.Sequential:
            node.schedule = dtypes.ScheduleType.Sequential

        # The CPU parallel-lowering rule for Reduce / ArgReduce / Scan / Copy / Fill lives in
        # :func:`~dace.transformation.auto.auto_optimize.apply_cpu_library_parallelism`, shared with
        # ``set_fast_implementations`` so the canonicalize and auto_optimize paths cannot drift onto
        # different implementations of the same node. It has the last word on the types it governs
        # (hence the ``continue``): a small Reduce would otherwise be clobbered by the matmul-size
        # override below, which only means to catch GEMMs.
        if device == dtypes.DeviceType.CPU and apply_cpu_library_parallelism(node, state, sdfg):
            continue
        # GPU ``Scan``: a scan INSIDE a kernel must stay ``pure`` -- ``ExpandCUDA`` emits a HOST-side
        # ``gpucub::DeviceScan`` call, which device code cannot issue. Everything else is host code and
        # takes the device expansion, a host loop around it included: there the ``pure`` lowering is
        # a host loop indexing ``GPU_Global`` operands, which is not slow but wrong (tsvc s256, an
        # affine scan under the outer loop). Decide by SCOPE, as the generic rule below does; the
        # schedule says ``Sequential`` for a host loop and a kernel alike.
        if isinstance(node, Scan) and device == dtypes.DeviceType.GPU:
            node.implementation = ('pure' if libnode_is_device_code(node, state, sdfg) else
                                   ('CUDA' if 'CUDA' in impls else node.implementation))
            continue
        # ``Transpose`` / ``TensorTranspose`` deliberately get NO override here. Our tiled kernel is
        # registered as ``CUDA`` and the priority list already puts ``cuBLAS`` and ``cuTENSOR`` ahead
        # of it, which is the right order: measured on an RTX 4050 at 8192x8192 float64, cuBLAS
        # ``geam`` transposes at 169 GB/s against our kernel's 150 and the pure map's 140 -- geam is
        # essentially at copy bandwidth. Ours is the fallback that matters when the vendor library is
        # absent (this box has a cuTENSOR that does not link), where it beats ``pure``.
        # GPU ``Symmetrize``: the ``pure`` expansion walks the triangle as nested maps, whose inner
        # extent depends on the row. That cannot be a thread-block dimension, so ``pure`` pins the
        # column axis Sequential and leaves one thread per ROW. At host level the node becomes a
        # kernel of its own and the bounding-box expansion is the right lowering -- both axes
        # parallel, constant extents. Inside a kernel there is no launch to configure and the
        # triangular walk is the cheaper one, so keep ``pure`` there.
        if isinstance(node, Symmetrize) and device == dtypes.DeviceType.GPU:
            node.implementation = ('pure' if libnode_is_device_code(node, state, sdfg) else
                                   ('CUDA' if 'CUDA' in impls else node.implementation))
            continue
        # ``set_fast_implementations`` leaves every ``Sequential`` GPU node ``pure``, reading
        # Sequential as "inside a kernel", where only device code may be emitted. A host loop and a
        # taskloop body are Sequential too, and there the pure expansion is HOST code over
        # ``GPU_Global`` operands -- polybench trisolv's Dot, npbench stockham_fft's Gemm and
        # TensorTranspose. Host code is where a device library call belongs, so decide by SCOPE.
        # A node with no device expansion falls through and keeps what it had.
        if device == dtypes.DeviceType.GPU and node.schedule == dtypes.ScheduleType.Sequential \
                and not libnode_is_device_code(node, state, sdfg):
            fast = next((impl for impl in gpu_priority if impl in impls), None)
            if fast is not None:
                node.implementation = fast
                continue
            # No device expansion to fall back on (Merge, Fill, and the other pure-only nodes).
            # Leaving the schedule Sequential is not a slow-but-correct choice here: the pure
            # expansion is a host map over ``GPU_Global`` operands, which validation rejects
            # outright ("stored as StorageType.GPU_Global but accessed on host", npbench bfs).
            # ``set_default_schedule_and_storage_types`` reads a host loop as Sequential the same
            # way it reads a kernel body, and this is the half of that verdict that is wrong: a
            # host-level node on a device graph is a kernel launch per iteration. The expansions
            # honour ``node.schedule``, so setting it here is what makes the map a kernel.
            node.schedule = dtypes.ScheduleType.GPU_Device
        # That same rule pins a node to ``pure`` without checking that the node HAS one:
        # ``CopyLibraryNode`` does not, and expansion then raises ``Unknown implementation``
        # (polybench durbin). Its own default is the lowering that reads the schedule.
        if node.implementation == 'pure' and 'pure' not in impls:
            node.implementation = type(node).default_implementation

        if 'pure' not in impls:
            continue
        # Only the row-wise (ikj) expansion: a vectorizable row update with a sequential K
        # accumulation and no atomic. The plain 'pure' nest is deliberately NOT selectable here --
        # it reduces through one ``reduce_atomic`` per multiply-add and measured slower than
        # OpenBLAS at every size tried. A node without 'rowwise' (e.g. MatMul) keeps its BLAS call.
        if 'rowwise' in impls and _all_matmul_extents_small(state, node, small_dim):
            node.implementation = 'rowwise'


def finalize_transient_storage(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Finalize the storage of a canonicalized (or vectorized) SDFG's transients, in place.

    The single home for transient storage finalization -- a value-preserving perf tail that any
    downstream producer (canonicalize's ``finalize_for_target``, or a caller that has additionally
    vectorized the SDFG) runs to allocate temporaries well. Three steps, mirroring
    ``auto_optimize``'s storage tail:

    1. **Length-1 transient arrays -> scalars** (:class:`ConvertLengthOneArraysToScalars`, at its
       default -- transient only): a single internal value belongs in a scalar, not a heap array.
       Non-transient length-1 arrays (SDFG-external returns / opaque handles) are left as arrays.
    2. **Small constant-size scratch -> registers** (:func:`move_small_arrays_to_stack`).
    3. **Independent top-level transients -> ``Persistent`` lifetime**
       (:func:`make_transients_persistent`, ``toplevel_only=True``): a state-struct member
       allocated once in ``__dace_init`` / freed in ``__dace_exit`` instead of a per-call
       ``malloc``/``free``. ``toplevel_only`` + ``get_parent_map``'s walk up across every
       nested-SDFG boundary excludes any per-thread buffer inside a parallel map body, so it is
       never collapsed to one shared copy; on GPU it also resets non-atomic WCR edges.

    A persistent size-1 WCR accumulator would land as ``__state->x`` -- not a valid OpenMP
    ``reduction(op:var)`` lvalue, so the parallel reduction fails to compile. Revert each promoted
    size-1 transient to a scope-lifetime register (a stack scalar).

    :param sdfg: SDFG whose transient storage is finalized in place.
    :param device: codegen device type (selects GPU WCR-reset / storage rules).
    """
    # Post-canonicalization cleanup: bind call-bearing loop ranges to symbols. The analysis passes
    # need ``int_ceil`` in the range to reason about a chunked stride; codegen cannot emit a call in
    # an OpenMP loop header. This is the seam between the two, so it runs after everything that
    # reads the range and before anything that emits it.
    HoistLoopRangeCalls().apply_pass(sdfg, {})
    ConvertLengthOneArraysToScalars(recursive=True).apply_pass(sdfg, {})
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    move_small_arrays_to_stack(sdfg)
    made_persistent = make_transients_persistent(sdfg, device)
    cfg_by_id = {sd.cfg_id: sd for sd in sdfg.all_sdfgs_recursive()}
    for cfg_id, names in made_persistent.items():
        sd = cfg_by_id.get(cfg_id)
        if sd is None:
            continue
        for name in names:
            desc = sd.arrays.get(name)
            if desc is None:
                continue
            if desc.total_size == 1:
                desc.lifetime = dtypes.AllocationLifetime.State
                desc.storage = dtypes.StorageType.Register
                continue
            # A shape naming a RUNTIME-SUPPLIED symbol cannot be allocated in ``__dace_init``:
            # that runs before any state, and ``__dace_num_threads`` is defined by a graph tasklet
            # once the program is running. Promoting such a buffer emits an init that references an
            # undeclared name and does not compile. Scope lifetime instead -- these are per-thread
            # seams of a few dozen elements, so a per-call allocation costs nothing.
            if any(symbolic.NUM_THREADS_SYMBOL in {str(x)
                                                   for x in symbolic.pystr_to_symbolic(str(dim)).free_symbols}
                   for dim in desc.shape):
                desc.lifetime = dtypes.AllocationLifetime.State
    sdfg.reset_cfg_list()


def recompute_fuse_for_gpu(sdfg: SDFG) -> int:
    """Collapse producer->consumer map chains into a single output-domain map, recomputing
    each intermediate inline and deleting its transient (``OTFMapFusion``, applied to fixpoint).

    This is the GPU strategy for a fused stencil pipeline: on the device, materializing the
    intermediates (``lap`` / ``flx`` / ``fly`` in hdiff, the half-step buffers in jacobi / heat)
    means an extra global-memory round-trip per stage, and memory traffic -- not ALU -- is the
    bound. Recomputing the producers in registers inside one kernel keeps the traffic to inputs
    and outputs only, so one fused map beats N materialized ones (measured ~2x on hdiff at scale,
    bandwidth-bound). ``OTFMapFusion`` only matches a producer map whose output a consumer map
    reads, and recomputes it per consumer iteration; it is numerically faithful to the
    materialized form (verified to machine epsilon on hdiff / jacobi_2d / heat_3d).

    CPU deliberately does NOT run this: there the intermediates are cache-resident and shared
    across the consumer maps, so materializing once and reading beats recomputing -- the four
    separate maps are the CPU strategy. Called by :func:`offload_to_gpu` as its first step, so the
    fusion sees plain (device-agnostic) maps and the single fused map is what gets offloaded.

    :param sdfg: The SDFG to fuse in place.
    :returns: The number of ``OTFMapFusion`` applications.
    """
    return sdfg.apply_transformations_repeated(OTFMapFusion, validate_all=False)


def offload_to_gpu(sdfg: SDFG) -> None:
    """Move a canonicalized SDFG onto the GPU, in place: recompute-fuse, device offload, block size.

    A SEPARATE step from :func:`finalize_for_target`, not part of canonicalization: the device move
    is where a caller's own scheduling decisions belong, so the pipeline is
    ``canonicalize(s, target='gpu')`` -> *(any passes the caller needs on a device-agnostic graph)*
    -> ``offload_to_gpu(s)`` -> ``finalize_for_target(s, 'gpu')``. Callers with their own offload
    recipe (CloudSC schedules the inner maps and keeps the nblocks map sequential) substitute it
    here and never call this function. Four steps, mirroring ``auto_optimize``'s GPU tail:

    0. **Structural cleanup** (:func:`~dace.transformation.passes.canonicalize.pipeline.
       run_structural_cleanup`): state fusion, empty/dead state elimination and redundant-ordering-
       edge removal, so the device move sees the reduced graph -- fewer states is fewer kernel
       launches and fewer host<->device copies, and a redundant ordering edge inside a state is one
       more thing the stream serialization would have to honour. It belongs HERE and not in the
       caller's recipe because the documented pipeline lets a caller run its own passes between
       ``canonicalize`` and this call, so the pipeline's own trailing cleanup is already stale by
       the time the offload starts; inside the function is the only place that can guarantee it.
    1. **Recompute-fuse** (:func:`recompute_fuse_for_gpu`): collapse producer chains into one map
       before the device move, so the single fused map is what lands on the device (register
       recompute beats the global-memory round-trip of materialized intermediates). CPU keeps the
       materialized maps, which is why this lives here and not in ``finalize_for_target``.
    2. **Full offload** (unconditional): put non-transient arrays in GPU global storage
       (:func:`apply_gpu_storage`) and run ``apply_gpu_transformations`` (host<->device copies +
       ``GPU_Device`` schedules on every eligible map). Run unconditionally -- a partially-offloaded
       input (some maps already ``GPU_Device``, others not) is COMPLETED rather than skipped;
       ``apply_gpu_transformations`` leaves already-device maps alone and offloads the rest. No extra
       ``simplify`` is run here (the canonicalized SDFG is already simplified; ``apply_gpu_storage`` +
       ``apply_gpu_transformations`` are the only offload steps needed).
    3. **Block size** (:func:`select_gpu_device_block_size`): pick a thread-block matching the
       iteration domain (``N x N`` -> ``16x16`` / ``32x16``; 1-D -> the ``128,1,1`` default, and a
       tree-reduction map a deep ``512,1,1``) on every ``GPU_Device`` map, run AFTER offload so
       each kernel map's final dimensionality is known.

    Streams: the canonicalized SDFG is offloaded onto the **single default stream**
    (``compiler.cuda.max_concurrent_streams = -1``). Canon produces many small device maps +
    host<->device copies with fine-grained dependences; DaCe's default (``0`` = a fresh stream
    per concurrent branch) then interleaves kernels and async copies across streams whose
    cross-stream events must be exactly right, and a single missed dependence is an illegal
    memory access. Serialising onto stream 0 (all work ordered, host-synchronous copies)
    removes that failure mode -- the correctness floor canon-GPU needs before any stream
    concurrency is layered back in. Set on the process Config here so the subsequent codegen
    (which reads the value) emits the single-stream form.
    """
    Config.set('compiler', 'cuda', 'max_concurrent_streams', value=-1)
    run_structural_cleanup(sdfg)
    recompute_fuse_for_gpu(sdfg)
    apply_gpu_storage(sdfg)
    sdfg.apply_gpu_transformations()
    # Between the offload and the block-size choice, and it has to be exactly here. The offload
    # assigns every nested scope ``Sequential``, so a map tagged ``is_warp_tile`` cannot become a
    # thread block before this point; and ``select_gpu_device_block_size`` skips a kernel that
    # already contains a thread-block map, so promoting after it would leave the kernel carrying
    # BOTH a declared block size and a thread-block map -- which is the conflict codegen refuses.
    PromoteWarpTiles().apply_pass(sdfg, {})
    select_gpu_device_block_size(sdfg)


def assert_offloaded(sdfg: SDFG) -> None:
    """Raise unless ``sdfg`` has actually been moved onto the device.

    :func:`finalize_for_target` with ``target='gpu'`` reads the device maps and ``GPU_Global``
    arrays that an offload creates; on a still-host-scheduled graph every GPU-specific step below
    it is a no-op and the result is a CPU graph wearing a GPU label. Fail loudly instead.

    Either signal counts, because the two legal offloads produce different mixes: the generic
    :func:`offload_to_gpu` sets both, while a caller-supplied recipe may schedule kernels without
    moving every non-transient (or vice versa for an all-library graph with no maps of its own).

    A graph with neither a map nor a library node anywhere is the one case this cannot be about: no
    offload could have placed anything, so the target has nothing to run (npbench crc16 is a purely
    sequential loop nest). Host code is then the right answer, not a missing call.

    :param sdfg: The SDFG to check, including nested SDFGs.
    :raises ValueError: If no ``GPU_Device`` map and no ``GPU_Global`` array is present.
    """
    offloadable = False
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device:
            return
        offloadable = offloadable or isinstance(node, (nodes.MapEntry, nodes.LibraryNode))
    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            if desc.storage == dtypes.StorageType.GPU_Global:
                return
    if not offloadable:
        return
    raise ValueError(f"finalize_for_target(sdfg, 'gpu') needs an already-offloaded SDFG, but '{sdfg.name}' has no "
                     "GPU_Device map and no GPU_Global array. Offload is a separate step so passes can run between "
                     "canonicalization and the device move: call offload_to_gpu(sdfg), or your own offload recipe, "
                     "before finalizing.")


def assert_no_nested_parallel_maps(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Post-pipeline invariant: a parallel scope of the target's device schedule must NEVER be
    nested inside another parallel scope of the same schedule.

    The device parallel schedule is ``CPU_Multicore`` on CPU and ``GPU_Device`` on GPU. Two stacked
    ``CPU_Multicore`` maps emit nested ``#pragma omp parallel for`` regions (T*T oversubscription, and
    for an inner reduction a re-forked team every outer iteration -- the "constant parallel reductions"
    catastrophe); a ``GPU_Device`` map inside another ``GPU_Device`` map is an illegal in-kernel kernel
    launch. The canonicalize policy is that ONLY top-level maps parallelize; every inner map -- and any
    library node re-entered inside a parallel map -- must carry a NON-device schedule (``Sequential`` on
    CPU; ``Sequential`` / a thread-block schedule on GPU). This is asserted once after finalization so a
    regression that leaves an inner map / libnode on the device schedule fails loudly instead of
    silently emitting nested parallelism. It is the counterpart of :func:`libnode_is_sequential`, which
    keeps a map-nested library node single-core during implementation selection.

    :param sdfg: the finalized SDFG to check (read-only).
    :param device: ``CPU`` -> forbid nested ``CPU_Multicore``; ``GPU`` -> forbid nested ``GPU_Device``.
    :raises ValueError: if any map / library node on the device parallel schedule has a map of that
        same schedule among its enclosing scopes (walked across nested-SDFG boundaries).
    """
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    parallel = (dtypes.ScheduleType.GPU_Device
                if device == dtypes.DeviceType.GPU else dtypes.ScheduleType.CPU_Multicore)
    for node, state in sdfg.all_nodes_recursive():
        # A library node is a "special tasklet": a ``node.schedule`` on the device parallel schedule
        # means it would open its own parallel region, exactly what must not happen inside a parallel map.
        if isinstance(node, nodes.MapEntry):
            label, node_sched = node.map.label, node.map.schedule
        elif isinstance(node, nodes.LibraryNode):
            label, node_sched = node.label, node.schedule
        else:
            continue
        if node_sched != parallel:
            continue
        for scope in get_parent_map_and_loop_scopes(sdfg, node, state):
            if isinstance(scope, nodes.MapEntry) and scope.map.schedule == parallel:
                kind = 'map' if isinstance(node, nodes.MapEntry) else 'library node'
                raise ValueError(
                    f"Nested {parallel.name}: {kind} '{label}' is nested inside {parallel.name} map "
                    f"'{scope.map.label}'. Inner maps / library nodes must not carry the device parallel "
                    f"schedule (only top-level maps parallelize) -- nesting emits stacked parallel regions.")


def finalize_for_target(sdfg: SDFG,
                        target: str = 'cpu',
                        validate: bool = True,
                        break_anti_dependence: bool = True) -> SDFG:
    """Apply the performance finalization tail to a canonicalized ``sdfg``.

    Selects fast library implementations (leaving the nodes un-expanded for
    codegen to lower), then moves small constant-size transients to the stack and
    independent transients to persistent allocation. Operates in place.

    ``target='gpu'`` finalizes an **already-offloaded** graph; it does not offload one. The device
    move is a separate step (:func:`offload_to_gpu`, or a caller's own recipe) so passes can run
    between canonicalization and it -- ``canonicalize(s, target='gpu'); offload_to_gpu(s);
    finalize_for_target(s, 'gpu')`` is the perf-path counterpart to ``auto_optimize(s,
    device=GPU)``.

    :param sdfg: A canonicalized SDFG; for ``target='gpu'``, an offloaded one.
    :param target: ``'cpu'`` or ``'gpu'`` (selects the fast-library priority).
    :param validate: Validate the SDFG once at the end.
    :param break_anti_dependence: Forwarded to the CPU specialization stage: chunk the
                                  anti-dependence snapshots canonicalization left behind. Pass
                                  ``False`` when canonicalization did not break anti dependences.
    :returns: The same ``sdfg`` instance, finalized.
    :raises ValueError: If ``target='gpu'`` and ``sdfg`` was never offloaded.
    """
    if target not in _TARGET_DEVICE:
        raise ValueError(f"target must be one of {sorted(_TARGET_DEVICE)}; got {target!r}")
    device = _TARGET_DEVICE[target]

    # Offload is NOT part of this tail: the caller runs it, so passes can be inserted between
    # canonicalization and the device move (see :func:`offload_to_gpu`). Everything below still
    # requires it to have happened -- the fast GPU library picks (cuBLAS/cuSolverDn/CUB) and the
    # GPU storage rules read the device maps and GPU_Global arrays the offload creates -- so a
    # host-scheduled graph is rejected here rather than quietly finalized as if it were CPU.
    if device == dtypes.DeviceType.GPU:
        assert_offloaded(sdfg)

    # Infer schedules BEFORE selecting library-node implementations so the selection can adhere to
    # each node's schedule: DaCe sets a library node nested in a parallel map (or re-entered per loop
    # iteration) to ``Sequential`` and a top-level one to the device default. A ``Sequential``
    # Reduce/Scan/Copy/Fill must lower to its efficient single-core expansion, NOT open its own
    # (nested) OpenMP region per outer iteration -- the "constant parallel reductions" slowdown.
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # Canonicalization stops at the maximally parallel form, so the target's specialization stage
    # runs here -- BEFORE library selection, so libnode_is_sequential sees the corrected schedules.
    # On CPU that is the whole cpu_specialize stage (calibration, anti-dependence chunking,
    # oversized-intermediate recompute, the fork/join cost model, transfer specialization); on GPU
    # it is the nested-kernel resolution. Both are idempotent, so finalizing an already-specialized
    # graph re-confirms the same verdicts rather than compounding them.
    if device == dtypes.DeviceType.GPU:
        SequentializeNestedDeviceScopes().apply_pass(sdfg, {})
    else:
        cpu_specialize(sdfg, break_anti_dependence=break_anti_dependence, validate=False)

    canonicalize_set_fast_implementations(sdfg, device)
    # Select the fast implementation per library node but DO NOT expand here: a library
    # node is expanded exactly once, at codegen (``compile()`` auto-expands using the
    # ``implementation`` chosen above). Expanding in canonicalization would re-introduce
    # the per-implementation shapes (BLAS scratch, reduction accumulators) into a form the
    # rest of the toolchain must then re-canonicalize; keeping one shape per computation
    # until codegen is the invariant every downstream pass relies on.
    infer_types.infer_connector_types(sdfg)
    finalize_transient_storage(sdfg, device)

    # Open the OpenMP team once per sequential loop rather than once per trip. It runs HERE, at the
    # very end, and not in the ``cpu_specialize`` band, because the step above is what decides which
    # transients live in the state struct: ``make_transients_persistent`` skips anything inside a
    # parallel map body (it would be a per-thread buffer), so a loop hoisted before it keeps a
    # scope-lifetime scratch array and pays a malloc per trip. Deciding storage first, then the
    # team, gets both. Connector types are re-inferred because the hoist outlines the loop into a
    # nested SDFG, whose boundary connectors are new.
    if device == dtypes.DeviceType.CPU:
        if HoistParallelRegion().apply_pass(sdfg, {}):
            infer_types.infer_connector_types(sdfg)

    if validate:
        assert_no_nested_parallel_maps(sdfg, device)
        sdfg.validate()
    return sdfg
