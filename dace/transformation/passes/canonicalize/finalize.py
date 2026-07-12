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

from dace import SDFG, dtypes
from dace.sdfg import infer_types, nodes
from dace.sdfg.state import LoopRegion
from dace.libraries.blas.environments import openblas
from dace.transformation.auto.auto_optimize import (apply_gpu_storage, make_transients_persistent,
                                                    move_small_arrays_to_stack, set_fast_implementations)
from dace.transformation.passes.gpu_block_size_selection import select_gpu_device_block_size
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.libraries.standard.nodes.reduce import Reduce
from dace.libraries.standard.nodes.scan import Scan
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode, select_copy_implementation
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode, select_memset_implementation

#: Map the canonicalize target string to the codegen device type.
_TARGET_DEVICE = {'cpu': dtypes.DeviceType.CPU, 'gpu': dtypes.DeviceType.GPU}

#: Per-dimension matmul extent at or below which canonicalization picks the inlined
#: ``'pure'`` expansion (a fusible/vectorizable sequential loop nest) over a BLAS call.
_SMALL_MATMUL_DIM = 256

#: The row-wise (ikj) ``'pure'`` variant is selected only for *tiny* matmuls -- every
#: dimension at most this. Its win (a vectorizable row update with a sequential K loop) is
#: a register/cache-blocking effect that only pays off at very small sizes; a larger
#: "small" matmul keeps the plain ``'pure'`` nest.
_ROWWISE_MATMUL_DIM = 64


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
    * GPU: ``cuBLAS``, ``cuSolverDn``, ``cuTENSOR``, ``GPUAuto``, ``CUB``, ``CUDA`` (``cub::DeviceScan``
      / device sort).

    Only impls whose environment is available on this host are listed, so forcing a pick never selects
    an unbuilt library. ``MatMul``/``Gemm`` still get the tiny-matmul ``pure``/``rowwise`` override in
    :func:`canonicalize_set_fast_implementations`.
    """
    if device == dtypes.DeviceType.GPU:
        # GPU perf runs where these are present; each node's own environment gates the actual build.
        return ['cuBLAS', 'cuSolverDn', 'cuTENSOR', 'GPUAuto', 'CUB', 'CUDA']
    prio = []
    if openblas.OpenBLAS.is_installed():
        prio.append('OpenBLAS')
    if 'HPTT_ROOT' in os.environ:
        prio.append('HPTT')
    prio.append('TTGT')
    prio += ['OpenMP', 'CPU']
    return prio


def libnode_is_sequential(node: nodes.LibraryNode, state, sdfg: SDFG) -> bool:
    """Whether ``node`` is re-entered inside an outer parallel/repeated scope and so must NOT open
    its own (nested) parallel region -- it lowers to its efficient single-core expansion instead.

    The storage-derived ``node.schedule`` is NOT a reliable signal here: DaCe's schedule inference
    (:func:`~dace.sdfg.infer_types.set_default_schedule_and_storage_types`) sets a library node's
    schedule from the *storage* of its neighbouring memlets (``CPU_Heap`` ->
    ``ScheduleType.CPU_Multicore`` via ``STORAGEDEFAULT_SCHEDULE``), NOT from the parallelism of the
    enclosing scope, so a ``Reduce`` nested in a parallel map can carry ``CPU_Multicore`` rather than
    ``Sequential`` and would then wrongly open a nested ``#pragma omp parallel`` per outer iteration
    -- the "constant parallel reductions" catastrophe. Determine sequentiality from SCOPE instead:
    a libnode is sequential if it has a parallel parent map, an enclosing loop (both re-enter it), or
    lives inside a ``Sequential``-scheduled nested SDFG (itself nested in a parallel scope).

    :func:`~dace.transformation.helpers.get_parent_map_and_loop_scopes` -- now LibraryNode-aware (a
    library node is a "special tasklet") -- yields every enclosing ``MapEntry`` / ``LoopRegion``
    across nested-SDFG boundaries, so deep nesting is handled by the existing helper rather than a
    bespoke climb. A genuinely top-level node (no enclosing parallel map / loop, not inside a
    ``Sequential`` nsdfg) returns ``False`` and is free to open its own OpenMP / device-parallel region.
    """
    # Function-local import: ``dace.transformation.helpers`` pulls in a chain that re-enters the
    # canonicalize package, so a top-level import here would be a cycle when ``finalize`` is imported
    # as the package's first submodule.
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    if node.schedule == dtypes.ScheduleType.Sequential:
        return True
    # A ``Sequential`` enclosing nested SDFG (itself nested in a parallel scope) makes the libnode
    # sequential too.
    parent_nsdfg = state.sdfg.parent_nsdfg_node
    if parent_nsdfg is not None and parent_nsdfg.schedule == dtypes.ScheduleType.Sequential:
        return True
    for scope in get_parent_map_and_loop_scopes(sdfg, node, state):
        if isinstance(scope, nodes.MapEntry):
            if scope.map.schedule != dtypes.ScheduleType.Sequential:
                return True
        elif isinstance(scope, LoopRegion):
            return True
    return False


def canonicalize_set_fast_implementations(sdfg: SDFG, device: dtypes.DeviceType, small_dim: int = _SMALL_MATMUL_DIM):
    """Select library-node implementations for the canonicalize perf tail.

    Delegates to :func:`~dace.transformation.auto.auto_optimize.set_fast_implementations` with the
    canonicalize priority (:func:`canonicalize_fast_library_priority`) -- OpenBLAS/LAPACK, HPTT/TTGT,
    OpenMP, cuBLAS/cuSolverDn/cuTENSOR/CUB, never MKL -- so EVERY library node the pipeline introduces
    (Reduce, Scan, Transpose, TensorTranspose, Symm, Cholesky, Solve, ...) lowers to its fast expansion
    rather than the serial ``pure`` loop. Then OVERRIDES any GEMM/MatMul whose every dimension is a
    known constant at most ``small_dim`` to the inlined ``'pure'`` expansion. A tiny matmul's
    BLAS/cuBLAS call is pure overhead, and -- unlike an opaque library call -- a sequential loop nest
    is fusible/vectorizable and keeps a loop of small matmuls sequential instead of issuing serialized
    library calls. Symbolic- or large-dimensioned matmuls keep the fast BLAS implementation.
    """
    set_fast_implementations(sdfg, device, blocklist=['MKL'], find_fast_library_fn=canonicalize_fast_library_priority)
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

        # ``Reduce``: parallel -> OpenMP privatized ``reduction(op:var)``; sequential -> the efficient
        # single-core ``pure`` reduction (a plain accumulate loop, never a contended ``omp atomic`` /
        # nested ``omp parallel``). ``find_fast_library`` omits both, so it would otherwise resolve to
        # ``pure`` anyway; the explicit pick makes the top-level case parallel.
        if isinstance(node, Reduce) and device == dtypes.DeviceType.CPU:
            if sequential:
                # ``pure-seq`` needs an ``identity`` the lifted node may not carry, so ``pure`` is the
                # robust single-core choice (it lowers to a plain accumulate loop when Sequential).
                node.implementation = 'pure'
            elif 'OpenMP' in impls:
                node.implementation = 'OpenMP'
            continue
        # ``Scan``: parallel -> ``CPU`` (OpenMP 5.0 ``#pragma omp parallel for simd reduction(inscan,..)``
        # + ``#pragma omp scan``); sequential -> the serial ``pure`` scan.
        if isinstance(node, Scan):
            if device == dtypes.DeviceType.CPU:
                node.implementation = 'pure' if sequential else ('CPU' if 'CPU' in impls else node.implementation)
                continue
            # GPU: a top-level parallel scan -> host-launched ``cub::DeviceScan``; a sequential
            # (device-level, or map-/loop-nested) scan MUST stay ``pure`` -- ``ExpandCUDA`` emits a
            # HOST-side ``cub::DeviceScan`` call that cannot be issued from inside a kernel (and it
            # rejects stride>1). Guarding on ``sequential`` mirrors the Reduce branch; without it a
            # device-level scan that ``set_fast_implementations`` correctly left ``pure`` was clobbered
            # to an uncompilable in-kernel ``cub::DeviceScan``.
            if device == dtypes.DeviceType.GPU:
                node.implementation = 'pure' if sequential else ('CUDA' if 'CUDA' in impls else node.implementation)
                continue
        # ``Copy`` / ``Memset``: a top-level node -> ``Auto`` (its own size gate picks the chunked
        # ``CPU_Multicore`` parallel transfer for large/symbolic sizes). A sequential (nested) node
        # must run single-core: ask the node's OWN selector for the correct expansion -- which routes
        # a non-contiguous / cross-storage transfer to ``MappedTasklet`` / ``pure`` rather than the
        # contiguous-only ``MemcpyCPU`` / ``CPU`` that would RAISE at codegen -- then downgrade the
        # parallel chunk-map variant to its serial sibling so a nested transfer opens no OpenMP region.
        if isinstance(node, CopyLibraryNode) and device == dtypes.DeviceType.CPU:
            if sequential:
                impl = select_copy_implementation(node, state, state.sdfg)
                node.implementation = 'MemcpyCPU' if impl == 'MemcpyParallelCPU' else impl
            else:
                node.implementation = 'Auto'
            continue
        if isinstance(node, MemsetLibraryNode) and device == dtypes.DeviceType.CPU:
            if sequential:
                impl = select_memset_implementation(node, state, state.sdfg)
                node.implementation = 'CPU' if impl == 'ParallelCPU' else impl
            else:
                node.implementation = 'Auto'
            continue
        if 'pure' not in impls:
            continue
        if _all_matmul_extents_small(state, node, small_dim):
            # Prefer the row-wise (ikj) pure expansion for TINY GEMMs (every dim <= 64): a
            # vectorizable row update with a sequential K accumulation. Larger small matmuls,
            # and nodes without a 'rowwise' impl (e.g. MatMul), keep the plain 'pure' nest.
            if 'rowwise' in impls and _all_matmul_extents_small(state, node, _ROWWISE_MATMUL_DIM):
                node.implementation = 'rowwise'
            else:
                node.implementation = 'pure'


def finalize_transient_storage(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Finalize the storage of a canonicalized (or vectorized) SDFG's transients, in place.

    The single home for transient storage finalization -- a value-preserving perf tail that any
    downstream producer (canonicalize's ``finalize_for_target``, or a caller that has additionally
    vectorized the SDFG) runs to allocate temporaries well. Three steps, mirroring
    ``auto_optimize``'s storage tail:

    1. **Length-1 transient arrays -> scalars** (:class:`ConvertLengthOneArraysToScalars`,
       ``transient_only=True``): a single internal value belongs in a scalar, not a heap array.
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
    ConvertLengthOneArraysToScalars(recursive=True, transient_only=True).apply_pass(sdfg, {})
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
            if desc is not None and desc.total_size == 1:
                desc.lifetime = dtypes.AllocationLifetime.Scope
                desc.storage = dtypes.StorageType.Register
    sdfg.reset_cfg_list()


def offload_to_gpu(sdfg: SDFG) -> None:
    """Move a canonicalized SDFG onto the GPU, in place: device offload then block-size choice.

    This is the canonicalize-GPU tail the user specified -- "canon-gpu runs after finalizing to
    GPU offloading and the block-size default pass" -- so ``canonicalize(s, target='gpu');
    finalize_for_target(s, 'gpu')`` becomes the perf-path counterpart to ``auto_optimize(s,
    device=GPU)``. Two steps, mirroring ``auto_optimize``'s GPU tail:

    1. **Full offload** (unconditional): put non-transient arrays in GPU global storage
       (:func:`apply_gpu_storage`) and run ``apply_gpu_transformations`` (host<->device copies +
       ``GPU_Device`` schedules on every eligible map). Run unconditionally -- a partially-offloaded
       input (some maps already ``GPU_Device``, others not) is COMPLETED rather than skipped;
       ``apply_gpu_transformations`` leaves already-device maps alone and offloads the rest. No extra
       ``simplify`` is run here (the canonicalized SDFG is already simplified; ``apply_gpu_storage`` +
       ``apply_gpu_transformations`` are the only offload steps needed).
    2. **Block size** (:func:`select_gpu_device_block_size`): pick a thread-block matching the
       iteration domain (``N x N`` -> ``16x16`` / ``32x16``; 1-D -> the ``128,1,1`` default) on every
       ``GPU_Device`` map, run AFTER offload so each kernel map's final dimensionality is known.
    """
    apply_gpu_storage(sdfg)
    sdfg.apply_gpu_transformations()
    select_gpu_device_block_size(sdfg)


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


def finalize_for_target(sdfg: SDFG, target: str = 'cpu', validate: bool = True) -> SDFG:
    """Apply the performance finalization tail to a canonicalized ``sdfg``.

    Selects fast library implementations (leaving the nodes un-expanded for
    codegen to lower), then moves small constant-size transients to the stack and
    independent transients to persistent allocation. Operates in place.

    :param sdfg: A canonicalized SDFG.
    :param target: ``'cpu'`` or ``'gpu'`` (selects the fast-library priority).
    :param validate: Validate the SDFG once at the end.
    :returns: The same ``sdfg`` instance, finalized.
    """
    if target not in _TARGET_DEVICE:
        raise ValueError(f"target must be one of {sorted(_TARGET_DEVICE)}; got {target!r}")
    device = _TARGET_DEVICE[target]

    # For GPU, offload to the device and choose thread-block dimensions BEFORE selecting library
    # implementations and storage: the fast GPU library picks (cuBLAS/cuSolverDn/CUB) and the GPU
    # storage rules must see the device maps and GPU_Global arrays the offload creates.
    if device == dtypes.DeviceType.GPU:
        offload_to_gpu(sdfg)

    # Infer schedules BEFORE selecting library-node implementations so the selection can adhere to
    # each node's schedule: DaCe sets a library node nested in a parallel map (or re-entered per loop
    # iteration) to ``Sequential`` and a top-level one to the device default. A ``Sequential``
    # Reduce/Scan/Copy/Memset must lower to its efficient single-core expansion, NOT open its own
    # (nested) OpenMP region per outer iteration -- the "constant parallel reductions" slowdown.
    infer_types.set_default_schedule_and_storage_types(sdfg, None)

    canonicalize_set_fast_implementations(sdfg, device)
    # Select the fast implementation per library node but DO NOT expand here: a library
    # node is expanded exactly once, at codegen (``compile()`` auto-expands using the
    # ``implementation`` chosen above). Expanding in canonicalization would re-introduce
    # the per-implementation shapes (BLAS scratch, reduction accumulators) into a form the
    # rest of the toolchain must then re-canonicalize; keeping one shape per computation
    # until codegen is the invariant every downstream pass relies on.
    infer_types.infer_connector_types(sdfg)
    finalize_transient_storage(sdfg, device)

    if validate:
        assert_no_nested_parallel_maps(sdfg, device)
        sdfg.validate()
    return sdfg
