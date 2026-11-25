# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-canonicalization target finalization for performance: the fast library implementation per node (expanded
once, at codegen) and the storage of transients. ``canonicalize(s); finalize_for_target(s)`` is the counterpart to
``auto_optimize(s)``.
"""

import os
from collections.abc import Callable

import sympy

from dace import SDFG, dtypes, symbolic
from dace import data as dt
from dace.config import Config
from dace.libraries.blas.environments import libxsmm, openblas
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul, _get_matmul_operands, _matrix_operand
from dace.libraries.fft.environments import fftw3
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose
from dace.libraries.standard.block_reduce import in_kernel_block_implementation, release_kernel_block_size
from dace.libraries.standard.nodes.scan import Scan
from dace.libraries.standard.nodes.symmetrize import Symmetrize
from dace.ordered import OrderedSet
from dace.sdfg import infer_types, nodes
from dace.sdfg.narrowing import as_range
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import helpers as xfh
from dace.transformation.auto.auto_optimize import (
    apply_cpu_library_parallelism,
    apply_gpu_storage,
    find_fast_library,
    libnode_is_sequential,
    make_transients_persistent,
    move_small_arrays_to_stack,
    set_fast_implementations,
)
from dace.transformation.dataflow import OTFMapFusion, RedundantArray, RedundantSecondArray
from dace.transformation.interstate import InlineSDFG
from dace.transformation.passes.bound_data_dependent_extents import BoundDataDependentExtents
from dace.transformation.passes.canonicalize.beta_fold import BetaFold
from dace.transformation.passes.canonicalize.hoist_loop_range_calls import HoistLoopRangeCalls
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import MoveLoopIntoMapGated
from dace.transformation.passes.canonicalize.pipeline import run_structural_cleanup
from dace.transformation.passes.canonicalize.shrink_map_local_transients import ShrinkMapLocalTransients
from dace.transformation.passes.canonicalize.split_map_local_transients import SplitMapLocalTransients
from dace.transformation.passes.canonicalize.transpose_into_blas_flag import transposes_as_blas_flags
from dace.transformation.passes.cpu_specialization.chunk_stream_compaction import GPU_BLOCK, ChunkStreamCompaction
from dace.transformation.passes.cpu_specialization.demote_level_maps import (
    demote_strided_level_maps,
    demote_uncoalesced_level_maps,
    rejoin_block_maps,
    small_maps,
)
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import HoistParallelRegion
from dace.transformation.passes.cpu_specialization.if_convert_map_bodies import IfConvertParallelMapBodies
from dace.transformation.passes.cpu_specialization.interchange_loops_into_lane_maps import InterchangeLoopsIntoLaneMaps
from dace.transformation.passes.cpu_specialization.pipeline import cpu_specialize
from dace.transformation.passes.cpu_specialization.strip_mine_interchange import StripMineInterchange
from dace.transformation.passes.equalize_symbol_dtypes import equalized
from dace.transformation.passes.fuse_maps import FuseMaps
from dace.transformation.passes.fusion_inline import FuseStates
from dace.transformation.passes.gpu_block_size_selection import select_gpu_device_block_size
from dace.transformation.passes.gpu_specialization.coarsen_scatter_kernels import CoarsenScatterKernels
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import gpu_specialize_offloaded
from dace.transformation.passes.gpu_specialization.promote_warp_tiles import PromoteWarpTiles
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.transformation.passes.minimize_transients import MinimizeTransients
from dace.transformation.passes.offloading.batch_map_products import BatchMapProducts
from dace.transformation.passes.offloading.batch_row_scans import BatchRowScans
from dace.transformation.passes.offloading.fuse_copy_chains import FuseCopyChains
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.specialize_small_gemms import (
    SMALL_MATMUL_DIM,
    SpecializeSmallGemms,
    matmul_extents_small,
)
from dace.transformation.passes.symbol_ssa import SymbolSSA
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

#: Map the canonicalize target string to the codegen device type.
TARGET_DEVICE = {"cpu": dtypes.DeviceType.CPU, "gpu": dtypes.DeviceType.GPU}


def blas_addresses(node: nodes.LibraryNode, state: SDFGState) -> bool:
    """Whether a BLAS call can take every matrix operand of ``node`` as it is laid out: BLAS names a matrix by a
    pointer and one leading dimension, so each operand needs a unit stride on one of its matrix axes."""
    try:
        operands = _get_matmul_operands(node, state, state.sdfg)
    except ValueError:
        return True
    return all(any(symbolic.equal_valued(1, s) for s in _matrix_operand(operand)[3]) for operand in operands)


def canonicalize_fast_library_priority(device: dtypes.DeviceType) -> list[str]:
    """The fast implementations canonicalize selects, in ``find_fast_library``'s order (the two pipelines are compared
    column against column), restricted to the libraries installed on this host."""
    if device == dtypes.DeviceType.GPU:
        # ``pure`` is auto_optimize's fallback, not a forced pick.
        return [impl for impl in find_fast_library(device) if impl != "pure"]
    prio: list[str] = []
    if openblas.OpenBLAS.is_installed():
        prio.append("OpenBLAS")
    if fftw3.FFTW3.is_installed():
        prio.append("FFTW3")
    if "HPTT_ROOT" in os.environ:
        prio.append("HPTT")
    prio.append("TTGT")
    prio.append("CPU")
    return prio


#: The fast lowerings canonicalize selects on the CPU when their library is present.
CPU_FAST_IMPLEMENTATIONS = ("OpenBLAS", "FFTW3", "HPTT", "TTGT", "CPU")


class SilentPureLowering(RuntimeError):
    """A top-level library node is left on its serial ``pure`` loop although a fast lowering exists for it."""


def refuse_silent_pure(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Raise :class:`SilentPureLowering` for every top-level library node left on ``pure`` whose type has one of the
    device's fast lowerings."""
    fast = (
        set(find_fast_library(device)) - {"pure"} if device == dtypes.DeviceType.GPU else set(CPU_FAST_IMPLEMENTATIONS)
    )
    silent = []
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode) or not node.auto_select_implementation:
            continue
        if (node.implementation or node.default_implementation) != "pure":
            continue
        vendors = sorted(fast & set(type(node).implementations))
        if not vendors or isinstance(node, Gemm) and not blas_addresses(node, state):
            continue
        if libnode_is_sequential(node, state, state.sdfg) or libnode_is_device_code(node, state, state.sdfg):
            continue
        silent.append(f"{type(node).__name__} {node.label!r} (fast lowerings: {', '.join(vendors)})")
    if silent:
        raise SilentPureLowering(
            "canonicalize left library nodes on their serial 'pure' loop; install the library or fix the selection: "
            + "; ".join(silent)
        )


def gemm_dtype(node: Gemm, state: SDFGState) -> dtypes.typeclass:
    """The element type of ``node``'s ``A`` operand."""
    edge = next(e for e in state.in_edges(node) if e.dst_conn == "_a")
    return state.sdfg.arrays[edge.data.data].dtype.base_type


def libxsmm_batched_product(node: BatchedMatMul, state: SDFGState) -> bool:
    """Whether ``node`` multiplies real matrices with no constant extent past
    :data:`~dace.symbolic_facts.SMALL_CONSTANT_EXTENT`: LIBXSMM hands a large shape to BLAS itself."""
    edges = [e for e in state.in_edges(node) if e.dst_conn in ("_a", "_b")]
    if len(edges) != 2:
        return False
    for edge in edges:
        shape = edge.data.subset.size()
        if state.sdfg.arrays[edge.data.data].dtype.base_type not in (dtypes.float32, dtypes.float64) or len(shape) < 2:
            return False
        if any(
            symbolic.pystr_to_symbolic(extent).is_number and not symbolic.is_small_extent(extent, None)
            for extent in shape[-2:]
        ):
            return False
    return True


def libnode_is_device_code(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> bool:
    """``node`` sits inside a GPU kernel, across every nested-SDFG boundary, so it must lower to device code."""
    return any(
        isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES
        for scope in xfh.get_parent_map_and_loop_scopes(sdfg, node, state)
    )


def libnode_on_host(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG) -> bool:
    """``node`` is host code whose operands all live in host memory, so it runs the CPU library."""
    operands = [edge.data.data for edge in state.all_edges(node) if not edge.data.is_empty()]
    return (
        bool(operands)
        and node.schedule not in dtypes.GPU_SCHEDULES
        and all(state.sdfg.arrays[name].storage not in dtypes.GPU_KERNEL_ACCESSIBLE_STORAGES for name in operands)
        and not libnode_is_device_code(node, state, sdfg)
    )


def cpu_library_choice(
    node: nodes.LibraryNode, state: SDFGState, sequential: bool, impls: dict, has_libxsmm: bool
) -> str | None:
    """The CPU implementation ``node`` takes over its fast default, or ``None`` to keep that."""
    # In a parallel region the in-thread transpose beats an HPTT plan per call.
    if sequential and isinstance(node, TensorTranspose):
        return "sequential"
    # In a parallel region a real GEMM is one small product per iteration: LIBXSMM, as OpenBLAS's per-call buffer
    # allocation contends across threads.
    if (
        sequential
        and isinstance(node, Gemm)
        and gemm_dtype(node, state) in (dtypes.float32, dtypes.float64)
        and has_libxsmm
    ):
        return "LIBXSMM"
    # A batch of real products not known to be large: a LIBXSMM kernel per product.
    if isinstance(node, BatchedMatMul) and libxsmm_batched_product(node, state) and has_libxsmm:
        return "LIBXSMM"
    # A top-level real GEMM takes OpenBLAS with a run-time shape plan (split-K, syrk for ``X.T @ X``).
    if (
        not sequential
        and isinstance(node, Gemm)
        and "Planned" in impls
        and gemm_dtype(node, state) in (dtypes.float32, dtypes.float64)
        and not ({"_alpha", "_beta"} & set(node.in_connectors))
        and openblas.OpenBLAS.is_installed()
    ):
        return "Planned"
    return None


def gpu_scope_implementation(node: nodes.LibraryNode, state: SDFGState, sdfg: SDFG, impls: dict) -> str | None:
    """A GPU ``Scan`` or ``Symmetrize``: ``pure`` inside a kernel (``ExpandCUDA`` emits a host-side call; the
    triangular walk is cheaper there), the ``CUDA`` expansion at host level, where ``pure`` would index ``GPU_Global``
    operands from the host."""
    if libnode_is_device_code(node, state, sdfg):
        return "pure"
    return "CUDA" if "CUDA" in impls else node.implementation


def canonicalize_set_fast_implementations(
    sdfg: SDFG,
    device: dtypes.DeviceType,
    small_dim: int = SMALL_MATMUL_DIM,
    selects: Callable[[nodes.LibraryNode, SDFGState], bool] | None = None,
) -> None:
    """Select every library node's fast implementation (:func:`canonicalize_fast_library_priority`), then the per-node
    overrides below; a GEMM/MatMul whose every extent is a constant of at most ``small_dim`` takes the inlined
    ``rowwise`` loop nest."""
    if device == dtypes.DeviceType.GPU and selects is None:
        canonicalize_set_fast_implementations(
            sdfg, dtypes.DeviceType.CPU, small_dim, lambda node, state: libnode_on_host(node, state, sdfg)
        )

        def on_device(node: nodes.LibraryNode, state: SDFGState) -> bool:
            return not libnode_on_host(node, state, sdfg)

        selects = on_device
    set_fast_implementations(
        sdfg, device, blocklist=["MKL"], find_fast_library_fn=canonicalize_fast_library_priority, selects=selects
    )
    gpu_priority = canonicalize_fast_library_priority(device) if device == dtypes.DeviceType.GPU else []
    # Probed once: without LIBXSMM_ROOT, locating LIBXSMM runs a library search per call.
    has_libxsmm = device == dtypes.DeviceType.CPU and libxsmm.LIBXSMM.is_installed()
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode) or selects is not None and not selects(node, state):
            continue
        select_implementation(node, state, sdfg, device, small_dim, gpu_priority, has_libxsmm)


def select_implementation(
    node: nodes.LibraryNode,
    state: SDFGState,
    sdfg: SDFG,
    device: dtypes.DeviceType,
    small_dim: int,
    gpu_priority: list[str],
    has_libxsmm: bool,
) -> None:
    """Override the lowering ``set_fast_implementations`` chose for ``node`` where the canonical form needs another."""
    impls = type(node).implementations
    # A node inside a parallel or repeated scope must not open its own parallel region (a fork/join per outer
    # iteration).
    sequential = libnode_is_sequential(node, state, sdfg)
    if sequential and node.schedule != dtypes.ScheduleType.Sequential:
        node.schedule = dtypes.ScheduleType.Sequential

    # A specialization pass already chose this node's lowering (``SpecializeSmallGemms``).
    if not node.auto_select_implementation:
        return

    # Inside a kernel, a node that can run as a block collective does: one block per outer iteration.
    if device == dtypes.DeviceType.GPU:
        block = in_kernel_block_implementation(node, state, share_private_outputs=True)
        if block is not None:
            node.implementation = block
            release_kernel_block_size(node, state)
            return

    # A GEMM no BLAS call can address takes the expansion that indexes its operands directly.
    if isinstance(node, Gemm) and not blas_addresses(node, state):
        node.implementation = "rowwise" if device == dtypes.DeviceType.CPU and "rowwise" in impls else "pure"
        if device == dtypes.DeviceType.GPU and not libnode_is_device_code(node, state, sdfg):
            node.schedule = dtypes.ScheduleType.GPU_Device
        return

    if device == dtypes.DeviceType.CPU:
        # Shared with auto_optimize so the two paths cannot drift; it has the last word on the types it governs.
        if apply_cpu_library_parallelism(node, state, sdfg):
            return
        choice = cpu_library_choice(node, state, sequential, impls, has_libxsmm)
        if choice is not None:
            node.implementation = choice
            return
    elif isinstance(node, (Scan, Symmetrize)):
        node.implementation = gpu_scope_implementation(node, state, sdfg, impls)
        return
    # ``set_fast_implementations`` reads every Sequential GPU node as kernel code, but a host loop is Sequential
    # too and belongs to a device library call.
    if (
        device == dtypes.DeviceType.GPU
        and node.schedule == dtypes.ScheduleType.Sequential
        and not libnode_is_device_code(node, state, sdfg)
    ):
        fast = next((impl for impl in gpu_priority if impl in impls), None)
        if fast is not None:
            node.implementation = fast
            return
        # No device expansion: the pure map over ``GPU_Global`` operands must become a kernel, as validation
        # refuses host access to device storage.
        node.schedule = dtypes.ScheduleType.GPU_Device
    # A node pinned to a ``pure`` it does not have (``CopyLibraryNode``) takes its own default.
    if node.implementation == "pure" and "pure" not in impls:
        node.implementation = type(node).default_implementation

    if "pure" not in impls:
        return
    # The row-wise (ikj) expansion only: the plain ``pure`` nest reduces through one atomic per multiply-add.
    if "rowwise" in impls and matmul_extents_small(state, node, small_dim):
        node.implementation = "rowwise"


def allocated_unconditionally(sdfg: SDFG, name: str) -> bool:
    """Whether ``name`` is accessed anywhere outside every :class:`ConditionalBlock` of ``sdfg``."""
    for state in sdfg.states():
        if not any(node.data == name for node in state.data_nodes()):
            continue
        block = state
        while block is not None and not isinstance(block, ConditionalBlock):
            block = block.parent_graph
        if block is None:
            return True
    return False


def finalize_transient_storage(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Finalize the storage of a canonicalized (or vectorized) SDFG's transients, in place: map-body-local transients
    shrink to the box they name, length-1 arrays become scalars, small constant-size scratch moves to registers,
    independent top-level transients become persistent, map- and loop-scoped arrays are pinned to the stack on the CPU,
    and device transients come from the stream-ordered pool on the GPU."""
    # Codegen cannot emit a call in an OpenMP loop header; the analyses before needed it in the range.
    HoistLoopRangeCalls().apply_pass(sdfg, {})
    # Before the length-one conversion and storage inference, so a per-iteration transient lands on one element.
    ShrinkMapLocalTransients().apply_pass(sdfg, {})
    # single_element: ShrinkMapLocalTransients leaves (1, ..., 1) boxes, which are scalars as much as a (1,) array.
    ConvertLengthOneArraysToScalars(recursive=True, single_element=True).apply_pass(sdfg, {})
    if device == dtypes.DeviceType.GPU:
        place_kernel_buffers_on_device(sdfg)
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
            # ``__state->x`` is no valid OpenMP ``reduction`` variable: a size-1 accumulator stays a stack scalar.
            if desc.total_size == 1 and desc.storage != dtypes.StorageType.GPU_Global:
                desc.lifetime = dtypes.AllocationLifetime.State
                desc.storage = dtypes.StorageType.Register(force=True)
                continue
            # A buffer sized by what its guard tests may have a negative size at init, where the branch never runs.
            if not (
                allocated_unconditionally(sd, name)
                or all(symbolic.provably_nonnegative(dim, sd.facts()) for dim in desc.shape)
            ):
                desc.total_size = sympy.Max(0, desc.total_size)
            # ``__dace_num_threads`` is defined only once the program runs, after ``__dace_init``.
            if any(
                symbolic.NUM_THREADS_SYMBOL in {str(x) for x in symbolic.pystr_to_symbolic(str(dim)).free_symbols}
                for dim in desc.shape
            ):
                desc.lifetime = dtypes.AllocationLifetime.State
    if device == dtypes.DeviceType.CPU:
        force_map_scoped_registers(sdfg)
        force_loop_scoped_registers(sdfg)
        force_registers(sdfg)
    else:
        # A kernel's loop allocates per thread too; MoveArrayOutOfKernel lifts the forced arrays out of the kernel.
        force_loop_scoped_registers(
            sdfg,
            [
                sd
                for sd in sdfg.all_sdfgs_recursive()
                if sd.parent_nsdfg_node is not None
                and is_devicelevel_gpu(sd.parent_sdfg, sd.parent, sd.parent_nsdfg_node)
            ],
        )
        # Host loops of a device program allocate per iteration as on the CPU: their register arrays go on the stack.
        force_registers(sdfg, host_only=True)
        pool_gpu_transients(sdfg)


def is_kernel_scope(node: nodes.Node) -> bool:
    return isinstance(node, (nodes.MapEntry, nodes.MapExit)) and node.map.schedule in dtypes.GPU_SCHEDULES


def place_kernel_buffers_on_device(sdfg: SDFG) -> list[str]:
    """Move to device memory every host transient array that only kernels and copies touch: one a kernel writes or
    reads at its boundary while the access node sits in host code, as in the body of a map over a few blocks that
    runs sequentially on the host and launches the maps inside it (a buffer MapFusion leaves between such a kernel
    and the copy out of it). Left on the host, the kernel writes host memory (SC40 CUDA icon_velocity_tendencies:
    an illegal memory access). An array host code also reads or writes (a tasklet, a library node) stays.

    :returns: the moved arrays, as ``<sdfg name>.<array>``."""
    moved = []
    host_storages = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)
    for nested in sdfg.all_sdfgs_recursive():
        kernel_touched: OrderedSet = OrderedSet()
        host_touched: OrderedSet = OrderedSet()
        for state in nested.states():
            for node in state.data_nodes():
                desc = nested.arrays[node.data]
                if (
                    not desc.transient
                    or type(desc) is not dt.Array
                    or desc.storage not in host_storages
                    or is_devicelevel_gpu(nested, state, node)
                ):
                    continue
                for edge in (*state.in_edges(node), *state.out_edges(node)):
                    if edge.data.is_empty():
                        continue
                    if is_kernel_scope(edge.dst if edge.src is node else edge.src):
                        kernel_touched.add(node.data)
                        continue
                    # Through the host scopes around it, to the node at the far end of the memlet path
                    path = state.memlet_path(edge)
                    far = path[-1].dst if edge.src is node else path[0].src
                    if is_devicelevel_gpu(nested, state, far):
                        kernel_touched.add(node.data)
                    elif not isinstance(far, nodes.AccessNode):
                        host_touched.add(node.data)
        for name in kernel_touched:
            if name not in host_touched:
                nested.arrays[name].storage = dtypes.StorageType.GPU_Global
                moved.append(f"{nested.name}.{name}")
    return moved


def pool_gpu_transients(sdfg: SDFG) -> int:
    """Allocate every device transient array (views and External ones excepted) from the stream-ordered memory
    pool; returns how many containers were marked."""
    marked = 0
    for nested in sdfg.all_sdfgs_recursive():
        for desc in nested.arrays.values():
            if (
                desc.transient
                and isinstance(desc, dt.Array)
                and not isinstance(desc, dt.View)
                and desc.storage == dtypes.StorageType.GPU_Global
                and desc.lifetime != dtypes.AllocationLifetime.External
                and not desc.pool
            ):
                desc.pool = True
                marked += 1
    return marked


def outermost_map(state: SDFGState, node: nodes.Node) -> nodes.EntryNode | None:
    """The outermost scope entry around ``node`` in ``state``, or ``None`` at the state's top level."""
    scopes = state.scope_dict()
    outer = scopes[node]
    while outer is not None and scopes[outer] is not None:
        outer = scopes[outer]
    return outer


def force_map_scoped_registers(sdfg: SDFG, in_map: bool = False) -> int:
    """Pin every transient array allocated inside a map body to ``StorageType.Register(force=True)``: it lives for one
    iteration, and unforced, codegen moves a symbolically sized one to the heap."""
    homes: dict[str, set[tuple[SDFGState, nodes.EntryNode | None]]] = {}
    pinned = 0
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                homes.setdefault(node.data, set()).add((state, outermost_map(state, node)))
            elif isinstance(node, nodes.NestedSDFG):
                pinned += force_map_scoped_registers(node.sdfg, in_map or outermost_map(state, node) is not None)
    for name, desc in sdfg.arrays.items():
        if not (
            # A Scalar too: storage inference puts a scope-lifetime one in a loop on the heap.
            type(desc) in (dt.Array, dt.Scalar)
            and desc.transient
            and desc.lifetime == dtypes.AllocationLifetime.Scope
            and (
                desc.storage in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)
                or (desc.storage == dtypes.StorageType.Register and not desc.storage.force)
            )
        ):
            continue
        scopes = homes.get(name, set())
        if in_map or (len(scopes) == 1 and next(iter(scopes))[1] is not None):
            desc.storage = dtypes.StorageType.Register(force=True)
            pinned += 1
    return pinned


def force_registers(sdfg: SDFG, host_only: bool = False) -> int:
    """Pin every remaining ``Register`` container to ``Register(force=True)``, so it stays on the stack."""
    pinned = 0
    for sd in sdfg.all_sdfgs_recursive():
        loose = [
            name
            for name, desc in sd.arrays.items()
            if desc.storage == dtypes.StorageType.Register and not desc.storage.force
        ]
        if loose and host_only:
            device = {n.data for state in sd.states() for n in state.data_nodes() if is_devicelevel_gpu(sd, state, n)}
            loose = [name for name in loose if name not in device]
        for name in loose:
            sd.arrays[name].storage = dtypes.StorageType.Register(force=True)
            pinned += 1
    return pinned


def enclosing_loops(state: SDFGState) -> list[LoopRegion]:
    """The loops of ``state``'s own SDFG enclosing it, innermost first."""
    loops = []
    region = state.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            loops.append(region)
        region = region.parent_graph
    return loops


def force_loop_scoped_registers(sdfg: SDFG, sdfgs: list[SDFG] | None = None) -> int:
    """Take every transient array a sequential loop allocates once per iteration off the per-iteration heap: one of a
    size fixed on entry to the program is allocated once per call, every other one becomes a stack VLA."""
    moved = 0
    for sd in sdfg.all_sdfgs_recursive() if sdfgs is None else sdfgs:
        homes: dict[str, set[SDFGState]] = {}
        for state in sd.states():
            for node in state.data_nodes():
                homes.setdefault(node.data, set()).add(state)
        assigned = {lhs for edge in sd.all_interstate_edges() for lhs in edge.data.assignments}
        for name, desc in sd.arrays.items():
            if not (
                type(desc) in (dt.Array, dt.Scalar)
                and desc.transient
                and desc.lifetime == dtypes.AllocationLifetime.Scope
                and (
                    desc.storage in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)
                    or (desc.storage == dtypes.StorageType.Register and not desc.storage.force)
                )
                and name in homes
            ):
                continue
            states = sorted(homes[name], key=lambda st: st.label)
            # The deepest loop enclosing every access: the one whose iterations allocate the array.
            common = [loop for loop in enclosing_loops(states[0]) if all(loop in enclosing_loops(st) for st in states)]
            if not common:
                continue
            if any(outermost_map(st, n) is not None for st in states for n in st.data_nodes() if n.data == name):
                continue
            names = {str(free) for free in symbolic.pystr_to_symbolic(desc.total_size).free_symbols}
            if sd is sdfg and not names & assigned:
                desc.lifetime = dtypes.AllocationLifetime.SDFG
            else:
                desc.storage = dtypes.StorageType.Register(force=True)
            moved += 1
    return moved


def fed_by_producer_map(state: SDFGState, node: nodes.LibraryNode) -> bool:
    """Whether a map in ``state`` writes a transient operand of ``node`` (spmv's gathered ``x[cols]``)."""
    return any(
        isinstance(edge.src, nodes.AccessNode)
        and edge.src.desc(state.sdfg).transient
        and any(isinstance(write.src, nodes.MapExit) for write in state.in_edges(edge.src))
        for edge in state.in_edges(node)
    )


def vector_operands(state: SDFGState, node: nodes.LibraryNode) -> bool:
    """Whether every input of ``node`` is a vector, so a ``MatMul`` specializes to a ``Dot``."""
    return all(
        len([extent for extent in as_range(edge.data.subset).size() if extent != 1]) <= 1
        for edge in state.in_edges(node)
    )


def expand_gathered_dots(sdfg: SDFG) -> int:
    """Expand each DOT a map of its own state feeds, then fuse that map into the product: a gathered operand has a
    per-row extent no kernel may allocate, and the expanded register sum takes the gather into the multiply."""
    count = 0
    while True:
        found = next(
            (
                (node, state)
                for node, state in sdfg.all_nodes_recursive()
                if isinstance(node, (Dot, MatMul))
                and fed_by_producer_map(state, node)
                and (isinstance(node, Dot) or vector_operands(state, node))
            ),
            None,
        )
        if found is None:
            break
        node, state = found
        if isinstance(node, Dot):
            node.implementation = "pure_accumulate"
            count += 1
        before = OrderedSet(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
        node.expand(state.sdfg, state)
        if isinstance(node, Dot):
            for nested in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and n not in before]:
                InlineSDFG.apply_to(state.sdfg, nested_sdfg=nested, verify=False, save=False)
    if count:
        FuseMaps(allow_intermediates_with_symbolic_size=True, partial_reduce_fusion=True, validate=False).apply_pass(
            sdfg, {}
        )
        sdfg.apply_transformations_repeated([RedundantArray, RedundantSecondArray], validate=False)
    return count


def recompute_fuse_for_gpu(sdfg: SDFG) -> int:
    """Fuse producer->consumer map chains into one map recomputing each intermediate in registers (``OTFMapFusion`` to
    fixpoint): on the device the intermediates' global-memory round trips bound the time."""
    # The offload validates once at its end.
    return sdfg.apply_transformations_repeated(OTFMapFusion, validate=False, validate_all=False)


def unroll_short_loops(sdfg: SDFG) -> None:
    """Fully unroll every constant-trip loop of at most ``optimizer.canonicalization.unroll_limit`` iterations
    (``ShortLoopUnroll``), in every SDFG, the innermost first."""
    for nested in reversed(list(sdfg.all_sdfgs_recursive())):
        ShortLoopUnroll().apply_pass(nested, {})
    UniqueLoopIterators().apply_pass(sdfg, {})
    SymbolSSA().apply_pass(sdfg, {})


def offload_to_gpu(sdfg: SDFG, validate: bool = False, tile_programs: bool = False) -> None:
    """Move a canonicalized SDFG onto the GPU, in place. A map the caller made ``Sequential`` around maps stays a host
    loop, the maps inside it the kernels, and so does a map over a small extent (:func:`small_maps`)."""
    Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)
    unroll_short_loops(sdfg)
    # Inside the function: a caller's passes since canonicalize leave the pipeline's own cleanup stale.
    run_structural_cleanup(sdfg)
    # Before the offload makes each copy of a chain its own transfer: independent copies are one kernel.
    FuseCopyChains().apply_pass(sdfg, {})
    # Before the offload reads a map around a device-wide scan as a host loop of launches.
    BatchRowScans().apply_pass(sdfg, {})
    # Likewise a map around a matrix-vector product or dot: one batched vendor call instead
    BatchMapProducts().apply_pass(sdfg, {})
    expand_gathered_dots(sdfg)
    recompute_fuse_for_gpu(sdfg)
    # Before the offload reads a map around a small GEMM as a host loop of vendor calls.
    SpecializeSmallGemms(dtypes.DeviceType.GPU).apply_pass(sdfg, {})
    # While schedules are Default, and before the species maps unroll into nests the lane analysis does not enter:
    # a loop (or the whole body) over maps sharing their lane axes becomes one kernel with the loop inside.
    InterchangeLoopsIntoLaneMaps(target="gpu").apply_pass(sdfg, {})
    # Before the offload makes a map over a handful of species the kernel: the planes in its body are the kernels.
    from dace.transformation.passes.unroll_small_maps import UnrollSmallMaps

    UnrollSmallMaps().apply_pass(sdfg, {})
    # Before the offload makes a level map the kernel: its planes, along the contiguous axis, are the threads.
    if demote_uncoalesced_level_maps(sdfg):
        # The level loops are new to the launch rule: one that can run inside its planes makes the nest one launch.
        MoveLoopIntoMapGated(target="gpu").apply_pass(sdfg, {})
    # Before the offload decides what a kernel can own: a buffer sized per iteration by data is allocated by the host.
    BoundDataDependentExtents().apply_pass(sdfg, {})
    # A compaction's mask and rank buffers span its iteration space, which can exceed the device.
    ChunkStreamCompaction(block=GPU_BLOCK, schedule=dtypes.ScheduleType.Default).apply_pass(sdfg, {})
    apply_gpu_storage(sdfg)
    # A map over a few blocks is no kernel: sequential before the offload, it stays a host map, and the maps in its
    # body are the kernels, each with a full grid.
    for entry in small_maps(sdfg):
        entry.map.schedule = dtypes.ScheduleType.Sequential
    # One validation for the whole offload, at the end.
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    SimplifyPass(pass_options={"FuseStates.aggressive_merges": True}).apply_pass(sdfg, {})
    # A host block map whose body the offload left as one kernel map: one launch over both, not one per block.
    rejoin_block_maps(sdfg)
    # After the offload, which makes nested scopes Sequential, and before the block size, which skips a kernel holding
    # a thread-block map.
    if not tile_programs:
        PromoteWarpTiles().apply_pass(sdfg, {})
    select_gpu_device_block_size(sdfg)
    # Sized by the block just chosen: a scatter kernel covers enough elements per block for its private copy
    if not tile_programs:
        CoarsenScatterKernels().apply_pass(sdfg, {})
    if validate:
        sdfg.validate()


def assert_offloaded(sdfg: SDFG) -> None:
    """Raise unless ``sdfg`` has a ``GPU_Device`` map or a ``GPU_Global`` array: finalizing a host graph for the GPU
    would yield a CPU graph with a GPU label."""
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
    raise ValueError(
        f"finalize_for_target(sdfg, 'gpu') needs an already-offloaded SDFG, but '{sdfg.name}' has no "
        "GPU_Device map and no GPU_Global array. Offload is a separate step so passes can run between "
        "canonicalization and the device move: call offload_to_gpu(sdfg), or your own offload recipe, "
        "before finalizing."
    )


def assert_no_nested_parallel_maps(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Raise when a map or library node on the device's parallel schedule (``CPU_Multicore``, ``GPU_Device``) sits
    inside a map of the same schedule, across nested SDFGs: only top-level maps parallelize."""
    from dace.transformation.helpers import get_parent_map_and_loop_scopes

    parallel = dtypes.ScheduleType.GPU_Device if device == dtypes.DeviceType.GPU else dtypes.ScheduleType.CPU_Multicore
    for node, state in sdfg.all_nodes_recursive():
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
                kind = "map" if isinstance(node, nodes.MapEntry) else "library node"
                raise ValueError(
                    f"Nested {parallel.name}: {kind} '{label}' is nested inside {parallel.name} map "
                    f"'{scope.map.label}'. Inner maps / library nodes must not carry the device parallel "
                    f"schedule (only top-level maps parallelize) -- nesting emits stacked parallel regions."
                )


def codegen_aids(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Rewrites that reshape only the code a compiler sees: on the CPU, a parallel map's per-element ``if``/``else``
    becomes selects (``IfConvertParallelMapBodies``)."""
    if device == dtypes.DeviceType.CPU:
        IfConvertParallelMapBodies().apply_pass(sdfg, {})


def finalize_for_target(sdfg: SDFG, target: str = "cpu", validate: bool = False) -> SDFG:
    """Apply the performance tail to a canonicalized ``sdfg``, in place: fast library implementations (expanded at
    codegen) and transient storage."""
    if target not in TARGET_DEVICE:
        raise ValueError(f"target must be one of {sorted(TARGET_DEVICE)}; got {target!r}")
    device = TARGET_DEVICE[target]
    with equalized(sdfg):
        if device == dtypes.DeviceType.CPU:
            unroll_short_loops(sdfg)  # the GPU path ran it before its offload
        finalize_stages(sdfg, device)
    if validate:
        assert_no_nested_parallel_maps(sdfg, device)
        sdfg.validate()
    return sdfg


def finalize_stages(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """The stages of :func:`finalize_for_target`, for the ``device`` it resolved."""
    if device == dtypes.DeviceType.GPU:
        assert_offloaded(sdfg)
    # The states the last canonicalization stages left adjacent and fusible
    FuseStates(aggressive_merges=True, order_array_seeds=True).apply_pass(sdfg, {})

    # Schedules before the library selection, which follows them.
    if device == dtypes.DeviceType.CPU:
        # While canonicalization's maps are still Default: a demoted level map's plane maps are decided top-level.
        demote_strided_level_maps(sdfg)
        # A sequential loop over maps of one lane axis becomes one parallel map over it, the loop inside
        InterchangeLoopsIntoLaneMaps().apply_pass(sdfg, {})
    else:
        infer_types.set_default_schedule_and_storage_types(sdfg, None)

    # The target's specialization, before the library selection reads its schedules. Both are idempotent.
    if device == dtypes.DeviceType.GPU:
        gpu_specialize_offloaded(sdfg)
    else:
        cpu_specialize(sdfg, validate=False)
        codegen_aids(sdfg, device)
        # The CPU lowering of a lifted stream compaction: per-block counts and ordered per-block writes instead of
        # the mask and rank buffers. After the stage, whose tests read the device-neutral three-phase form.
        ChunkStreamCompaction().apply_pass(sdfg, {})

    # ``X = X + T`` onto the contraction writing ``T``: once on the lifted nodes, once on the BLAS nodes the
    # selection specializes them to, before either expands to a call.
    BetaFold().apply_pass(sdfg, {})
    transposes_as_blas_flags(sdfg)
    canonicalize_set_fast_implementations(sdfg, device)
    BetaFold().apply_pass(sdfg, {})
    refuse_silent_pure(sdfg, device)
    # Library nodes stay unexpanded: ``compile()`` expands each once, keeping one shape per computation until codegen.
    infer_types.infer_connector_types(sdfg)
    # Before storage is decided: one map-local name shared across states is otherwise one SDFG-wide buffer
    # the threads of a parallel map share.
    SplitMapLocalTransients().apply_pass(sdfg, {})
    finalize_transient_storage(sdfg, device)
    # Once storage and lifetimes are final, and before the parallel-region hoist reads dependences by name.
    if device == dtypes.DeviceType.CPU:
        MinimizeTransients().apply_pass(sdfg, {})

    # One OpenMP team per sequential loop instead of one per trip
    if device == dtypes.DeviceType.CPU:
        # Strip-mining takes the carried nests it can tile: one worksharing construct outside the carry.
        tiled = StripMineInterchange().apply_pass(sdfg, {})
        if HoistParallelRegion().apply_pass(sdfg, {}) is not None or tiled is not None:
            infer_types.infer_connector_types(sdfg)
    # The states specialization adds beside states they fuse with: a flag set ahead of a privatized loop, an
    # anti-dependence prologue, an outlined team, an if-conversion's branches, an expansion's transposes ahead of its
    # GEMM. The GPU lane barriers are side effects and keep their states apart.
    FuseStates(aggressive_merges=True, order_array_seeds=True).apply_pass(sdfg, {})
