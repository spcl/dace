# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC GPU offload: schedule assignment + host/device mirroring with dual-resident constants.

CloudSC-specific, so it lives with the corpus rather than in ``dace/transformation/passes``. Ported
from the velocity-tendencies ``OffloadVelocityToGPU`` (SC26-Layout-AD E7) and adapted to CloudSC's
block structure. Four phases, in order:

1. **Assign schedules.** CloudSC's outermost map is the per-block loop (``DO IBL = 1, NBLOCKS``),
   which must NOT become a kernel -- it orchestrates one kernel launch per block. So instead of
   "top-level map -> ``GPU_Device``" this pass offloads the outermost map *strictly inside* a block
   map. Block maps get ``Sequential``; every map below an offloaded map gets ``Sequential`` too.
2. **Classify constant data** (:func:`constant_offload_data`): non-transient arrays that are read-only,
   or written exactly once host-side and only read thereafter.
3. **Mirror kernel-side non-transients to ``gpu_<name>``** on ``GPU_Global``, with a copy-in head
   state and a copy-out terminal state. Constants are **dual-resident**: copied in once, never copied
   out, and the host original stays live and valid.
4. **Promote transients** to ``GPU_Global`` (scalars to ``Register``) and propagate that storage
   through NestedSDFG connector bindings. A transient read on a host interstate edge (a branch guard
   or edge assignment) is the exception: it is kept host and given a ``gpu_<name>`` mirror instead
   (:func:`mirror_host_needed_transients`), since a device buffer can't be read on the host edge.

``Sequential``, not ``Default``, for the block map: ``dace.sdfg.infer_types`` resolves a
``Default``-schedule map from the storage of its incident memlets (``SCOPEDEFAULT_SCHEDULE``,
``GPU_Global -> GPU_Device``). After phase 3 the block map's edges carry ``gpu_*`` GPU_Global data, so
``Default`` would silently promote the block loop onto the device.

Wired into :mod:`tests.corpus.cloudsc.pipelines` as an OPT-IN terminal phase (``offload=True``, for
EVERY variant, so all three offloaded recipes can be benchmarked and the fastest kept). Off by
default: ``canon_gpu`` stops before offload so the
graph stays CPU-runnable and every phase is numeric-checked on the host. The offload phase itself is
checked by ``validate()`` + CUDA code generation instead.
"""
import copy
from typing import Dict, FrozenSet, Iterable, Optional, Set, Tuple

import dace
from dace import data, dtypes, subsets, symbolic
from dace.config import Config
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.writeset_underapproximation import UnderapproximateWrites
from dace.sdfg.sdfg import InterstateEdge
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from dace.sdfg.state import SDFGState
from dace.transformation.passes.analysis.analysis import FindAccessNodes, StateReachability

#: Symbols whose appearance in a map's range marks it as the host-side per-block orchestrator. The
#: CloudSC Fortran driver loops ``DO IBL = 1, NBLOCKS``; the GPU SCC k-caching driver loops
#: ``DO JKGLO = 1, NGPTOT, NPROMA`` instead, so that frontend passes ``('ngptot', )``. A name signal is
#: needed because after canonicalization the block map is not distinguishable from a horizontal map by
#: shape alone.
BLOCK_MAP_SYMBOLS: Tuple[str, ...] = ('nblocks', )

CPU_STORAGES = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register)


def offload_cloudsc_to_gpu(sdfg: dace.SDFG,
                           block_map_symbols: Iterable[str] = BLOCK_MAP_SYMBOLS,
                           exclude_from_offload: Iterable[str] = ()) -> None:
    """Make ``sdfg`` GPU-compilable with the block loop kept on the host.

    :param sdfg: The CloudSC SDFG, canonicalized/parallelized (maps already formed).
    :param block_map_symbols: Range symbols marking the per-block orchestrator map (see
                              :data:`BLOCK_MAP_SYMBOLS`).
    :param exclude_from_offload: Non-transient array names that must stay host-only regardless of
                                 where they are accessed.
    """
    block_symbols = frozenset(block_map_symbols)
    pin_default_gpu_stream()
    # A length-1 input (e.g. ptsphy) read on BOTH host (per-block scalar inits) and device must become
    # a Scalar -> Register: passed into kernels by value and readable on the host, so it never needs a
    # host/device split. ``preserve_abi`` is required to reach the non-transient inputs at all -- left
    # clear, the pass skips non-transients entirely and ptsphy stays an Array. It stages rather than
    # rewrites them, so the SDFG signature is byte-identical afterwards: the caller still passes a
    # 1-element buffer for every program output (ngpblks), which gets its copy-out state.
    ConvertLengthOneArraysToScalars(preserve_abi=True, recursive=True).apply_pass(sdfg, {})
    symbolize_readonly_range_scalars(sdfg)
    assign_schedules(sdfg, block_symbols)
    mirror_nontransients_to_gpu(sdfg, frozenset(exclude_from_offload))
    mirror_host_needed_transients(sdfg)
    promote_transients_to_gpu(sdfg)
    propagate_gpu_storage_into_nested_sdfgs(sdfg)
    sdfg.validate()


# -- Phase 0: default stream and dynamic map ranges -----------------------------------------------


def pin_default_gpu_stream() -> None:
    """Force every launch and async transfer onto the default (null) stream.

    ``compiler.cuda.max_concurrent_streams = -1`` makes the CUDA target fill the ``gpu_streams``
    array with ``nullptr`` instead of creating streams
    (``dace/codegen/targets/experimental_cuda.py``, the ``== -1`` branch of the stream-allocation
    choice), so ``gpu_streams[0]`` IS the default stream and every ``cudaLaunchKernel`` /
    ``cudaMemcpyAsync`` the graph emits runs on it. Set here rather than left to the caller's
    environment so the offloaded graph has one stream regime wherever it is built.
    """
    Config.set('compiler', 'cuda', 'max_concurrent_streams', value=-1)


def readonly_range_scalars(graph: dace.SDFG) -> Set[str]:
    """Integer Scalars that a map range reads ONLY through a dynamic-range connector, never written.

    CloudSC's horizontal bounds ``kidia``/``kfdia`` arrive from the Fortran frontend as scalar
    program arguments, and every klon map reads them as a dynamic map range. That shape is what
    :func:`symbolize_readonly_range_scalars` rewrites; see there for why it has to go.

    Admitted only when the value is provably loop-invariant: no non-empty write edge anywhere (the
    empty in-edges the canon pipeline leaves behind are ordering, not data -- they carry no subset),
    and every non-empty read edge lands on a MapEntry's same-named dynamic-range connector. A single
    ordinary reader disqualifies the name, because turning it into a symbol would leave that reader
    without a container.
    """
    candidates = {
        name
        for name, desc in graph.arrays.items()
        if isinstance(desc, data.Scalar) and desc.dtype in (dace.int32, dace.int64)
    }
    if not candidates:
        return set()
    in_range: Set[str] = set()
    for state in graph.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                in_range |= {str(s) for s in node.map.range.free_symbols}
    candidates &= in_range

    for state in graph.states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and node.data in candidates):
                continue
            if any(not edge.data.is_empty() for edge in state.in_edges(node)):
                candidates.discard(node.data)
                continue
            for edge in state.out_edges(node):
                if edge.data.is_empty():
                    continue
                if not (isinstance(edge.dst, nodes.MapEntry) and edge.dst_conn == node.data):
                    candidates.discard(node.data)
                    break
    return candidates


def symbolize_readonly_range_scalars(sdfg: dace.SDFG) -> int:
    """Turn read-only dynamic map ranges into SDFG symbols. Returns how many were converted.

    WHY: a ``GPU_Device`` map whose range is a dynamic connector does not code-generate. The CUDA
    target emits one local per dynamic input at the launch site
    (``dace/codegen/targets/experimental_cuda.py``, the ``dyn_inputs`` loop that writes
    ``memlet_definition`` into ``callsite_stream``), and CloudSC's connectors are named after the
    very containers they read, so the declaration shadows its own initializer -- ``int kidia =
    kidia[0];``. Worse, a range symbol that reaches only the GRID expression and no kernel argument
    is dropped from the wrapper signature altogether, so the emitted ``.cu`` references an undefined
    ``kidia``. Measured on the canonicalized CloudSC graph: 100 ``identifier "kidia" is undefined``
    errors plus the whole host-side shadowing set, all of which disappear once the range is a symbol.

    Both are code-generator defects, not graph defects -- the graph is valid and the CPU target
    builds it. They live in ``dace/codegen``, outside what this pass may change, so the workaround
    is to deny the codegen the shape it mishandles: a symbolic range is passed into the kernel by
    the ordinary symbol path (exactly how ``klev``/``klon`` already arrive) and emits no launch-site
    locals at all.

    The rewrite is only sound for a name :func:`readonly_range_scalars` admits. Its access nodes are
    then left with nothing but empty (ordering) in-edges and no readers; a sink that orders nothing
    downstream is dropped with them. The descriptor becomes a symbol of the same name and dtype, so
    the caller still passes ``kidia=`` -- a symbol keyword rather than a scalar buffer.
    """
    converted = 0
    for graph in sdfg.all_sdfgs_recursive():
        names = readonly_range_scalars(graph)
        if not names:
            continue
        for state in graph.states():
            for node in list(state.nodes()):
                if not (isinstance(node, nodes.AccessNode) and node.data in names):
                    continue
                for edge in list(state.out_edges(node)):
                    if edge.data.is_empty():
                        continue
                    edge.dst.remove_in_connector(edge.dst_conn)
                    state.remove_edge(edge)
                if state.out_degree(node) > 0:
                    continue
                for edge in list(state.in_edges(node)):
                    state.remove_edge(edge)
                state.remove_node(node)
        for name in sorted(names):
            dtype = graph.arrays[name].dtype
            graph.remove_data(name, validate=False)
            graph.add_symbol(name, dtype)
            converted += 1
    return converted


# -- Phase 1: schedules -------------------------------------------------------------------------


def is_block_map(entry: nodes.MapEntry, block_symbols: FrozenSet[str]) -> bool:
    """A block map iterates over the block count, and stays on the host whatever its body holds.

    The body shape must not enter this decision: the canon pipelines flatten the block map's body to
    bare tasklets, so a "body holds a map or NestedSDFG" test reads the per-block orchestrator as a
    leaf compute map and offloads the whole loop as one grid.
    """
    return bool({str(s) for s in entry.map.range.free_symbols} & block_symbols)


def enclosed_by_kernel(node: nodes.Node, sdict: Dict, block_symbols: FrozenSet[str]) -> bool:
    """True iff some enclosing map of ``node`` was offloaded, i.e. is a non-block map."""
    parent = sdict[node]
    while parent is not None:
        if isinstance(parent, nodes.MapEntry) and not is_block_map(parent, block_symbols):
            return True
        parent = sdict[parent]
    return False


def assign_schedules(sdfg: dace.SDFG, block_symbols: FrozenSet[str], in_kernel: bool = False) -> None:
    """Offload the outermost non-block map; keep block maps and everything under a kernel sequential.

    ``in_kernel`` carries the "already inside a device scope" flag across NestedSDFG boundaries, where
    the per-state scope dict restarts -- a map at the top of an NSDFG that sits inside a kernel is
    device-level, not a fresh kernel.
    """
    for state in sdfg.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, (nodes.MapEntry, nodes.LibraryNode)):
                nested = in_kernel or enclosed_by_kernel(node, sdict, block_symbols)
                host = nested or (isinstance(node, nodes.MapEntry) and is_block_map(node, block_symbols))
                schedule = dtypes.ScheduleType.Sequential if host else dtypes.ScheduleType.GPU_Device
                if isinstance(node, nodes.MapEntry):
                    node.map.schedule = schedule
                else:
                    node.schedule = schedule
            elif isinstance(node, nodes.NestedSDFG):
                below = in_kernel or enclosed_by_kernel(node, sdict, block_symbols)
                assign_schedules(node.sdfg, block_symbols, below)


# -- Phase 2: constant classification ------------------------------------------------------------


def constant_offload_data(sdfg: dace.SDFG, candidates: Set[str]) -> Dict[str, Optional[SDFGState]]:
    """Which of ``candidates`` are constant, and where they are produced.

    Constant = the host copy is complete before the first device read and never invalidated
    afterwards, so the mirror can be copied in once and both copies stay valid (dual residency, no
    copy-out). Two admitted shapes:

    * **read-only** -- no write access node anywhere; maps to ``None``.
    * **write-once** -- exactly one writing access node, host-side, in a state that runs at most once,
      whose write *fully* covers the array under the write-set UNDER-approximation, with every read
      in a state reachable from it. Maps to that state.

    Under-approximation is the safe direction here: a partial or unprovable write leaves the name out
    of the result, and it is then mirrored with the normal copy-in/copy-out round trip.

    Reuses :class:`~dace.transformation.passes.analysis.analysis.FindAccessNodes` (read/write access
    nodes per state), :class:`~dace.transformation.passes.analysis.analysis.StateReachability`
    (write-before-read ordering) and
    :class:`~dace.sdfg.analysis.writeset_underapproximation.UnderapproximateWrites` (full-coverage
    proof).
    """
    access = FindAccessNodes().apply_pass(sdfg, {})[sdfg.cfg_id]
    reachable = StateReachability().apply_pass(sdfg, {})[sdfg.cfg_id]
    approximation = UnderapproximateWrites().apply_pass(sdfg, {})[sdfg.cfg_id].approximation
    acyclic = not sdfg.has_cycles()

    constants: Dict[str, Optional[SDFGState]] = {}
    for name in candidates:
        per_state = access[name]
        writers = [(state, node) for state, (_, writes) in per_state.items() for node in writes]
        if not writers:
            constants[name] = None
            continue
        if len(writers) != 1 or not acyclic:
            continue
        state, node = writers[0]
        # Runs at most once: directly in the root graph (not in a loop or branch region) and the root
        # graph itself has no back edges.
        if state.parent_graph is not sdfg:
            continue
        # A read in the producing state would be served by a mirror that does not exist yet.
        if per_state[state][0]:
            continue
        # A device-side write leaves the host copy stale, so copying host -> device would clobber it.
        if is_kernel_side(node, state, state.scope_dict()):
            continue
        if not fully_written(state, node, sdfg.arrays[name], approximation):
            continue
        readers = {s for s, (reads, _) in per_state.items() if reads}
        if not readers <= reachable[state]:
            continue
        constants[name] = state
    return constants


def fully_written(state: SDFGState, node: nodes.AccessNode, desc: data.Data, approximation: Dict) -> bool:
    """True iff the incoming edges provably write the whole of ``desc``.

    Two proofs, both under-approximating (an unprovable write just leaves the array out of the
    constant set and it round-trips):

    * one edge whose approximated subset covers the array on its own;
    * the edges together enumerate every element -- see :func:`elementwise_writes_cover`.
    """
    full = subsets.Range.from_array(desc)
    written = []
    for edge in state.in_edges(node):
        approximated = approximation.get(edge)
        if approximated is None or approximated.subset is None:
            continue  # unknown writes only ADD coverage, so dropping them stays under-approximate
        if approximated.subset.covers(full):
            return True
        written.append(approximated.subset)
    return elementwise_writes_cover(written, desc)


def elementwise_writes_cover(written: Iterable[subsets.Subset], desc: data.Data) -> bool:
    """True iff ``written`` is a set of constant single-element writes hitting every element.

    ``SubsetUnion.covers`` asks whether ONE member covers the argument, so N per-element writes --
    what an unrolled assignment loop leaves behind -- never prove they cover the array between them.
    Enumerating them does, exactly: collect the constant index tuples, and if the distinct in-bounds
    ones number the whole array they partition it. Restricted to single-element writes on purpose;
    the count is bounded by the number of edges, so nothing is enumerated that the graph does not
    already spell out. Anything else (symbolic index, multi-element subset, symbolic shape) refuses.
    """
    shape = [as_constant(dim) for dim in desc.shape]
    if any(dim is None for dim in shape):
        return False
    total = 1
    for dim in shape:
        total *= dim
    if total == 0:
        return False

    covered: Set[Tuple[int, ...]] = set()
    for subset in written:
        members = subset.subset_list if isinstance(subset, subsets.SubsetUnion) else [subset]
        for member in members:
            if member.num_elements() != 1 or member.dims() != len(shape):
                return False
            index = tuple(as_constant(begin) for begin, _, _ in member.ranges)
            if any(i is None for i in index):
                return False
            if any(not 0 <= i < dim for i, dim in zip(index, shape)):
                return False
            covered.add(index)
    return len(covered) == total


def as_constant(expression) -> Optional[int]:
    """``expression`` as a Python int, or None if it is not a compile-time constant."""
    value = symbolic.simplify(symbolic.pystr_to_symbolic(expression))
    return int(value) if value.is_Integer else None


# -- Phase 3: mirror kernel-side non-transients ---------------------------------------------------


def mirror_nontransients_to_gpu(sdfg: dace.SDFG, excluded: FrozenSet[str]) -> None:
    """Give every kernel-side non-transient Array a ``gpu_<name>`` sibling on ``GPU_Global``.

    Non-constants round-trip: copy-in in a new head state, copy-out in a new terminal state.
    Constants are dual-resident: copy-in only, placed right after their producer (or in the head state
    when read-only), and the host original is left untouched.
    """
    mirrored = arrays_needing_gpu_mirror(sdfg) - excluded
    if not mirrored:
        return
    constants = constant_offload_data(sdfg, mirrored)

    old_start = sdfg.start_block
    head = sdfg.add_state('gpu_copy_in', is_start_block=True)
    sdfg.add_edge(head, old_start, InterstateEdge())
    sinks = sdfg.sink_nodes()
    tail = sdfg.add_state('gpu_copy_out')
    for sink in sinks:
        sdfg.add_edge(sink, tail, InterstateEdge())
    # One copy-in state per constant producer, so the mirror is filled after the host value is final.
    after_producer = {
        state: sdfg.add_state_after(state, 'gpu_const_copy_in')
        for state in set(constants.values()) - {None}
    }
    copy_states = {head, tail} | set(after_producer.values())

    for name in sorted(mirrored):
        desc = sdfg.arrays[name]
        gpu_name = 'gpu_' + name
        assert gpu_name not in sdfg.arrays, f'{gpu_name!r} already exists; offload ran twice?'
        gpu_desc = copy.deepcopy(desc)
        gpu_desc.transient = True
        gpu_desc.storage = dtypes.StorageType.GPU_Global
        gpu_desc.lifetime = dtypes.AllocationLifetime.SDFG
        sdfg.add_datadesc(gpu_name, gpu_desc)

        producer = constants.get(name)
        copy_in = head if producer is None else after_producer[producer]
        add_full_copy(copy_in, name, desc, gpu_name)
        if name not in constants:
            add_full_copy(tail, gpu_name, gpu_desc, name)

    retargeted = set()
    for state in sdfg.states():
        if state in copy_states:
            continue
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in mirrored and is_kernel_side(node, state, sdict):
                node.data = 'gpu_' + node.data
                retargeted.add(id(node))

    for state in sdfg.states():
        if state in copy_states:
            continue
        sdict = state.scope_dict()
        for edge in state.edges():
            if edge.data is None or edge.data.data not in mirrored:
                continue
            if edge_is_kernel_side(edge, state, sdict, retargeted):
                edge.data.data = 'gpu_' + edge.data.data


def add_full_copy(state: SDFGState, src: str, src_desc: data.Data, dst: str) -> None:
    """A whole-array AccessNode -> AccessNode copy. Fresh Memlet per edge."""
    state.add_edge(state.add_read(src), None, state.add_write(dst), None, Memlet.from_array(src, src_desc))


def arrays_needing_gpu_mirror(sdfg: dace.SDFG) -> Set[str]:
    """Non-transient CPU-storage Arrays that are actually read/written inside a kernel. Host-only
    arrays are left alone -- including those that merely *feed* a NestedSDFG but are used only on the
    host inside it (which the ``is_kernel_side`` boundary test would wrongly flag)."""
    candidates = {
        name
        for name, desc in sdfg.arrays.items()
        if isinstance(desc, data.Array) and not desc.transient and desc.storage in CPU_STORAGES
    }
    return candidates & device_touched_names(sdfg) if candidates else set()


def is_device_boundary(node: nodes.Node) -> bool:
    """A Map/Library boundary node scheduled on ``GPU_Device`` -- data crossing it is device data."""
    if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
        return node.map.schedule == dtypes.ScheduleType.GPU_Device
    if isinstance(node, nodes.LibraryNode):
        return node.schedule == dtypes.ScheduleType.GPU_Device
    return False


def touches_device(node: nodes.Node, state: SDFGState, sdict: Dict, in_kernel: bool = False) -> bool:
    """True iff ``node`` executes on the device: already inside a kernel (``in_kernel``, carried across
    NestedSDFG boundaries), inside a ``GPU_Device`` scope, or staged at top level directly against a
    ``GPU_Device`` map/library boundary.

    ``in_kernel`` is essential and easy to miss: ``scope_dict`` restarts per NestedSDFG, and
    :func:`assign_schedules` marks maps *below* a kernel ``Sequential``. So a node inside an NSDFG that
    sits inside a kernel has no ``GPU_Device`` scope above it *within its own state* and would look
    host-side to a local-only test.
    """
    if in_kernel:
        return True
    parent = sdict[node]
    while parent is not None:
        if isinstance(parent, nodes.MapEntry) and parent.map.schedule == dtypes.ScheduleType.GPU_Device:
            return True
        parent = sdict[parent]
    if not isinstance(node, nodes.AccessNode):
        return False
    return (any(is_device_boundary(e.src) for e in state.in_edges(node))
            or any(is_device_boundary(e.dst) for e in state.out_edges(node)))


def device_touched_per_sdfg(graph: dace.SDFG,
                            in_kernel: bool = False,
                            out: Optional[Dict[int, Set[str]]] = None) -> Dict[int, Set[str]]:
    """Map ``id(sdfg) -> local data names that reach a ``GPU_Device`` computation``, for ``graph`` and
    every SDFG below it.

    Computed top-down in one walk so the "already inside a kernel" flag reaches nested SDFGs -- a
    per-graph query cannot recover it (see :func:`touches_device`). A name is device-touched when it is
    accessed on the device or bound to a NestedSDFG connector that is itself device-touched. Precise
    where :func:`is_kernel_side` over-approximates: an array that only feeds a NestedSDFG but is used
    solely on the host inside it is not device-touched, so it is neither mirrored nor promoted.

    Edges are scanned as well as nodes: an array can reach a kernel purely through a memlet path
    (``pin -> block_map -> device_map -> tasklet``) without any AccessNode of its own inside a device
    scope, and a node-only scan would call it host-only.
    """
    if out is None:
        out = {}
    touched: Set[str] = set()
    for state in graph.states():
        sdict = state.scope_dict()
        for edge in state.edges():
            if edge.data is None or edge.data.data is None:
                continue
            if any(
                    is_device_boundary(end) or touches_device(end, state, sdict, in_kernel)
                    for end in (edge.src, edge.dst)):
                touched.add(edge.data.data)
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if touches_device(node, state, sdict, in_kernel):
                    touched.add(node.data)
            elif isinstance(node, nodes.NestedSDFG):
                below = in_kernel or touches_device(node, state, sdict)
                device_touched_per_sdfg(node.sdfg, below, out)
                inner = out[id(node.sdfg)]
                for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
                    if edge.data is None or edge.data.data is None:
                        continue
                    conn = edge.dst_conn if edge.dst is node else edge.src_conn
                    if conn in inner:
                        touched.add(edge.data.data)
    out[id(graph)] = touched
    return out


def device_touched_names(graph: dace.SDFG) -> Set[str]:
    """Device-touched local names of ``graph`` itself, treating it as a top-level (host) SDFG."""
    return device_touched_per_sdfg(graph)[id(graph)]


def device_written_per_sdfg(graph: dace.SDFG,
                            in_kernel: bool = False,
                            out: Optional[Dict[int, Set[str]]] = None) -> Dict[int, Set[str]]:
    """Names WRITTEN by device code, per SDFG, following NestedSDFG connector bindings.

    The write-side twin of :func:`device_touched_per_sdfg`, and whole-SDFG for the same reason: a
    per-graph tasklet scan cannot see a write performed inside a NestedSDFG, which is how a
    device-written array slips past the "two masters" guard in :func:`mirror_host_needed_transients`
    and gets a host mirror whose master is never filled.
    """
    if out is None:
        out = {}
    written: Set[str] = set()
    for state in graph.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                if state.in_degree(node) == 0:
                    continue
                if (touches_device(node, state, sdict, in_kernel)
                        or any(is_device_boundary(edge.src) for edge in state.in_edges(node))):
                    written.add(node.data)
            elif isinstance(node, nodes.NestedSDFG):
                below = in_kernel or touches_device(node, state, sdict)
                device_written_per_sdfg(node.sdfg, below, out)
                inner = out[id(node.sdfg)]
                # Only out-edges: a NestedSDFG writes an outer array through an output connector.
                for edge in state.out_edges(node):
                    if edge.data is not None and edge.data.data is not None and edge.src_conn in inner:
                        written.add(edge.data.data)
    out[id(graph)] = written
    return out


def sdfgs_inside_kernels(graph: dace.SDFG, in_kernel: bool = False, out: Optional[Set[int]] = None) -> Set[int]:
    """``id(sdfg)`` for every SDFG below ``graph`` that executes inside a ``GPU_Device`` kernel. Its
    top-level nodes are device nodes even though ``scope_dict`` shows no enclosing scope."""
    if out is None:
        out = set()
    if in_kernel:
        out.add(id(graph))
    for state in graph.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                sdfgs_inside_kernels(node.sdfg, in_kernel or touches_device(node, state, sdict), out)
    return out


def device_facing(node: nodes.Node, state: SDFGState, sdict: Dict) -> bool:
    """True iff ``node`` runs device code, or is a NestedSDFG whose interior may. The NestedSDFG case is
    deliberately conservative -- its inner storage is settled by
    :func:`propagate_gpu_storage_into_nested_sdfgs`, not here."""
    if isinstance(node, nodes.NestedSDFG) or is_device_boundary(node):
        return True
    return touches_device(node, state, sdict)


def is_kernel_side(node: nodes.AccessNode, state: SDFGState, sdict: Dict) -> bool:
    """True iff this access is produced or consumed on the device, resolved along the memlet PATH so a
    map's pass-through connector cannot hide the real endpoint.

    Sitting inside a scope is NOT the test. The canon pipelines put bare host tasklets straight into the
    Sequential block map's body, so an any-scope test retargets those host reads onto the device mirror
    and reintroduces the very GPU_Global-read-on-host it was meant to prevent.
    """
    if touches_device(node, state, sdict):
        return True
    if any(device_facing(state.memlet_path(e)[0].src, state, sdict) for e in state.in_edges(node)):
        return True
    return any(device_facing(state.memlet_path(e)[-1].dst, state, sdict) for e in state.out_edges(node))


def edge_is_kernel_side(edge, state: SDFGState, sdict: Dict, retargeted: Set[int]) -> bool:
    """Kernel-side iff an endpoint was retargeted, or the memlet path this edge lies on ends in device
    code. Edges between two host-side nodes keep the original name."""
    if id(edge.src) in retargeted or id(edge.dst) in retargeted:
        return True
    path = state.memlet_path(edge)
    return device_facing(path[0].src, state, sdict) or device_facing(path[-1].dst, state, sdict)


# -- Phase 4: transient promotion and NSDFG storage propagation -----------------------------------


def interstate_read_arrays(graph: dace.SDFG) -> Set[str]:
    """Transient Array names read on any interstate edge within ``graph``. Interstate edges evaluate
    on the host, so these must keep a host-readable copy -- they cannot be promoted to ``GPU_Global``."""
    names: Set[str] = set()
    arrays = set(graph.arrays)
    for edge in graph.all_interstate_edges():
        for name in edge.data.free_symbols & arrays:
            desc = graph.arrays[name]
            if isinstance(desc, data.Array) and desc.transient:
                names.add(name)
    return names


def tasklet_accessed_arrays(graph: dace.SDFG, in_kernel: bool, device_side: bool, writing: bool) -> Set[str]:
    """Transient Array names a Tasklet on the requested side of the host/device split accesses.

    A host-side access of either direction pins the master to the host. Only ``writing=True,
    device_side=True`` carries the extra meaning of disqualifying an array from mirroring -- two
    masters cannot both be authoritative, which a device read does not create.

    The tasklet is the far end of the memlet PATH, not the AccessNode's neighbour: a tasklet inside a
    map scope reaches the array through the map's pass-through connector, so an adjacency test misses
    it entirely.

    ``in_kernel`` comes from :func:`sdfgs_inside_kernels`; inside a kernel every tasklet is device
    code no matter what the per-state scope dict says (it restarts at each NestedSDFG).
    """
    names: Set[str] = set()
    for state in graph.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = graph.arrays.get(node.data)
            if not (isinstance(desc, data.Array) and desc.transient):
                continue
            for edge in (state.in_edges(node) if writing else state.out_edges(node)):
                path = state.memlet_path(edge)
                tasklet = path[0].src if writing else path[-1].dst
                if not isinstance(tasklet, nodes.Tasklet):
                    continue
                if (in_kernel or touches_device(tasklet, state, sdict)) is device_side:
                    names.add(node.data)
                    break
    return names


def host_pinned_arrays(graph: dace.SDFG, in_kernel: bool) -> Set[str]:
    """Transient Arrays in ``graph`` that must keep a host-resident master, because host code accesses
    them: on an interstate edge, or from a bare tasklet reading or writing. The master cannot move to
    ``GPU_Global``; device users get a ``gpu_<name>`` mirror instead."""
    return (interstate_read_arrays(graph) | tasklet_accessed_arrays(graph, in_kernel, False, writing=True)
            | tasklet_accessed_arrays(graph, in_kernel, False, writing=False))


def mirror_host_needed_transients(sdfg: dace.SDFG) -> int:
    """Dual-resident a transient that host code touches AND a kernel uses.

    Two ways a transient gets pinned to the host, both instances of one rule -- a host-side access
    means the master must stay host-readable:

    * **read on a host interstate edge**: ``imelt`` (``imelt[i] = -99`` then a ``jn = imelt[0]``
      guard) and ``zvqx`` (``zvqx[k] > 0`` guards). The edge would read device memory.
    * **written by a bare host tasklet**: ``iphase`` in the canon pipelines, whose ``(5, )`` phase
      classification is LICM-hoisted to a host init and then read only inside the kernel. The write
      would store to device memory.

    The fix mirrors :func:`mirror_nontransients_to_gpu` without a copy-out -- keep the host array
    (writers, guards and host readers stay on it), add a ``gpu_<name>`` ``GPU_Global`` sibling, copy
    host -> device right after EVERY writer state, and retarget the kernel-side reads to the mirror.
    Copy-after-every-writer is correct in the cyclic per-block loop: on any path the copy that follows
    the last write fills the mirror.

    An array the device also *writes* is skipped: the mirror would be a second master and the host
    copy would silently go stale. Those stay unmirrored and fail validation loudly rather than
    miscompiling. Returns the number of transients mirrored.
    """
    count = 0
    per_graph = device_touched_per_sdfg(sdfg)
    written_per_graph = device_written_per_sdfg(sdfg)
    inside = sdfgs_inside_kernels(sdfg)
    for graph in sdfg.all_sdfgs_recursive():
        in_kernel = id(graph) in inside
        device_written = written_per_graph.get(id(graph), set())
        mirrored = (host_pinned_arrays(graph, in_kernel) & per_graph.get(id(graph), set())) - device_written
        if not mirrored:
            continue
        copy_states: Set[SDFGState] = set()
        for name in sorted(mirrored):
            desc = graph.arrays[name]
            gpu_name = 'gpu_' + name
            if gpu_name in graph.arrays:
                continue
            gpu_desc = copy.deepcopy(desc)
            gpu_desc.transient = True
            gpu_desc.storage = dtypes.StorageType.GPU_Global
            gpu_desc.lifetime = dtypes.AllocationLifetime.SDFG
            graph.add_datadesc(gpu_name, gpu_desc)
            writer_states = [
                state for state in graph.states() if any(
                    isinstance(n, nodes.AccessNode) and n.data == name and state.in_edges(n) for n in state.nodes())
            ]
            for wstate in writer_states:
                cstate = wstate.parent_graph.add_state_after(wstate, f'gpu_copy_{name}')
                add_full_copy(cstate, name, desc, gpu_name)
                copy_states.add(cstate)
        for name in sorted(mirrored):
            retarget_kernel_side_reads(graph, name, 'gpu_' + name, copy_states)
        count += len(mirrored)
    return count


def retarget_kernel_side_reads(graph: dace.SDFG, name: str, gpu_name: str, skip: Set[SDFGState]) -> None:
    """Point kernel-side accesses of ``name`` at ``gpu_name`` (mirrors the retarget in
    :func:`mirror_nontransients_to_gpu`). Host writers and host interstate/tasklet readers are left on
    ``name``; the host -> device copy states in ``skip`` are untouched so the copy source stays host.

    Classified per EDGE, not per node: a top-level AccessNode can fan out to both a device kernel and a
    host tasklet from the same scope, and a node-granularity test would drag the host edge's read onto
    the device mirror along with the legitimately kernel-side one. A node with mixed edges is therefore
    SPLIT -- a second ``gpu_name`` AccessNode takes only the device-facing edges, the original keeps the
    rest -- instead of renamed whole.
    """
    retargeted: Set[int] = set()
    for state in graph.states():
        if state in skip:
            continue
        sdict = state.scope_dict()
        for node in list(state.nodes()):
            if not (isinstance(node, nodes.AccessNode) and node.data == name):
                continue
            # Empty memlets are happens-before ordering, not data: they carry no subset, so they
            # neither vote on the device/host split nor move. Naming one would make it non-empty, and
            # the copy-insertion pass -- which skips empty memlets -- would then materialize it into a
            # real copy with subsets derived from the shapes.
            in_edges = [e for e in state.in_edges(node) if not e.data.is_empty()]
            out_edges = [e for e in state.out_edges(node) if not e.data.is_empty()]
            in_device = [device_facing(state.memlet_path(e)[0].src, state, sdict) for e in in_edges]
            out_device = [device_facing(state.memlet_path(e)[-1].dst, state, sdict) for e in out_edges]
            if not any(in_device) and not any(out_device):
                continue
            if all(in_device) and all(out_device):
                node.data = gpu_name
                retargeted.add(id(node))
                continue
            mirror = state.add_access(gpu_name)
            retargeted.add(id(mirror))
            for edge, device in zip(in_edges, in_device):
                if not device:
                    continue
                memlet = copy.deepcopy(edge.data)
                memlet.data = gpu_name
                state.remove_edge(edge)
                state.add_edge(edge.src, edge.src_conn, mirror, edge.dst_conn, memlet)
            for edge, device in zip(out_edges, out_device):
                if not device:
                    continue
                memlet = copy.deepcopy(edge.data)
                memlet.data = gpu_name
                state.remove_edge(edge)
                state.add_edge(mirror, edge.src_conn, edge.dst, edge.dst_conn, memlet)
    for state in graph.states():
        if state in skip:
            continue
        sdict = state.scope_dict()
        for edge in state.edges():
            if edge.data is None or edge.data.data != name:
                continue
            if edge_is_kernel_side(edge, state, sdict, retargeted):
                edge.data.data = gpu_name


def promote_transients_to_gpu(sdfg: dace.SDFG) -> None:
    """Transient Arrays -> ``GPU_Global``; Scalars -> ``Register`` at ``State`` lifetime.

    Non-transient scalars are included because an NSDFG's inner descriptor shadows its outer binding,
    and a CPU_Heap inner scalar fed to a kernel trips DaCe's ``IllegalCopy`` dispatch. Leaving them
    host instead is measurably worse, not merely different: on the canonicalized CloudSC graph it
    turns 2 host compile errors into 742 (``invalid types 'int[int]'``, ``int`` vs ``const int*``),
    because the callee side expects the by-value form the Register storage produces.

    ``State`` rather than the default ``Scope`` lifetime: a Register scalar is a C++ local, and at
    Scope lifetime the declaration lands inside the braces of the dataflow component that writes it.
    CloudSC's LICM preheaders put the write (``neg_ydcst_rlvtt = -ydcst_rlvtt``) and the kernel
    launch that consumes it in two components of the SAME state, so the launch referenced a name
    whose block had already closed -- ``'neg_ydcst_rlvtt' was not declared in this scope``. State
    lifetime hoists the declaration to the state's own block, which spans both components.
    """
    per_graph = device_touched_per_sdfg(sdfg)
    inside = sdfgs_inside_kernels(sdfg)
    for graph in sdfg.all_sdfgs_recursive():
        host_needed = host_pinned_arrays(graph, id(graph) in inside)
        device = per_graph.get(id(graph), set())
        for name, desc in graph.arrays.items():
            if desc.storage not in CPU_STORAGES:
                continue
            if isinstance(desc, data.Array) and desc.transient:
                # Only device-used transients go on the GPU. A host-only transient stays host (else
                # its host readers fault); a host-pinned one is kept host + mirrored instead.
                if name not in device or name in host_needed:
                    continue
                desc.storage = dtypes.StorageType.GPU_Global
            elif isinstance(desc, data.Scalar):
                desc.storage = dtypes.StorageType.Register
                desc.lifetime = dtypes.AllocationLifetime.State


def propagate_gpu_storage_into_nested_sdfgs(sdfg: dace.SDFG) -> None:
    """Give an NSDFG's inner descriptor the ``GPU_Global`` storage of its outer binding, except where
    a host interstate edge reads that name -- interstate edges evaluate on the host, and the validator
    rejects device data there."""
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            host_only = names_used_on_interstate_edges(node.sdfg)
            for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
                if edge.data is None or edge.data.data is None:
                    continue
                outer = state.sdfg.arrays.get(edge.data.data)
                if outer is None or outer.storage != dtypes.StorageType.GPU_Global:
                    continue
                conn = edge.dst_conn if edge.dst is node else edge.src_conn
                if conn is None or conn in host_only:
                    continue
                inner = node.sdfg.arrays.get(conn)
                if isinstance(inner, data.Array):
                    inner.storage = dtypes.StorageType.GPU_Global
            propagate_gpu_storage_into_nested_sdfgs(node.sdfg)


def names_used_on_interstate_edges(sdfg: dace.SDFG) -> Set[str]:
    """Data names referenced by any interstate edge in ``sdfg`` or its descendants."""
    names: Set[str] = set()
    for graph in sdfg.all_sdfgs_recursive():
        arrays = set(graph.arrays)
        for edge in graph.all_interstate_edges():
            names |= edge.data.free_symbols & arrays
    return names
