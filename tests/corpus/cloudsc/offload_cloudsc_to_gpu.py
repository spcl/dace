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
   through NestedSDFG connector bindings.

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
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.writeset_underapproximation import UnderapproximateWrites
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import SDFGState
from dace.transformation.passes.analysis.analysis import FindAccessNodes, StateReachability

#: Symbols whose appearance in a map's range marks it as the host-side per-block orchestrator. The
#: CloudSC Fortran driver loops ``DO IBL = 1, NBLOCKS``; the GPU SCC k-caching driver loops
#: ``DO JKGLO = 1, NGPTOT, NPROMA`` instead, so that frontend passes ``('ngptot', )``. A name signal is
#: needed because after canonicalization the block map is not distinguishable from a horizontal map by
#: shape alone -- but it is only trusted together with the structural guard in :func:`is_block_map`.
BLOCK_MAP_SYMBOLS: Tuple[str, ...] = ('nblocks', )

CPU_STORAGES = (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register)

BOUNDARY_NODE_TYPES = (nodes.MapEntry, nodes.MapExit, nodes.NestedSDFG)


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
    assign_schedules(sdfg, block_symbols)
    mirror_nontransients_to_gpu(sdfg, frozenset(exclude_from_offload))
    promote_transients_to_gpu(sdfg)
    propagate_gpu_storage_into_nested_sdfgs(sdfg)
    sdfg.validate()


# -- Phase 1: schedules -------------------------------------------------------------------------


def encloses_compute(entry: nodes.MapEntry, state: SDFGState) -> bool:
    """True iff ``entry``'s body holds a map or a NestedSDFG -- i.e. there is something inside it to
    offload. Without this guard a leaf map whose range happens to mention a block symbol would be
    demoted to the host with nothing taking its place on the device."""
    body = state.scope_subgraph(entry, include_entry=False, include_exit=False)
    return any(isinstance(n, (nodes.MapEntry, nodes.NestedSDFG)) for n in body.nodes())


def is_block_map(entry: nodes.MapEntry, state: SDFGState, block_symbols: FrozenSet[str]) -> bool:
    """A block map iterates over the block count AND encloses further compute."""
    if not {str(s) for s in entry.map.range.free_symbols} & block_symbols:
        return False
    return encloses_compute(entry, state)


def enclosed_by_kernel(node: nodes.Node, state: SDFGState, sdict: Dict, block_symbols: FrozenSet[str]) -> bool:
    """True iff some enclosing map of ``node`` was offloaded, i.e. is a non-block map."""
    parent = sdict[node]
    while parent is not None:
        if isinstance(parent, nodes.MapEntry) and not is_block_map(parent, state, block_symbols):
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
                nested = in_kernel or enclosed_by_kernel(node, state, sdict, block_symbols)
                host = nested or (isinstance(node, nodes.MapEntry) and is_block_map(node, state, block_symbols))
                schedule = dtypes.ScheduleType.Sequential if host else dtypes.ScheduleType.GPU_Device
                if isinstance(node, nodes.MapEntry):
                    node.map.schedule = schedule
                else:
                    node.schedule = schedule
            elif isinstance(node, nodes.NestedSDFG):
                below = in_kernel or enclosed_by_kernel(node, state, sdict, block_symbols)
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
            if edge_is_kernel_side(edge, sdict, retargeted):
                edge.data.data = 'gpu_' + edge.data.data


def add_full_copy(state: SDFGState, src: str, src_desc: data.Data, dst: str) -> None:
    """A whole-array AccessNode -> AccessNode copy. Fresh Memlet per edge."""
    state.add_edge(state.add_read(src), None, state.add_write(dst), None, Memlet.from_array(src, src_desc))


def arrays_needing_gpu_mirror(sdfg: dace.SDFG) -> Set[str]:
    """Non-transient CPU-storage Arrays with at least one kernel-side access node. Host-only arrays
    (touched solely by top-level tasklets) are left alone."""
    candidates = {
        name
        for name, desc in sdfg.arrays.items()
        if isinstance(desc, data.Array) and not desc.transient and desc.storage in CPU_STORAGES
    }
    if not candidates:
        return set()
    needed = set()
    for state in sdfg.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in candidates and is_kernel_side(node, state, sdict):
                needed.add(node.data)
    return needed


def is_kernel_side(node: nodes.AccessNode, state: SDFGState, sdict: Dict) -> bool:
    """Inside any scope, or at top level but wired to a map/NSDFG boundary. Block maps count: data
    entering a block map still reaches the kernels nested inside it."""
    if sdict[node] is not None:
        return True
    if any(isinstance(e.src, BOUNDARY_NODE_TYPES) for e in state.in_edges(node)):
        return True
    return any(isinstance(e.dst, BOUNDARY_NODE_TYPES) for e in state.out_edges(node))


def edge_is_kernel_side(edge, sdict: Dict, retargeted: Set[int]) -> bool:
    """Kernel-side iff an endpoint was retargeted, the edge sits inside a scope, or it touches a scope
    boundary. Edges between two host-side nodes keep the original name."""
    if id(edge.src) in retargeted or id(edge.dst) in retargeted:
        return True
    if sdict[edge.src] is not None or sdict[edge.dst] is not None:
        return True
    return isinstance(edge.src, BOUNDARY_NODE_TYPES) or isinstance(edge.dst, BOUNDARY_NODE_TYPES)


# -- Phase 4: transient promotion and NSDFG storage propagation -----------------------------------


def promote_transients_to_gpu(sdfg: dace.SDFG) -> None:
    """Transient Arrays -> ``GPU_Global``; Scalars -> ``Register``. Non-transient scalars are included
    because an NSDFG's inner descriptor shadows its outer binding, and a CPU_Heap inner scalar fed to a
    kernel trips DaCe's ``IllegalCopy`` dispatch."""
    for graph in sdfg.all_sdfgs_recursive():
        for desc in graph.arrays.values():
            if desc.storage not in CPU_STORAGES:
                continue
            if isinstance(desc, data.Array) and desc.transient:
                desc.storage = dtypes.StorageType.GPU_Global
            elif isinstance(desc, data.Scalar):
                desc.storage = dtypes.StorageType.Register


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
