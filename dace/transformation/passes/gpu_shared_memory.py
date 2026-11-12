# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Passes that place the shared memory of GPU kernels, run by the CUDA code generator before code generation.

Every transient ``GPU_Shared`` container of a kernel is placed either in static shared memory (declared as a
``__shared__`` array) or in dynamic shared memory (a flat buffer sized at kernel launch). The choice is the container's
``StorageType.GPU_Shared(dynamic=...)`` attribute, or, if left to the code generator, whether it still fits in the
static shared memory of a thread-block. Dynamic containers become views of a flat ``uint8`` buffer at their offsets, so
the layout of dynamic shared memory is part of the SDFG.

The passes run in this order: ``PrivatizeKernelSharedMemory``, ``LowerDynamicMapState``,
``SharedMemoryInterference``, ``PlanSharedMemory``, ``LowerDynamicSharedMemory``.
"""

import copy
import itertools
import warnings
from dataclasses import dataclass, field

import networkx as nx

from dace import config, dtypes, properties, subsets, symbolic
from dace import data as dt
from dace.codegen import common
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.scope import THREAD_SCHEDULES, is_in_scope, is_thread_block_level, kernel_scope
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import gpu_helpers
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import StateReachability

#: Alignment of every container in dynamic shared memory, which suffices for all types up to 16-byte vectors
DYNAMIC_SHARED_MEMORY_ALIGNMENT = 16

#: Name of the symbol that holds the offset of a nested SDFG's part of dynamic shared memory, if not a constant
DYNAMIC_SHARED_MEMORY_BASE = "__dace_dynsmem_base"

#: Name of the flat dynamic shared memory buffers
DYNAMIC_SHARED_MEMORY_BUFFER = "__dace_dynsmem"


def _aligned(size: int) -> int:
    """The bytes a static shared memory container of ``size`` bytes occupies (they are aligned like dynamic ones)."""
    return -(-size // DYNAMIC_SHARED_MEMORY_ALIGNMENT) * DYNAMIC_SHARED_MEMORY_ALIGNMENT


def _nested_sdfg_nodes(sdfg: SDFG) -> list[nodes.NestedSDFG]:
    """Returns the nested SDFG nodes directly in ``sdfg``, in a deterministic order."""
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]


def _max(a: symbolic.SymbolicType, b: symbolic.SymbolicType) -> symbolic.SymbolicType:
    """The larger of two (symbolic) sizes."""
    if not symbolic.issymbolic(a) and not symbolic.issymbolic(b):
        return max(a, b)
    return symbolic.sympy.Max(symbolic.pystr_to_symbolic(a), symbolic.pystr_to_symbolic(b))


def _kernel_is_device_map(sdfg: SDFG, state: SDFGState, node: nodes.Node) -> bool:
    """Whether the kernel around a node is a ``GPU_Device`` scope (a persistent ``GPU_Persistent`` one synchronizes
    its states differently)."""
    found = kernel_scope(sdfg, state, node)
    return found is not None and found[1].schedule == dtypes.ScheduleType.GPU_Device


def is_shared_container(desc: dt.Data) -> bool:
    """
    Returns whether a data descriptor is a shared memory container that the passes of this module place, i.e., a
    transient ``GPU_Shared`` array that is not a view.

    :param desc: The data descriptor.
    :return: True if the descriptor is such a container.
    """
    return (
        desc.transient
        and isinstance(desc, dt.Array)
        and not isinstance(desc, dt.View)
        and desc.storage == dtypes.StorageType.GPU_Shared
    )


def dynamic_map_state_elements(fine_grained: bool, block_size: int) -> int:
    """
    Returns the size of the shared scheduling state of ``dace::DynamicMap`` (``shared_type`` in ``dynmap.cuh``), in
    elements of its index type: the union of four indices for thread-block scheduling, and either two arrays of
    ``WARP_SIZE`` squared indices per warp (fine-grained) or two indices. The warp size is that of the GPU backend.

    :param fine_grained: Whether the fine-grained schedule is used.
    :param block_size: The total thread-block size.
    :return: The number of index elements.
    """
    warp_size = common.gpu_warp_size()
    fine_grained_elements = 2 * (block_size // warp_size) * warp_size * warp_size if fine_grained else 2
    return max(4, fine_grained_elements)


@properties.make_properties
class PrivatizeKernelSharedMemory(ppl.Pass):
    """
    Gives every GPU kernel its own shared memory containers. Shared memory does not outlive a kernel, so a container
    that several kernels of an SDFG access is a separate container in each of them: every kernel but the first that
    accesses it receives a copy, and its accesses are renamed to the copy. Afterwards, every shared memory container of
    an SDFG belongs to one kernel, and the kernels' shared memory can be placed independently.
    """

    CATEGORY: str = "GPU"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> dict[str, list[str]] | None:
        owned: dict[SDFG, set[str]] = {}
        copies: dict[str, list[str]] = {}
        for kernel_sdfg, kernel_state, kernel_entry in gpu_helpers.gpu_kernels(sdfg):
            kernel_scope = kernel_state.scope_subgraph(kernel_entry)
            owned_here = owned.setdefault(kernel_sdfg, set())
            names = sorted(
                {
                    n.data
                    for n in kernel_scope.nodes()
                    if isinstance(n, nodes.AccessNode) and is_shared_container(kernel_sdfg.arrays[n.data])
                }
            )
            for name in names:
                if name not in owned_here:
                    owned_here.add(name)
                    continue
                copy_name = kernel_sdfg.add_datadesc(name, copy.deepcopy(kernel_sdfg.arrays[name]), find_new_name=True)
                for node in kernel_scope.nodes():
                    if isinstance(node, nodes.AccessNode) and node.data == name:
                        node.data = copy_name
                for edge in kernel_scope.edges():
                    if edge.data.data == name:
                        edge.data.data = copy_name
                owned_here.add(copy_name)
                copies.setdefault(name, []).append(copy_name)
        return copies or None


@properties.make_properties
class LowerDynamicMapState(ppl.Pass):
    """
    Adds the shared scheduling state of the dynamic thread-block maps (``dace::DynamicMap``) of each GPU kernel as a
    ``GPU_Shared`` container, so that it is placed with the rest of the kernel's shared memory. One container is added
    per SDFG that contains dynamic maps, which the maps of that SDFG share (they run one after the other). The name of
    the container is stored in the ``_cuda_dynmap_state`` attribute of each dynamic map entry.
    """

    CATEGORY: str = "GPU"

    def depends_on(self):
        return {PrivatizeKernelSharedMemory}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> dict[str, int] | None:
        fine_grained = config.Config.get_bool("compiler", "cuda", "dynamic_map_fine_grained")
        elements = dynamic_map_state_elements(fine_grained, gpu_helpers.dynamic_map_block_size())
        added = {}
        for kernel_sdfg, kernel_state, kernel_entry in gpu_helpers.gpu_kernels(sdfg):
            dynamic_maps = self._dynamic_maps(kernel_sdfg, kernel_state, kernel_state.scope_subgraph(kernel_entry))
            if not dynamic_maps:
                continue
            index_type = common.gpu_dynamic_map_index_type(kernel_sdfg, kernel_state, kernel_entry)
            state_names: dict[SDFG, str] = {}
            for map_sdfg, map_state, map_entry in dynamic_maps:
                if map_sdfg not in state_names:
                    state_names[map_sdfg], _ = map_sdfg.add_array(
                        "__dace_dynmap_state",
                        [elements],
                        index_type,
                        storage=dtypes.StorageType.GPU_Shared,
                        transient=True,
                        find_new_name=True,
                    )
                    added[state_names[map_sdfg]] = elements
                name = state_names[map_sdfg]

                # An access node that precedes the dynamic map, in the scope that encloses it
                parent = map_state.entry_node(map_entry)
                access = map_state.add_access(name)
                if parent is not None:
                    map_state.add_nedge(parent, access, Memlet())
                map_state.add_nedge(access, map_entry, Memlet())
                map_entry._cuda_dynmap_state = name
        return added or None

    @staticmethod
    def _dynamic_maps(sdfg: SDFG, state: SDFGState, graph) -> list[tuple[SDFG, SDFGState, nodes.MapEntry]]:
        result = []
        for node in graph.nodes():
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
                result.append((sdfg, state, node))
            elif isinstance(node, nodes.NestedSDFG):
                for nstate in node.sdfg.states():
                    result.extend(LowerDynamicMapState._dynamic_maps(node.sdfg, nstate, nstate))
        return result


#: A shared memory container of a GPU kernel that the passes of this module place (see ``is_shared_container``): the
#: SDFG that owns it, and its name
SharedContainer = tuple[SDFG, str]


def _level_containers(sdfg: SDFG, names: list[str], nested: list[nodes.NestedSDFG]) -> list[tuple[SDFG, list[str]]]:
    """Returns the shared containers of an SDFG level and, recursively, of its nested SDFGs, in layout order."""
    result = [(sdfg, names)]
    for nsdfg_node in nested:
        nsdfg = nsdfg_node.sdfg
        nested_names = sorted(n for n, d in nsdfg.arrays.items() if is_shared_container(d))
        result.extend(_level_containers(nsdfg, nested_names, _nested_sdfg_nodes(nsdfg)))
    return result


def kernel_shared_containers(sdfg: SDFG, state: SDFGState, kernel_entry: nodes.MapEntry) -> list[SharedContainer]:
    """
    Returns the shared containers of a GPU kernel in layout order: those of the kernel's SDFG that the kernel accesses,
    then, recursively, those of the nested SDFGs in the kernel.

    :param sdfg: The SDFG of the kernel map.
    :param state: The state of the kernel map.
    :param kernel_entry: The kernel map.
    :return: The containers, as pairs of their SDFG and name.
    """
    kernel_nodes = state.scope_subgraph(kernel_entry).nodes()
    names = sorted(
        {n.data for n in kernel_nodes if isinstance(n, nodes.AccessNode) and is_shared_container(sdfg.arrays[n.data])}
    )
    nested = [n for n in kernel_nodes if isinstance(n, nodes.NestedSDFG)]
    return [(level, name) for level, level_names in _level_containers(sdfg, names, nested) for name in level_names]


@dataclass
class SharedMemoryConflicts:
    """Which shared memory containers of a GPU kernel may not occupy the same bytes (see ``SharedMemoryInterference``)."""

    #: The containers of the kernel, in layout order
    containers: list = field(default_factory=list)
    #: The pairs of containers that conflict, as frozensets
    pairs: set = field(default_factory=set)
    #: The containers that conflict with every other one
    pinned: set = field(default_factory=set)

    def conflict(self, a: SharedContainer, b: SharedContainer) -> bool:
        """Whether two different containers of the kernel may not share bytes."""
        return a in self.pinned or b in self.pinned or frozenset((a, b)) in self.pairs


class _PinAll(Exception):
    """The analysis cannot tell which container an access uses, or whether the threads of a block take the same branches:
    no container of the kernel shares memory."""


@dataclass
class _Atom:
    """A top-level part of a state that runs as a whole: a code node, a scope, or a copy between two access nodes."""

    first: nodes.Node
    last: nodes.Node
    #: The nested SDFG node, if the atom is one
    nested: nodes.NestedSDFG | None
    reads: set
    writes: set
    #: The containers accessed within a scope (read and written)
    inside: set

    @property
    def accesses(self) -> set:
        return self.reads | self.writes | self.inside


class _KernelInterference:
    """The analysis of ``SharedMemoryInterference`` for one kernel."""

    def __init__(self, sdfg: SDFG, state: SDFGState, kernel_entry: nodes.MapEntry, reachability):
        self.sdfg, self.state, self.entry = sdfg, state, kernel_entry
        self.result = SharedMemoryConflicts(kernel_shared_containers(sdfg, state, kernel_entry))
        self.containers = set(self.result.containers)
        self.reachability = reachability  # A function that returns the state reachability of the kernel's SDFG

    def run(self, reuse: bool) -> SharedMemoryConflicts:
        if not reuse or len(self.containers) < 2 or not self._analyzable():
            self.result.pinned = set(self.containers)
            return self.result
        try:
            self.reach = self.reachability()
            zeroed = self._zeroed(self.sdfg, [self.state], [c for c in self.containers if c[0] is self.sdfg])
            atoms = self._atoms(self.sdfg, self.state, self.entry)
            # Nothing is in use when a kernel starts (see ``PrivatizeKernelSharedMemory``), nor after it
            self._state(self.sdfg, self.state, atoms, zeroed, zeroed, check_barrier=False)
        except _PinAll:
            self.result.pinned = set(self.containers)
        return self.result

    def _analyzable(self) -> bool:
        """Whether the kernel's code is that of thread blocks in a ``GPU_Device`` kernel without collaborative copies."""
        if self.entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False
        children = [n for n in self.state.scope_children()[self.entry] if not isinstance(n, nodes.ExitNode)]
        if not children or not is_thread_block_level(self.sdfg, self.state, children[0]):
            return False
        return not self._has_collaborative_copy()

    def _has_collaborative_copy(self) -> bool:
        """Whether the kernel's block-level code copies between GPU containers without a node (a collaborative copy, after
        which the code generator leaves out the barrier at the start of the next state, see
        ``gpu_helpers.state_starts_with_block_barrier``)."""
        graphs = [(self.sdfg, self.state, self.state.scope_subgraph(self.entry))]
        for node in graphs[0][2].nodes():
            if isinstance(node, nodes.NestedSDFG):
                graphs.extend((s.sdfg, s, s) for nsdfg in node.sdfg.all_sdfgs_recursive() for s in nsdfg.states())
        gpu_storage = (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared)
        for sdfg, state, graph in graphs:
            for e in graph.edges():
                if not isinstance(e.src, nodes.AccessNode) or not isinstance(e.dst, nodes.AccessNode):
                    continue
                src, dst = sdfg.arrays[e.src.data], sdfg.arrays[e.dst.data]
                if isinstance(src, (dt.View, dt.Stream)) or isinstance(dst, (dt.View, dt.Stream)):
                    continue  # A view of a container, or a queue operation
                if (
                    src.storage in gpu_storage
                    and dst.storage in gpu_storage
                    and not is_in_scope(sdfg, state, e.dst, THREAD_SCHEDULES)
                ):
                    return True
        return False

    # Containers of accesses

    def _resolve(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> set:
        """The container of the kernel an access node accesses (through views and nested SDFG connectors), as a set of
        at most one."""
        desc = sdfg.arrays[node.data]
        if isinstance(desc, dt.View):
            viewed = sdutil.get_last_view_node(state, node)
            if viewed is None:
                if desc.storage == dtypes.StorageType.GPU_Shared:
                    raise _PinAll
                return set()
            return self._resolve(sdfg, state, viewed)
        return self._resolve_name(sdfg, node.data)

    def _resolve_name(self, sdfg: SDFG, name: str) -> set:
        """The container of the kernel that a data descriptor of an SDFG is, as a set of at most one."""
        if (sdfg, name) in self.containers:
            return {(sdfg, name)}
        desc = sdfg.arrays[name]
        if isinstance(desc, dt.View):
            raise _PinAll  # Only an access node tells what a view views
        if desc.transient:
            return set()  # Not shared memory that these passes place
        # Data of an enclosing SDFG of the kernel, passed to this one (whatever storage its descriptor has so far)
        shared = desc.storage == dtypes.StorageType.GPU_Shared
        if sdfg is not self.sdfg and sdfg.parent_nsdfg_node is not None:
            nsdfg_node, state, outer = sdfg.parent_nsdfg_node, sdfg.parent, sdfg.parent_sdfg
            for e in state.in_edges(nsdfg_node):
                if e.dst_conn == name and isinstance(state.memlet_path(e)[0].src, nodes.AccessNode):
                    return self._resolve(outer, state, state.memlet_path(e)[0].src)
            for e in state.out_edges(nsdfg_node):
                if e.src_conn == name and isinstance(state.memlet_path(e)[-1].dst, nodes.AccessNode):
                    return self._resolve(outer, state, state.memlet_path(e)[-1].dst)
        if shared:
            raise _PinAll
        return set()

    def _sources(self, sdfg: SDFG, state: SDFGState, node: nodes.Node) -> set:
        return set().union(
            *(self._resolve(sdfg, state, e.src) for e in state.in_edges(node) if isinstance(e.src, nodes.AccessNode))
        )

    def _destinations(self, sdfg: SDFG, state: SDFGState, node: nodes.Node) -> set:
        return set().union(
            *(self._resolve(sdfg, state, e.dst) for e in state.out_edges(node) if isinstance(e.dst, nodes.AccessNode))
        )

    def _pin_nested(self, nsdfg: SDFG) -> None:
        """Pins the containers of a nested SDFG that threads run on their own (and of those it nests)."""
        inner = set(nsdfg.all_sdfgs_recursive())
        self.result.pinned |= {c for c in self.containers if c[0] in inner}

    def _atoms(self, sdfg: SDFG, state: SDFGState, scope: nodes.EntryNode | None) -> list[_Atom]:
        """The atoms of a state, or of the kernel's scope."""
        atoms = []
        for node in state.scope_children()[scope]:
            if isinstance(node, nodes.EntryNode):
                exit_ = state.exit_node(node)
                inside = set()
                for inner in state.scope_subgraph(node, include_entry=False, include_exit=False).nodes():
                    if isinstance(inner, nodes.AccessNode):
                        inside |= self._resolve(sdfg, state, inner)
                    elif isinstance(inner, nodes.NestedSDFG):
                        self._pin_nested(inner.sdfg)
                reads, writes = self._sources(sdfg, state, node), self._destinations(sdfg, state, exit_)
                atoms.append(_Atom(node, exit_, None, reads, writes, inside))
            elif isinstance(node, nodes.CodeNode):
                nested = node if isinstance(node, nodes.NestedSDFG) else None
                reads, writes = self._sources(sdfg, state, node), self._destinations(sdfg, state, node)
                atoms.append(_Atom(node, node, nested, reads, writes, set()))
            elif isinstance(node, nodes.AccessNode):
                for e in state.out_edges(node):
                    if isinstance(e.dst, nodes.AccessNode) and not self._is_view_edge(sdfg, state, e):
                        reads, writes = self._resolve(sdfg, state, node), self._resolve(sdfg, state, e.dst)
                        atoms.append(_Atom(node, e.dst, None, reads, writes, set()))
        return atoms

    @staticmethod
    def _is_view_edge(sdfg: SDFG, state: SDFGState, edge) -> bool:
        return any(
            isinstance(sdfg.arrays[n.data], dt.View) and sdutil.get_view_edge(state, n) is edge
            for n in (edge.src, edge.dst)
        )

    @staticmethod
    def _zeroed(sdfg: SDFG, states, containers) -> set:
        """The containers that are zeroed where they are allocated: in use in their whole SDFG."""
        names = {n.data for s in states for n in s.data_nodes() if n.setzero}
        return {c for c in containers if c[1] in names}

    # The analysis

    def _conflict(self, containers: set) -> None:
        self.result.pairs.update(frozenset(pair) for pair in itertools.combinations(containers, 2))

    def _conflict_between(self, first: set, second: set) -> None:
        self.result.pairs.update(frozenset((a, b)) for a in first for b in second if a != b)

    def _state(
        self, sdfg: SDFG, state: SDFGState, atoms: list[_Atom], def_start: set, use_end: set, check_barrier: bool
    ):
        """
        Adds the conflicts of a state of a block-level SDFG, or of the kernel's scope, and analyzes its nested SDFGs.

        :param def_start: The containers that may hold a value written before the state.
        :param use_end: The containers that may be read after the state.
        :param check_barrier: Whether the containers that leaf atoms access must be protected by a barrier at the start
                              of the state (see ``gpu_helpers.state_starts_with_block_barrier``).
        :return: The containers that threads of the block may still access when others leave the state.
        """
        after = {id(a): nx.descendants(state.nx, a.last) for a in atoms}

        def precedes(a: _Atom, b: _Atom) -> bool:
            return b.first is a.last or b.first in after[id(a)]

        leaves = [a for a in atoms if a.nested is None]
        touched = set().union(*(a.accesses for a in leaves))
        self._conflict(touched)  # No barrier separates the leaves of a state
        if check_barrier and touched and not gpu_helpers.state_starts_with_block_barrier(sdfg, state):
            self.result.pinned |= touched
        hot = set()
        for v in atoms:
            before = [u for u in atoms if u is not v and not precedes(v, u)]
            later = [u for u in atoms if u is not v and not precedes(u, v)]
            defined = set(def_start).union(*(u.writes | u.inside for u in before))
            used = set(use_end).union(*(u.reads | u.inside for u in later))
            if v.nested is None:
                self._conflict(v.accesses | (defined & used))  # What it accesses, and what holds a value across it
            else:
                nested_hot = self._sdfg(v.nested.sdfg, defined, used)
                hot |= nested_hot
                for u in later:
                    if u.nested is None:
                        self._conflict_between(nested_hot, u.accesses)
        if gpu_helpers.state_ends_with_block_barrier(state):
            return set()
        return touched | hot

    def _sdfg(self, sdfg: SDFG, def_in: set, use_out: set) -> set:
        """
        Adds the conflicts of a nested SDFG that the threads of a block run together.

        :param def_in: The containers that may hold a value when the SDFG starts.
        :param use_out: The containers that may be read after the SDFG.
        :return: The containers that threads of the block may still access when others leave the SDFG.
        """
        states = list(sdfg.states())
        zeroed = self._zeroed(sdfg, states, [c for c in self.containers if c[0] is sdfg])
        def_in, use_out = def_in | zeroed, use_out | zeroed
        control = self._control_reads(sdfg)
        use_out = use_out.union(*(self._resolve_name(sdfg, name) for name in control))
        reach = self.reach[sdfg.cfg_id]
        atoms = {s: self._atoms(sdfg, s, None) for s in states}
        reads = {s: set().union(*(a.reads | a.inside for a in atoms[s])) for s in states}
        writes = {s: set().union(*(a.writes | a.inside for a in atoms[s])) for s in states}
        hot = set()
        for s in states:
            # The virtual source state writes def_in and reaches every state; every state reaches the virtual sink,
            # which reads use_out
            def_start = def_in.union(*(writes[t] for t in states if s in reach[t]))
            use_end = use_out.union(*(reads[t] for t in reach[s]))
            hot |= self._state(sdfg, s, atoms[s], def_start, use_end, check_barrier=True)
        return hot

    def _control_reads(self, sdfg: SDFG) -> set[str]:
        """The data descriptors that the control flow of an SDFG reads (interstate edges, loops, branch conditions). If
        one is written by the threads of a thread-block map, the threads of a block may take different branches."""
        arrays = set(sdfg.arrays.keys())
        names = set()
        for region in sdfg.all_control_flow_regions():
            for e in region.edges():
                names |= e.data.free_symbols & arrays
            exprs = []
            if isinstance(region, LoopRegion):
                exprs = [region.loop_condition, region.init_statement, region.update_statement]
            elif isinstance(region, ConditionalBlock):
                exprs = [condition for condition, _ in region.branches]
            for expr in exprs:
                if expr is not None:
                    names |= expr.get_free_symbols() & arrays
        for name in names:
            if sdfg.arrays[name].storage not in (dtypes.StorageType.Register, dtypes.StorageType.Default):
                continue
            for state in sdfg.states():
                for node in state.data_nodes():
                    if node.data != name:
                        continue
                    for e in state.in_edges(node):
                        if is_in_scope(sdfg, state, state.memlet_path(e)[0].src, THREAD_SCHEDULES):
                            raise _PinAll  # Per-thread values decide the control flow of the block
        return names


@properties.make_properties
class SharedMemoryInterference(ppl.Pass):
    """
    Finds which shared memory containers of each GPU kernel may share bytes: those never in use at the same time.

    A container is in use where it is *live* (it holds a value that may still be read: written before, on some path
    or earlier in the state, and read after) or *hot* (threads of the block may still access it since the last barrier).
    The analysis follows the dataflow of the thread blocks' code of a ``GPU_Device`` kernel, through nested SDFGs that
    the threads of a block run together, and relies on where the code generator synchronizes the block (see
    ``gpu_helpers.state_starts_with_block_barrier`` and ``state_ends_with_block_barrier``): no barrier separates the
    top-level nodes of a state, which therefore all conflict, and a state that accesses shared memory starts with one.
    Containers of exclusive branches never conflict, which relies on the threads of a block taking the same branches,
    as the barriers in them require; the analysis gives up on a kernel whose control flow reads values that the threads
    of a thread-block map wrote. Containers that threads run on their own (in nested SDFGs within thread-block maps),
    and all containers of other kernels (persistent ones, those whose threads are not scheduled by thread-block maps, or
    with collaborative copies) conflict with every other.

    No property declares that a container may share memory: a workspace whose last use precedes the first write of a
    result, in a later state, shares its bytes with the result.
    """

    CATEGORY: str = "GPU"

    reuse = properties.Property(
        dtype=bool,
        allow_none=True,
        default=None,
        desc="Whether containers may share memory (if None, the compiler.cuda.shared_memory_reuse configuration entry)",
    )

    def __init__(self, reuse: bool | None = None):
        super().__init__()
        self.reuse = reuse

    def depends_on(self):
        return {LowerDynamicMapState}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> dict[nodes.MapEntry, SharedMemoryConflicts] | None:
        reuse = self.reuse
        if reuse is None:
            reuse = config.Config.get_bool("compiler", "cuda", "shared_memory_reuse")
        reachability: dict[int, dict] = {}

        def reach_of(kernel_sdfg: SDFG):
            def compute():
                if id(kernel_sdfg) not in reachability:
                    reachability[id(kernel_sdfg)] = StateReachability().apply_pass(kernel_sdfg, None)
                return reachability[id(kernel_sdfg)]

            return compute

        result = {}
        for kernel_sdfg, kernel_state, kernel_entry in gpu_helpers.gpu_kernels(sdfg):
            analysis = _KernelInterference(kernel_sdfg, kernel_state, kernel_entry, reach_of(kernel_sdfg))
            result[kernel_entry] = analysis.run(reuse)
        return result or None


@dataclass
class SharedMemoryLevel:
    """The dynamic shared memory of one SDFG within a GPU kernel."""

    #: The SDFG
    sdfg: SDFG
    #: The offset and size (in bytes, in the symbols of ``sdfg``) of each of its dynamic containers
    containers: dict[str, tuple[symbolic.SymbolicType, symbolic.SymbolicType]] = field(default_factory=dict)
    #: The end of the part of dynamic shared memory that the SDFG and its nested SDFGs use (in the symbols of ``sdfg``)
    end: symbolic.SymbolicType = 0


@dataclass
class _Overlap:
    """A shared container of an SDFG that the threads of a block run together, which holds no values in some of its
    states, as nothing writes it before them: their nested SDFGs may use its memory."""

    name: str
    #: The states, and in each, the nested SDFG that writes the container first (whose shared containers may overlap
    #: its outputs), or None if no node of the state accesses the container
    states: dict = field(default_factory=dict)


@dataclass
class KernelSharedMemory:
    """The shared memory layout of a GPU kernel."""

    #: The bytes of static shared memory the kernel's containers use
    static_bytes: int = 0
    #: The bytes of dynamic shared memory to launch the kernel with, in the symbols of the kernel's SDFG
    dynamic_bytes: symbolic.SymbolicType = 0
    #: The dynamic shared memory of each SDFG in the kernel that has any
    levels: list[SharedMemoryLevel] = field(default_factory=list)


@properties.make_properties
class PlanSharedMemory(ppl.Pass):
    """
    Decides, for every transient ``GPU_Shared`` container of each GPU kernel, whether it is placed in static or dynamic
    shared memory, and lays out dynamic shared memory.

    A container whose storage specifies ``dynamic`` keeps it (a statically placed container must have a constant size).
    Otherwise, containers of symbolic size are placed dynamically, and the rest stay static while the kernel's static
    shared memory fits in ``compiler.cuda.max_static_shared_memory``. The decision is written back to the storage type of
    each container. In dynamic shared memory, the containers of each SDFG are laid out contiguously, followed by those
    of its nested SDFGs; a nested SDFG receives the offset of its part as the symbol ``__dace_dynsmem_base`` unless the
    offset is a constant. The parts of nested SDFGs in different states of an SDFG that the threads of a block run
    together (see ``is_thread_block_level``) start at the same offset: they never run at the same time, as a state of a
    thread block's code that accesses shared memory starts with a barrier (e.g., the tasks in the branches of a
    persistent kernel share their memory). Those in one state are laid out one after the other. The number of bytes to launch each kernel with is stored in the
    ``_cuda_dynamic_shared_memory`` attribute of the kernel map entry, and its static shared memory in
    ``_cuda_static_shared_memory``.
    """

    CATEGORY: str = "GPU"

    def depends_on(self):
        return {LowerDynamicMapState}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> dict[nodes.MapEntry, KernelSharedMemory] | None:
        result = {}
        for kernel_sdfg, kernel_state, kernel_entry in gpu_helpers.gpu_kernels(sdfg):
            plan = self._plan_kernel(kernel_sdfg, kernel_state, kernel_entry)
            kernel_entry._cuda_dynamic_shared_memory = plan.dynamic_bytes
            kernel_entry._cuda_static_shared_memory = plan.static_bytes
            if plan.levels or plan.static_bytes:
                result[kernel_entry] = plan
        return result or None

    def _plan_kernel(self, sdfg: SDFG, state: SDFGState, kernel_entry: nodes.MapEntry) -> KernelSharedMemory:
        kernel_nodes = state.scope_subgraph(kernel_entry).nodes()
        kernel_label = kernel_entry.map.label

        # The shared containers of each SDFG in the kernel. At the kernel's level, only those accessed in the kernel
        kernel_level = sorted(
            {
                n.data
                for n in kernel_nodes
                if isinstance(n, nodes.AccessNode) and is_shared_container(sdfg.arrays[n.data])
            }
        )
        top_nested = [n for n in kernel_nodes if isinstance(n, nodes.NestedSDFG)]

        # Decide the placement of each container
        plan = KernelSharedMemory()
        limit = common.gpu_max_static_shared_memory()
        candidates: list[tuple[SDFG, str, int]] = []
        dynamic: set[tuple[SDFG, str]] = set()
        # The containers whose memory nested SDFGs may use where they hold no values, which must be dynamic
        overlaps: dict[int, _Overlap] = {}
        found = self._overlap_in_kernel(sdfg, state, kernel_entry, kernel_level)
        if found is not None:
            overlaps[id(sdfg)] = found
        for level_sdfg, _ in self._containers(sdfg, kernel_level, top_nested)[1:]:
            found = self._overlap(level_sdfg)
            if found is not None:
                overlaps[id(level_sdfg)] = found
        for level_sdfg, names in self._containers(sdfg, kernel_level, top_nested):
            for name in names:
                desc = level_sdfg.arrays[name]
                size = desc.total_size_in_bytes
                is_symbolic = symbolic.issymbolic(size, level_sdfg.constants)
                placement = desc.storage.dynamic
                if placement is False:
                    if is_symbolic:
                        raise ValueError(
                            f'Shared memory container "{name}" of kernel "{kernel_label}" is placed in '
                            f"static shared memory, which requires a constant size (got {size} bytes). "
                            "Place it in dynamic shared memory with "
                            "`StorageType.GPU_Shared(dynamic=True)`, or leave the placement to the code "
                            "generator with `StorageType.GPU_Shared`."
                        )
                    plan.static_bytes += _aligned(int(symbolic.evaluate(size, level_sdfg.constants)))
                elif placement is True or is_symbolic or overlaps.get(id(level_sdfg), _Overlap("")).name == name:
                    dynamic.add((level_sdfg, name))
                else:
                    candidates.append((level_sdfg, name, _aligned(int(symbolic.evaluate(size, level_sdfg.constants)))))

        for level_sdfg, name, size in candidates:
            if plan.static_bytes + size <= limit:
                plan.static_bytes += size
            else:
                warnings.warn(
                    f'Shared memory container "{name}" ({size} bytes) of kernel "{kernel_label}" does not '
                    f"fit in the static shared memory of a thread-block ({plan.static_bytes} of {limit} "
                    "bytes already used), so it is placed in dynamic shared memory."
                )
                dynamic.add((level_sdfg, name))
        # Make every decision explicit in the SDFG
        for level_sdfg, names in self._containers(sdfg, kernel_level, top_nested):
            for name in names:
                desc = level_sdfg.arrays[name]
                desc.storage = dtypes.StorageType.GPU_Shared(
                    dynamic=(level_sdfg, name) in dynamic, may_overlap_outputs=desc.storage.may_overlap_outputs
                )

        # Lay out dynamic shared memory
        if dynamic:
            plan.dynamic_bytes = self._layout(sdfg, kernel_level, top_nested, 0, dynamic, plan.levels, False, overlaps)
            self._check_launch_symbols(state, kernel_entry, kernel_nodes, plan.dynamic_bytes)
        return plan

    def _containers(self, sdfg: SDFG, names: list[str], nested: list[nodes.NestedSDFG]) -> list[tuple[SDFG, list[str]]]:
        """Returns the shared containers of an SDFG level and, recursively, of its nested SDFGs, in layout order."""
        result = [(sdfg, names)]
        for nsdfg_node in nested:
            nsdfg = nsdfg_node.sdfg
            nested_names = sorted(n for n, d in nsdfg.arrays.items() if is_shared_container(d))
            result.extend(self._containers(nsdfg, nested_names, _nested_sdfg_nodes(nsdfg)))
        return result

    def _layout(
        self,
        sdfg: SDFG,
        names: list[str],
        nested: list[nodes.NestedSDFG],
        base: symbolic.SymbolicType,
        dynamic: set[tuple[SDFG, str]],
        levels: list[SharedMemoryLevel],
        block_level: bool,
        overlaps: dict[int, _Overlap],
    ) -> symbolic.SymbolicType:
        """
        Lays out the dynamic containers of an SDFG, followed by those of its nested SDFGs, starting at ``base``.

        :param block_level: Whether the threads of a block run the SDFG together, so that the nested SDFGs of different
                            states may share their part (see the class documentation).
        :param overlaps: The container of every SDFG whose memory nested SDFGs may use (see ``_overlap``).
        :return: The end of the layout, in the symbols of ``sdfg``.
        """
        overlap = overlaps.get(id(sdfg))  # Its states are those of ``groups``: of a block-level SDFG, or the kernel
        if overlap is not None and (overlap.name not in names or (sdfg, overlap.name) not in dynamic):
            overlap = None
        if overlap is not None:  # The container last: what nested SDFGs place at its offset may extend past it
            names = [n for n in names if n != overlap.name] + [overlap.name]
        level = SharedMemoryLevel(sdfg)
        # The base is aligned, and stays so across containers whose size is a multiple of the alignment
        cursor, cursor_alignment = base, DYNAMIC_SHARED_MEMORY_ALIGNMENT
        for name in names:
            if (sdfg, name) not in dynamic:
                continue
            desc = sdfg.arrays[name]
            alignment = max(DYNAMIC_SHARED_MEMORY_ALIGNMENT, desc.dtype.bytes)
            offset = cursor if cursor_alignment % alignment == 0 else symbolic.align(cursor, alignment)
            size = desc.total_size_in_bytes
            level.containers[name] = (offset, size)
            shared_at = offset
            cursor = offset + size
            cursor_alignment = alignment if symbolic.is_multiple(size, alignment) else 1

        # The nested SDFGs that may share their part: those of each state of a block-level SDFG
        state_of = {n: s for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.NestedSDFG)}
        groups: dict = {}
        for nsdfg_node in nested:
            groups.setdefault(state_of.get(nsdfg_node) if block_level else None, []).append(nsdfg_node)
        start, start_alignment = cursor, cursor_alignment
        end = start if overlap is not None else None
        for key, group in groups.items():
            if overlap is not None and key in overlap.states:
                # The container holds no values in this state: its nested SDFGs start at it, the one that writes it
                # first, if any, before the others (which then start after the container as well)
                first = overlap.states[key]
                rest = [n for n in group if n is not first]
                rest_start, rest_alignment = shared_at, DYNAMIC_SHARED_MEMORY_ALIGNMENT
                group_end = shared_at
                if first is not None and first in group:
                    group_end = self._layout_nested(
                        sdfg, [first], state_of, shared_at, DYNAMIC_SHARED_MEMORY_ALIGNMENT, dynamic, levels, overlaps
                    )
                    rest_start, rest_alignment = _max(start, group_end), 1
                group_end = _max(
                    group_end,
                    self._layout_nested(sdfg, rest, state_of, rest_start, rest_alignment, dynamic, levels, overlaps),
                )
            else:
                group_end = self._layout_nested(
                    sdfg, group, state_of, start, start_alignment, dynamic, levels, overlaps
                )
            end = group_end if end is None else _max(end, group_end)
        if end is not None:
            cursor = end

        level.end = cursor
        if level.containers:
            levels.append(level)
        return cursor

    def _layout_nested(
        self,
        sdfg: SDFG,
        nested: list[nodes.NestedSDFG],
        state_of: dict,
        cursor: symbolic.SymbolicType,
        cursor_alignment: int,
        dynamic: set[tuple[SDFG, str]],
        levels: list[SharedMemoryLevel],
        overlaps: dict[int, _Overlap],
    ) -> symbolic.SymbolicType:
        """Lays out the parts of nested SDFGs one after the other, starting at ``cursor``.

        :return: The end of the last part, in the symbols of ``sdfg``.
        """
        for nsdfg_node in nested:
            nsdfg = nsdfg_node.sdfg
            nested_names = sorted(n for n, d in nsdfg.arrays.items() if is_shared_container(d))
            if not any(
                (level_sdfg, name) in dynamic
                for level_sdfg, level_names in self._containers(nsdfg, nested_names, _nested_sdfg_nodes(nsdfg))
                for name in level_names
            ):
                continue
            nested_base = (
                cursor
                if cursor_alignment % DYNAMIC_SHARED_MEMORY_ALIGNMENT == 0
                else symbolic.align(cursor, DYNAMIC_SHARED_MEMORY_ALIGNMENT)
            )
            if symbolic.issymbolic(nested_base):
                # Pass the offset of the nested SDFG's part as a symbol
                symbol_name = DYNAMIC_SHARED_MEMORY_BASE
                if symbol_name not in nsdfg.symbols:
                    nsdfg.add_symbol(symbol_name, dtypes.int64)
                nsdfg_node.symbol_mapping[symbol_name] = nested_base
                inner_base = symbolic.symbol(symbol_name, dtypes.int64)
            else:
                inner_base = nested_base
            state = state_of[nsdfg_node]
            inner_block_level = is_thread_block_level(sdfg, state, nsdfg_node) and _kernel_is_device_map(
                sdfg, state, nsdfg_node
            )
            inner_end = self._layout(
                nsdfg, nested_names, _nested_sdfg_nodes(nsdfg), inner_base, dynamic, levels, inner_block_level, overlaps
            )

            # Translate the end of the nested part to the symbols of this SDFG
            mapping = {symbolic.symbol(k): symbolic.pystr_to_symbolic(v) for k, v in nsdfg_node.symbol_mapping.items()}
            cursor = (
                symbolic.pystr_to_symbolic(inner_end).subs(mapping) if symbolic.issymbolic(inner_end) else inner_end
            )
            cursor_alignment = 1
        return cursor

    def _overlap(self, sdfg: SDFG) -> _Overlap | None:
        """
        The shared container of an SDFG whose memory its nested SDFGs may use most often, or None. A container holds no
        values in the states that nothing writes it before (on any path, and in no loop around them). In such a state
        that does not access the container, nested SDFGs may use its memory. In such a state that does, they may too if
        a single node writes it, a nested SDFG that the block's threads run together whose shared containers may
        overlap its outputs (``StorageType.GPU_Shared.may_overlap_outputs``), and that does not read the container:
        that nested SDFG's part starts at the container, which is laid out last.
        """
        candidates = sorted(
            n for n, d in sdfg.arrays.items() if is_shared_container(d) and d.storage.dynamic is not False
        )
        states = list(sdfg.states())
        if not candidates or not any(isinstance(n, nodes.NestedSDFG) for s in states for n in s.nodes()):
            return None
        reach = StateReachability().apply_pass(sdfg, None)[sdfg.cfg_id]
        best, best_count = None, 0
        for name in candidates:
            accesses = {s: [n for n in s.data_nodes() if n.data == name] for s in states}
            written = [s for s, ns in accesses.items() if any(not e.data.is_empty() for n in ns for e in s.in_edges(n))]
            found = _Overlap(name)
            for s in states:
                if s in reach[s] or any(s in reach[t] for t in written if t is not s):
                    continue  # It may hold values written before
                if not accesses[s]:
                    found.states[s] = None
                    continue
                writers = {e.src for n in accesses[s] for e in s.in_edges(n) if not e.data.is_empty()}
                if len(writers) == 1 and self._writes_first(sdfg, s, next(iter(writers)), name, None):
                    found.states[s] = next(iter(writers))
            count = sum(1 for writer in found.states.values() if writer is not None)
            if count > best_count:
                best, best_count = found, count
        return best

    def _overlap_in_kernel(
        self, sdfg: SDFG, state: SDFGState, kernel_entry: nodes.MapEntry, names: list[str]
    ) -> _Overlap | None:
        """The shared container of a kernel's level whose memory the nested SDFG that writes it may use (see
        ``_overlap``), or None. Such a container holds no values before the kernel (see
        ``PrivatizeKernelSharedMemory``); the nested SDFGs of the kernel are one group of the layout."""
        kernel_nodes = state.scope_subgraph(kernel_entry).nodes()
        for name in names:
            if sdfg.arrays[name].storage.dynamic is False:
                continue
            accesses = [n for n in kernel_nodes if isinstance(n, nodes.AccessNode) and n.data == name]
            writers = {e.src for n in accesses for e in state.in_edges(n) if not e.data.is_empty()}
            if len(writers) == 1 and self._writes_first(sdfg, state, next(iter(writers)), name, kernel_entry):
                return _Overlap(name, {None: next(iter(writers))})
        return None

    def _writes_first(self, sdfg: SDFG, state: SDFGState, writer: nodes.Node, name: str, scope) -> bool:
        """Whether the only writer of a container in a state, which holds no values before it, may place its part at
        the container: a nested SDFG in the given scope (not repeated in an inner one) that the block's threads run
        together, whose shared containers may overlap its outputs, and that does not read the container."""
        return (
            isinstance(writer, nodes.NestedSDFG)
            and state.entry_node(writer) is scope
            and not any(isinstance(e.src, nodes.AccessNode) and e.src.data == name for e in state.in_edges(writer))
            and is_thread_block_level(sdfg, state, writer)
            and self._overlaps_its_outputs(writer)
        )

    def _overlaps_its_outputs(self, nsdfg_node: nodes.NestedSDFG) -> bool:
        """Whether a nested SDFG has shared containers, all of which may overlap its outputs (and none deeper)."""
        nsdfg = nsdfg_node.sdfg
        names = sorted(n for n, d in nsdfg.arrays.items() if is_shared_container(d))
        deeper = self._containers(nsdfg, names, _nested_sdfg_nodes(nsdfg))[1:]
        return (
            bool(names)
            and all(nsdfg.arrays[n].storage.may_overlap_outputs for n in names)
            and not any(level_names for _, level_names in deeper)
        )

    @staticmethod
    def _check_launch_symbols(
        state: SDFGState, kernel_entry: nodes.MapEntry, kernel_nodes, size: symbolic.SymbolicType
    ) -> None:
        """Raises if the dynamic shared memory size of a kernel depends on symbols defined only within the kernel."""
        if not symbolic.issymbolic(size):
            return
        inner = set()
        for node in kernel_nodes:
            if isinstance(node, nodes.MapEntry):
                inner |= set(node.map.params)
        free = {str(s) for s in symbolic.pystr_to_symbolic(size).free_symbols}
        defined = set(state.symbols_defined_at(kernel_entry).keys())
        defined |= {e.dst_conn for e in sdutil.dynamic_map_inputs(state, kernel_entry)}
        undefined = free & inner | (free - defined - set(state.sdfg.constants.keys()))
        if undefined:
            raise ValueError(
                f'The dynamic shared memory of kernel "{kernel_entry.map.label}" ({size} bytes) depends '
                f"on {', '.join(sorted(undefined))}, which are not known when the kernel is launched."
            )


@properties.make_properties
class LowerDynamicSharedMemory(ppl.Pass):
    """
    Turns every container that ``PlanSharedMemory`` placed in dynamic shared memory into a view of a flat ``uint8``
    buffer (``StorageType.GPU_Shared(dynamic=True)``) at its offset, one buffer per SDFG. After this pass, the only
    dynamically placed containers that are not views are these buffers, which the code generator binds to the dynamic
    shared memory of the kernel.
    """

    CATEGORY: str = "GPU"

    def depends_on(self):
        return {PlanSharedMemory}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        plans: dict[nodes.MapEntry, KernelSharedMemory] = pipeline_results.get(PlanSharedMemory.__name__) or {}
        converted = 0
        for plan in plans.values():
            for level in plan.levels:
                buffer, _ = level.sdfg.add_array(
                    DYNAMIC_SHARED_MEMORY_BUFFER,
                    [level.end],
                    dtypes.uint8,
                    storage=dtypes.StorageType.GPU_Shared(dynamic=True),
                    transient=True,
                    find_new_name=True,
                )
                for name, (offset, size) in level.containers.items():
                    sdutil.convert_to_view(level.sdfg, name, buffer, subsets.Range([(offset, offset + size - 1, 1)]))
                    converted += 1
        return converted or None


def is_dynamic_shared_memory_buffer(desc: dt.Data) -> bool:
    """
    Returns whether a data descriptor is a flat dynamic shared memory buffer created by ``LowerDynamicSharedMemory``.

    :param desc: The data descriptor.
    :return: True if the descriptor is such a buffer.
    """
    return (
        not isinstance(desc, dt.View) and desc.storage == dtypes.StorageType.GPU_Shared and desc.storage.dynamic is True
    )


def plan_gpu_shared_memory(sdfg: SDFG) -> dict[nodes.MapEntry, KernelSharedMemory]:
    """
    Runs the shared memory passes on an SDFG.

    :param sdfg: The SDFG, modified in place.
    :return: The shared memory layout of each GPU kernel that uses shared memory.
    """
    pipeline = ppl.Pipeline(
        [PrivatizeKernelSharedMemory(), LowerDynamicMapState(), PlanSharedMemory(), LowerDynamicSharedMemory()]
    )
    results = pipeline.apply_pass(sdfg, {}) or {}
    return results.get(PlanSharedMemory.__name__) or {}
