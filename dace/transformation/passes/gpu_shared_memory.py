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
from dace.sdfg.scope import THREAD_SCHEDULES, is_in_scope, is_thread_block_level
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
    """The larger of two (symbolic) sizes or offsets in bytes, which are in terms of sizes of arrays: the one that is
    larger whatever the (nonnegative) values of their symbols and of the functions of them (e.g., rounded sizes), if
    there is one."""
    if not symbolic.issymbolic(a) and not symbolic.issymbolic(b):
        return max(a, b)
    a, b = symbolic.pystr_to_symbolic(a), symbolic.pystr_to_symbolic(b)
    difference = b - a
    terms = list(difference.atoms(symbolic.sympy.Function)) + list(difference.free_symbols)
    difference = difference.xreplace({t: symbolic.sympy.Dummy(integer=True, nonnegative=True) for t in terms})
    if difference.is_nonnegative:
        return b
    if difference.is_nonpositive:
        return a
    return symbolic.sympy.Max(a, b)


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
class KernelSharedMemory:
    """The shared memory layout of a GPU kernel."""

    #: The bytes of static shared memory the kernel's containers use
    static_bytes: int = 0
    #: The bytes of dynamic shared memory to launch the kernel with, in the symbols of the kernel's SDFG
    dynamic_bytes: symbolic.SymbolicType = 0
    #: The dynamic shared memory of each SDFG in the kernel that has any
    levels: list[SharedMemoryLevel] = field(default_factory=list)


def _align(value: symbolic.SymbolicType, alignment: int) -> symbolic.SymbolicType:
    """``value`` rounded up to a multiple of ``alignment``."""
    if symbolic.issymbolic(value):
        return symbolic.align(value, alignment)
    return -(-int(value) // alignment) * alignment


def _ranges_overlap(a: tuple, b: tuple) -> bool:
    """Whether two ranges ``(offset, size)`` of constant bytes overlap (False if either is not constant)."""
    if any(symbolic.issymbolic(v) for v in (*a, *b)):
        return False
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


@properties.make_properties
class PlanSharedMemory(ppl.Pass):
    """
    Decides, for every transient ``GPU_Shared`` container of each GPU kernel, whether it is placed in static or dynamic
    shared memory, and lays out dynamic shared memory.

    Containers that are never in use at the same time (see ``SharedMemoryInterference``) may occupy the same bytes of
    dynamic shared memory. Containers of constant size are placed first, largest first, each at the lowest offset where
    it overlaps no placed container it conflicts with; containers of symbolic size follow, after the containers they
    conflict with. Offsets are those of the kernel's dynamic shared memory; a nested SDFG receives the offset of a
    container as a symbol (``__dace_dynsmem_base`` and following names) unless it is a constant.

    A container whose storage specifies ``dynamic`` keeps it (a statically placed container, which shares no memory,
    must have a constant size). Otherwise, containers of symbolic size, and those that share memory with others, are
    placed dynamically; the rest stay static while the kernel's static shared memory fits in
    ``compiler.cuda.max_static_shared_memory``. The decision is written back to the storage type of each container. The
    number of bytes to launch each kernel with is stored in the ``_cuda_dynamic_shared_memory`` attribute of the kernel
    map entry, and its static shared memory in ``_cuda_static_shared_memory``.
    """

    CATEGORY: str = "GPU"

    def depends_on(self):
        return {SharedMemoryInterference}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> dict[nodes.MapEntry, KernelSharedMemory] | None:
        conflicts = (pipeline_results or {}).get(SharedMemoryInterference.__name__)
        if conflicts is None:
            conflicts = SharedMemoryInterference().apply_pass(sdfg, pipeline_results) or {}
        result = {}
        for kernel_sdfg, kernel_state, kernel_entry in gpu_helpers.gpu_kernels(sdfg):
            kernel_conflicts = conflicts.get(kernel_entry)
            if kernel_conflicts is None:  # Every container conflicts with every other
                containers = kernel_shared_containers(kernel_sdfg, kernel_state, kernel_entry)
                kernel_conflicts = SharedMemoryConflicts(containers, pinned=set(containers))
            plan = self._plan_kernel(kernel_sdfg, kernel_state, kernel_entry, kernel_conflicts)
            kernel_entry._cuda_dynamic_shared_memory = plan.dynamic_bytes
            kernel_entry._cuda_static_shared_memory = plan.static_bytes
            if plan.levels or plan.static_bytes:
                result[kernel_entry] = plan
        return result or None

    def _plan_kernel(
        self, sdfg: SDFG, state: SDFGState, kernel_entry: nodes.MapEntry, conflicts: SharedMemoryConflicts
    ) -> KernelSharedMemory:
        kernel_label = kernel_entry.map.label
        containers = conflicts.containers
        sizes = {c: self._size_in_kernel(sdfg, c) for c in containers}

        # Decide the placement of each container
        plan = KernelSharedMemory()
        limit = common.gpu_max_static_shared_memory()
        dynamic: set = set()
        undecided: list = []  # Left to the code generator, of constant size
        for container in containers:
            level_sdfg, name = container
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
            elif placement is True or is_symbolic:
                dynamic.add(container)
            else:
                undecided.append(container)

        # Those left to the code generator are placed dynamically where they then share memory with others
        if undecided:
            trial = self._place(sdfg, [c for c in containers if c in dynamic or c in undecided], sizes, conflicts)
            dynamic |= {
                c for c in undecided if any(other != c and _ranges_overlap(trial[c], r) for other, r in trial.items())
            }
        for container in undecided:
            if container in dynamic:
                continue
            size = _aligned(int(sizes[container]))
            if plan.static_bytes + size <= limit:
                plan.static_bytes += size
            else:
                warnings.warn(
                    f'Shared memory container "{container[1]}" ({size} bytes) of kernel "{kernel_label}" does not '
                    f"fit in the static shared memory of a thread-block ({plan.static_bytes} of {limit} "
                    "bytes already used), so it is placed in dynamic shared memory."
                )
                dynamic.add(container)

        # Make every decision explicit in the SDFG
        for level_sdfg, name in containers:
            desc = level_sdfg.arrays[name]
            desc.storage = dtypes.StorageType.GPU_Shared(dynamic=(level_sdfg, name) in dynamic)

        # Lay out dynamic shared memory
        if dynamic:
            placed = self._place(sdfg, [c for c in containers if c in dynamic], sizes, conflicts)
            ends = [offset + size for offset, size in placed.values()]
            plan.dynamic_bytes = ends[0]
            for end in ends[1:]:
                plan.dynamic_bytes = _max(plan.dynamic_bytes, end)
            plan.levels = self._levels(sdfg, [c for c in containers if c in dynamic], placed)
            self._check_launch_symbols(
                state, kernel_entry, state.scope_subgraph(kernel_entry).nodes(), plan.dynamic_bytes
            )
        return plan

    @staticmethod
    def _size_in_kernel(sdfg: SDFG, container: SharedContainer) -> symbolic.SymbolicType:
        """The size in bytes of a container, in the symbols of the kernel's SDFG ``sdfg``."""
        level_sdfg, name = container
        size = level_sdfg.arrays[name].total_size_in_bytes
        if not symbolic.issymbolic(size, level_sdfg.constants):
            return int(symbolic.evaluate(size, level_sdfg.constants))
        size = symbolic.pystr_to_symbolic(size)
        while level_sdfg is not sdfg:
            node = level_sdfg.parent_nsdfg_node
            mapping = {symbolic.symbol(k): symbolic.pystr_to_symbolic(v) for k, v in node.symbol_mapping.items()}
            size = size.subs(mapping)
            level_sdfg = level_sdfg.parent_sdfg
        if not symbolic.issymbolic(size, sdfg.constants):
            return int(symbolic.evaluate(size, sdfg.constants))
        return size

    @staticmethod
    def _place(sdfg: SDFG, containers: list, sizes: dict, conflicts: SharedMemoryConflicts) -> dict:
        """
        Places containers in dynamic shared memory (see the class documentation).

        :return: The offset and size in bytes of each container, in the symbols of the kernel's SDFG ``sdfg``.
        """
        constant = [c for c in containers if not symbolic.issymbolic(sizes[c])]
        constant.sort(key=lambda c: -_aligned(sizes[c]))  # Stable: in layout order otherwise
        order = constant + [c for c in containers if symbolic.issymbolic(sizes[c])]
        placed: dict = {}
        for container in order:
            size = sizes[container]
            alignment = max(DYNAMIC_SHARED_MEMORY_ALIGNMENT, container[0].arrays[container[1]].dtype.bytes)
            others = [placed[other] for other in placed if conflicts.conflict(container, other)]
            if not symbolic.issymbolic(size) and not any(symbolic.issymbolic(v) for r in others for v in r):
                # The lowest offset where it overlaps none of them
                candidates = sorted({0} | {_align(offset + other_size, alignment) for offset, other_size in others})
                offset = next(
                    c for c in candidates if all(c + size <= o or c >= o + other_size for o, other_size in others)
                )
            else:
                offset = 0
                for other_offset, other_size in others:
                    offset = _max(offset, other_offset + other_size)
                offset = _align(offset, alignment)
            placed[container] = (offset, size)
        return placed

    @staticmethod
    def _levels(sdfg: SDFG, containers: list, placed: dict) -> list[SharedMemoryLevel]:
        """The dynamic shared memory of each SDFG with dynamically placed containers. The offset of a container of a
        nested SDFG that is not a constant is passed to it as a symbol."""
        levels: dict[int, SharedMemoryLevel] = {}
        used_names: set[str] = set()
        for container in containers:
            level_sdfg, name = container
            offset = placed[container][0]
            if symbolic.issymbolic(offset) and level_sdfg is not sdfg:
                offset = PlanSharedMemory._pass_offset(sdfg, level_sdfg, offset, used_names)
            size = level_sdfg.arrays[name].total_size_in_bytes
            level = levels.get(id(level_sdfg))
            if level is None:
                level = levels[id(level_sdfg)] = SharedMemoryLevel(level_sdfg, end=offset + size)
            else:
                level.end = _max(level.end, offset + size)
            level.containers[name] = (offset, size)
        return list(levels.values())

    @staticmethod
    def _pass_offset(sdfg: SDFG, level_sdfg: SDFG, offset, used_names: set[str]) -> symbolic.SymbolicType:
        """Passes an offset (in the symbols of the kernel's SDFG ``sdfg``) down to a nested SDFG as a new symbol, through
        every nested SDFG on the way, and returns the symbol."""
        path = []
        current = level_sdfg
        while current is not sdfg:
            path.append(current.parent_nsdfg_node)
            current = current.parent_sdfg
        path.reverse()
        index = 0
        while True:
            symbol_name = DYNAMIC_SHARED_MEMORY_BASE if index == 0 else f"{DYNAMIC_SHARED_MEMORY_BASE}_{index}"
            if symbol_name not in used_names and all(
                symbol_name not in n.sdfg.symbols and symbol_name not in n.sdfg.arrays for n in path
            ):
                break
            index += 1
        used_names.add(symbol_name)
        value = offset
        for nsdfg_node in path:
            nsdfg_node.sdfg.add_symbol(symbol_name, dtypes.int64)
            nsdfg_node.symbol_mapping[symbol_name] = value
            value = symbolic.symbol(symbol_name, dtypes.int64)
        return value

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
        [
            PrivatizeKernelSharedMemory(),
            LowerDynamicMapState(),
            SharedMemoryInterference(),
            PlanSharedMemory(),
            LowerDynamicSharedMemory(),
        ]
    )
    results = pipeline.apply_pass(sdfg, {}) or {}
    return results.get(PlanSharedMemory.__name__) or {}
