# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Passes that place the shared memory of GPU kernels, run by the CUDA code generator before code generation.

Every transient ``GPU_Shared`` container of a kernel is placed either in static shared memory (declared as a
``__shared__`` array) or in dynamic shared memory (a flat buffer sized at kernel launch). The choice is the container's
``StorageType.GPU_Shared(dynamic=...)`` attribute, or, if left to the code generator, whether it still fits in the
static shared memory of a thread-block. Dynamic containers become views of a flat ``uint8`` buffer at their offsets, so
the layout of dynamic shared memory is part of the SDFG.

The passes run in this order: ``PrivatizeKernelSharedMemory``, ``LowerDynamicMapState``, ``PlanSharedMemory``,
``LowerDynamicSharedMemory``.
"""

import copy
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from dace import config, data as dt, dtypes, properties, subsets, symbolic
from dace.codegen import common
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.transformation import gpu_helpers, pass_pipeline as ppl

#: Alignment of every container in dynamic shared memory, which suffices for all types up to 16-byte vectors
DYNAMIC_SHARED_MEMORY_ALIGNMENT = 16

#: Name of the symbol that holds the offset of a nested SDFG's part of dynamic shared memory, if not a constant
DYNAMIC_SHARED_MEMORY_BASE = "__dace_dynsmem_base"

#: Name of the flat dynamic shared memory buffers
DYNAMIC_SHARED_MEMORY_BUFFER = "__dace_dynsmem"


def _nested_sdfg_nodes(sdfg: SDFG) -> List[nodes.NestedSDFG]:
    """Returns the nested SDFG nodes directly in ``sdfg``, in a deterministic order."""
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]


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

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, List[str]]]:
        owned: Dict[SDFG, Set[str]] = {}
        copies: Dict[str, List[str]] = {}
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

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, int]]:
        fine_grained = config.Config.get_bool("compiler", "cuda", "dynamic_map_fine_grained")
        elements = dynamic_map_state_elements(fine_grained, gpu_helpers.dynamic_map_block_size())
        added = {}
        for kernel_sdfg, kernel_state, kernel_entry in gpu_helpers.gpu_kernels(sdfg):
            dynamic_maps = self._dynamic_maps(kernel_sdfg, kernel_state, kernel_state.scope_subgraph(kernel_entry))
            if not dynamic_maps:
                continue
            index_type = common.gpu_dynamic_map_index_type(kernel_sdfg, kernel_state, kernel_entry)
            state_names: Dict[SDFG, str] = {}
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
    def _dynamic_maps(sdfg: SDFG, state: SDFGState, graph) -> List[Tuple[SDFG, SDFGState, nodes.MapEntry]]:
        result = []
        for node in graph.nodes():
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
                result.append((sdfg, state, node))
            elif isinstance(node, nodes.NestedSDFG):
                for nstate in node.sdfg.states():
                    result.extend(LowerDynamicMapState._dynamic_maps(node.sdfg, nstate, nstate))
        return result


@dataclass
class SharedMemoryLevel:
    """The dynamic shared memory of one SDFG within a GPU kernel."""

    #: The SDFG
    sdfg: SDFG
    #: The offset and size (in bytes, in the symbols of ``sdfg``) of each of its dynamic containers
    containers: Dict[str, Tuple[symbolic.SymbolicType, symbolic.SymbolicType]] = field(default_factory=dict)
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
    levels: List[SharedMemoryLevel] = field(default_factory=list)


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
    offset is a constant. The number of bytes to launch each kernel with is stored in the
    ``_cuda_dynamic_shared_memory`` attribute of the kernel map entry, its static bytes in ``_cuda_static_shared_memory``.
    """

    CATEGORY: str = "GPU"

    def depends_on(self):
        return {LowerDynamicMapState}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[nodes.MapEntry, KernelSharedMemory]]:
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
        candidates: List[Tuple[SDFG, str, int]] = []
        dynamic: Set[Tuple[SDFG, str]] = set()
        for level_sdfg, names in self._containers(sdfg, kernel_level, top_nested):
            for name in names:
                desc = level_sdfg.arrays[name]
                size = desc.total_size_in_bytes
                is_symbolic = symbolic.issymbolic(size, level_sdfg.constants)
                placement = dtypes.is_dynamic_shared(desc.storage)
                if placement is False:
                    if is_symbolic:
                        raise ValueError(
                            f'Shared memory container "{name}" of kernel "{kernel_label}" is placed in '
                            f"static shared memory, which requires a constant size (got {size} bytes). "
                            "Place it in dynamic shared memory with "
                            "`StorageType.GPU_Shared(dynamic=True)`, or leave the placement to the code "
                            "generator with `StorageType.GPU_Shared`."
                        )
                    plan.static_bytes += int(symbolic.evaluate(size, level_sdfg.constants))
                elif placement is True or is_symbolic:
                    dynamic.add((level_sdfg, name))
                else:
                    candidates.append((level_sdfg, name, int(symbolic.evaluate(size, level_sdfg.constants))))

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
                level_sdfg.arrays[name].storage = dtypes.StorageType.GPU_Shared(dynamic=(level_sdfg, name) in dynamic)

        # Lay out dynamic shared memory
        if dynamic:
            plan.dynamic_bytes = self._layout(sdfg, kernel_level, top_nested, 0, dynamic, plan.levels)
            self._check_launch_symbols(state, kernel_entry, kernel_nodes, plan.dynamic_bytes)
        return plan

    def _containers(self, sdfg: SDFG, names: List[str], nested: List[nodes.NestedSDFG]) -> List[Tuple[SDFG, List[str]]]:
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
        names: List[str],
        nested: List[nodes.NestedSDFG],
        base: symbolic.SymbolicType,
        dynamic: Set[Tuple[SDFG, str]],
        levels: List[SharedMemoryLevel],
    ) -> symbolic.SymbolicType:
        """
        Lays out the dynamic containers of an SDFG, followed by those of its nested SDFGs, starting at ``base``.

        :return: The end of the layout, in the symbols of ``sdfg``.
        """
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
            cursor = offset + size
            cursor_alignment = alignment if symbolic.is_multiple(size, alignment) else 1

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
            inner_end = self._layout(nsdfg, nested_names, _nested_sdfg_nodes(nsdfg), inner_base, dynamic, levels)

            # Translate the end of the nested part to the symbols of this SDFG
            mapping = {symbolic.symbol(k): symbolic.pystr_to_symbolic(v) for k, v in nsdfg_node.symbol_mapping.items()}
            cursor = (
                symbolic.pystr_to_symbolic(inner_end).subs(mapping) if symbolic.issymbolic(inner_end) else inner_end
            )
            cursor_alignment = 1

        level.end = cursor
        if level.containers:
            levels.append(level)
        return cursor

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

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[int]:
        plans: Dict[nodes.MapEntry, KernelSharedMemory] = pipeline_results.get(PlanSharedMemory.__name__) or {}
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
        not isinstance(desc, dt.View)
        and desc.storage == dtypes.StorageType.GPU_Shared
        and dtypes.is_dynamic_shared(desc.storage) is True
    )


def plan_gpu_shared_memory(sdfg: SDFG) -> Dict[nodes.MapEntry, KernelSharedMemory]:
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
