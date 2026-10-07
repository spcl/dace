# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple

from ordered_set import OrderedSet

from dace import data, dtypes, subsets, symbolic
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.state import ControlFlowRegion, ReturnBlock
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.sdfg.utils import get_last_view_node
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace import utils


def remove_empty_return_entries(entries: List[Tuple[ControlFlowRegion, SDFGState]]) -> None:
    """Remove each entry of ``separate_early_returns`` still empty, wiring its predecessors to its successor."""
    for region, entry in entries:
        successors = list(region.out_edges(entry))
        # Copies follow the entry on a plain edge; any other shape stays as it is.
        plain = len(successors) <= 1 and all(
            edge.data.is_unconditional() and not edge.data.assignments for edge in successors
        )
        if entry.number_of_nodes() > 0 or not plain:
            continue
        for edge in list(region.in_edges(entry)):
            for successor in successors:
                region.add_edge(edge.src, successor.dst, edge.data)
        region.remove_node(entry)


def separate_early_returns(sdfg: SDFG) -> List[Tuple[ControlFlowRegion, SDFGState]]:
    """Put an empty state before each return, for its copy-backs; return the (region, state) pairs."""
    entries: List[Tuple[ControlFlowRegion, SDFGState]] = []
    for region in list(sdfg.all_control_flow_regions()):
        for block in [block for block in region.nodes() if isinstance(block, ReturnBlock)]:
            entry = region.add_state_before(block, "return_entry", is_start_block=block is region.start_block)
            entries.append((region, entry))
    return entries


def link_early_returns(IR: OffloadingIRNode) -> None:
    """Tie each state leading into a return to the level's end, whose copy-backs the return must run first."""
    entries: List[OffloadingIRNode] = []

    def collect(node: OffloadingIRNode) -> None:
        if (
            node.type == OffloadingIRNode.STATE
            and isinstance(node.block, SDFGState)
            and any(isinstance(edge.dst, ReturnBlock) for edge in node.block.parent_graph.out_edges(node.block))
        ):
            entries.append(node)

    traverse_IR(IR, collect)
    for node in entries:
        if IR.close not in node.next:
            node.append_node(IR.close)


def get_sdfg_scope_dict(sdfg: SDFG) -> Dict[SDFGState, Dict[nodes.Node, Optional[nodes.Node]]]:
    """``scope_dict`` of every state, built once: it is expensive."""
    return {state: state.scope_dict() for state in sdfg.states()}


#: Connectors the Python frontend wires to ``__pystate`` around a callback, to block reordering.
PYSTATE_CONNECTORS = frozenset({"__istate", "__ostate"})


def callback_symbol_names(sdfg: SDFG) -> OrderedSet[str]:
    """Names of the ``dace.callback`` symbols declared in ``sdfg`` or any SDFG nested in it."""
    names: OrderedSet[str] = OrderedSet()
    for scope in sdfg.all_sdfgs_recursive():
        for name, stype in scope.symbols.items():
            if isinstance(stype, dtypes.callback):
                names.add(name)
    return names


def is_callback_tasklet(node: nodes.Node, sdfg: SDFG, callback_names: Optional[OrderedSet[str]] = None) -> bool:
    """A tasklet calling back into Python (``__pystate`` connectors or a ``dace.callback`` symbol): host only."""
    if not isinstance(node, nodes.Tasklet):
        return False
    if PYSTATE_CONNECTORS & (OrderedSet(node.in_connectors) | OrderedSet(node.out_connectors)):
        return True
    names = callback_symbol_names(sdfg) if callback_names is None else callback_names
    code = node.code.as_string or ""
    return any(name in code for name in names)


def scope_holds_callback(
    state: SDFGState,
    entry: Optional[nodes.MapEntry],
    scope_children: Dict[Optional[nodes.Node], List[nodes.Node]],
    sdfg: SDFG,
    callback_names: Optional[OrderedSet[str]] = None,
) -> bool:
    """``entry``'s scope contains a callback, at any depth, so the scope is host code."""
    names = callback_symbol_names(sdfg) if callback_names is None else callback_names
    for node in scope_children.get(entry, ()):
        if is_callback_tasklet(node, sdfg, names):
            return True
        if isinstance(node, nodes.MapEntry) and scope_holds_callback(state, node, scope_children, sdfg, names):
            return True
        if isinstance(node, nodes.NestedSDFG) and sdfg_holds_callback(node.sdfg):
            return True
    return False


def sdfg_holds_callback(sdfg: SDFG) -> bool:
    names = callback_symbol_names(sdfg)
    for scope in sdfg.all_sdfgs_recursive():
        for state in scope.states():
            for node in state.nodes():
                if is_callback_tasklet(node, scope, names):
                    return True
    return False


def sdfg_holds_gpu_schedule(sdfg: SDFG) -> bool:
    """Any map or library node anywhere in ``sdfg`` scheduled on the device."""
    return any(
        isinstance(node, (nodes.MapEntry, nodes.LibraryNode)) and node.schedule in dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )


def is_device_work(node: nodes.Node) -> bool:
    """A map or library node scheduled on the device, or a nested SDFG holding one."""
    if isinstance(node, nodes.NestedSDFG):
        return sdfg_holds_gpu_schedule(node.sdfg)
    return isinstance(node, (nodes.MapEntry, nodes.LibraryNode)) and node.schedule in dtypes.GPU_SCHEDULES


def scope_nodes(state: SDFGState, entry: nodes.MapEntry) -> List[nodes.Node]:
    return state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()


#: Expansions of a library node that emit code for inside a kernel; any other chosen expansion is a call
#: only host code can issue (a cub device reduce, a vendor BLAS call).
IN_KERNEL_IMPLEMENTATIONS = frozenset({"pure", "pure-seq", "CUDA (block)", "CUDA (block allreduce)"})


def is_device_wide_libnode(node: nodes.Node) -> bool:
    """A library node with a chosen expansion that host code issues, so no kernel can contain it."""
    from dace.libraries.standard.nodes.copy import CopyLibraryNode  # Avoid import loop
    from dace.libraries.standard.nodes.fill import FillLibraryNode  # Avoid import loop

    return (
        isinstance(node, nodes.LibraryNode)
        and not isinstance(node, (CopyLibraryNode, FillLibraryNode))
        and node.implementation is not None
        and node.implementation not in IN_KERNEL_IMPLEMENTATIONS
    )


def holds_device_wide_libnode(node: nodes.Node) -> bool:
    if isinstance(node, nodes.NestedSDFG):
        return any(is_device_wide_libnode(inner) for inner, _ in node.sdfg.all_nodes_recursive())
    return is_device_wide_libnode(node)


def is_array_stored_on_GPU(sdfg: SDFG, array_name: str) -> bool:
    return sdfg.arrays[array_name].storage in GPU_RESIDENT_STORAGES


def host_name(name: str) -> str:
    """Host twin of ``name``; a ``__return`` twin is a buffer, so it leaves the reserved namespace."""
    if name.startswith("__return"):
        return f"buffer__return{name[8:]}_host"
    return f"{name}_host"


def gpu_name(name: str) -> str:
    """Device twin of ``name``; see :func:`host_name`."""
    if name.startswith("__return"):
        return f"buffer__return{name[8:]}_gpu"
    return f"{name}_gpu"


def twin_name(sdfg: SDFG, name: str) -> str:
    """The name of ``name``'s copy on the side it does not live on."""
    return host_name(name) if is_array_stored_on_GPU(sdfg, name) else gpu_name(name)


def read_anywhere(sdfg: SDFG, name: str) -> bool:
    """Anything reads ``name``: an access node with an out-edge, or an interstate edge naming it."""
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            if any(node.data == name and state.out_degree(node) > 0 for node in state.data_nodes()):
                return True
        if any(name in edge.data.used_arrays(nested.arrays) for edge in nested.all_interstate_edges()):
            return True
    return False


def written_in_full(sdfg: SDFG, name: str) -> bool:
    """One write covers all of ``name``: by subset and by volume (an indirect write covers by subset only)."""
    desc = sdfg.arrays[name]
    whole = subsets.Range.from_array(desc)
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data != name:
                continue
            for edge in state.in_edges(node):
                memlet = edge.data
                if memlet.is_empty() or memlet.dynamic or memlet.wcr is not None:
                    continue
                written = memlet.get_dst_subset(edge, state)
                if (
                    written is not None
                    and written.covers(whole)
                    and symbolic.equal(memlet.volume, desc.total_size, is_length=False)
                ):
                    return True
    return False


def overwritten_before_any_read(sdfg: SDFG) -> OrderedSet[str]:
    """Host-resident signature arrays nothing reads and one write covers: staging them down is dead work."""
    dead: OrderedSet[str] = OrderedSet()
    for name, desc in sdfg.arrays.items():
        if desc.transient or not is_array(name, sdfg) or is_array_stored_on_GPU(sdfg, name):
            continue
        if not read_anywhere(sdfg, name) and written_in_full(sdfg, name):
            dead.add(name)
    return dead


def containers_written(sdfg: SDFG) -> OrderedSet:
    """Every container some access node of ``sdfg`` writes."""
    written: OrderedSet[str] = OrderedSet()
    for state in sdfg.states():
        for node in state.data_nodes():
            if state.in_degree(node) > 0:
                written.add(node.data)
    return written


def is_unoffloadable(data_name: str, sdfg: SDFG) -> bool:
    """A structure or container of containers: no single buffer to place."""
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, (data.Structure, data.StructureView, data.ContainerArray, data.ContainerView))


def is_array(data_name: str, sdfg: SDFG) -> bool:
    """A buffer with a location of its own: not a view (placed with its container), not a container kind, and
    not a constant (declared on both sides)."""
    desc = sdfg.arrays[data_name]
    return (
        isinstance(desc, data.Array)
        and not isinstance(desc, data.View)
        and not is_unoffloadable(data_name, sdfg)
        and data_name not in sdfg.constants
    )


def enclosing_kernel(scopes: Dict[nodes.Node, Optional[nodes.Node]], node: nodes.Node) -> Optional[nodes.MapEntry]:
    """The nearest enclosing map with a GPU schedule, or None outside every kernel."""
    scope = scopes[node]
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES:
            return scope
        scope = scopes[scope]
    return None


def data_written_by_device_code(sdfg: SDFG) -> OrderedSet[str]:
    """Descriptors a kernel writes that outlive it: through its exit, or inside it and accessed elsewhere too."""
    through_the_exit: OrderedSet[str] = OrderedSet()
    written_inside: OrderedSet[str] = OrderedSet()
    kernels_per_data: Dict[str, OrderedSet[Optional[nodes.MapEntry]]] = {}
    for state in sdfg.states():
        scopes = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and node.schedule in dtypes.GPU_SCHEDULES:
                through_the_exit |= get_data_used_by_access_nodes(
                    sdfg, state, node, downstream=True, include_scalars=True, ordering=False, through_copies=False
                )
            if not isinstance(node, nodes.AccessNode) or node.data not in sdfg.arrays:
                continue
            kernel = enclosing_kernel(scopes, node)
            kernels_per_data.setdefault(node.data, OrderedSet()).add(kernel)
            if kernel is not None and state.in_degree(node) > 0:
                written_inside.add(node.data)
    return through_the_exit | OrderedSet(name for name in written_inside if len(kernels_per_data[name]) > 1)


def device_resident(sdfg: SDFG) -> OrderedSet[str]:
    """Every container left in a GPU storage, qualified by the id of the SDFG that holds it."""
    placed: OrderedSet[str] = OrderedSet()
    for nested in sdfg.all_sdfgs_recursive():
        for name, desc in nested.arrays.items():
            if desc.storage in GPU_RESIDENT_STORAGES:
                placed.add(f"{nested.cfg_id}.{name}")
    return placed


def refuse_by_value_scalars_the_device_writes(sdfg: SDFG) -> None:
    """Raise if a Scalar a kernel writes would reach that kernel by value, which discards the write."""
    offenders = [
        name
        for name in data_written_by_device_code(sdfg)
        if isinstance(sdfg.arrays[name], data.Scalar) and sdfg.arrays[name].storage != dtypes.StorageType.GPU_Global
    ]
    if offenders:
        raise ValueError(
            f"device code writes {offenders}, still Scalars in host storage; a kernel takes those "
            "by value, so the write would be lost"
        )


def register_kernel_local_transients(sdfg: SDFG, placed_on_gpu: OrderedSet[str]) -> None:
    """Make a register of every transient (Default, or put on the device by this pass) that only one kernel accesses."""
    for nested in sdfg.all_sdfgs_recursive():
        kernels: Dict[str, OrderedSet[Optional[nodes.MapEntry]]] = {}
        for state in nested.states():
            scopes = state.scope_dict()
            for node in state.data_nodes():
                desc = nested.arrays.get(node.data)
                if (
                    desc is None
                    or not desc.transient
                    or isinstance(desc, (data.View, data.Stream))
                    or not (
                        desc.storage == dtypes.StorageType.Default or (nested is sdfg and node.data in placed_on_gpu)
                    )
                ):
                    continue
                kernels.setdefault(node.data, OrderedSet()).add(enclosing_kernel(scopes, node))
        for name, owners in kernels.items():
            if len(owners) == 1 and owners[0] is not None:
                nested.arrays[name].storage = dtypes.StorageType.Register


def view_origin(state: SDFGState, node: nodes.AccessNode) -> Optional[str]:
    """The container ``node`` ultimately aliases, following a chain of views, or None."""
    viewed = get_last_view_node(state, node)
    return viewed.data if viewed is not None else None


def is_length1_array(data_name: str, sdfg: SDFG) -> bool:
    """A length-1 array, not a view (a Scalar cannot carry the ``views`` edge)."""
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return is_array(data_name, sdfg) and len(desc.shape) == 1 and desc.shape[0] == 1


def traverse_IR(IR: OffloadingIRNode, method: Callable[[OffloadingIRNode], None]) -> None:
    """Pre-order walk calling ``method`` once per node; iterative, since the IR has a node per block."""
    visited_set: OrderedSet[OffloadingIRNode] = OrderedSet()
    stack = [IR]
    while stack:
        node = stack.pop()
        if node in visited_set:
            continue
        visited_set.add(node)
        method(node)
        stack.extend(reversed(node.next))


def traverse_IR_after_predecessors(IR: OffloadingIRNode, method: Callable[[OffloadingIRNode], None]) -> None:
    """Call ``method`` on each node after all its predecessors, so a join hears every arm first."""
    waiting: Dict[OffloadingIRNode, int] = {}

    def count(node: OffloadingIRNode) -> None:
        for next in node.next:
            waiting[next] = waiting.get(next, 0) + 1

    traverse_IR(IR, count)
    stack = [IR]
    while stack:
        node = stack.pop()
        method(node)
        ready = []
        for next in node.next:
            waiting[next] -= 1
            if waiting[next] == 0:
                ready.append(next)
        stack.extend(reversed(ready))
    stuck = [node.debug_name for node, pending in waiting.items() if pending]
    if stuck:
        raise RuntimeError(f"the offloading IR is not a DAG: {stuck} are never reached by all predecessors")


def traverse_same_level(IR: OffloadingIRNode, method: Callable[[OffloadingIRNode], None]) -> None:  # DFS
    queue = IR.next.copy()
    while queue:
        curr = queue.pop()
        if curr.type == OffloadingIRNode.STATE or curr.type == OffloadingIRNode.EDGE:  # data node
            method(curr)
            queue += curr.next

        elif curr.is_open_node():
            method(curr)
            queue += curr.close.next

        elif curr.type == OffloadingIRNode.CLOSE:
            break

        else:
            raise ValueError(f"unhandled IR node type {OffloadingIRNode.get_type_as_str(curr.type)}")


def get_data_used_by_access_nodes(
    sdfg: SDFG,
    state: SDFGState,
    node: nodes.Node,
    downstream: bool,
    include_scalars: bool = False,
    ordering: bool = True,
    through_copies: bool = True,
) -> OrderedSet[str]:
    """Arrays reachable from ``node`` through access nodes, following empty memlets and copies if asked."""
    arrays: OrderedSet[str] = OrderedSet()
    # Visited, because an access node and a view of it can refer to each other.
    visited: OrderedSet[nodes.Node] = OrderedSet()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        children: List[nodes.Node] = []
        stops = False
        if isinstance(current, nodes.AccessNode):
            name = current.data
            if is_array(name, sdfg) or (include_scalars and isinstance(sdfg.arrays[name], data.Scalar)):
                arrays.add(name)
            children = view_origin_nodes(sdfg, state, current)
            stops = not through_copies and not isinstance(sdfg.arrays[name], data.View)
        if not stops:
            children += neighboring_access_nodes(state, current, downstream, ordering)
        # Reversed, so they pop in order: the same visits in the same order as a recursion.
        stack.extend(reversed(children))
    return arrays


def view_origin_nodes(sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> List[nodes.AccessNode]:
    """The access node a view aliases, or nothing: a chain that reaches no access node has no origin to place."""
    if not isinstance(sdfg.arrays[node.data], data.View):
        return []
    origin = get_last_view_node(state, node)
    return [] if origin is None else [origin]


def neighboring_access_nodes(
    state: SDFGState, node: nodes.Node, downstream: bool, ordering: bool
) -> List[nodes.AccessNode]:
    edges = state.out_edges(node) if downstream else state.in_edges(node)
    neighbors = [(edge.dst if downstream else edge.src, edge) for edge in edges]
    return [
        neighbor
        for neighbor, edge in neighbors
        if isinstance(neighbor, nodes.AccessNode) and (ordering or not edge.data.is_empty())
    ]


def get_new_map_identifiers(state: SDFGState, map_label: str, map_param: str) -> Tuple[str, str]:
    """A map label new to the state and a parameter new to every symbol, descriptor and map parameter in reach."""
    sdfg = state.sdfg
    taken: OrderedSet = OrderedSet()
    for node in state.nodes():
        if isinstance(
            node, (nodes.MapEntry, nodes.MapExit, nodes.Tasklet, nodes.AccessNode, nodes.LibraryNode, nodes.NestedSDFG)
        ):
            taken.add(node.label)

    symbols: OrderedSet = OrderedSet()
    for scope in [sdfg] + list(sdfg.all_sdfgs_recursive()):
        symbols |= OrderedSet(scope.symbols)
        symbols |= OrderedSet(scope.arrays)
        for scope_state in scope.states():
            for node in scope_state.nodes():
                if isinstance(node, nodes.MapEntry):
                    symbols |= OrderedSet(node.map.params)
    parent = sdfg.parent_sdfg
    while parent is not None:
        symbols |= OrderedSet(parent.symbols)
        parent = parent.parent_sdfg

    return utils.find_new_name(map_label, taken), utils.find_new_name(map_param, symbols)
