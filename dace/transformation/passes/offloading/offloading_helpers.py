# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, List, Optional, Tuple

from dace.ordered import OrderedSet

from dace import dtypes, data
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg.state import ControlFlowRegion, ReturnBlock
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg.utils import get_last_view_node


def remove_empty_return_entries(entries: List[Tuple[ControlFlowRegion, SDFGState]]) -> None:
    """Remove each entry of ``separate_early_returns`` still empty, wiring its predecessors to its successor."""
    for region, entry in entries:
        successors = list(region.out_edges(entry))
        # Copies follow the entry on a plain edge; any other shape stays as it is.
        plain = len(successors) <= 1 and all(edge.data.is_unconditional() and not edge.data.assignments
                                             for edge in successors)
        if entry.number_of_nodes() > 0 or not plain:
            continue
        for edge in list(region.in_edges(entry)):
            for successor in successors:
                region.add_edge(edge.src, successor.dst, edge.data)
        region.remove_node(entry)


def separate_early_returns(sdfg: SDFG) -> List[Tuple[ControlFlowRegion, SDFGState]]:
    """Put an empty state before each return on ``sdfg``'s own level, so its copies run on the return's path alone.

    :return: every region given such a state, with that state; ``remove_empty_return_entries`` takes them out again.
    """
    entries: List[Tuple[ControlFlowRegion, SDFGState]] = []
    for region in list(sdfg.all_control_flow_regions()):
        for block in [block for block in region.nodes() if isinstance(block, ReturnBlock)]:
            entry = region.add_state_before(block, 'return_entry', is_start_block=block is region.start_block)
            entries.append((region, entry))
    return entries


def link_early_returns(IR: OffloadingIRNode) -> None:
    """Tie each state leading into a return to the level's end, whose copy-backs the return must run first."""
    entries: List[OffloadingIRNode] = []

    def collect(node: OffloadingIRNode) -> None:
        if node.type == OffloadingIRNode.STATE and isinstance(node.block, SDFGState) and any(
                isinstance(edge.dst, ReturnBlock) for edge in node.block.parent_graph.out_edges(node.block)):
            entries.append(node)

    traverse_IR(IR, collect)
    for node in entries:
        if IR.close not in node.next:
            node.append_node(IR.close)


# Scope Dict
# is expensive to generate, should be cached

# Checking Common Conditions

#: Connectors the Python frontend wires to ``__pystate`` around a callback, to block reordering.
PYSTATE_CONNECTORS = frozenset({'__istate', '__ostate'})
PYSTATE = '__pystate'


def callback_symbol_names(sdfg: SDFG) -> OrderedSet:
    """Names of the ``dace.callback`` symbols declared in ``sdfg`` or any SDFG nested in it."""
    names: OrderedSet[str] = OrderedSet()
    for scope in sdfg.all_sdfgs_recursive():
        for name, stype in scope.symbols.items():
            if isinstance(stype, dtypes.callback):
                names.add(name)
    return names


def is_callback_tasklet(node: nodes.Node, sdfg: SDFG, callback_names: Optional[OrderedSet] = None) -> bool:
    """A tasklet that calls back into Python, so it can only run on the host.

    Neither kind of callback can be offloaded: a Python callback needs the interpreter, and a GPU
    callback is itself a launch, so a kernel cannot issue it. Two markers, because either can be
    absent -- the frontend wires ``__pystate`` through ``__istate``/``__ostate`` to pin the
    ordering, and the callee itself is a ``dace.callback`` symbol the tasklet's code names.

    :param callback_names: :func:`callback_symbol_names` of ``sdfg``, when the caller asks per node.
    """
    if not isinstance(node, nodes.Tasklet):
        return False
    if PYSTATE_CONNECTORS & (OrderedSet(node.in_connectors) | OrderedSet(node.out_connectors)):
        return True
    names = callback_symbol_names(sdfg) if callback_names is None else callback_names
    if not names:
        return False
    code = node.code.as_string or ''
    return any(name in code for name in names)


def scope_holds_callback(state: SDFGState,
                         entry: Optional[nodes.MapEntry],
                         scope_children: Dict[Optional[nodes.Node], List[nodes.Node]],
                         sdfg: SDFG,
                         callback_names: Optional[OrderedSet] = None) -> bool:
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
    for scope in sdfg.all_sdfgs_recursive():
        for state in scope.states():
            for node in state.nodes():
                if is_callback_tasklet(node, scope):
                    return True
    return False


def view_origin(state: SDFGState, node: nodes.AccessNode) -> Optional[str]:
    """The container ``node`` ultimately aliases, following a chain of views, or None."""
    viewed = get_last_view_node(state, node)
    return viewed.data if viewed is not None else None


# SDFG Traversal

# Get Arrays Used by Access Nodes

# Map Creation Helper


def enclosing_kernel(scopes: dict, node: nodes.Node) -> Optional[nodes.MapEntry]:
    """The nearest enclosing map with a GPU schedule, or None outside every kernel."""
    scope = scopes[node]
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES:
            return scope
        scope = scopes[scope]
    return None


def get_predecessors(state, node):
    return OrderedSet(e.src for e in state.in_edges(node))


def is_scalar(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Scalar)


def is_stream(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Stream)


def is_view(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.View)


def is_array(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Array)


def is_length1_array(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Array) and len(desc.shape) == 1 and desc.shape[0] == 1


def traverse_IR(IR: OffloadingIRNode, method):
    # ITERATIVE for the same reason :meth:`OffloadingIRNode.get_all_tails` is: the IR chain is as
    # long as the program has blocks, and a recursive pre-order walk overran Python's stack on the
    # first application-sized graph. The explicit stack keeps the recursion's own order -- children
    # pushed REVERSED so they pop in ``node.next`` order -- so ``method`` sees the same sequence.
    visited_set = OrderedSet()
    stack = [IR]
    while stack:
        node = stack.pop()
        if node in visited_set:
            continue
        visited_set.add(node)
        method(node)
        stack.extend(reversed(node.next))


def traverse_same_level(IR: OffloadingIRNode, method):  #DFS
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
            raise ValueError(f'unhandled IR node type {OffloadingIRNode.get_type_as_str(curr.type)}')


def has_GPU_schedule(node):
    return get_schedule(node) in dtypes.GPU_SCHEDULES


def get_schedule(node):
    if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
        return node.map.schedule
    if isinstance(node, nodes.LibraryNode):
        return node.schedule
    raise TypeError(f'node {node} of type {type(node).__name__} carries no schedule')


def is_array_stored_on_GPU(sdfg, array_name):
    storage = sdfg.arrays[array_name].storage
    if storage in GPU_RESIDENT_STORAGES:
        return True
    elif storage in {
            dtypes.StorageType.Default, dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.CPU_Pinned, dtypes.StorageType.CPU_ThreadLocal
    }:
        return False
    else:
        raise NotImplementedError(f"array {array_name!r} lives in {storage}, which this pass does not offload")


def register_kernel_local_transients(sdfg: SDFG) -> None:
    """Storage for a transient every access of which is inside one kernel: a register.

    The copy analysis places the containers that CROSS the host/device boundary and leaves the
    rest at ``Default``, which is host memory. A transient the kernel both writes and reads --
    the scalar a fused map keeps its intermediate in -- is then a host allocation named only by
    device code, and the copy into it is host-to-device inside a kernel: the generator's
    dispatcher answers that pattern with ``IllegalCopy``. It never actually emits one here, but
    registering the target and not using it trips the code generator's own consistency check.
    The offloading this pass replaced made the same descriptors registers.
    """
    for nested in sdfg.all_sdfgs_recursive():
        local: OrderedSet[str] = OrderedSet()
        escapes: OrderedSet[str] = OrderedSet()
        for state in nested.states():
            scopes = state.scope_dict()
            for node in state.data_nodes():
                if node.data not in nested.arrays:
                    continue
                desc = nested.arrays[node.data]
                if not desc.transient or desc.storage in GPU_RESIDENT_STORAGES:
                    continue
                if isinstance(desc, (data.View, data.Stream)):
                    continue
                if enclosing_kernel(scopes, node):
                    local.add(node.data)
                else:
                    escapes.add(node.data)
        for name in local - escapes:
            nested.arrays[name].storage = dtypes.StorageType.Register


def get_data_used_by_incoming_access_nodes(sdfg: SDFG,
                                           state: SDFGState,
                                           node: nodes.Node,
                                           include_scalars: bool = False) -> OrderedSet[str]:

    def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]):
        # the visited set is necessary for edge cases, e.g. an access node A whose predecessor B is a view node
        # refering back to A
        if node in visited_set:
            return OrderedSet()
        visited_set.add(node)

        # find accessed arrays
        arrays: OrderedSet[str] = OrderedSet()
        if isinstance(node, nodes.AccessNode):
            data_name = node.data
            if is_array(data_name, sdfg):
                arrays.add(data_name)

            elif is_view(data_name, sdfg):  # trace it if it is a view
                original = get_last_view_node(
                    state, node
                )  # once the view access node is known, its original access node can be found and it's data added
                arrays |= recursion(original, visited_set)

            elif include_scalars and is_scalar(data_name, sdfg):
                arrays.add(data_name)

        # check if more access nodes UPstream
        for n in get_predecessors(state, node):
            if isinstance(n, nodes.AccessNode):
                arrays |= recursion(n, visited_set)

        return arrays

    return recursion(node, OrderedSet())


def get_data_used_by_outgoing_access_nodes(sdfg: SDFG,
                                           state: SDFGState,
                                           node: nodes.Node,
                                           include_scalars: bool = False,
                                           ordering: bool = True,
                                           through_copies: bool = True) -> OrderedSet[str]:
    """Data of the access nodes downstream of ``node``; ``ordering`` follows empty memlets too.

    Placement follows them, and relies on it (tsvc_2_5 ``reduce_inner_carry`` keeps its taskloop's
    output on the device that way). A write analysis must not: an empty memlet only orders, so the
    scalars CloudSC orders after ``zpsupsatsrce`` are not written by the kernel before them. Nor
    does it follow ``through_copies``: past the first non-view access node an edge is a copy the
    host issues, so polybench durbin's staged ``alpha_host`` is not written by the kernel before it.
    """

    def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]) -> OrderedSet[str]:
        # the visited set is necessary for edge cases, e.g. an access node A whose successor B is a view node
        # refering back to A
        if node in visited_set:
            return OrderedSet()
        visited_set.add(node)

        # find accessed arrays
        arrays: OrderedSet[str] = OrderedSet()
        if isinstance(node, nodes.AccessNode):
            data_name = node.data

            if is_array(data_name, sdfg):
                arrays.add(data_name)

            elif is_view(data_name, sdfg):  # trace it if it is a view
                original = get_last_view_node(
                    state, node
                )  # once the view access node is known, its original access node can be found and it's data added
                arrays |= recursion(original, visited_set)

            elif include_scalars and is_scalar(data_name, sdfg):
                arrays.add(data_name)

            if not through_copies and not is_view(data_name, sdfg):
                return arrays

        # check if more access nodes DOWNstream
        for edge in state.out_edges(node):
            if isinstance(edge.dst, nodes.AccessNode) and (ordering or not edge.data.is_empty()):
                arrays |= recursion(edge.dst, visited_set)

        return arrays

    return recursion(node, OrderedSet())


def get_new_map_identifiers(state: SDFGState, map_label: str, map_param: str):
    existing_labels = OrderedSet(node.label for node in state.nodes())
    existing_params = OrderedSet()
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            existing_params |= OrderedSet(node.map.params)

    suffix = 0
    new_label = map_label
    while new_label in existing_labels:
        suffix += 1
        new_label = f"{map_label}_{suffix}"

    suffix = 0
    new_param = map_param
    while new_param in existing_params:
        suffix += 1
        new_param = f"{map_param}_{suffix}"

    return new_label, new_param
