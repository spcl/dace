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


##################################################
###                Scope Dict                  ###
### is expensive to generate, should be cached ###
##################################################


def get_sdfg_scope_dict(sdfg: SDFG) -> Dict[SDFGState, Dict[nodes.Node, Optional[nodes.Node]]]:
    scopes = {}
    for state in sdfg.states():
        scopes[state] = state.scope_dict()
    return scopes


###################################
###  Checking Common Conditions ###
###################################


def has_GPU_schedule(node: nodes.Node) -> bool:
    schedule = None
    if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
        schedule = node.map.schedule
    elif isinstance(node, nodes.LibraryNode):
        schedule = node.schedule
    else:
        assert False
    return schedule in dtypes.GPU_SCHEDULES


#: Connectors the Python frontend wires to ``__pystate`` around a callback, to block reordering.
PYSTATE_CONNECTORS = frozenset({'__istate', '__ostate'})
PYSTATE = '__pystate'


def callback_symbol_names(sdfg: SDFG) -> OrderedSet[str]:
    """Names of the ``dace.callback`` symbols declared in ``sdfg`` or any SDFG nested in it."""
    names: OrderedSet[str] = OrderedSet()
    for scope in sdfg.all_sdfgs_recursive():
        for name, stype in scope.symbols.items():
            if isinstance(stype, dtypes.callback):
                names.add(name)
    return names


def is_callback_tasklet(node: nodes.Node, sdfg: SDFG, callback_names: Optional[OrderedSet[str]] = None) -> bool:
    """A tasklet that calls back into Python, so it can only run on the host.

    Neither kind of callback can be offloaded: a Python callback needs the interpreter, and a GPU
    callback is itself a launch. The frontend wires ``__pystate`` through ``__istate``/``__ostate``,
    and the callee is a ``dace.callback`` symbol the code names; either marker can be absent.

    :param callback_names: :func:`callback_symbol_names` of ``sdfg``, when the caller asks per node.
    """
    if not isinstance(node, nodes.Tasklet):
        return False
    if PYSTATE_CONNECTORS & (OrderedSet(node.in_connectors) | OrderedSet(node.out_connectors)):
        return True
    names = callback_symbol_names(sdfg) if callback_names is None else callback_names
    code = node.code.as_string or ''
    return any(name in code for name in names)


def scope_holds_callback(state: SDFGState,
                         entry: Optional[nodes.MapEntry],
                         scope_children: Dict[Optional[nodes.Node], List[nodes.Node]],
                         sdfg: SDFG,
                         callback_names: Optional[OrderedSet[str]] = None) -> bool:
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
    """Any map or library node anywhere in ``sdfg`` that the schedule phase put on the device."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)) and has_GPU_schedule(node):
            return True
    return False


def is_array_stored_on_GPU(sdfg: SDFG, array_name: str) -> bool:
    storage = sdfg.arrays[array_name].storage
    return storage == dtypes.StorageType.GPU_Global or storage in dtypes.GPU_STORAGES


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
    """One write to ``name`` provably touches every element of its descriptor.

    A covering subset is not enough: an indirect write carries the whole array as its subset while
    its volume counts what it writes, so both must match the descriptor.
    """
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
                if written is not None and written.covers(whole) and symbolic.equal(
                        memlet.volume, desc.total_size, is_length=False):
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
    """Every container this SDFG writes, read off the graph rather than off the placement IR.

    An incoming edge is the write, whichever node carries it: a tasklet, a map exit and a nested
    SDFG's output connector all reach the container the same way. What is never written keeps the
    contents it was called with, so its home copy stays valid for the whole run.
    """
    written: OrderedSet[str] = OrderedSet()
    for state in sdfg.states():
        for node in state.data_nodes():
            if state.in_degree(node) > 0:
                written.add(node.data)
    return written


def is_unoffloadable(data_name: str, sdfg: SDFG) -> bool:
    """A descriptor this pass does not place: a structure, or a container of containers.

    These have no single buffer whose location can be decided and copied -- a ``Structure`` is a
    record of other descriptors, and a ``ContainerArray`` an array of them -- so they are skipped
    rather than classified. ``ContainerArray`` needs saying explicitly because it derives from
    ``Array`` and would otherwise read as an ordinary buffer.
    """
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, (data.Structure, data.StructureView, data.ContainerArray, data.ContainerView))


def is_scalar(data_name: str, sdfg: SDFG) -> bool:
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Scalar)


def is_array(data_name: str, sdfg: SDFG) -> bool:
    """A buffer with a location of its own.

    ``ArrayView``, ``ContainerView`` and ``ContainerArray`` all derive from ``Array``, so the bare
    isinstance answers True for an alias and for a container of containers. A view is placed with
    the container it aliases, and the container kinds are not placed at all.
    """
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return (isinstance(desc, data.Array) and not isinstance(desc, data.View) and not is_unoffloadable(data_name, sdfg))


def is_view(data_name: str, sdfg: SDFG) -> bool:
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.View)


def enclosing_kernel(scopes: Dict[nodes.Node, Optional[nodes.Node]], node: nodes.Node) -> Optional[nodes.MapEntry]:
    """The nearest enclosing map with a GPU schedule, or None outside every kernel."""
    scope = scopes[node]
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES:
            return scope
        scope = scopes[scope]
    return None


def data_written_by_device_code(sdfg: SDFG) -> OrderedSet[str]:
    """Every descriptor a GPU-scheduled scope writes whose value has to outlive that scope.

    Either through the scope's exit, or by an access node inside a kernel that is also accessed
    outside it or under another kernel: a size-1 wrapper can pull a tasklet and the node it writes
    into one kernel, leaving nothing at the exit.
    """
    through_the_exit: OrderedSet[str] = OrderedSet()
    written_inside: OrderedSet[str] = OrderedSet()
    kernels_per_data: Dict[str, OrderedSet[Optional[nodes.MapEntry]]] = {}
    for state in sdfg.states():
        scopes = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and has_GPU_schedule(node):
                through_the_exit |= get_data_used_by_outgoing_access_nodes(sdfg,
                                                                           state,
                                                                           node,
                                                                           include_scalars=True,
                                                                           ordering=False,
                                                                           through_copies=False)
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
                placed.add(f'{nested.cfg_id}.{name}')
    return placed


def refuse_by_value_scalars_the_device_writes(sdfg: SDFG) -> None:
    """Raise if a Scalar a kernel writes would reach that kernel by value, which discards the write."""
    offenders = [
        name for name in data_written_by_device_code(sdfg)
        if is_scalar(name, sdfg) and sdfg.arrays[name].storage != dtypes.StorageType.GPU_Global
    ]
    if offenders:
        raise ValueError(f'device code writes {offenders}, still Scalars in host storage; a kernel takes those '
                         'by value, so the write would be lost')


def register_kernel_local_transients(sdfg: SDFG) -> None:
    """Storage for a transient every access of which is inside one kernel: a register.

    The copy analysis places the containers that CROSS the host/device boundary and leaves the rest
    at ``Default``, which is host memory. A transient the kernel both writes and reads -- the scalar
    a fused map keeps its intermediate in -- is then a host allocation named only by device code,
    and the copy into it is host-to-device inside a kernel: the generator's dispatcher answers that
    pattern with ``IllegalCopy``. It never emits one here, but registering the target and not using
    it trips the code generator's own consistency check, which fires as a bare AssertionError naming
    nothing. The transformation this pass replaced made the same descriptors registers.
    """
    for nested in sdfg.all_sdfgs_recursive():
        local: OrderedSet = OrderedSet()
        escapes: OrderedSet = OrderedSet()
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


def view_origin(state: SDFGState, node: nodes.AccessNode) -> Optional[str]:
    """The container ``node`` ultimately aliases, following a chain of views, or None."""
    viewed = get_last_view_node(state, node)
    return viewed.data if viewed is not None else None


def is_stream(data_name: str, sdfg: SDFG) -> bool:
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Stream)


def is_length1_array(data_name: str, sdfg: SDFG) -> bool:
    """A length-1 buffer that could be held as a scalar instead.

    A view is excluded: a ``Scalar`` cannot carry the ``views`` alias edge, which is why
    ``ConvertLengthOneArraysToScalars`` exempts one as well -- offering it a view to convert asks it
    for a rewrite it refuses.
    """
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return is_array(data_name, sdfg) and len(desc.shape) == 1 and desc.shape[0] == 1


#######################
###  SDFG Traversal ###
#######################


def get_children(state: SDFGState, node: nodes.Node) -> OrderedSet:
    return OrderedSet(e.dst for e in state.out_edges(node))


def get_predecessors(state: SDFGState, node: nodes.Node) -> OrderedSet:
    return OrderedSet(e.src for e in state.in_edges(node))


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
    """Call ``method`` on each node once all of its predecessors had it, else in :func:`traverse_IR` order.

    A join (the close node of a conditional) must hear from every arm before it forwards anything.
    """
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
        raise RuntimeError(f'the offloading IR is not a DAG: {stuck} are never reached by all predecessors')


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
            raise ValueError(f'unhandled IR node type {OffloadingIRNode.get_type_as_str(curr.type)}')


########################################
###  Get Arrays Used by Access Nodes ###
########################################


def get_data_used_by_incoming_access_nodes(sdfg: SDFG,
                                           state: SDFGState,
                                           node: nodes.Node,
                                           include_scalars: bool = False) -> OrderedSet[str]:

    def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]) -> OrderedSet:
        if node in visited_set:  # the visited set is necessary for edge cases, e.g. an access node A whose predecessor B is a view node refering back to A
            return OrderedSet()
        visited_set.add(node)

        # find accessed arrays
        arrays: OrderedSet[str] = OrderedSet()
        if isinstance(node, nodes.AccessNode):
            data_name = node.data
            if is_array(data_name, sdfg):
                arrays.add(data_name)

            elif is_view(data_name, sdfg):  # trace it if it is a view
                # once the view access node is known, its original access node can be found and its
                # data added. A chain that reaches no access node has no origin to place, and
                # recursing on None asks the state for the edges of a node it does not hold.
                original = get_last_view_node(state, node)
                if original is not None:
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
    """Data of the access nodes downstream of ``node``.

    Placement follows empty memlets (``ordering``) and container-to-container copies
    (``through_copies``); a write analysis must not, since neither is a write by ``node``.
    """

    def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]) -> OrderedSet:
        if node in visited_set:  # the visited set is necessary for edge cases, e.g. an access node A whose successor B is a view node refering back to A
            return OrderedSet()
        visited_set.add(node)

        # find accessed arrays
        arrays: OrderedSet[str] = OrderedSet()
        if isinstance(node, nodes.AccessNode):
            data_name = node.data

            if is_array(data_name, sdfg):
                arrays.add(data_name)

            elif is_view(data_name, sdfg):  # trace it if it is a view
                # once the view access node is known, its original access node can be found and its
                # data added. A chain that reaches no access node has no origin to place, and
                # recursing on None asks the state for the edges of a node it does not hold.
                original = get_last_view_node(state, node)
                if original is not None:
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


############################
###  Map Creation Helper ###
############################


def get_new_map_identifiers(state: SDFGState, map_label: str, map_param: str) -> Tuple[str, str]:
    """A label and a map parameter that collide with nothing the SDFG already knows.

    The parameter becomes a SYMBOL, so uniqueness has to be checked against every name that could
    already carry assumptions -- the SDFG's own symbol table and its parent's, the data descriptors,
    and the parameters of every map in the SDFG, not just this state's. Reusing a name that is
    already a symbol elsewhere would give one string two meanings with two sets of assumptions,
    which resolves differently depending on which one a later pass reaches for.
    """
    sdfg = state.sdfg
    taken: OrderedSet = OrderedSet()
    for node in state.nodes():
        if isinstance(
                node,
            (nodes.MapEntry, nodes.MapExit, nodes.Tasklet, nodes.AccessNode, nodes.LibraryNode, nodes.NestedSDFG)):
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
