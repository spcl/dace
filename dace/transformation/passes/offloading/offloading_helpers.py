# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple

from ordered_set import OrderedSet

from dace import dtypes, data
from dace.sdfg import nodes, SDFG, SDFGState
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.sdfg.utils import get_last_view_node
from dace import utils

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


def is_callback_tasklet(node: nodes.Node, sdfg: SDFG) -> bool:
    """A tasklet that calls back into Python, so it can only run on the host.

    Neither kind of callback can be offloaded: a Python callback needs the interpreter, and a GPU
    callback is itself a launch, so a kernel cannot issue it. Two markers, because either can be
    absent -- the frontend wires ``__pystate`` through ``__istate``/``__ostate`` to pin the
    ordering, and the callee itself is a ``dace.callback`` symbol the tasklet's code names.
    """
    if not isinstance(node, nodes.Tasklet):
        return False
    if PYSTATE_CONNECTORS & (OrderedSet(node.in_connectors) | OrderedSet(node.out_connectors)):
        return True
    code = node.code.as_string or ''
    for scope in [sdfg] + list(sdfg.all_sdfgs_recursive()):
        for name, stype in scope.symbols.items():
            if isinstance(stype, dtypes.callback) and name in code:
                return True
    return False


def scope_holds_callback(state: SDFGState, entry: Optional[nodes.MapEntry],
                         scope_children: Dict[Optional[nodes.Node], List[nodes.Node]], sdfg: SDFG) -> bool:
    """``entry``'s scope contains a callback, at any depth, so the scope is host code."""
    for node in scope_children.get(entry, ()):
        if is_callback_tasklet(node, sdfg):
            return True
        if isinstance(node, nodes.MapEntry) and scope_holds_callback(state, node, scope_children, sdfg):
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


def sdfg_holds_gpu_schedule(sdfg: SDFG) -> bool:
    """Any map or library node anywhere in ``sdfg`` that the schedule phase put on the device."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)) and has_GPU_schedule(node):
            return True
    return False


def is_array_stored_on_GPU(sdfg: SDFG, array_name: str) -> bool:
    storage = sdfg.arrays[array_name].storage
    return storage == dtypes.StorageType.GPU_Global or storage in dtypes.GPU_STORAGES


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
    from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES

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

    def recursion(node: OffloadingIRNode, visited_set: OrderedSet) -> None:
        if node in visited_set:
            return
        visited_set.add(node)

        method(node)

        for next in node.next:
            recursion(next, visited_set)

    return recursion(IR, OrderedSet())


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
            assert False


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
                                           include_scalars: bool = False) -> OrderedSet[str]:

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

        # check if more access nodes DOWNstream
        for n in get_children(state, node):
            if isinstance(n, nodes.AccessNode):
                arrays |= recursion(n, visited_set)

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
