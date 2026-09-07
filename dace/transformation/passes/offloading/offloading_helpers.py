# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from ordered_set import OrderedSet

from dace import dtypes, data
from dace.sdfg import nodes, SDFG, SDFGState
from dace.transformation.passes.offloading.offloading_ir_node import OffloadingIRNode
from dace.sdfg.utils import get_last_view_node

##################################################
###                Scope Dict                  ###
### is expensive to generate, should be cached ###
##################################################


def get_sdfg_scope_dict(sdfg):
    scopes = {}
    for state in sdfg.states():
        scopes[state] = state.scope_dict()
    return scopes


###################################
###  Checking Common Conditions ###
###################################


def has_GPU_schedule(node):
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
    if PYSTATE_CONNECTORS & (set(node.in_connectors) | set(node.out_connectors)):
        return True
    code = node.code.as_string or ''
    for scope in [sdfg] + list(sdfg.all_sdfgs_recursive()):
        for name, stype in scope.symbols.items():
            if isinstance(stype, dtypes.callback) and name in code:
                return True
    return False


def scope_holds_callback(state: SDFGState, entry, scope_children: dict, sdfg: SDFG) -> bool:
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
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            owner = parent.sdfg if hasattr(parent, 'sdfg') else sdfg
            if is_callback_tasklet(node, owner):
                return True
    return False


def sdfg_holds_gpu_schedule(sdfg: SDFG) -> bool:
    """Any map or library node anywhere in ``sdfg`` that the schedule phase put on the device."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)) and has_GPU_schedule(node):
            return True
    return False


def is_array_stored_on_GPU(sdfg, array_name):
    storage = sdfg.arrays[array_name].storage
    return storage == dtypes.StorageType.GPU_Global or storage in dtypes.GPU_STORAGES


def is_scalar(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Scalar)


def is_array(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Array)


def is_view(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.View)


def enclosing_kernel(scopes: dict, node: nodes.Node):
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


def is_stream(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Stream)


def is_length1_array(data_name: str, sdfg: SDFG):
    assert data_name in sdfg.arrays
    desc = sdfg.arrays[data_name]
    return isinstance(desc, data.Array) and len(desc.shape) == 1 and desc.shape[0] == 1


#######################
###  SDFG Traversal ###
#######################


def get_children(state, node):
    return OrderedSet(e.dst for e in state.out_edges(node))


def get_predecessors(state, node):
    return OrderedSet(e.src for e in state.in_edges(node))


def traverse_IR(IR: OffloadingIRNode, method):

    def recursion(node, visited_set):
        if node in visited_set:
            return
        visited_set.add(node)

        method(node)

        for next in node.next:
            recursion(next, visited_set)

    return recursion(IR, OrderedSet())


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
            assert False


########################################
###  Get Arrays Used by Access Nodes ###
########################################


def get_data_used_by_incoming_access_nodes(sdfg: SDFG,
                                           state: SDFGState,
                                           node: nodes.Node,
                                           include_scalars: bool = False) -> OrderedSet[str]:

    def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]):
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
                                           include_scalars: bool = False) -> OrderedSet[str]:

    def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]):
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
                original = get_last_view_node(
                    state, node
                )  # once the view access node is known, its original access node can be found and it's data added
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


def get_new_map_identifiers(state: SDFGState, map_label: str, map_param: str):
    existing_labels = OrderedSet(getattr(node, "label", None) for node in state.nodes())
    existing_params = OrderedSet()
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            existing_params |= set(node.map.params)

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
