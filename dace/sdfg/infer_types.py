# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import re
from collections import defaultdict
from dace import data, dtypes, symbolic
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, nodes, validation
from dace.sdfg.state import LoopRegion
from dace.sdfg import nodes
from dace.sdfg.graph import Edge, SubgraphView
from dace.sdfg.utils import dfs_topological_sort
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

#############################################################################
# Connector type inference


def infer_out_connector_type(sdfg: SDFG, state: SDFGState, node: nodes.CodeNode,
                             cname: str) -> Optional[dtypes.typeclass]:
    """
    Tries to infer a single output connector type on a Tasklet or Nested SDFG node.

    :param sdfg: The SDFG to infer in.
    :param state: The state in which the code node resides.
    :param node: The tasklet to infer.
    :param cname: An input/output connector name to infer.
    :return: The connector type, or None if a type could not be inferred.
    """
    e = next(state.out_edges_by_connector(node, cname))
    if cname is None:
        return None
    scalar = (bool(e.data.subset) and e.data.subset.num_elements() == 1
              and (not e.data.dynamic or (e.data.dynamic and e.data.wcr is not None)))
    if e.data.data is not None:
        allocated_as_scalar = (sdfg.arrays[e.data.data].storage is not dtypes.StorageType.GPU_Global)
    else:
        allocated_as_scalar = True

    # If nested SDFG, try to use internal array type
    if isinstance(node, nodes.NestedSDFG):
        scalar = (isinstance(node.sdfg.arrays[cname], data.Scalar) and allocated_as_scalar)
        dtype = node.sdfg.arrays[cname].dtype
        ctype = (dtype if scalar else dtypes.pointer(dtype))
    elif e.data.data is not None:  # Obtain type from memlet
        scalar |= isinstance(sdfg.arrays[e.data.data], data.Scalar)
        if isinstance(node, nodes.LibraryNode):
            scalar &= allocated_as_scalar
        dtype = sdfg.arrays[e.data.data].dtype
        ctype = (dtype if scalar else dtypes.pointer(dtype))
    else:
        return None

    return ctype


def infer_connector_types(sdfg: SDFG):
    """
    Infers connector types throughout an SDFG and its nested SDFGs in-place.

    :param sdfg: The SDFG to infer.
    """
    # Loop over states, and in a topological sort over each state's nodes
    for state in sdfg.states():
        for node in dfs_topological_sort(state):
            # Try to infer input connector type from node type or previous edges
            for e in state.in_edges(node):
                cname = e.dst_conn
                if cname is None:
                    continue
                scalar = bool(e.data.subset) and e.data.subset.num_elements() == 1
                if e.data.data is not None:
                    allocated_as_scalar = (sdfg.arrays[e.data.data].storage is not dtypes.StorageType.GPU_Global)
                else:
                    allocated_as_scalar = True

                if node.in_connectors[cname].type is None:
                    # If nested SDFG, try to use internal array type
                    if isinstance(node, nodes.NestedSDFG):
                        # NOTE: Scalars allocated on the host can be read by GPU kernels. Therefore, we do not need
                        # to use the `allocated_as_scalar` check here.
                        scalar = isinstance(node.sdfg.arrays[cname], data.Scalar)
                        struct = isinstance(node.sdfg.arrays[cname], data.Structure)
                        dtype = node.sdfg.arrays[cname].dtype
                        ctype = (dtype if scalar or struct else dtypes.pointer(dtype))
                    elif e.data.data is not None:  # Obtain type from memlet
                        scalar |= isinstance(sdfg.arrays[e.data.data], data.Scalar)
                        if isinstance(node, nodes.LibraryNode):
                            scalar &= allocated_as_scalar
                        dtype = sdfg.arrays[e.data.data].dtype
                        ctype = (dtype if scalar else dtypes.pointer(dtype))
                    else:  # Code->Code
                        src_edge = state.memlet_path(e)[0]
                        sconn = src_edge.src.out_connectors[src_edge.src_conn]
                        if sconn.type is None:
                            raise TypeError('Ambiguous or uninferable type in'
                                            ' connector "%s" of node "%s"' % (sconn, src_edge.src))
                        ctype = sconn
                    node.in_connectors[cname] = ctype

            # Try to infer outputs from output edges
            for e in state.out_edges(node):
                cname = e.src_conn
                if cname is None:
                    continue

                if node.out_connectors[cname].type is None:
                    ctype = infer_out_connector_type(sdfg, state, node, cname)
                    if ctype is not None:
                        node.out_connectors[cname] = ctype

            # Let the node infer other output types on its own
            node.infer_connector_types(sdfg, state)

            # If there are any remaining uninferable connectors, fail
            for e in state.out_edges(node):
                cname = e.src_conn
                if cname and node.out_connectors[cname] is None:
                    raise TypeError('Ambiguous or uninferable type in'
                                    ' connector "%s" of node "%s"' % (cname, node))


#############################################################################
# Default schedule and storage type inference


def set_default_schedule_and_storage_types(scope: Union[SDFG, SDFGState, nodes.EntryNode],
                                           parent_schedules: List[dtypes.ScheduleType] = None,
                                           use_parent_schedule: bool = False,
                                           state: SDFGState = None,
                                           child_nodes: Dict[nodes.Node, List[nodes.Node]] = None):
    """
    Sets default storage and schedule types throughout SDFG in-place.
    Replaces ``ScheduleType.Default`` and ``StorageType.Default``
    with the corresponding types according to the parent scope's schedule.

    The defaults for storage types are determined by the
    ``dtypes.SCOPEDEFAULT_STORAGE`` dictionary (for example, a GPU device
    schedule, by default, will allocate containers on the shared memory).
    Following storage type inference for a scope, nested scopes (e.g., map entry, nested SDFG)
    are evaluated using the ``dtypes.STORAGEDEFAULT_SCHEDULE`` dictionary (for example, a
    default map with only GPU arrays connected to it will execute on the GPU). This decision
    is superseded if the schedule is specified in ``dtypes.SCOPEDEFAULT_SCHEDULE`` (e.g.,
    a map nested in a CPU multi-core map will by default run within a single thread).
    If no default schedule is found while traversing the parent scopes, the chosen schedule will be
    determined based on the SDFG's device, as specified in ``dtypes.DEFAULT_TOPLEVEL_STORAGE`` and
    ``dtypes.DEFAULT_TOPLEVEL_SCHEDULE``.
    May raise ``InvalidSDFGNodeError`` if a default scope is ambiguous based on surrounding
    storage types.

    :param scope: The SDFG, state, or scope to infer.
    :param parent_schedules: A list of ScheduleType elements representing
                             an ordered list of schedules, from the global schedule
                             on the top-level SDFG (usually ``None``), up to this
                             point.
    :param use_parent_schedule: If True, uses the parent scope's schedule type
                                directly, instead of the default schedule type.
                                Used when expanding nested SDFGs to preserve their
                                top-level schedule.
    :param state: (Use when working with a single scope) The parent state.
    :param child_nodes: (Use when working with a single scope) A mapping of each scope entry
                        node to its children.
    """
    parent_schedules = parent_schedules or [None]

    if isinstance(scope, SDFG):
        # Set device for default top-level schedules and storages
        for state in scope.states():
            set_default_schedule_and_storage_types(state,
                                                   parent_schedules,
                                                   use_parent_schedule=use_parent_schedule,
                                                   state=state,
                                                   child_nodes=state.scope_children())

        # Take care of remaining scalars without access nodes
        for aname, desc in scope.arrays.items():
            # If not transient in a nested SDFG, take storage from parent, regardless of current type
            if not desc.transient and scope.parent_sdfg is not None:
                desc.storage = _get_storage_from_parent(aname, scope)
            elif ((desc.transient or scope.parent_sdfg is None) and desc.storage == dtypes.StorageType.Default):
                # Indeterminate storage type, set to register
                desc.storage = dtypes.StorageType.Register
        return

    # Setup arguments
    parent_node = None if isinstance(scope, SDFGState) else scope
    if state is None:
        if isinstance(scope, SDFGState):
            state = scope
        else:
            raise ValueError('SDFG state cannot be None when inferring a scope')
    if child_nodes is None:
        child_nodes = state.scope_children()

    ############################################

    # Set default storage types in this scope
    _set_default_storage_in_scope(state, parent_node, parent_schedules, child_nodes)

    # Set default schedules in this scope based on parent schedule and inferred storage types
    nested_scopes = _set_default_schedule_in_scope(state, parent_node, parent_schedules, child_nodes,
                                                   use_parent_schedule)

    # Loop over internal nested SDFGs and scope entry nodes
    for nnode in nested_scopes:
        # Continue through nested SDFGs
        if isinstance(nnode, nodes.NestedSDFG):
            nscope = nnode.sdfg
            child_nodes = None
            extra_parent_schedules = []
        else:
            nscope = nnode
            extra_parent_schedules = [nnode.schedule]
        set_default_schedule_and_storage_types(nscope,
                                               parent_schedules + extra_parent_schedules,
                                               use_parent_schedule=False,
                                               state=state,
                                               child_nodes=child_nodes)


def _determine_child_schedule(parent_schedules: List[dtypes.ScheduleType]) -> Optional[dtypes.ScheduleType]:
    for sched in reversed(parent_schedules):
        if sched is not None and sched in dtypes.SCOPEDEFAULT_SCHEDULE:
            child_sched = dtypes.SCOPEDEFAULT_SCHEDULE[sched]
            if child_sched is not None:
                return child_sched
    return None


def _determine_child_storage(parent_schedules: List[dtypes.ScheduleType]) -> Optional[dtypes.StorageType]:
    for sched in reversed(parent_schedules):
        if (sched is not None and sched in dtypes.SCOPEDEFAULT_STORAGE and sched != dtypes.ScheduleType.Sequential):
            child_sched = dtypes.SCOPEDEFAULT_STORAGE[sched]
            if child_sched is not None:
                return child_sched
    return None


def _determine_schedule_from_storage(state: SDFGState, node: nodes.Node) -> Optional[dtypes.ScheduleType]:
    child_schedule = None
    memlets: Set[str] = set()
    if node is None or isinstance(node, nodes.NestedSDFG):  # State or nested SDFG
        pass
    elif isinstance(node, nodes.EntryNode):
        # Test for storage of the scope by collecting all neighboring memlets
        memlets = set(e.data.data for e in state.out_edges(node) if not e.data.is_empty())
        exit_node = state.exit_node(node)
        memlets.update(e.data.data for e in state.in_edges(exit_node) if not e.data.is_empty())
    else:
        # A library node's ``host_connectors`` are the ones its expansion writes from host code even
        # when the rest of it runs on the device -- ArgReduce's CUB call answers into host scalars.
        # Their storage says nothing about where the node runs, and counting it makes every such
        # node read as conflicted, so a CUDA ArgReduce could not be compiled at all.
        host = node.host_connectors if isinstance(node, nodes.LibraryNode) else frozenset()
        memlets = set(e.data.data for e in state.all_edges(node)
                      if not e.data.is_empty() and (e.dst_conn if e.dst is node else e.src_conn) not in host)

    # From memlets, use non-scalar data descriptors for decision
    constraints: Set[dtypes.ScheduleType] = set()
    sdfg = state.parent
    for dname in memlets:
        if isinstance(sdfg.arrays[dname], data.Scalar):
            continue  # Skip scalars

        storage = sdfg.arrays[dname].storage
        if storage not in dtypes.STORAGEDEFAULT_SCHEDULE:
            continue
        sched = dtypes.STORAGEDEFAULT_SCHEDULE[storage]
        if sched is None:
            continue
        constraints.add(sched)

    # Copy/Fill library nodes legitimately bridge storages; schedule on the GPU if involved.
    from dace.libraries.standard.nodes.copy import CopyLibraryNode
    from dace.libraries.standard.nodes.fill import FillLibraryNode
    if isinstance(node, (CopyLibraryNode, FillLibraryNode)) and dtypes.ScheduleType.GPU_Device in constraints:
        return dtypes.ScheduleType.GPU_Device

    if not constraints:  # No constraints found
        child_schedule = None
    elif len(constraints) > 1:
        raise validation.InvalidSDFGNodeError(
            f'Cannot determine default schedule for node {node}. '
            'Multiple arrays that point to it say that it should be the following schedules: '
            f'{constraints}', state.parent, state.parent.node_id(state), state.node_id(node))
    else:
        child_schedule = next(iter(constraints))

    # If no valid schedules are found and there are no conflicts with storage, use default top-level schedule
    if child_schedule is None:
        child_schedule = dtypes.SCOPEDEFAULT_SCHEDULE[None]

    return child_schedule


#: ``name = expression``, with the comparisons that are not assignments (``<=``, ``==``, ...) left
#: alone, so a loop condition keeps its whole text and a loop step yields only its right-hand side.
ASSIGNMENT = re.compile(r'^\s*[A-Za-z_]\w*\s*=(?!=)\s*(?P<rhs>.+)$', re.S)


def assigned_expression(statement: str) -> str:
    """The right-hand side of ``statement`` when it assigns, else ``statement`` unchanged."""
    match = ASSIGNMENT.match(statement)
    return match.group('rhs') if match else statement


def symbol_origins(definitions: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """``symbol -> the symbols its value is a function of``, resolved transitively.

    ``definitions`` pairs a symbol with the source text that gives it a value: a nested SDFG's
    symbol mapping, or a loop's init / condition / update. A loop iterator bounded by ``p`` is a
    function of ``p``, which is what makes a write indexed by that iterator move with ``p``.
    """
    direct: Dict[str, Set[str]] = {}
    for name, expression in definitions:
        try:
            symbols = {str(s) for s in symbolic.pystr_to_symbolic(expression).free_symbols}
        except Exception:  # not a symbolic expression (a call, a subscript): nothing to resolve
            symbols = set()
        direct.setdefault(name, set()).update(symbols - {name})

    origins: Dict[str, Set[str]] = {}
    for name in direct:
        reached: Set[str] = set()
        stack = list(direct[name])
        while stack:
            symbol = stack.pop()
            if symbol in reached:
                continue
            reached.add(symbol)
            stack.extend(direct.get(symbol, ()))
        origins[name] = reached or {name}
    return origins


#: How far :func:`map_scope_carries_dependency` follows a write down through nested scopes before it
#: gives up and refuses the schedule. Three levels reach the shape canonicalization produces (map ->
#: NestedSDFG -> inner map or loop body).
WRITE_DESCENT_LIMIT = 3


def write_moves_with(state: SDFGState, edge, varying: Set[str], depth: int = 0) -> Optional[bool]:
    """Whether the write arriving on ``edge`` moves with every symbol in ``varying``.

    ``None`` means undecidable, which the caller reads as a conflict.

    A memlet arriving at a scope exit FROM A NESTED SCOPE has been propagated over that scope: the
    per-iteration ``aa[j, 0:N]`` of a row-wise map reads back as the whole ``aa[0:N, 0:N]``, whose
    free symbols name no map parameter at all. Judging that memlet directly calls every such map a
    carried dependency, which is how the gate below turned seven parallel TSVC kernels sequential.
    So a write that comes out of a nested scope is followed to the accesses that produced it, and a
    write this cannot follow stays undecided rather than being read off the widened subset.
    """
    memlet = edge.data
    if isinstance(edge.src, (nodes.MapExit, nodes.NestedSDFG)) and depth >= WRITE_DESCENT_LIMIT:
        return None

    if isinstance(edge.src, nodes.MapExit):
        inner = [e for e in state.in_edges(edge.src) if not e.data.is_empty() and e.data.data == memlet.data]
        if not inner:
            return None
        verdicts = [write_moves_with(state, e, varying, depth + 1) for e in inner]
        return None if None in verdicts else all(verdicts)

    if isinstance(edge.src, nodes.NestedSDFG):
        # Inside the nested SDFG the container is named by the connector and indexed in the nested
        # SDFG's own symbols, so a write index is asked what its value is a FUNCTION of rather than
        # which names it spells. A skewed wavefront writes ``a[_loop_it_0, _loop_it_1]``, whose
        # symbols are inner loop iterators; they move with the enclosing map because the loops that
        # define them are bounded by its parameter.
        conn = edge.src_conn
        if conn is None:
            return None
        definitions = [(str(name), str(outer)) for name, outer in edge.src.symbol_mapping.items()]
        for region in edge.src.sdfg.all_control_flow_regions(recursive=True):
            if not isinstance(region, LoopRegion) or not region.loop_variable:
                continue
            for block in (region.init_statement, region.loop_condition, region.update_statement):
                if block is not None:
                    definitions.append((region.loop_variable, assigned_expression(block.as_string)))
        origins = symbol_origins(definitions)

        verdicts = []
        for nested_state in edge.src.sdfg.all_states():
            for access in nested_state.data_nodes():
                if access.data != conn:
                    continue
                for inner_edge in nested_state.in_edges(access):
                    if inner_edge.data.is_empty() or inner_edge.data.wcr is not None:
                        continue
                    if inner_edge.data.subset is None:
                        return None
                    written_by = set()
                    for symbol in inner_edge.data.subset.free_symbols:
                        written_by |= origins.get(str(symbol), {str(symbol)})
                    verdicts.append(not (varying - written_by))
        if not verdicts:
            return None
        return all(verdicts)

    if memlet.subset is None:
        return None
    return not (varying - {str(s) for s in memlet.subset.free_symbols})


def map_scope_carries_dependency(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether ``entry``'s scope reads a container it also writes at an index that does not move with
    every map parameter -- a loop-carried dependency, which no parallel schedule preserves.

    Only the edges INSIDE the scope are read. The ones outside it are propagated, so a per-iteration
    write ``B[i]`` reads back as the whole ``B[0:N]`` and every map would look conflicted. The same
    widening happens one level down, on a write arriving from a nested scope, and there
    :func:`write_moves_with` follows the write to the accesses that produced it. A write that
    carries write-conflict resolution is a reduction the code generator already makes atomic, and a
    parameter with a single iteration cannot carry anything.
    """
    varying = {param for param, size in zip(entry.map.params, entry.map.range.size()) if size != 1}
    if not varying:
        return False
    try:
        exit_node = state.exit_node(entry)
    except (KeyError, StopIteration):  # scope without an exit: undecidable, so refuse the schedule
        return True
    read = {e.data.data for e in state.out_edges(entry) if not e.data.is_empty()}
    for edge in state.in_edges(exit_node):
        memlet = edge.data
        if memlet.is_empty() or memlet.wcr is not None or memlet.data not in read:
            continue
        if write_moves_with(state, edge, varying) is not True:
            return True
    return False


def _set_default_schedule_in_scope(state: SDFGState,
                                   parent_node: nodes.Node,
                                   parent_schedules: List[dtypes.ScheduleType],
                                   child_nodes: Dict[nodes.Node, List[nodes.Node]],
                                   use_parent_schedule: bool = False) -> List[Union[nodes.EntryNode, nodes.NestedSDFG]]:
    nested_scopes: List[Union[nodes.EntryNode, nodes.NestedSDFG]] = []

    # Try to determine schedule based on parent schedule(s)
    if use_parent_schedule:
        child_schedule = parent_schedules[-1]
    else:
        child_schedule = _determine_child_schedule(parent_schedules)

        # Special case for dynamic thread-block neighboring schedules
        if child_schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            from dace.transformation.helpers import gpu_map_has_explicit_dyn_threadblocks  # Avoid import loops
            if gpu_map_has_explicit_dyn_threadblocks(state, parent_node):
                child_schedule = dtypes.ScheduleType.GPU_ThreadBlock_Dynamic

    # Set child schedule type in scope
    for node in child_nodes[parent_node]:
        # Set default schedule types
        if isinstance(node, (nodes.EntryNode, nodes.NestedSDFG)):
            nested_scopes.append(node)
            if isinstance(node, nodes.EntryNode) and node.schedule == dtypes.ScheduleType.Default:
                # If parent schedules do not determine child schedule,
                # test for storage of the scope by collecting all neighboring memlets
                if child_schedule is None:
                    local_child_schedule = _determine_schedule_from_storage(state, node)
                else:
                    local_child_schedule = child_schedule
                # An OpenMP team over a loop-carried dependency is a data race, and the wrong answer
                # it gives is silent. Never CHOOSE that schedule here; an explicit one is the
                # author's to defend.
                if (local_child_schedule in (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent)
                        and isinstance(node, nodes.MapEntry) and map_scope_carries_dependency(state, node)):
                    local_child_schedule = dtypes.ScheduleType.Sequential
                node.schedule = local_child_schedule
        elif isinstance(node, nodes.LibraryNode):
            if node.schedule == dtypes.ScheduleType.Default:
                if child_schedule is None:
                    local_child_schedule = _determine_schedule_from_storage(state, node)
                else:
                    local_child_schedule = child_schedule
                node.schedule = local_child_schedule

    return nested_scopes


def _set_default_storage_in_scope(state: SDFGState, parent_node: Optional[nodes.Node],
                                  parent_schedules: List[dtypes.ScheduleType], child_nodes: Dict[nodes.Node,
                                                                                                 List[nodes.Node]]):
    # Special case for GPU maps without explicit thread-block assignment
    if (dtypes.ScheduleType.GPU_Device in parent_schedules
            and dtypes.ScheduleType.GPU_ThreadBlock not in parent_schedules
            and dtypes.ScheduleType.GPU_ThreadBlock_Dynamic not in parent_schedules):
        from dace.transformation.helpers import gpu_map_has_explicit_threadblocks  # Avoid import loops
        # Find GPU scopes without thread-block maps
        if not gpu_map_has_explicit_threadblocks(state, parent_node):
            # Do not modify external list
            parent_schedules = parent_schedules + [dtypes.ScheduleType.GPU_ThreadBlock]
    # End of special case

    sdfg = state.parent
    child_storage = _determine_child_storage(parent_schedules)
    if child_storage is None:
        child_storage = dtypes.SCOPEDEFAULT_STORAGE[None]

    exit_nodes = [state.exit_node(n) for n in child_nodes[parent_node] if isinstance(n, nodes.EntryNode)]
    scope_subgraph = SubgraphView(state, child_nodes[parent_node] + exit_nodes)

    # Loop over access nodes
    for node in scope_subgraph.nodes():
        if not isinstance(node, nodes.AccessNode):
            continue
        desc = node.desc(sdfg)
        # If not transient in a nested SDFG, take storage from parent, regardless of current type
        if not desc.transient and sdfg.parent is not None:
            desc.storage = _get_storage_from_parent(node.data, sdfg)
        elif desc.storage == dtypes.StorageType.Default:
            desc.storage = child_storage

    # Take care of code->code edges that do not have access nodes
    for edge in scope_subgraph.edges():
        if not edge.data.is_empty():
            desc = sdfg.arrays[edge.data.data]
            # If not transient in a nested SDFG, take storage from parent, regardless of current type
            if not desc.transient and sdfg.parent is not None:
                desc.storage = _get_storage_from_parent(edge.data.data, sdfg)
            elif desc.storage == dtypes.StorageType.Default:
                desc.storage = child_storage


def _get_storage_from_parent(data_name: str, sdfg: SDFG) -> dtypes.StorageType:
    """
    Retrieves the storage type of an array from its parent SDFG.

    :param data_name: The name of the data descriptor.
    :param sdfg: The parent SDFG.
    :return: The storage type of the data descriptor.
    """
    nsdfg_node = sdfg.parent_nsdfg_node
    parent_state = sdfg.parent
    parent_sdfg = parent_state.parent

    # Find data descriptor in parent SDFG
    # NOTE: Assuming that all members of a Structure have the same storage type.
    data_name = data_name.split('.')[0]
    if data_name in nsdfg_node.in_connectors:
        e = next(iter(parent_state.in_edges_by_connector(nsdfg_node, data_name)))
        return parent_sdfg.arrays[e.data.data].storage
    elif data_name in nsdfg_node.out_connectors:
        e = next(iter(parent_state.out_edges_by_connector(nsdfg_node, data_name)))
        return parent_sdfg.arrays[e.data.data].storage

    raise ValueError(f'Could not find data descriptor {data_name} in parent SDFG')


def infer_aliasing(node: nodes.NestedSDFG, sdfg: SDFG, state: SDFGState) -> None:
    """
    Infers aliasing information on nested SDFG arrays based on external edges and connectors.
    Operates in-place on nested SDFG node.

    :param node: The nested SDFG node.
    :param sdfg: Parent SDFG of the nested SDFG node.
    :param state: Parent state of the nested SDFG node.
    """
    data_to_conn: Dict[str, Set[str]] = defaultdict(set)

    def _infer_aliased_connectors(
        get_edges: Callable[[nodes.NestedSDFG], List[Edge[Memlet]]],
        get_conn: Callable[[Edge[Memlet]], str],
        outgoing: bool,
    ):
        for e in get_edges(node):
            if e.data.is_empty():  # Skip empty memlets
                continue

            # Get all addressed arrays (through views)
            dnames = _get_addressed_arrays(state, e, outgoing=outgoing)

            # Register data name mapping to matching connectors
            conn = get_conn(e)
            for dname in dnames:
                data_to_conn[dname].add(conn)

    # Infer for input arrays
    _infer_aliased_connectors(state.in_edges, lambda e: e.dst_conn, False)

    # Infer for output arrays
    _infer_aliased_connectors(state.out_edges, lambda e: e.src_conn, True)

    # If array is already connected to the nested SDFG in multiple, different connector names;
    # it may alias with others.
    for dname, conns in data_to_conn.items():
        # If the original array may alias already, set the child to alias too
        if len(conns) > 1 or sdfg.arrays[dname].may_alias:
            for aname in conns:
                # Modify internal arrays
                if aname in node.sdfg.arrays:
                    desc = node.sdfg.arrays[aname]
                    if isinstance(desc, data.Array):  # The only data type where may_alias can be set
                        desc.may_alias = True


def _get_addressed_arrays(state: SDFGState, edge: Edge[Memlet], outgoing: bool) -> Set[str]:
    """
    Helper function that returns the actual array data descriptor name from a memlet.
    Traces the memlet path out, including through views.
    """
    # Avoid import loop
    from dace.sdfg import utils as sdutil

    mpath = state.memlet_path(edge)
    last_node = mpath[-1].dst if outgoing else mpath[0].src
    if not isinstance(last_node, nodes.AccessNode):
        return {edge.data.data}

    # If access node, find viewed node
    last_node = sdutil.get_all_view_nodes(state, last_node)
    if last_node is None:
        return {edge.data.data}
    return set(n.data for n in last_node)
