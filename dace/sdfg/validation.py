# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Exception classes and methods for validation of SDFGs. """

import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Set

import networkx as nx

from dace import dtypes, subsets, symbolic, data
from dace.dtypes import DebugInfo

if TYPE_CHECKING:
    import dace
    from dace.memlet import Memlet
    from dace.sdfg import SDFG
    from dace.sdfg import graph as gr
    from dace.sdfg.state import ControlFlowRegion

###########################################
# Validation


def validate(graph: 'dace.sdfg.graph.SubgraphView'):
    from dace.sdfg import SDFG, SDFGState
    from dace.sdfg.graph import SubgraphView
    gtype = graph.parent if isinstance(graph, SubgraphView) else graph
    if isinstance(gtype, SDFG):
        validate_sdfg(graph)
    elif isinstance(gtype, SDFGState):
        validate_state(graph)


def validate_control_flow_region(sdfg: 'SDFG',
                                 region: 'ControlFlowRegion',
                                 initialized_transients: Set[str],
                                 symbols: dict,
                                 references: Set[int] = None,
                                 **context: bool):
    from dace.sdfg.state import SDFGState, ControlFlowRegion, ConditionalBlock, LoopRegion
    from dace.sdfg.scope import is_in_scope
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState

    if len(region.source_nodes()) > 1:
        try:
            region.start_block
        except:
            raise InvalidSDFGError("Starting block is ambiguous or undefined.", sdfg, None)

    in_default_scope = None

    # Check every state separately
    start_block = region.start_block
    visited = set()
    visited_edges = set()
    # Run through blocks via DFS, ensuring that only the defined symbols are available for validation
    for edge in region.dfs_edges(start_block):
        # Source -> inter-state definition -> Destination
        ##########################################
        visited_edges.add(edge)

        # Reference check
        if id(edge) in references:
            raise InvalidSDFGInterstateEdgeError(
                f'Duplicate inter-state edge object detected: "{edge}". Please '
                'copy objects rather than using multiple references to the same one', sdfg, region.edge_id(edge))
        references.add(id(edge))
        if id(edge.data) in references:
            raise InvalidSDFGInterstateEdgeError(
                f'Duplicate inter-state edge object detected: "{edge}". Please '
                'copy objects rather than using multiple references to the same one', sdfg, region.edge_id(edge))
        references.add(id(edge.data))

        # Source
        if edge.src not in visited:
            visited.add(edge.src)
            if isinstance(edge.src, SDFGState):
                validate_state(edge.src, region.node_id(edge.src), sdfg, symbols, initialized_transients, references,
                               **context)
            elif isinstance(edge.src, ConditionalBlock):
                for _, r in edge.src.branches:
                    if r is not None:
                        validate_control_flow_region(sdfg, r, initialized_transients, symbols, references, **context)
            elif isinstance(edge.src, ControlFlowRegion):
                lsyms = copy.copy(symbols)
                if isinstance(edge.src, LoopRegion) and not edge.src.loop_variable in lsyms:
                    lsyms[edge.src.loop_variable] = None
                validate_control_flow_region(sdfg, edge.src, initialized_transients, lsyms, references, **context)

        ##########################################
        # Edge
        # Check inter-state edge for undefined symbols
        undef_syms = set(edge.data.free_symbols) - set(symbols.keys())
        if len(undef_syms) > 0:
            eid = region.edge_id(edge)
            raise InvalidSDFGInterstateEdgeError(
                f'Undefined symbols in edge: {undef_syms}. Add those with '
                '`sdfg.add_symbol()` or define outside with `dace.symbol()`', sdfg, eid)

        # Validate inter-state edge names
        issyms = edge.data.new_symbols(sdfg, symbols)
        if any(not dtypes.validate_name(s) for s in issyms):
            invalid = next(s for s in issyms if not dtypes.validate_name(s))
            eid = region.edge_id(edge)
            raise InvalidSDFGInterstateEdgeError("Invalid interstate symbol name %s" % invalid, sdfg, eid)

        # Ensure accessed data containers in assignments and conditions are accessible in this context
        ise_memlets = edge.data.get_read_memlets(sdfg.arrays)
        for memlet in ise_memlets:
            container = memlet.data
            if not _accessible(sdfg, container, context):
                # Check context w.r.t. maps
                if in_default_scope is None:  # Lazy-evaluate in_default_scope
                    in_default_scope = False
                    if sdfg.parent_nsdfg_node is not None:
                        if is_in_scope(sdfg.parent_sdfg, sdfg.parent, sdfg.parent_nsdfg_node,
                                       [dtypes.ScheduleType.Default]):
                            in_default_scope = True
                if in_default_scope is False:
                    eid = region.edge_id(edge)
                    raise InvalidSDFGInterstateEdgeError(
                        f'Trying to read an inaccessible data container "{container}" '
                        f'(Storage: {sdfg.arrays[container].storage}) in host code interstate edge', sdfg, eid)

        # Check for race conditions on edge assignments
        for aname, aval in edge.data.assignments.items():
            syms = symbolic.free_symbols_and_functions(aval)
            also_assigned = (syms & edge.data.assignments.keys()) - {aname}
            if also_assigned:
                eid = region.edge_id(edge)
                raise InvalidSDFGInterstateEdgeError(
                    f'Race condition: inter-state assignment {aname} = {aval} uses '
                    f'variables {also_assigned}, which are also modified in the same '
                    'edge.', sdfg, eid)

        # Add edge symbols into defined symbols
        symbols.update(issyms)

        ##########################################
        # Destination
        if edge.dst not in visited:
            visited.add(edge.dst)
            if isinstance(edge.dst, SDFGState):
                validate_state(edge.dst, region.node_id(edge.dst), sdfg, symbols, initialized_transients, references,
                               **context)
            elif isinstance(edge.dst, ConditionalBlock):
                for _, r in edge.dst.branches:
                    if r is not None:
                        validate_control_flow_region(sdfg, r, initialized_transients, symbols, references, **context)
            elif isinstance(edge.dst, ControlFlowRegion):
                lsyms = copy.copy(symbols)
                if isinstance(edge.dst, LoopRegion) and not edge.dst.loop_variable in lsyms:
                    lsyms[edge.dst.loop_variable] = None
                validate_control_flow_region(sdfg, edge.dst, initialized_transients, lsyms, references, **context)
    # End of block DFS

    # If there is only one block, the DFS will miss it
    if start_block not in visited:
        if isinstance(start_block, SDFGState):
            validate_state(start_block, region.node_id(start_block), sdfg, symbols, initialized_transients, references,
                           **context)
        elif isinstance(start_block, ConditionalBlock):
            for _, r in start_block.branches:
                if r is not None:
                    validate_control_flow_region(sdfg, r, initialized_transients, symbols, references, **context)
        elif isinstance(start_block, ControlFlowRegion):
            lsyms = copy.copy(symbols)
            if isinstance(start_block, LoopRegion) and not start_block.loop_variable in lsyms:
                lsyms[start_block.loop_variable] = None
            validate_control_flow_region(sdfg, start_block, initialized_transients, lsyms, references, **context)

    # Validate all inter-state edges (including self-loops not found by DFS)
    for eid, edge in enumerate(region.edges()):
        if edge in visited_edges:
            continue

        # Reference check
        if id(edge) in references:
            raise InvalidSDFGInterstateEdgeError(
                f'Duplicate inter-state edge object detected: "{edge}". Please '
                'copy objects rather than using multiple references to the same one', sdfg, eid)
        references.add(id(edge))
        if id(edge.data) in references:
            raise InvalidSDFGInterstateEdgeError(
                f'Duplicate inter-state edge object detected: "{edge}". Please '
                'copy objects rather than using multiple references to the same one', sdfg, eid)
        references.add(id(edge.data))

        issyms = edge.data.assignments.keys()
        if any(not dtypes.validate_name(s) for s in issyms):
            invalid = next(s for s in issyms if not dtypes.validate_name(s))
            raise InvalidSDFGInterstateEdgeError("Invalid interstate symbol name %s" % invalid, sdfg, eid)

        # Ensure accessed data containers in assignments and conditions are accessible in this context
        ise_memlets = edge.data.get_read_memlets(sdfg.arrays)
        for memlet in ise_memlets:
            container = memlet.data
            if not _accessible(sdfg, container, context):
                # Check context w.r.t. maps
                if in_default_scope is None:  # Lazy-evaluate in_default_scope
                    in_default_scope = False
                    if sdfg.parent_nsdfg_node is not None:
                        if is_in_scope(sdfg.parent_sdfg, sdfg.parent, sdfg.parent_nsdfg_node,
                                       [dtypes.ScheduleType.Default]):
                            in_default_scope = True
                if in_default_scope is False:
                    raise InvalidSDFGInterstateEdgeError(
                        f'Trying to read an inaccessible data container "{container}" '
                        f'(Storage: {sdfg.arrays[container].storage}) in host code interstate edge', sdfg, eid)

    # Check for interstate edges that write to scalars or arrays
    _no_writes_to_scalars_or_arrays_on_interstate_edges(sdfg)


def validate_sdfg(sdfg: 'dace.sdfg.SDFG', references: Set[int] = None, **context: bool):
    """ Verifies the correctness of an SDFG by applying multiple tests.

        :param sdfg: The SDFG to verify.
        :param references: An optional set keeping seen IDs for object
                           miscopy validation.
        :param context: An optional dictionary of boolean attributes
                        used to understand the context of this validation
                        (e.g., is this in a GPU kernel).

        Raises an InvalidSDFGError with the erroneous node/edge
        on failure.
    """
    # Avoid import loop
    from dace import data as dt
    from dace.sdfg.scope import is_devicelevel_gpu
    from dace.sdfg.state import ConditionalBlock

    references = references or set()

    # Reference check
    if id(sdfg) in references:
        raise InvalidSDFGError(
            f'Duplicate SDFG detected: "{sdfg.name}". Please copy objects '
            'rather than using multiple references to the same one', sdfg, None)
    references.add(id(sdfg))

    try:
        # SDFG-level checks
        if not dtypes.validate_name(sdfg.name):
            raise InvalidSDFGError("Invalid name", sdfg, None)

        for cfg in sdfg.all_control_flow_regions():
            if isinstance(cfg, ConditionalBlock):
                continue
            blocks = cfg.nodes()
            if len(blocks) != len(set([s.label for s in blocks])):
                raise InvalidSDFGError('Found multiple blocks with the same name in ' + cfg.name, sdfg, None)

        # Check the names of data descriptors and co.
        seen_names: Set[str] = set()
        for obj_names in [sdfg.arrays.keys(), sdfg.symbols.keys(), sdfg._rdistrarrays.keys(), sdfg._subarrays.keys()]:
            if not seen_names.isdisjoint(obj_names):
                raise InvalidSDFGError(
                    f'Found duplicated names: "{seen_names.intersection(obj_names)}". Please ensure '
                    'that the names of symbols, data descriptors, subarrays and rdistarrays are unique.', sdfg, None)
            seen_names.update(obj_names)

        # Ensure that there is a mentioning of constants in either the array or symbol.
        for const_name, (const_type, _) in sdfg.constants_prop.items():
            if const_name in sdfg.arrays:
                if const_type.dtype != sdfg.arrays[const_name].dtype:
                    # This should actually be an error, but there is a lots of code that depends on it.
                    warnings.warn(f'Mismatch between constant and data descriptor of "{const_name}", '
                                  f'expected to find "{const_type}" but found "{sdfg.arrays[const_name]}".')
            elif const_name in sdfg.symbols:
                if const_type.dtype != sdfg.symbols[const_name]:
                    # This should actually be an error, but there is a lots of code that depends on it.
                    warnings.warn(f'Mismatch between constant and symbol type of "{const_name}", '
                                  f'expected to find "{const_type}" but found "{sdfg.symbols[const_name]}".')

        # Validate data descriptors
        for name, desc in sdfg._arrays.items():
            if id(desc) in references:
                raise InvalidSDFGError(
                    f'Duplicate data descriptor object detected: "{name}". Please copy objects '
                    'rather than using multiple references to the same one', sdfg, None)
            references.add(id(desc))

            # Because of how the code generator works Scalars can not be return values.
            #  TODO: Remove this limitation as the CompiledSDFG contains logic for that.
            if (sdfg.parent is None and isinstance(desc, dt.Scalar) and name.startswith("__return")
                    and not desc.transient):
                raise InvalidSDFGError(
                    f'Cannot use scalar data descriptor ("{name}") as return value of a top-level function.', sdfg,
                    None)

            # Check for UndefinedSymbol in transient data shape (needed for memory allocation)
            if desc.transient:
                # Check dimensions
                for i, dim in enumerate(desc.shape):
                    if symbolic.is_undefined(dim):
                        raise InvalidSDFGError(
                            f'Transient data container "{name}" contains undefined symbol in dimension {i}, '
                            f'which is required for memory allocation', sdfg, None)

                # Check strides if array
                if hasattr(desc, 'strides'):
                    for i, stride in enumerate(desc.strides):
                        if symbolic.is_undefined(stride):
                            raise InvalidSDFGError(
                                f'Transient data container "{name}" contains undefined symbol in stride {i}, '
                                f'which is required for memory allocation', sdfg, None)

                # Check total size
                if hasattr(desc, 'total_size') and symbolic.is_undefined(desc.total_size):
                    raise InvalidSDFGError(
                        f'Transient data container "{name}" has undefined total size, '
                        f'which is required for memory allocation', sdfg, None)

                # Check any other undefined symbols in the data descriptor
                if any(symbolic.is_undefined(s) for s in desc.used_symbols(all_symbols=False)):
                    raise InvalidSDFGError(
                        f'Transient data container "{name}" has undefined symbols, '
                        f'which are required for memory allocation', sdfg, None)

            # Validate array names
            if name is not None and not dtypes.validate_name(name):
                raise InvalidSDFGError("Invalid array name %s" % name, sdfg, None)
            # Allocation lifetime checks
            if (desc.lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External)
                    and desc.storage == dtypes.StorageType.Register):
                raise InvalidSDFGError(
                    "Array %s cannot be both persistent/external and use Register as "
                    "storage type. Please use a different storage location." % name, sdfg, None)

        # Check if SDFG is located within a GPU kernel
        context['in_gpu'] = is_devicelevel_gpu(sdfg, None, None)

        initialized_transients = {'__pystate'}
        initialized_transients.update(sdfg.constants_prop.keys())
        symbols = copy.deepcopy(sdfg.symbols)
        symbols.update(sdfg.arrays)
        symbols.update({k: v for k, (v, _) in sdfg.constants_prop.items()})
        for desc in sdfg.arrays.values():
            for sym in desc.free_symbols:
                symbols[str(sym)] = sym.dtype

        if len(sdfg.nodes()) == 0:
            raise InvalidSDFGError("SDFGs are required to contain at least one state.", sdfg, None)

        validate_control_flow_region(sdfg, sdfg, initialized_transients, symbols, references, **context)

    except InvalidSDFGError as ex:
        # If the SDFG is invalid, save it
        fpath = os.path.join('_dacegraphs', 'invalid.sdfgz')
        sdfg.save(fpath, exception=ex, compress=True)
        ex.path = fpath
        raise


def _accessible(sdfg: 'dace.sdfg.SDFG', container: str, context: Dict[str, bool]):
    """
    Helper function that returns False if a data container cannot be accessed in the current SDFG context.
    """
    storage = sdfg.arrays[container].storage
    if storage == dtypes.StorageType.GPU_Global or storage in dtypes.GPU_STORAGES:
        return context.get('in_gpu', False)

    return True


def _is_scalar(edge: 'gr.MultiConnectorEdge[Memlet]', memlet_path: List['gr.MultiConnectorEdge[Memlet]']):
    """
    Helper function that determines if a memlet is going to dereference a scalar value.
    Returns False in any case the memlet _may not_ be dereferenced (but could be).
    """
    # If any of the connectors is a pointer, it takes precedence
    src_conn = memlet_path[0].src_conn
    if src_conn and src_conn in memlet_path[0].src.out_connectors:
        src_conntype = memlet_path[0].src.out_connectors[src_conn]
    else:
        src_conntype = None
    dst_conn = memlet_path[-1].dst_conn
    if dst_conn and dst_conn in memlet_path[0].dst.in_connectors:
        dst_conntype = memlet_path[-1].dst.in_connectors[dst_conn]
    else:
        dst_conntype = None
    for conntype in (src_conntype, dst_conntype):
        if isinstance(conntype, dtypes.pointer):
            return False

    # If the memlet is dynamically accessed, it may also not be a scalar
    if edge.data.dynamic and (edge.data.volume == -1 or edge.data.volume == 0):
        return False

    # If the memlet has more than one element, it is definitely not a scalar
    if edge.data.num_elements() != 1:
        return False

    # Otherwise, we can assume this is a scalar
    return True


def validate_state(state: 'dace.sdfg.SDFGState',
                   state_id: int = None,
                   sdfg: 'dace.sdfg.SDFG' = None,
                   symbols: Dict[str, dtypes.typeclass] = None,
                   initialized_transients: Set[str] = None,
                   references: Set[int] = None,
                   **context: bool):
    """ Verifies the correctness of an SDFG state by applying multiple
        tests. Raises an InvalidSDFGError with the erroneous node on
        failure.
    """
    # Avoid import loops
    from dace import data as dt
    from dace import subsets as sbs
    from dace.config import Config
    from dace.sdfg import SDFG
    from dace.sdfg import nodes as nd
    from dace.sdfg import utils as sdutil
    from dace.sdfg.scope import is_devicelevel_gpu, scope_contains_scope

    sdfg = sdfg or state.parent
    state_id = state_id if state_id is not None else state.parent_graph.node_id(state)
    symbols = symbols or {}
    initialized_transients = (initialized_transients if initialized_transients is not None else {'__pystate'})
    references = references or set()

    # Obtain whether we are already in an accelerator context
    if not hasattr(context, 'in_gpu'):
        context['in_gpu'] = is_devicelevel_gpu(sdfg, state, None)

    # Reference check
    if id(state) in references:
        raise InvalidSDFGError(
            f'Duplicate SDFG state detected: "{state.label}". Please copy objects '
            'rather than using multiple references to the same one', sdfg, state_id)
    references.add(id(state))

    if not dtypes.validate_name(state._label):
        raise InvalidSDFGError("Invalid state name", sdfg, state_id)

    if state.sdfg != sdfg:
        raise InvalidSDFGError("State does not point to the correct "
                               "parent", sdfg, state_id)

    # Unreachable
    ########################################
    if (sdfg.number_of_nodes() > 1 and sdfg.in_degree(state) == 0 and sdfg.out_degree(state) == 0):
        raise InvalidSDFGError("Unreachable state", sdfg, state_id)

    if state.has_cycles():
        raise InvalidSDFGError('State should be acyclic but contains cycles', sdfg, state_id)

    scope = state.scope_dict()

    for nid, node in enumerate(state.nodes()):
        # Reference check
        if id(node) in references:
            raise InvalidSDFGNodeError(
                f'Duplicate node detected: "{node}". Please copy objects '
                'rather than using multiple references to the same one', sdfg, state_id, nid)
        references.add(id(node))

        # Node validation
        try:
            if isinstance(node, nd.NestedSDFG):
                node.validate(sdfg, state, references, **context)
            else:
                node.validate(sdfg, state)
        except InvalidSDFGError:
            raise
        except Exception as ex:
            raise InvalidSDFGNodeError("Node validation failed: " + str(ex), sdfg, state_id, nid) from ex

        # Isolated nodes
        ########################################
        if state.in_degree(node) + state.out_degree(node) == 0:
            # One corner case: OK if this is a code node
            if isinstance(node, nd.CodeNode):
                pass
            else:
                raise InvalidSDFGNodeError("Isolated node", sdfg, state_id, nid)

        # Scope tests
        ########################################
        if isinstance(node, nd.EntryNode):
            try:
                state.exit_node(node)
            except StopIteration:
                raise InvalidSDFGNodeError(
                    "Entry node does not have matching "
                    "exit node",
                    sdfg,
                    state_id,
                    nid,
                )

        if isinstance(node, (nd.EntryNode, nd.ExitNode)):
            for iconn in node.in_connectors:
                if (iconn is not None and iconn.startswith("IN_") and ("OUT_" + iconn[3:]) not in node.out_connectors):
                    raise InvalidSDFGNodeError(
                        "No match for input connector %s in output "
                        "connectors" % iconn,
                        sdfg,
                        state_id,
                        nid,
                    )
            for oconn in node.out_connectors:
                if (oconn is not None and oconn.startswith("OUT_") and ("IN_" + oconn[4:]) not in node.in_connectors):
                    raise InvalidSDFGNodeError(
                        "No match for output connector %s in input "
                        "connectors" % oconn,
                        sdfg,
                        state_id,
                        nid,
                    )

        # Node-specific tests
        ########################################
        if isinstance(node, nd.AccessNode):
            if node.data not in sdfg.arrays:
                raise InvalidSDFGNodeError(
                    "Access node must point to a valid array name in the SDFG",
                    sdfg,
                    state_id,
                    nid,
                )
            arr = sdfg.arrays[node.data]

            # Verify View references
            if isinstance(arr, dt.View):
                if sdutil.get_view_edge(state, node) is None:
                    raise InvalidSDFGNodeError("Ambiguous or invalid edge to/from a View access node", sdfg, state_id,
                                               nid)

            # Find uninitialized transients
            if node.data not in initialized_transients:
                if isinstance(arr, dt.Reference):  # References are considered more conservatively
                    if any(e.dst_conn == 'set' for e in state.in_edges(node)):
                        initialized_transients.add(node.data)
                    else:
                        raise InvalidSDFGNodeError(
                            'Reference data descriptor was used before it was set. Set '
                            'it with an incoming memlet to the "set" connector', sdfg, state_id, nid)
                elif (arr.transient and state.in_degree(node) == 0 and state.out_degree(node) > 0
                      # Streams do not need to be initialized
                      and not isinstance(arr, dt.Stream)):
                    if node.setzero == False:
                        warnings.warn('WARNING: Use of uninitialized transient "%s" in state "%s"' %
                                      (node.data, state.label))

                # Register initialized transients
                if arr.transient and state.in_degree(node) > 0:
                    initialized_transients.add(node.data)

            nsdfg_node = sdfg.parent_nsdfg_node
            if nsdfg_node is not None:
                # Find unassociated non-transients access nodes
                node_data = node.data.split('.')[0]
                if (not arr.transient and node_data not in nsdfg_node.in_connectors
                        and node_data not in nsdfg_node.out_connectors):
                    raise InvalidSDFGNodeError(
                        f'Data descriptor "{node_data}" is not transient and used in a nested SDFG, '
                        'but does not have a matching connector on the outer SDFG node.', sdfg, state_id, nid)

                # Find writes to input-only arrays
                only_empty_inputs = all(e.data.is_empty() for e in state.in_edges(node))
                if (not arr.transient) and (not only_empty_inputs):
                    if node_data not in nsdfg_node.out_connectors:
                        raise InvalidSDFGNodeError(
                            'Data descriptor %s is '
                            'written to, but only given to nested SDFG as an '
                            'input connector' % node.data, sdfg, state_id, nid)

        if (isinstance(node, nd.ConsumeEntry) and "IN_stream" not in node.in_connectors):
            raise InvalidSDFGNodeError("Consume entry node must have an input stream", sdfg, state_id, nid)
        if (isinstance(node, nd.ConsumeEntry) and "OUT_stream" not in node.out_connectors):
            raise InvalidSDFGNodeError(
                "Consume entry node must have an internal stream",
                sdfg,
                state_id,
                nid,
            )

        # Connector tests
        ########################################
        # Tasklet connector tests
        if not isinstance(node, (nd.NestedSDFG, nd.LibraryNode)):
            # Check for duplicate connector names (unless it's a nested SDFG)
            if len(node.in_connectors.keys() & node.out_connectors.keys()) > 0:
                dups = node.in_connectors.keys() & node.out_connectors.keys()
                raise InvalidSDFGNodeError("Duplicate connectors: " + str(dups), sdfg, state_id, nid)

            for conn in node.in_connectors.keys() | node.out_connectors.keys():
                if conn in (sdfg.constants_prop.keys() | sdfg.symbols.keys() | sdfg.arrays.keys()):
                    if not isinstance(node, nd.EntryNode):  # Special case for dynamic map inputs
                        raise InvalidSDFGNodeError(
                            "Connector name '%s' is already used as a symbol, constant, or array name" % conn, sdfg,
                            state_id, nid)

        # Check for dangling connectors (incoming)
        for conn in node.in_connectors:
            incoming_edges = 0
            for e in state.in_edges(node):
                # Connector found
                if e.dst_conn == conn:
                    incoming_edges += 1

            if incoming_edges == 0:
                raise InvalidSDFGNodeError("Dangling in-connector %s" % conn, sdfg, state_id, nid)
            # Connectors may have only one incoming edge
            # Due to input connectors of scope exit, this is only correct
            # in some cases:
            if incoming_edges > 1 and not isinstance(node, nd.ExitNode):
                raise InvalidSDFGNodeError(
                    "Connector '%s' cannot have more "
                    "than one incoming edge, found %d" % (conn, incoming_edges),
                    sdfg,
                    state_id,
                    nid,
                )

        # Check for dangling connectors (outgoing)
        for conn in node.out_connectors:
            outgoing_edges = 0
            for e in state.out_edges(node):
                # Connector found
                if e.src_conn == conn:
                    outgoing_edges += 1

            if outgoing_edges == 0:
                raise InvalidSDFGNodeError("Dangling out-connector %s" % conn, sdfg, state_id, nid)

            # In case of scope exit or code node, only one outgoing edge per
            # connector is allowed.
            if outgoing_edges > 1 and isinstance(node, (nd.ExitNode, nd.CodeNode)):
                raise InvalidSDFGNodeError(
                    "Connector '%s' cannot have more "
                    "than one outgoing edge, found %d" % (conn, outgoing_edges),
                    sdfg,
                    state_id,
                    nid,
                )

        # Check for edges to nonexistent connectors
        for e in state.in_edges(node):
            if e.dst_conn is not None and e.dst_conn not in node.in_connectors:
                raise InvalidSDFGNodeError(
                    ("Memlet %s leading to " + "nonexistent connector %s") % (str(e.data), e.dst_conn),
                    sdfg,
                    state_id,
                    nid,
                )
        for e in state.out_edges(node):
            if e.src_conn is not None and e.src_conn not in node.out_connectors:
                raise InvalidSDFGNodeError(
                    ("Memlet %s coming from " + "nonexistent connector %s") % (str(e.data), e.src_conn),
                    sdfg,
                    state_id,
                    nid,
                )
        ########################################

    for eid, e in enumerate(state.edges()):
        # Reference check
        if id(e) in references:
            raise InvalidSDFGEdgeError(
                f'Duplicate memlet detected: "{e}". Please copy objects '
                'rather than using multiple references to the same one', sdfg, state_id, eid)
        references.add(id(e))
        if e.data.is_empty():
            pass
        elif id(e.data) in references:
            raise InvalidSDFGEdgeError(
                f'Duplicate memlet detected: "{e.data}". Please copy objects '
                'rather than using multiple references to the same one', sdfg, state_id, eid)
        references.add(id(e.data))

        # Edge validation
        try:
            e.data.validate(sdfg, state)
        except InvalidSDFGError:
            raise
        except Exception as ex:
            raise InvalidSDFGEdgeError("Edge validation failed: " + str(ex), sdfg, state_id, eid)

        # If the edge is a connection between two AccessNodes check if the subset has negative size.
        # NOTE: We _should_ do this check in `Memlet.validate()` however, this is not possible,
        #  because the connection between am AccessNode and a MapEntry, with a negative size, is
        #  legal because, the Map will not run in that case. However, this constellation can not
        #  be tested for in the Memlet's validation function, so we have to do it here.
        # NOTE: Zero size is explicitly allowed because it is essentially `memcpy(dst, src, 0)`
        #  which is save.
        # TODO: The AN to AN connection is the most obvious one, but it should be extended.
        if isinstance(e.src, nd.AccessNode) and isinstance(e.dst, nd.AccessNode):
            e_memlet: dace.Memlet = e.data
            if e_memlet.subset is not None:
                if any((ss < 0) == True for ss in e_memlet.subset.size()):
                    raise InvalidSDFGEdgeError(
                        f'`subset` of an AccessNode to AccessNode Memlet contains a negative size; the size was {e_memlet.subset.size()}',
                        sdfg, state_id, eid)
            if e_memlet.other_subset is not None:
                if any((ss < 0) == True for ss in e_memlet.other_subset.size()):
                    raise InvalidSDFGEdgeError(
                        f'`other_subset` of an AccessNode to AccessNode Memlet contains a negative size; the size was {e_memlet.other_subset.size()}',
                        sdfg, state_id, eid)

        # For every memlet, obtain its full path in the DFG
        path = state.memlet_path(e)
        src_node = path[0].src
        dst_node = path[-1].dst

        # NestedSDFGs must connect to AccessNodes
        if not e.data.is_empty():
            if isinstance(src_node, nd.NestedSDFG) and not isinstance(dst_node, nd.AccessNode):
                raise InvalidSDFGEdgeError("Nested SDFG source nodes must be AccessNodes", sdfg, state_id, eid)
            if isinstance(dst_node, nd.NestedSDFG) and not isinstance(src_node, nd.AccessNode):
                raise InvalidSDFGEdgeError("Nested SDFG destination nodes must be AccessNodes", sdfg, state_id, eid)

        # Set up memlet-specific SDFG context
        memlet_context = copy.copy(context)
        for pe in path:
            for pn in (pe.src, pe.dst):
                if isinstance(pn, (nd.EntryNode, nd.ExitNode)):
                    if pn.schedule in dtypes.GPU_SCHEDULES:
                        memlet_context['in_gpu'] = True
                        break
                    if pn.schedule == dtypes.ScheduleType.Default:
                        # Default schedule memlet accessibility validation is deferred
                        # to after schedule/storage inference
                        memlet_context['in_default'] = True
                        break

        # Check if memlet data matches src or dst nodes
        name = e.data.data
        if isinstance(src_node, nd.AccessNode) and isinstance(sdfg.arrays[src_node.data], dt.Structure):
            name = None
        if isinstance(dst_node, nd.AccessNode) and isinstance(sdfg.arrays[dst_node.data], dt.Structure):
            name = None
        if (name is not None and (isinstance(src_node, nd.AccessNode) or isinstance(dst_node, nd.AccessNode))
                and (not isinstance(src_node, nd.AccessNode) or (name != src_node.data and name != e.src_conn))
                and (not isinstance(dst_node, nd.AccessNode) or (name != dst_node.data and name != e.dst_conn))):
            raise InvalidSDFGEdgeError(
                "Memlet data does not match source or destination "
                "data nodes",
                sdfg,
                state_id,
                eid,
            )

        # Check accessibility of scalar memlet data in tasklets and dynamic map ranges
        if (not e.data.is_empty() and _is_scalar(e, path)
                and (isinstance(e.src, nd.Tasklet) or isinstance(e.dst, nd.Tasklet) or isinstance(e.dst, nd.MapEntry))):
            if not memlet_context.get('in_default', False) and not _accessible(sdfg, e.data.data, memlet_context):
                # Rerun slightly more expensive but foolproof test
                memlet_context['in_gpu'] = is_devicelevel_gpu(sdfg, state, e.dst)
                if not _accessible(sdfg, e.data.data, memlet_context):
                    raise InvalidSDFGEdgeError(
                        f'Data container "{e.data.data}" is stored as {sdfg.arrays[e.data.data].storage} '
                        'but accessed on host', sdfg, state_id, eid)

        # Ensure empty memlets are properly connected to tasklets:
        # Empty memlets may only connect two adjacent tasklets
        if e.data.is_empty():
            if len(path) == 1 and isinstance(src_node, nd.Tasklet) and isinstance(dst_node, nd.Tasklet):
                pass
            elif isinstance(dst_node, nd.Tasklet) and path[-1].dst_conn:
                raise InvalidSDFGEdgeError(
                    f'Empty memlet connected to tasklet input connector "{path[-1].dst_conn}". This '
                    'is only allowed when connecting two adjacent tasklets.', sdfg, state_id, eid)
            elif isinstance(src_node, nd.Tasklet) and path[0].src_conn:
                raise InvalidSDFGEdgeError(
                    f'Empty memlet connected to tasklet output connector "{path[0].src_conn}". This '
                    'is only allowed when connecting two adjacent tasklets.', sdfg, state_id, eid)

        # Check memlet subset validity with respect to source/destination nodes
        if e.data.data is not None and e.data.allow_oob == False:
            subset_node = (dst_node
                           if isinstance(dst_node, nd.AccessNode) and e.data.data == dst_node.data else src_node)
            other_subset_node = (dst_node
                                 if isinstance(dst_node, nd.AccessNode) and e.data.data != dst_node.data else src_node)

            if isinstance(subset_node, nd.AccessNode):
                arr = sdfg.arrays[e.data.data]
                # Dimensionality
                if e.data.subset.dims() != len(arr.shape):
                    raise InvalidSDFGEdgeError(
                        "Memlet subset does not match node dimension "
                        "(expected %d, got %d)" % (len(arr.shape), e.data.subset.dims()), sdfg, state_id, eid)

                # Bounds
                if any(((minel + off) < 0) == True for minel, off in zip(e.data.subset.min_element(), arr.offset)):
                    # In case of dynamic memlet, only output a warning
                    if e.data.dynamic:
                        warnings.warn(f'Potential negative out-of-bounds memlet subset: {e}')
                    else:
                        raise InvalidSDFGEdgeError("Memlet subset negative out-of-bounds", sdfg, state_id, eid)
                if any(((maxel + off) >= s) == True
                       for maxel, s, off in zip(e.data.subset.max_element(), arr.shape, arr.offset)):
                    if e.data.dynamic:
                        warnings.warn(f'Potential out-of-bounds memlet subset: {e}')
                    else:
                        raise InvalidSDFGEdgeError("Memlet subset out-of-bounds", sdfg, state_id, eid)

            # Test other_subset as well
            if e.data.other_subset is not None and isinstance(other_subset_node, nd.AccessNode):
                arr = sdfg.arrays[other_subset_node.data]
                # Dimensionality
                if e.data.other_subset.dims() != len(arr.shape):
                    raise InvalidSDFGEdgeError(
                        "Memlet other_subset does not match node dimension "
                        "(expected %d, got %d)" % (len(arr.shape), e.data.other_subset.dims()), sdfg, state_id, eid)

                # Bounds
                if any(
                    ((minel + off) < 0) == True for minel, off in zip(e.data.other_subset.min_element(), arr.offset)):
                    if e.data.dynamic:
                        warnings.warn(f'Potential negative out-of-bounds memlet other_subset: {e}')
                    else:
                        raise InvalidSDFGEdgeError("Memlet other_subset negative out-of-bounds", sdfg, state_id, eid)
                if any(((maxel + off) >= s) == True
                       for maxel, s, off in zip(e.data.other_subset.max_element(), arr.shape, arr.offset)):
                    if e.data.dynamic:
                        warnings.warn(f'Potential out-of-bounds memlet other_subset: {e}')
                    else:
                        raise InvalidSDFGEdgeError("Memlet other_subset out-of-bounds", sdfg, state_id, eid)

            # Test subset and other_subset for undefined symbols
            if Config.get_bool('experimental', 'validate_undefs'):
                # TODO: Traverse by scopes and accumulate data
                defined_symbols = state.symbols_defined_at(e.dst)
                undefs = (e.data.subset.free_symbols - set(defined_symbols.keys()))
                if len(undefs) > 0:
                    raise InvalidSDFGEdgeError('Undefined symbols %s found in memlet subset' % undefs, sdfg, state_id,
                                               eid)
                if e.data.other_subset is not None:
                    undefs = (e.data.other_subset.free_symbols - set(defined_symbols.keys()))
                    if len(undefs) > 0:
                        raise InvalidSDFGEdgeError('Undefined symbols %s found in memlet '
                                                   'other_subset' % undefs, sdfg, state_id, eid)
        #######################################

        # Memlet path scope lifetime checks
        # If scope(src) == scope(dst): OK
        if scope[src_node] == scope[dst_node] or src_node == scope[dst_node]:
            pass
        # If scope(src) contains scope(dst), then src must be a data node,
        # unless the memlet is empty in order to connect to a scope
        elif scope_contains_scope(scope, src_node, dst_node):
            pass
        # If scope(dst) contains scope(src), then dst must be a data node,
        # unless the memlet is empty in order to connect to a scope
        elif scope_contains_scope(scope, dst_node, src_node):
            if not isinstance(dst_node, nd.AccessNode):
                # It is also possible that edge leads to a tasklet that has no incoming or outgoing memlet
                # since the check is to be performed for all edges leading to the dst_node, it is sufficient
                # to check for the memlets of outgoing edges
                if e.data.is_empty():
                    if isinstance(dst_node, nd.ExitNode):
                        pass
                    if isinstance(dst_node, nd.Tasklet) and all(
                        {oe.data.is_empty()
                         for oe in state.out_edges(dst_node)}):
                        pass
                else:
                    raise InvalidSDFGEdgeError(
                        f"Memlet creates an invalid path (sink node {dst_node}"
                        " should be a data node)", sdfg, state_id, eid)
        # If scope(dst) is disjoint from scope(src), it's an illegal memlet
        else:
            raise InvalidSDFGEdgeError("Illegal memlet between disjoint scopes", sdfg, state_id, eid)

        # Check dimensionality of memory access
        if isinstance(e.data.subset, sbs.Range):
            if e.data.subset.dims() != len(sdfg.arrays[e.data.data].shape):
                raise InvalidSDFGEdgeError(
                    "Memlet subset uses the wrong dimensions"
                    " (%dD for a %dD data node)" % (e.data.subset.dims(), len(sdfg.arrays[e.data.data].shape)),
                    sdfg,
                    state_id,
                    eid,
                )

        # Verify that source and destination subsets contain the same
        # number of elements
        if not e.data.allow_oob and e.data.other_subset is not None and not (
            (isinstance(src_node, nd.AccessNode) and isinstance(sdfg.arrays[src_node.data], dt.Stream)) or
            (isinstance(dst_node, nd.AccessNode) and isinstance(sdfg.arrays[dst_node.data], dt.Stream))):
            src_expr = (e.data.src_subset.num_elements() * sdfg.arrays[src_node.data].veclen)
            dst_expr = (e.data.dst_subset.num_elements() * sdfg.arrays[dst_node.data].veclen)
            if symbolic.inequal_symbols(src_expr, dst_expr):
                error = InvalidSDFGEdgeError('Dimensionality mismatch between src/dst subsets', sdfg, state_id, eid)
                # NOTE: Make an exception for Views and reference sets
                from dace.sdfg import utils
                if (isinstance(sdfg.arrays[src_node.data], dt.View) and utils.get_view_edge(state, src_node) is e):
                    warnings.warn(error.message)
                    continue
                if (isinstance(sdfg.arrays[dst_node.data], dt.View) and utils.get_view_edge(state, dst_node) is e):
                    warnings.warn(error.message)
                    continue
                if e.dst_conn == 'set':
                    continue
                raise error

    if Config.get_bool('experimental.check_race_conditions'):
        node_labels = []
        write_accesses = defaultdict(list)
        read_accesses = defaultdict(list)
        for node in state.data_nodes():
            node_labels.append(node.label)
            write_accesses[node.label].extend([{
                'subset': e.data.dst_subset,
                'node': node,
                'wcr': e.data.wcr
            } for e in state.in_edges(node)])
            read_accesses[node.label].extend([{
                'subset': e.data.src_subset,
                'node': node
            } for e in state.out_edges(node)])

        for node_label in node_labels:
            writes = write_accesses[node_label]
            reads = read_accesses[node_label]
            # Check write-write data races.
            for i in range(len(writes)):
                for j in range(i + 1, len(writes)):
                    same_or_unreachable_nodes = (writes[i]['node'] == writes[j]['node']
                                                 or not nx.has_path(state.nx, writes[i]['node'], writes[j]['node']))
                    no_wcr = writes[i]['wcr'] is None and writes[j]['wcr'] is None
                    if same_or_unreachable_nodes and no_wcr:
                        subsets_intersect = subsets.intersects(writes[i]['subset'], writes[j]['subset'])
                        if subsets_intersect:
                            warnings.warn(f'Memlet range overlap while writing to "{node}" in state "{state.label}"')
            # Check read-write data races.
            for write in writes:
                for read in reads:
                    if (not nx.has_path(state.nx, read['node'], write['node'])
                            and subsets.intersects(write['subset'], read['subset'])):
                        warnings.warn(f'Memlet range overlap while writing to "{node}" in state "{state.label}"')

    ########################################


###########################################
# Exception classes


class InvalidSDFGError(Exception):
    """ A class of exceptions thrown when SDFG validation fails. """

    def __init__(self, message: str, sdfg: 'SDFG', state_id: int):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.path = None

    def _getlineinfo(self, obj) -> str:
        """
        Tries to retrieve the source line information of an entity, if exists.

        :param obj: The entity to retrieve.
        :return: A string that contains the file and line of the issue, or an empty string if
                 cannot be evaluated.
        """
        if not hasattr(obj, 'debuginfo'):
            return ''

        lineinfo: DebugInfo = obj.debuginfo
        if lineinfo is None or not lineinfo.filename:
            return ''

        if lineinfo.start_line >= 0:
            if lineinfo.start_column > 0:
                return (f'File "{lineinfo.filename}", line {lineinfo.start_line}, '
                        f'column {lineinfo.start_column}')
            return f'File "{lineinfo.filename}", line {lineinfo.start_line}'

        return f'File "{lineinfo.filename}"'

    def to_json(self):
        return dict(message=self.message, cfg_id=self.sdfg.cfg_id, state_id=self.state_id)

    def __str__(self):
        if self.state_id is not None:
            state = self.sdfg.node(self.state_id)
            locinfo = self._getlineinfo(state)
            suffix = f' (at state {state.label})'
        else:
            suffix = ''
            if self.sdfg.number_of_nodes() >= 1:
                locinfo = self._getlineinfo(self.sdfg.node(0))
            else:
                locinfo = ''

        if locinfo:
            locinfo = '\nOriginating from source code at ' + locinfo

        if self.path:
            locinfo += f'\nInvalid SDFG saved for inspection in {os.path.abspath(self.path)}'

        return f'{self.message}{suffix}{locinfo}'


class InvalidSDFGInterstateEdgeError(InvalidSDFGError):
    """ Exceptions of invalid inter-state edges in an SDFG. """

    def __init__(self, message: str, sdfg: 'SDFG', edge_id: int):
        self.message = message
        self.sdfg = sdfg
        self.edge_id = edge_id
        self.path = None

    def to_json(self):
        return dict(message=self.message, cfg_id=self.sdfg.cfg_id, isedge_id=self.edge_id)

    def __str__(self):
        if self.edge_id is not None:
            e = self.sdfg.edges()[self.edge_id]
            edgestr = ' (at edge "%s" (%s -> %s)' % (
                e.data.label,
                str(e.src),
                str(e.dst),
            )
            locinfo_src = self._getlineinfo(e.src)
            locinfo_dst = self._getlineinfo(e.dst)
        else:
            edgestr = ''
            locinfo_src = locinfo_dst = ''

        if locinfo_src or locinfo_dst:
            if locinfo_src == locinfo_dst:
                locinfo = f'at {locinfo_src}'
            elif locinfo_src and not locinfo_dst:
                locinfo = f'at {locinfo_src}'
            elif locinfo_dst and not locinfo_src:
                locinfo = f'at {locinfo_src}'
            else:
                locinfo = f'between\n {locinfo_src}\n and\n {locinfo_dst}'

            locinfo = f'\nOriginating from source code {locinfo}'
        else:
            locinfo = ''

        if self.path:
            locinfo += f'\nInvalid SDFG saved for inspection in {os.path.abspath(self.path)}'

        return f'{self.message}{edgestr}{locinfo}'


class InvalidSDFGNodeError(InvalidSDFGError):
    """ Exceptions of invalid nodes in an SDFG state. """

    def __init__(self, message: str, sdfg: 'SDFG', state_id: int, node_id: int):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.node_id = node_id
        self.path = None

    def to_json(self):
        return dict(message=self.message, cfg_id=self.sdfg.cfg_id, state_id=self.state_id, node_id=self.node_id)

    def __str__(self):
        state = self.sdfg.node(self.state_id)
        locinfo = ''

        if self.node_id is not None:
            from dace.sdfg.nodes import Node
            node: Node = state.node(self.node_id)
            nodestr = f', node {node}'
            locinfo = self._getlineinfo(node)
        else:
            nodestr = ''
            locinfo = self._getlineinfo(state)

        if locinfo:
            locinfo = '\nOriginating from source code at ' + locinfo

        if self.path:
            locinfo += f'\nInvalid SDFG saved for inspection in {os.path.abspath(self.path)}'

        return f'{self.message} (at state {state.label}{nodestr}){locinfo}'


class NodeNotExpandedError(InvalidSDFGNodeError):
    """
    Exception that is raised whenever a library node was not expanded
    before code generation.
    """

    def __init__(self, sdfg: 'SDFG', state_id: int, node_id: int):
        super().__init__('Library node not expanded', sdfg, state_id, node_id)


class InvalidSDFGEdgeError(InvalidSDFGError):
    """ Exceptions of invalid edges in an SDFG state. """

    def __init__(self, message: str, sdfg: 'SDFG', state_id: int, edge_id: int):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.edge_id = edge_id
        self.path = None

    def to_json(self):
        return dict(message=self.message, cfg_id=self.sdfg.cfg_id, state_id=self.state_id, edge_id=self.edge_id)

    def __str__(self):
        state = self.sdfg.node(self.state_id)

        if self.edge_id is not None:
            e = state.edges()[self.edge_id]
            edgestr = ", edge %s (%s:%s -> %s:%s)" % (
                str(e.data),
                str(e.src),
                e.src_conn,
                str(e.dst),
                e.dst_conn,
            )
            locinfo = self._getlineinfo(e.data)
        else:
            edgestr = ''
            locinfo = self._getlineinfo(state)

        if locinfo:
            locinfo = '\nOriginating from source code at ' + locinfo

        if self.path:
            locinfo += f'\nInvalid SDFG saved for inspection in {os.path.abspath(self.path)}'

        return f'{self.message} (at state {state.label}{edgestr}){locinfo}'


def validate_memlet_data(memlet_data: str, access_data: str) -> bool:
    """ Validates that the src/dst access node data matches the memlet data.

        :param memlet_data: The data of the memlet.
        :param access_data: The data of the access node.
        :return: True if the memlet data matches the access node data.
    """
    if memlet_data == access_data:
        return True
    if memlet_data is None or access_data is None:
        return False
    access_tokens = access_data.split('.')
    memlet_tokens = memlet_data.split('.')
    mem_root = '.'.join(memlet_tokens[:len(access_tokens)])
    return mem_root == access_data


def _no_writes_to_scalars_or_arrays_on_interstate_edges(cfg: 'dace.ControlFlowRegion'):
    from dace.sdfg import InterstateEdge
    for edge in cfg.edges():
        if edge.data is not None and isinstance(edge.data, InterstateEdge):
            # sdfg.arrays return arrays and scalars, it is invalid to write to them
            if any([key in cfg.sdfg.arrays for key in edge.data.assignments]):
                raise InvalidSDFGInterstateEdgeError(
                    f'Assignment to a scalar or an array detected in an interstate edge: "{edge}"', cfg.sdfg,
                    cfg.edge_id(edge))
