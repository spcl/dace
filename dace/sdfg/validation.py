# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Exception classes and methods for validation of SDFGs. """
import copy
from dace.dtypes import DebugInfo, StorageType
import os
from typing import TYPE_CHECKING, Dict, Set, Tuple, Union
import warnings
from dace import dtypes, data as dt, subsets
from dace import symbolic

if TYPE_CHECKING:
    import dace
    from dace.sdfg import SDFG

###########################################
# Validation


def validate(graph: 'dace.sdfg.graph.SubgraphView'):
    from dace.sdfg import SDFG, SDFGState, SubgraphView
    gtype = graph.parent if isinstance(graph, SubgraphView) else graph
    if isinstance(gtype, SDFG):
        validate_sdfg(graph)
    elif isinstance(gtype, SDFGState):
        validate_state(graph)


def validate_sdfg(sdfg: 'dace.sdfg.SDFG', references: Set[int] = None):
    """ Verifies the correctness of an SDFG by applying multiple tests.
    
        :param sdfg: The SDFG to verify.
        :param references: An optional set keeping seen IDs for object
                           miscopy validation.

        Raises an InvalidSDFGError with the erroneous node/edge
        on failure.
    """
    # Avoid import loop
    from dace.codegen.targets import fpga

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

        if len(sdfg.source_nodes()) > 1 and sdfg.start_state is None:
            raise InvalidSDFGError("Starting state undefined", sdfg, None)

        if len(set([s.label for s in sdfg.nodes()])) != len(sdfg.nodes()):
            raise InvalidSDFGError("Found multiple states with the same name", sdfg, None)

        # Validate data descriptors
        for name, desc in sdfg._arrays.items():
            if id(desc) in references:
                raise InvalidSDFGError(
                    f'Duplicate data descriptor object detected: "{name}". Please copy objects '
                    'rather than using multiple references to the same one', sdfg, None)
            references.add(id(desc))

            # Validate array names
            if name is not None and not dtypes.validate_name(name):
                raise InvalidSDFGError("Invalid array name %s" % name, sdfg, None)
            # Allocation lifetime checks
            if (desc.lifetime is dtypes.AllocationLifetime.Persistent and desc.storage is dtypes.StorageType.Register):
                raise InvalidSDFGError(
                    "Array %s cannot be both persistent and use Register as "
                    "storage type. Please use a different storage location." % name, sdfg, None)

            # Check for valid bank assignments
            try:
                bank_assignment = fpga.parse_location_bank(desc)
            except ValueError as e:
                raise InvalidSDFGError(str(e), sdfg, None)
            if bank_assignment is not None:
                if bank_assignment[0] == "DDR" or bank_assignment[0] == "HBM":
                    try:
                        tmp = subsets.Range.from_string(bank_assignment[1])
                    except SyntaxError:
                        raise InvalidSDFGError(
                            "Memory bank specifier must be convertible to subsets.Range"
                            f" for array {name}", sdfg, None)
                    try:
                        low, high = fpga.get_multibank_ranges_from_subset(bank_assignment[1], sdfg)
                    except ValueError as e:
                        raise InvalidSDFGError(str(e), sdfg, None)
                    if (high - low < 1):
                        raise InvalidSDFGError(
                            "Memory bank specifier must at least define one bank to be used"
                            f" for array {name}", sdfg, None)
                    if (high - low > 1 and (high - low != desc.shape[0] or len(desc.shape) < 2)):
                        raise InvalidSDFGError(
                            "Arrays that use a multibank access pattern must have the size of the first dimension equal"
                            f" the number of banks and have at least 2 dimensions for array {name}", sdfg, None)

        # Check every state separately
        start_state = sdfg.start_state
        initialized_transients = {'__pystate'}
        initialized_transients.update(sdfg.constants_prop.keys())
        symbols = copy.deepcopy(sdfg.symbols)
        symbols.update(sdfg.arrays)
        symbols.update({k: v for k, (v, _) in sdfg.constants_prop.items()})
        for desc in sdfg.arrays.values():
            for sym in desc.free_symbols:
                symbols[str(sym)] = sym.dtype
        visited = set()
        visited_edges = set()
        # Run through states via DFS, ensuring that only the defined symbols
        # are available for validation
        for edge in sdfg.dfs_edges(start_state):
            # Source -> inter-state definition -> Destination
            ##########################################
            visited_edges.add(edge)

            # Reference check
            if id(edge) in references:
                raise InvalidSDFGInterstateEdgeError(
                    f'Duplicate inter-state edge object detected: "{edge}". Please '
                    'copy objects rather than using multiple references to the same one', sdfg, sdfg.edge_id(edge))
            references.add(id(edge))
            if id(edge.data) in references:
                raise InvalidSDFGInterstateEdgeError(
                    f'Duplicate inter-state edge object detected: "{edge}". Please '
                    'copy objects rather than using multiple references to the same one', sdfg, sdfg.edge_id(edge))
            references.add(id(edge.data))

            # Source
            if edge.src not in visited:
                visited.add(edge.src)
                validate_state(edge.src, sdfg.node_id(edge.src), sdfg, symbols, initialized_transients, references)

            ##########################################
            # Edge
            # Check inter-state edge for undefined symbols
            undef_syms = set(edge.data.free_symbols) - set(symbols.keys())
            if len(undef_syms) > 0:
                eid = sdfg.edge_id(edge)
                raise InvalidSDFGInterstateEdgeError(
                    f'Undefined symbols in edge: {undef_syms}. Add those with '
                    '`sdfg.add_symbol()` or define outside with `dace.symbol()`', sdfg, eid)

            # Validate inter-state edge names
            issyms = edge.data.new_symbols(sdfg, symbols)
            if any(not dtypes.validate_name(s) for s in issyms):
                invalid = next(s for s in issyms if not dtypes.validate_name(s))
                eid = sdfg.edge_id(edge)
                raise InvalidSDFGInterstateEdgeError("Invalid interstate symbol name %s" % invalid, sdfg, eid)

            # Add edge symbols into defined symbols
            symbols.update(issyms)

            ##########################################
            # Destination
            if edge.dst not in visited:
                visited.add(edge.dst)
                validate_state(edge.dst, sdfg.node_id(edge.dst), sdfg, symbols, initialized_transients, references)
        # End of state DFS

        # If there is only one state, the DFS will miss it
        if start_state not in visited:
            validate_state(start_state, sdfg.node_id(start_state), sdfg, symbols, initialized_transients, references)

        # Validate all inter-state edges (including self-loops not found by DFS)
        for eid, edge in enumerate(sdfg.edges()):
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
            if any(s not in symbols for s in issyms):
                invalid = {s for s in issyms if s not in symbols}
                raise InvalidSDFGInterstateEdgeError(
                    f'Symbols {invalid} are used by inter-state edges, but are not '
                    'defined in `sdfg.symbols`', sdfg, eid)

    except InvalidSDFGError as ex:
        # If the SDFG is invalid, save it
        sdfg.save(os.path.join('_dacegraphs', 'invalid.sdfg'), exception=ex)
        raise


def validate_state(state: 'dace.sdfg.SDFGState',
                   state_id: int = None,
                   sdfg: 'dace.sdfg.SDFG' = None,
                   symbols: Dict[str, dtypes.typeclass] = None,
                   initialized_transients: Set[str] = None,
                   references: Set[int] = None):
    """ Verifies the correctness of an SDFG state by applying multiple
        tests. Raises an InvalidSDFGError with the erroneous node on
        failure.
    """
    # Avoid import loops
    from dace import data as dt
    from dace import subsets as sbs
    from dace.codegen.targets import fpga
    from dace.config import Config
    from dace.sdfg import SDFG
    from dace.sdfg import nodes as nd
    from dace.sdfg import utils as sdutil
    from dace.sdfg.scope import scope_contains_scope

    sdfg = sdfg or state.parent
    state_id = state_id or sdfg.node_id(state)
    symbols = symbols or {}
    initialized_transients = (initialized_transients if initialized_transients is not None else {'__pystate'})
    references = references or set()
    scope_local_constants: dict[nd.MapEntry, list[str]] = dict()
    scope = state.scope_dict()

    # Reference check
    if id(state) in references:
        raise InvalidSDFGError(
            f'Duplicate SDFG state detected: "{state.label}". Please copy objects '
            'rather than using multiple references to the same one', sdfg, state_id)
    references.add(id(state))

    if not dtypes.validate_name(state._label):
        raise InvalidSDFGError("Invalid state name", sdfg, state_id)

    if state._parent != sdfg:
        raise InvalidSDFGError("State does not point to the correct "
                               "parent", sdfg, state_id)

    # Unreachable
    ########################################
    if (sdfg.number_of_nodes() > 1 and sdfg.in_degree(state) == 0 and sdfg.out_degree(state) == 0):
        raise InvalidSDFGError("Unreachable state", sdfg, state_id)

    if state.has_cycles():
        raise InvalidSDFGError('State should be acyclic but contains cycles', sdfg, state_id)

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
                node.validate(sdfg, state, references)
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
                if (arr.transient and state.in_degree(node) == 0 and state.out_degree(node) > 0
                        # Streams do not need to be initialized
                        and not isinstance(arr, dt.Stream)):
                    if node.setzero == False:
                        warnings.warn('WARNING: Use of uninitialized transient "%s" in state %s' %
                                      (node.data, state.label))

                # Register initialized transients
                if arr.transient and state.in_degree(node) > 0:
                    initialized_transients.add(node.data)

            nsdfg_node = sdfg.parent_nsdfg_node
            if nsdfg_node is not None:
                # Find unassociated non-transients access nodes
                if (not arr.transient and node.data not in nsdfg_node.in_connectors
                        and node.data not in nsdfg_node.out_connectors):
                    raise InvalidSDFGNodeError(
                        f'Data descriptor "{node.data}" is not transient and used in a nested SDFG, '
                        'but does not have a matching connector on the outer SDFG node.', sdfg, state_id, nid)

                # Find writes to input-only arrays
                only_empty_inputs = all(e.data.is_empty() for e in state.in_edges(node))
                if (not arr.transient) and (not only_empty_inputs):
                    if node.data not in nsdfg_node.out_connectors:
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

        # Tasklets may only access 1 HBM bank at a time
        if isinstance(node, nd.Tasklet):
            for attached in state.all_edges(node):
                if attached.data.data in sdfg.arrays:
                    if fpga.is_multibank_array_with_distributed_index(sdfg.arrays[attached.data.data]):
                        low, high, _ = attached.data.subset[0]
                        if (low != high):
                            raise InvalidSDFGNodeError(
                                "Tasklets may only be directly connected"
                                " to HBM-memlets accessing only one bank", sdfg, state_id, nid)

        # Connector tests
        ########################################
        # Check for duplicate connector names (unless it's a nested SDFG)
        if (len(node.in_connectors.keys() & node.out_connectors.keys()) > 0
                and not isinstance(node, (nd.NestedSDFG, nd.LibraryNode))):
            dups = node.in_connectors.keys() & node.out_connectors.keys()
            raise InvalidSDFGNodeError("Duplicate connectors: " + str(dups), sdfg, state_id, nid)

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

    # Memlet checks
    for eid, e in enumerate(state.edges()):
        # Reference check
        if id(e) in references:
            raise InvalidSDFGEdgeError(
                f'Duplicate memlet detected: "{e}". Please copy objects '
                'rather than using multiple references to the same one', sdfg, state_id, eid)
        references.add(id(e))
        if id(e.data) in references:
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

        # For every memlet, obtain its full path in the DFG
        path = state.memlet_path(e)
        src_node = path[0].src
        dst_node = path[-1].dst

        # Check if memlet data matches src or dst nodes
        if (e.data.data is not None and (isinstance(src_node, nd.AccessNode) or isinstance(dst_node, nd.AccessNode))
                and (not isinstance(src_node, nd.AccessNode) or e.data.data != src_node.data)
                and (not isinstance(dst_node, nd.AccessNode) or e.data.data != dst_node.data)):
            raise InvalidSDFGEdgeError(
                "Memlet data does not match source or destination "
                "data nodes)",
                sdfg,
                state_id,
                eid,
            )

        # Check memlet subset validity with respect to source/destination nodes
        if e.data.data is not None and e.data.allow_oob == False:
            subset_node = (dst_node
                           if isinstance(dst_node, nd.AccessNode) and e.data.data == dst_node.data else src_node)
            other_subset_node = (dst_node
                                 if isinstance(dst_node, nd.AccessNode) and e.data.data != dst_node.data else src_node)

            if isinstance(subset_node, nd.AccessNode):
                arr = sdfg.arrays[subset_node.data]
                # Dimensionality
                if e.data.subset.dims() != len(arr.shape):
                    raise InvalidSDFGEdgeError(
                        "Memlet subset does not match node dimension "
                        "(expected %d, got %d)" % (len(arr.shape), e.data.subset.dims()),
                        sdfg,
                        state_id,
                        eid,
                    )

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
                        "(expected %d, got %d)" % (len(arr.shape), e.data.other_subset.dims()),
                        sdfg,
                        state_id,
                        eid,
                    )

                # Bounds
                if any(
                    ((minel + off) < 0) == True for minel, off in zip(e.data.other_subset.min_element(), arr.offset)):
                    raise InvalidSDFGEdgeError(
                        "Memlet other_subset negative out-of-bounds",
                        sdfg,
                        state_id,
                        eid,
                    )
                if any(((maxel + off) >= s) == True
                       for maxel, s, off in zip(e.data.other_subset.max_element(), arr.shape, arr.offset)):
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
                if e.data.is_empty() and isinstance(dst_node, nd.ExitNode):
                    pass
                else:
                    raise InvalidSDFGEdgeError(
                        f"Memlet creates an invalid path (sink node {dst_node}"
                        " should be a data node)", sdfg, state_id, eid)
        # If scope(dst) is disjoint from scope(src), it's an illegal memlet
        else:
            raise InvalidSDFGEdgeError("Illegal memlet between disjoint scopes", sdfg, state_id, eid)

        # Check dimensionality of memory access
        if isinstance(e.data.subset, (sbs.Range, sbs.Indices)):
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
                # NOTE: Make an exception for Views
                from dace.sdfg import utils
                if (isinstance(sdfg.arrays[src_node.data], dt.View) and utils.get_view_edge(state, src_node) is e):
                    warnings.warn(error.message)
                    continue
                if (isinstance(sdfg.arrays[dst_node.data], dt.View) and utils.get_view_edge(state, dst_node) is e):
                    warnings.warn(error.message)
                    continue
                raise error

    ########################################


###########################################
# Exception classes


class InvalidSDFGError(Exception):
    """ A class of exceptions thrown when SDFG validation fails. """

    def __init__(self, message: str, sdfg: 'SDFG', state_id: int):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id

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
        return dict(message=self.message, sdfg_id=self.sdfg.sdfg_id, state_id=self.state_id)

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

        return f'{self.message}{suffix}{locinfo}'


class InvalidSDFGInterstateEdgeError(InvalidSDFGError):
    """ Exceptions of invalid inter-state edges in an SDFG. """

    def __init__(self, message: str, sdfg: 'SDFG', edge_id: int):
        self.message = message
        self.sdfg = sdfg
        self.edge_id = edge_id

    def to_json(self):
        return dict(message=self.message, sdfg_id=self.sdfg.sdfg_id, isedge_id=self.edge_id)

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

        return f'{self.message}{edgestr}{locinfo}'


class InvalidSDFGNodeError(InvalidSDFGError):
    """ Exceptions of invalid nodes in an SDFG state. """

    def __init__(self, message: str, sdfg: 'SDFG', state_id: int, node_id: int):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.node_id = node_id

    def to_json(self):
        return dict(message=self.message, sdfg_id=self.sdfg.sdfg_id, state_id=self.state_id, node_id=self.node_id)

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

    def to_json(self):
        return dict(message=self.message, sdfg_id=self.sdfg.sdfg_id, state_id=self.state_id, edge_id=self.edge_id)

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

        return f'{self.message} (at state {state.label}{edgestr}){locinfo}'
