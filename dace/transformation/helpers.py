# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Transformation helper API. """
import copy
import itertools
import warnings
from networkx import MultiDiGraph

from dace.properties import CodeBlock
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, ReturnBlock
from dace.subsets import Range, Subset, union
import dace.subsets as subsets
from typing import Dict, Iterable, List, Optional, Tuple, Set, Union

from dace import data, dtypes, symbolic
from dace.codegen import control_flow as cf
from dace.sdfg import nodes, utils
from dace.sdfg.graph import Edge, SubgraphView, MultiConnectorEdge
from dace.sdfg.scope import ScopeSubgraphView, ScopeTree
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg import graph
from dace.memlet import Memlet


def nest_sdfg_subgraph(sdfg: SDFG, subgraph: SubgraphView, start: Optional[SDFGState] = None) -> SDFGState:
    """
    Nests an SDFG subgraph (SDFGStates and InterstateEdges).

    :param sdfg: The SDFG containing the subgraph.
    :param subgraph: The SubgraphView description of the subgraph.
    :param start: The start state of the subgraph.
    :return: The SDFGState containing the NestedSDFG node (containing the nested SDFG subgraph).
    """

    # Nest states
    blocks: List[ControlFlowBlock] = subgraph.nodes()
    return_state = None
    if len(blocks) > 1 or isinstance(blocks[0], AbstractControlFlowRegion):
        # Avoid cyclic imports
        from dace.transformation.passes.analysis import loop_analysis

        graph: ControlFlowRegion = blocks[0].parent_graph
        if start is not None:
            source_node = start
        else:
            src_nodes = subgraph.source_nodes()
            if len(src_nodes) != 1:
                raise ValueError('Ambiguous start state of the SDFG subgraph. '
                                 'Please provide the start state via the "start" argument.')
            source_node = src_nodes[0]

        sink_nodes = subgraph.sink_nodes()
        if len(sink_nodes) != 1:
            raise NotImplementedError
        sink_node = sink_nodes[0]

        all_blocks: List[ControlFlowBlock] = []
        is_edges: List[Edge[InterstateEdge]] = []
        for b in blocks:
            if isinstance(b, AbstractControlFlowRegion):
                all_blocks.append(b)
                for nb in b.all_control_flow_blocks():
                    all_blocks.append(nb)
                for e in b.all_interstate_edges():
                    is_edges.append(e)
            else:
                all_blocks.append(b)
        states: List[SDFGState] = [b for b in all_blocks if isinstance(b, SDFGState)]
        for src in blocks:
            for dst in blocks:
                for edge in graph.edges_between(src, dst):
                    is_edges.append(edge)
        return_blocks: Set[ReturnBlock] = set([b for b in all_blocks if isinstance(b, ReturnBlock)])
        if len(return_blocks) > 0:
            did_return_inner = '_did_ret_from_nsdfg'
            did_return_inner = sdfg._find_new_name(did_return_inner)
            sdfg.add_scalar(did_return_inner, dtypes.int32, transient=True)
        else:
            did_return_inner = None

        # Find read/write sets
        read_set, write_set = set(), set()
        for state in states:
            rset, wset = state.read_and_write_sets()
            read_set |= rset
            write_set |= wset
            # Add to write set also scalars between tasklets
            for src_node in state.nodes():
                if not isinstance(src_node, nodes.Tasklet):
                    continue
                for dst_node in state.nodes():
                    if src_node is dst_node:
                        continue
                    if not isinstance(dst_node, nodes.Tasklet):
                        continue
                    for e in state.edges_between(src_node, dst_node):
                        if e.data.data and e.data.data in sdfg.arrays:
                            write_set.add(e.data.data)
        # Add data from edges
        for edge in is_edges:
            for s in edge.data.free_symbols:
                if s in sdfg.arrays:
                    read_set.add(s)
        for blk in all_blocks:
            if isinstance(blk, ConditionalBlock):
                for c, _ in blk.branches:
                    if c is not None:
                        for s in c.get_free_symbols():
                            if s in sdfg.arrays:
                                read_set.add(s)
            elif isinstance(blk, LoopRegion):
                for s in blk.loop_condition.get_free_symbols():
                    if s in sdfg.arrays:
                        read_set.add(s)

        # Find NestedSDFG's unique data
        rw_set = read_set | write_set
        unique_set = set()
        for name in rw_set:
            if not sdfg.arrays[name].transient:
                continue
            found = False
            for state in sdfg.states():
                if state in blocks:
                    continue
                for node in state.nodes():
                    if (isinstance(node, nodes.AccessNode) and node.data == name):
                        found = True
                        break
            if not found:
                unique_set.add(name)

        # Find NestedSDFG's connectors
        read_set = {n for n in read_set if n not in unique_set or not sdfg.arrays[n].transient}
        write_set = {n for n in write_set if n not in unique_set or not sdfg.arrays[n].transient}

        # Find defined subgraph symbols
        defined_symbols = set()
        strictly_defined_symbols = set()
        for e in is_edges:
            defined_symbols.update(set(e.data.assignments.keys()))
            for k, v in e.data.assignments.items():
                try:
                    if k not in sdfg.symbols and k not in {str(a) for a in symbolic.pystr_to_symbolic(v).args}:
                        strictly_defined_symbols.add(k)
                except AttributeError:
                    # `symbolic.pystr_to_symbolic` may return bool, which doesn't have attribute `args`
                    pass
        for b in all_blocks:
            if isinstance(b, LoopRegion) and b.loop_variable:
                defined_symbols.add(b.loop_variable)
                if b.loop_variable not in sdfg.symbols:
                    if b.init_statement:
                        init_assignment = loop_analysis.get_init_assignment(b)
                        if b.loop_variable not in {str(s) for s in symbolic.pystr_to_symbolic(init_assignment).args}:
                            strictly_defined_symbols.add(b.loop_variable)
                    else:
                        strictly_defined_symbols.add(b.loop_variable)

        return_state = new_state = graph.add_state('nested_sdfg_parent')

        # If there is a return that is being nested in, a conditional return is added right after the new nested SDFG
        # which will be taken if the inner, nested return was hit.
        ret_cond = None
        if len(return_blocks) > 0:
            ret_cond = ConditionalBlock('return_' + sdfg.label + '_from_nested', sdfg, graph)
            graph.add_node(ret_cond, ensure_unique_name=True)
            ret_branch = ControlFlowRegion('return_' + sdfg.label + '_from_nested_body', sdfg, ret_cond)
            ret_block = ReturnBlock('return', sdfg, ret_branch)
            ret_branch.add_node(ret_block)
            ret_cond.add_branch(CodeBlock(did_return_inner), ret_branch)

        nsdfg = SDFG("nested_sdfg", constants=sdfg.constants_prop, parent=new_state)
        nsdfg.add_node(source_node, is_start_state=True)
        nsdfg.add_nodes_from([s for s in blocks if s is not source_node])
        for e in subgraph.edges():
            nsdfg.add_edge(e.src, e.dst, e.data)

        # Annotate any transitions to return blocks in the inner, nested SDFG by first setting the added transient
        # scalar to 1 / true to detect that the inner SDFG returned.
        if len(return_blocks) > 0:
            for blk in nsdfg.all_control_flow_blocks():
                if blk in return_blocks:
                    pre_state = blk.parent_graph.add_state_before(blk)
                    did_ret_tasklet = pre_state.add_tasklet('__did_ret_set', {}, {'out'}, 'out = 1')
                    did_ret_access = pre_state.add_access(did_return_inner)
                    pre_state.add_edge(did_ret_tasklet, 'out', did_ret_access, None, Memlet(did_return_inner + '[0]'))
                    write_set.add(did_return_inner)

        if ret_cond is not None:
            pre_state = graph.add_state('before_nested_sdfg_parent')
            for e in graph.in_edges(source_node):
                graph.add_edge(e.src, pre_state, e.data)
            did_ret_tasklet = pre_state.add_tasklet('__did_ret_init', {}, {'out'}, 'out = 0')
            did_ret_access = pre_state.add_access(did_return_inner)
            pre_state.add_edge(did_ret_tasklet, 'out', did_ret_access, None, Memlet(did_return_inner + '[0]'))
            graph.add_edge(pre_state, new_state, InterstateEdge())
            graph.add_edge(new_state, ret_cond, InterstateEdge())
            for e in graph.out_edges(sink_node):
                graph.add_edge(ret_cond, e.dst, e.data)
        else:
            for e in graph.in_edges(source_node):
                graph.add_edge(e.src, new_state, e.data)
            for e in graph.out_edges(sink_node):
                graph.add_edge(new_state, e.dst, e.data)

        graph.remove_nodes_from(blocks)

        # Add NestedSDFG arrays
        for name in read_set | write_set:
            nsdfg.arrays[name] = copy.deepcopy(sdfg.arrays[name])
            nsdfg.arrays[name].transient = False
        for name in unique_set:
            nsdfg.arrays[name] = sdfg.arrays[name]
            del sdfg.arrays[name]

        # If there are symbols (new or not) assigned value in the subgraph, their values will be propagated to the
        # (outer) SDFG through a symbol-scalar-symbol conversion. This is happening in two parts: (1) inside the nested
        # SDFG (symbol -> scalar), (2) in the outer SDFG, just after the SDFGState containing the NestedSDFG node
        # (scalar -> symbol).
        ndefined_symbols = set()
        out_mapping = {}
        out_state = None
        for e in nsdfg.all_interstate_edges():
            ndefined_symbols.update(set(e.data.assignments.keys()))
        for b in all_blocks:
            if isinstance(b, LoopRegion) and b.loop_variable is not None and b.loop_variable != '' and b.init_statement:
                ndefined_symbols.add(b.loop_variable)
        if ndefined_symbols:
            out_state = nsdfg.add_state('symbolic_output')
            nsdfg.add_edge(sink_node, out_state, InterstateEdge())
            for s in ndefined_symbols:
                if s in nsdfg.symbols:
                    dtype = nsdfg.symbols[s]
                else:
                    dtype = sdfg.symbols[s]
                name, _ = sdfg.add_scalar(f"__sym_out_{s}", dtype, transient=True, find_new_name=True)
                out_mapping[s] = name
                nname, ndesc = nsdfg.add_scalar(f"__sym_out_{s}", dtype, find_new_name=True)
                # Part (1)
                tasklet = out_state.add_tasklet(f"set_{nname}", {}, {'__out'}, f'__out = {s}')
                acc = out_state.add_access(nname)
                out_state.add_edge(tasklet, '__out', acc, None, Memlet.from_array(nname, ndesc))
                write_set.add(name)

        # Add NestedSDFG node
        fsymbols = sdfg.symbols.keys() | nsdfg.free_symbols
        fsymbols.update(defined_symbols)
        fsymbols = fsymbols - strictly_defined_symbols
        mapping = {s: s for s in fsymbols}
        cnode = new_state.add_nested_sdfg(nsdfg, read_set, write_set, mapping)
        for s in strictly_defined_symbols:
            if s in sdfg.symbols:
                sdfg.remove_symbol(s)

        # Connect input/output data of the subgraph to the NestedSDFG node.
        for name in read_set:
            r = new_state.add_read(name)
            new_state.add_edge(r, None, cnode, name, Memlet.from_array(name, sdfg.arrays[name]))
        for name in write_set:
            w = new_state.add_write(name)
            new_state.add_edge(cnode, name, w, None, Memlet.from_array(name, sdfg.arrays[name]))

        # Part (2)
        if out_state is not None:
            extra_state = graph.add_state('symbolic_output')
            for e in graph.out_edges(new_state):
                graph.add_edge(extra_state, e.dst, e.data)
                graph.remove_edge(e)
            graph.add_edge(new_state, extra_state, InterstateEdge(assignments=out_mapping))
            new_state = extra_state

    else:
        return_state = blocks[0]

    return return_state


def nest_sdfg_control_flow(sdfg: SDFG):
    """
    Partitions the SDFG to subgraphs and nests them.

    :param sdfg: The SDFG to be partitioned.
    """
    for nd in sdfg.nodes():
        if isinstance(nd, AbstractControlFlowRegion):
            nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, [nd]))
            sdfg.reset_cfg_list()


def nest_state_subgraph(sdfg: SDFG,
                        state: SDFGState,
                        subgraph: SubgraphView,
                        name: Optional[str] = None,
                        full_data: bool = False) -> nodes.NestedSDFG:
    """ Turns a state subgraph into a nested SDFG. Operates in-place.

        :param sdfg: The SDFG containing the state subgraph.
        :param state: The state containing the subgraph.
        :param subgraph: Subgraph to nest.
        :param name: An optional name for the nested SDFG.
        :param full_data: If True, nests entire input/output data.
        :return: The nested SDFG node.
        :raise KeyError: Some or all nodes in the subgraph are not located in
                         this state, or the state does not belong to the given
                         SDFG.
        :raise ValueError: The subgraph is contained in more than one scope.
    """
    if state.sdfg != sdfg:
        raise KeyError('State does not belong to given SDFG')
    if subgraph is not state and subgraph.graph is not state:
        raise KeyError('Subgraph does not belong to given state')

    # Find the top-level scope
    scope_tree = state.scope_tree()
    scope_dict = state.scope_dict()
    scope_dict_children = state.scope_children()
    top_scopenode = -1  # Initialized to -1 since "None" already means top-level

    for node in subgraph.nodes():
        if node not in scope_dict:
            raise KeyError('Node not found in state')

        # If scope entry/exit, ensure entire scope is in subgraph
        if isinstance(node, nodes.EntryNode):
            scope_nodes = scope_dict_children[node]
            if any(n not in subgraph.nodes() for n in scope_nodes):
                raise ValueError('Subgraph contains partial scopes (entry)')
        elif isinstance(node, nodes.ExitNode):
            entry = state.entry_node(node)
            scope_nodes = scope_dict_children[entry] + [entry]
            if any(n not in subgraph.nodes() for n in scope_nodes):
                raise ValueError('Subgraph contains partial scopes (exit)')

        scope_node = scope_dict[node]
        if scope_node not in subgraph.nodes():
            if top_scopenode != -1 and top_scopenode != scope_node:
                raise ValueError('Subgraph is contained in more than one scope')
            top_scopenode = scope_node

    scope = scope_tree[top_scopenode]
    ###

    # Consolidate edges in top scope
    utils.consolidate_edges(sdfg, scope)
    snodes = subgraph.nodes()

    # Collect inputs and outputs of the nested SDFG
    inputs: List[MultiConnectorEdge] = []
    outputs: List[MultiConnectorEdge] = []
    for node in snodes:
        for edge in state.in_edges(node):
            if edge.src not in snodes:
                inputs.append(edge)
        for edge in state.out_edges(node):
            if edge.dst not in snodes:
                outputs.append(edge)

    # Collect transients not used outside of subgraph (will be removed of
    # top-level graph)
    data_in_subgraph = set(n.data for n in subgraph.nodes() if isinstance(n, nodes.AccessNode))
    # Find other occurrences in SDFG
    other_nodes = set(n.data for s in sdfg.states() for n in s.nodes()
                      if isinstance(n, nodes.AccessNode) and n not in subgraph.nodes())
    subgraph_transients = set()
    for data in data_in_subgraph:
        datadesc = sdfg.arrays[data]
        if datadesc.transient and data not in other_nodes:
            subgraph_transients.add(data)

    # All transients of edges between code nodes are also added to nested graph
    for edge in subgraph.edges():
        if (isinstance(edge.src, nodes.CodeNode) and isinstance(edge.dst, nodes.CodeNode)):
            subgraph_transients.add(edge.data.data)

    # Collect data used in access nodes within subgraph (will be referenced in
    # full upon nesting)
    input_arrays = set()
    output_arrays = {}
    for node in subgraph.nodes():
        if (isinstance(node, nodes.AccessNode) and node.data not in subgraph_transients):
            if node.has_reads(state):
                input_arrays.add(node.data)
            if node.has_writes(state):
                output_arrays[node.data] = state.in_edges(node)[0].data.wcr

    # Create the nested SDFG
    nsdfg = SDFG(name or 'nested_' + state.label)

    # Transients are added to the nested graph as-is
    for name in subgraph_transients:
        nsdfg.add_datadesc(name, sdfg.arrays[name])

    # Input/output data that are not source/sink nodes are added to the graph
    # as non-transients
    for name in (input_arrays | output_arrays.keys()):
        datadesc = copy.deepcopy(sdfg.arrays[name])
        datadesc.transient = False
        nsdfg.add_datadesc(name, datadesc)

    # Connected source/sink nodes outside subgraph become global data
    # descriptors in nested SDFG
    input_names = {}
    output_names = {}
    global_subsets: Dict[str, Tuple[str, Subset]] = {}
    for edge in inputs:
        if edge.data.data is None:  # Skip edges with an empty memlet
            continue
        name = edge.data.data
        if name not in global_subsets:
            datadesc = copy.deepcopy(sdfg.arrays[edge.data.data])
            datadesc.transient = False
            if not full_data:
                datadesc.shape = edge.data.subset.size()
            new_name = nsdfg.add_datadesc(name, datadesc, find_new_name=True)
            global_subsets[name] = (new_name, edge.data.subset)
        else:
            new_name, subset = global_subsets[name]
            if not full_data:
                new_subset = union(subset, edge.data.subset)
                if new_subset is None:
                    new_subset = Range.from_array(sdfg.arrays[name])
                global_subsets[name] = (new_name, new_subset)
                nsdfg.arrays[new_name].shape = new_subset.size()
        input_names[edge] = new_name
    for edge in outputs:
        if edge.data.data is None:  # Skip edges with an empty memlet
            continue
        name = edge.data.data
        if name not in global_subsets:
            datadesc = copy.deepcopy(sdfg.arrays[edge.data.data])
            datadesc.transient = False
            if not full_data:
                datadesc.shape = edge.data.subset.size()
            new_name = nsdfg.add_datadesc(name, datadesc, find_new_name=True)
            global_subsets[name] = (new_name, edge.data.subset)
        else:
            new_name, subset = global_subsets[name]
            if not full_data:
                new_subset = union(subset, edge.data.subset)
                if new_subset is None:
                    new_subset = Range.from_array(sdfg.arrays[name])
                global_subsets[name] = (new_name, new_subset)
                nsdfg.arrays[new_name].shape = new_subset.size()
        output_names[edge] = new_name
    ###################

    # Add scope symbols to the nested SDFG
    symbols_at_top = state.symbols_defined_at(top_scopenode)
    defined_vars = set(
        symbolic.pystr_to_symbolic(s) for s in (state.symbols_defined_at(top_scopenode).keys()
                                                | sdfg.symbols))
    for v in defined_vars:
        if v in sdfg.symbols:
            sym = sdfg.symbols[v]
            nsdfg.add_symbol(v, sym.dtype)

    # Add constants to nested SDFG
    for cstname, cstval in sdfg.constants.items():
        nsdfg.add_constant(cstname, cstval)

    # Create nested state
    nstate = nsdfg.add_state()

    # Add subgraph nodes and edges to nested state
    nstate.add_nodes_from(subgraph.nodes())
    for e in subgraph.edges():
        nstate.add_edge(e.src, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))

    # Modify nested SDFG parents in subgraph
    for node in subgraph.nodes():
        if isinstance(node, nodes.NestedSDFG):
            node.sdfg.parent = nstate
            node.sdfg.parent_sdfg = nsdfg
            node.sdfg.parent_nsdfg_node = node

    # Add access nodes and edges as necessary
    edges_to_offset = []
    for edge, name in input_names.items():
        node = nstate.add_read(name)
        new_edge = copy.deepcopy(edge.data)
        new_edge.data = name
        edges_to_offset.append((edge, nstate.add_edge(node, None, edge.dst, edge.dst_conn, new_edge)))
    for edge, name in output_names.items():
        node = nstate.add_write(name)
        new_edge = copy.deepcopy(edge.data)
        new_edge.data = name
        edges_to_offset.append((edge, nstate.add_edge(edge.src, edge.src_conn, node, None, new_edge)))

    # Offset memlet paths inside nested SDFG according to subsets
    for original_edge, new_edge in edges_to_offset:
        for edge in nstate.memlet_tree(new_edge):
            edge.data.data = new_edge.data.data
            if not full_data:
                edge.data.subset.offset(global_subsets[original_edge.data.data][1], True)
                edge.data.subset.offset(nsdfg.arrays[edge.data.data].offset, True)

    # Add nested SDFG node to the input state
    nested_sdfg = state.add_nested_sdfg(nsdfg,
                                        set(input_names.values()) | input_arrays,
                                        set(output_names.values()) | output_arrays.keys())

    # Reconnect memlets to nested SDFG
    reconnected_in = set()
    reconnected_out = set()
    empty_input = None
    empty_output = None
    for edge in inputs:
        if edge.data.data is None:
            empty_input = edge
            continue

        name = input_names[edge]
        if name in reconnected_in:
            continue
        if full_data:
            data = Memlet.from_array(edge.data.data, sdfg.arrays[edge.data.data])
        else:
            data = copy.deepcopy(edge.data)
            data.subset = copy.deepcopy(global_subsets[edge.data.data][1])
        state.add_edge(edge.src, edge.src_conn, nested_sdfg, name, data)
        reconnected_in.add(name)

    for edge in outputs:
        if edge.data.data is None:
            empty_output = edge
            continue

        name = output_names[edge]
        if name in reconnected_out:
            continue
        if full_data:
            data = Memlet.from_array(edge.data.data, sdfg.arrays[edge.data.data])
        else:
            data = copy.deepcopy(edge.data)
            data.subset = copy.deepcopy(global_subsets[edge.data.data][1])
        data.wcr = edge.data.wcr
        state.add_edge(nested_sdfg, name, edge.dst, edge.dst_conn, data)
        reconnected_out.add(name)

    # Connect access nodes to internal input/output data as necessary
    entry = scope.entry
    exit = scope.exit
    for name in input_arrays:
        node = state.add_read(name)
        if entry is not None:
            state.add_nedge(entry, node, Memlet())
        state.add_edge(node, None, nested_sdfg, name, Memlet.from_array(name, sdfg.arrays[name]))
    for name, wcr in output_arrays.items():
        node = state.add_write(name)
        if exit is not None:
            state.add_nedge(node, exit, Memlet())
        state.add_edge(nested_sdfg, name, node, None, Memlet(data=name, wcr=wcr))

    # Graph was not reconnected, but needs to be
    if state.in_degree(nested_sdfg) == 0 and empty_input is not None:
        state.add_edge(empty_input.src, empty_input.src_conn, nested_sdfg, None, empty_input.data)
    if state.out_degree(nested_sdfg) == 0 and empty_output is not None:
        state.add_edge(nested_sdfg, None, empty_output.dst, empty_output.dst_conn, empty_output.data)

    # Remove subgraph nodes from graph
    state.remove_nodes_from(subgraph.nodes())

    # Remove subgraph transients from top-level graph
    for transient in subgraph_transients:
        del sdfg.arrays[transient]

    # Remove newly isolated nodes due to memlet consolidation
    for edge in inputs:
        if state.in_degree(edge.src) + state.out_degree(edge.src) == 0:
            state.remove_node(edge.src)
    for edge in outputs:
        if state.in_degree(edge.dst) + state.out_degree(edge.dst) == 0:
            state.remove_node(edge.dst)

    sdfg.reset_cfg_list()

    return nested_sdfg


def state_fission(
    subgraph: graph.SubgraphView,
    label: Optional[str] = None,
    allow_isolated_nodes: bool = True,
) -> SDFGState:
    """Splits the state into two connected states such that `subgraph` is located in the first/top state.

    The function will create a new state before the state in which `subgraph` is located in, this
    new state is also returned. Then the function will then distribute the dataflow onto these two
    states. In general it will move the dataflow defined by `subgraph`, together with all its
    dependencies into the first/top state and the rest will be in the second state (which is the
    current state).
    Beside the dependencies of `subgraph` the function might also move nodes located downstream of
    `subgraph` into the first state. The first reason is that the "cut" at that location is not
    possible, i.e. there must be a non view AccessNode to transmit data from the first to the
    second state. Another reason is that the function maintains the number of "writing nodes", i.e
    AccessNodes with incoming connections, this behaviour is analogous to `isolate_nested_sdfg()`.
    Another reason why more nodes are moved to the first state are scopes. It is only possible to
    split a state at the top scope, not as an example, within a Map. Thus if a node inside `subgraph`
    if not at the global scope the function will include its containing state. Thus if you only
    pass a Tasklet, with a Map, as `subgraph` then the function will add the surrounding Map
    to the first state as well.

    Note, the split might result in a state where nodes become isolated inside a state. By default
    the function allows this to happen in either state. However, it will cause a validation error.
    By  setting `allow_isolated_nodes` to `False` the function will remove all isolated nodes in
    both states.

    :param subgraph: The graph that describes the split location.
    :param label: The label to use for the new state.
    :param allow_isolated_nodes: If `True`, the default, then the function might create isolated
        nodes. If it is `False` all isolated nodes will be removed after the split in both states.

    :return: The newly created state that is located before the one containing `subgraph`.

    :note: Currently scopes are not handled, if for example `subgraph` only contains a Tasklet from
        inside a Map, then the function will add all dependencies, i.e. the MapEntry, but will
        most likely fail to pick up the MapExit.

    :todo: Handle partial Map scopes.
    """
    state: SDFGState = subgraph.graph
    sdfg: SDFG = state.sdfg

    # State fissions can not occur within a scope, i.e., the MapEntry of a Map scope can not end up
    #  in the first state while the MapExit lands in the second state. Extend the set of nodes to
    #  make sure we have only top level nodes and their scope.
    initial_first_nodes: Set[nodes.Node] = set()
    scope_dict = state.scope_dict()
    for node in subgraph.nodes():
        containing_scope = scope_dict[node]

        if node in initial_first_nodes:
            continue  # Already added by a previous node.

        elif isinstance(node, nodes.EntryNode):
            # Add the whole body of the Map to the first state. ExitNodes are handled by the scope
            #  implementation, as they are located inside the scope of their associated EntryNode.
            initial_first_nodes.update(state.scope_subgraph(node).nodes())

        elif containing_scope is None:
            initial_first_nodes.add(node)  # Global node just add it.

        else:
            # Note at global scope, find the top most scope node and add the defining subgraph.
            while scope_dict[containing_scope] is not None:
                containing_scope = scope_dict[containing_scope]
            top_entry_node = containing_scope
            initial_first_nodes.update(state.scope_subgraph(top_entry_node).nodes())

    # Notes that should end up in the first state.
    first_nodes: Set[nodes.Node] = set()
    for node in initial_first_nodes:
        utils.find_upstream_nodes(
            node_to_start=node,
            state=state,
            seen=first_nodes,  # Inplace update.
        )

    # For semantic reasons we can only split certain Memlets, i.e. Memlets that started at an
    #  AccessNode. We now have to inspect the boundary of the nodes defining `first_nodes`.
    #  If we found a Memlet that can not be split, then we add the node also to `first_nodes`.
    nodes_to_scan: List[nodes.Node] = list(first_nodes)
    boundary_nodes: Set[nodes.Node] = set()
    pure_first_nodes: Set[nodes.Node] = set()

    while len(nodes_to_scan) > 0:
        node_to_scan = nodes_to_scan.pop()
        non_first_state_consumer = [
            oedge.dst for oedge in state.out_edges(node_to_scan)
            if (oedge.dst not in first_nodes) and (not oedge.data.is_empty())
        ]

        if len(non_first_state_consumer) == 0:
            # There are no consumer that are not inside `first_nodes` thus it is inside `first_nodes`
            #  and it is "pure", i.e. not a boundary node.
            assert node_to_scan in first_nodes
            pure_first_nodes.add(node_to_scan)
            boundary_nodes.discard(node_to_scan)

        elif isinstance(node_to_scan, nodes.AccessNode) and (not isinstance(node_to_scan.desc(sdfg), data.View)):
            # There are consumer that are not inside the first state, but `node_to_scan` is a non
            #  view AccessNode, it therefore qualifies to be a boundary node.
            assert node_to_scan in first_nodes
            boundary_nodes.add(node_to_scan)

        else:
            # There are non first state consumer and `node_to_scan` does not qualify to be a boundary
            #  node. Thus we have to add all its consumer and their producer, to the first state as well.
            new_first_nodes: Set[nodes.Node] = set()
            for new_first_state_node_seed in non_first_state_consumer:
                utils.find_upstream_nodes(
                    node_to_start=new_first_state_node_seed,
                    state=state,
                    seen=new_first_nodes,  # Inplace update!
                )

            # All nodes that we found are first state nodes. However, we have to reevaluate if they
            #  are still boundary nodes or have become pure nodes.
            boundary_nodes.difference_update(new_first_nodes)
            nodes_to_scan.extend(new_first_nodes)
            first_nodes.update(new_first_nodes)

    # Ensure that all dependencies are assigned to the first state.
    assert len(first_nodes) >= 1
    assert all(all(iedge.src in first_nodes for iedge in state.in_edges(first_node)) for first_node in first_nodes)
    assert all(all(iedge.src in first_nodes for iedge in state.in_edges(bnode)) for bnode in boundary_nodes)
    assert boundary_nodes.isdisjoint(pure_first_nodes)
    assert boundary_nodes.union(pure_first_nodes) == first_nodes

    if len(first_nodes) == 1:
        warnings.warn(
            f"While splitting `{state.label}` the supplied `subgraph` leads to isolated nodes in the first state" +
            (", that will be removed." if not allow_isolated_nodes else "."))

    # Now create the new states on which we will operate. It is important that the newly created
    #  state is at the top.
    # NOTE: There is the case that `subgraph` contains the entire dataflow in `state`, in that case
    #   one might think, that we could perform an optimization, such as just create a new state and
    #   keep the nodes. However, this is not possible, since the function only returns the new state
    #   and code assumes that this is the first/top state containing the `subgraph` dataflow. Thus
    #   we have to perform the relocation.
    second_state = state
    first_state = state.parent_graph.add_state_before(state, label=label)

    # This map maps the nodes from the old/original state to the corresponding node in the first/new
    #  state. The pure nodes are the exact same objects, but the we copy the boundary nodes, since
    #  they have also to be present inside the second state.
    first_nodes_map: Dict[nodes.Node, nodes.Node] = {node: node for node in pure_first_nodes}
    first_nodes_map.update({old_bnode: copy.deepcopy(old_bnode) for old_bnode in boundary_nodes})

    # Save the edges that we have to copy.
    subgraph_to_copy = graph.SubgraphView(second_state, first_nodes)
    edges_to_copy = list(subgraph_to_copy.edges())

    # Now remove and the nodes from the first state and add them to the second state.
    second_state.remove_nodes_from(pure_first_nodes)
    first_state.add_nodes_from(first_nodes_map.values())

    # Now recreate the edge structure in the first state based on the first state.
    for original_edge in edges_to_copy:
        first_state.add_edge(
            first_nodes_map[original_edge.src],
            original_edge.src_conn,
            first_nodes_map[original_edge.dst],
            original_edge.dst_conn,
            original_edge.data,
        )

    # Cleaning the connectors of the boundary nodes. In the first state we have to remove all
    #  outgoing connectors and in the second state all incoming.
    # NOTE: I am actually not sure why this is here, it was in the original code, so I retained it.
    #   Since, only non view AccessNodes can transmit data across states, the connectors for them
    #   should be `None` anyway.
    for second_state_boundary_node in boundary_nodes:
        first_state_boundary_node = first_nodes_map[second_state_boundary_node]
        assert second_state.in_degree(second_state_boundary_node) == 0
        first_state_boundary_node._out_connectors.clear()
        second_state_boundary_node._in_connectors.clear()

    # Remove isolated nodes.
    # TODO(phimuell): Is this the best approach, it might be a bit too far reaching.
    if not allow_isolated_nodes:
        warning_was_issued = False
        for state_to_clean in [first_state, second_state]:
            for node in list(state_to_clean.nodes()):
                # This is how the validation defines isolated.
                if isinstance(node, nodes.CodeNode):
                    continue
                elif state_to_clean.degree(node) == 0:
                    if (not warning_was_issued) and (state_to_clean is second_state):
                        # Warning for isolated nodes in the second state if we do not delete it.
                        #  is a bit expensive, this is why we do not issue a warning in that case.
                        warnings.warn(
                            f"While splitting state `{state.label}` some nodes in the original state, got isolated and are now deleted."
                        )
                        warning_was_issued = True
                    state_to_clean.remove_node(node)

    return first_state


def isolate_nested_sdfg(
    state: SDFGState,
    nsdfg_node: nodes.NestedSDFG,
    test_if_applicable: bool = False,
) -> Union[Tuple[SDFGState, SDFGState, SDFGState], bool]:
    """Isolate the nested SDFG.

    The function will split `state` into three states:
    - Pre State: Contains the data flow that is needed to compute the dependency
        of the nested SDFG.
    - Middle State: This state contains the nested SDFG and the nodes that are needed
        as input or output to the nested SDFG.
    - Post State: Contains the data flow that uses the data that was computed
        by the nested SDFG.

    The important aspect of this function is, that it will not increase the
    number of AccessNodes that are written to, it might, however, add new AccessNodes
    that read data.

    :param state: The state on which we operate.
    :param nested_sdfg: The nested SDFG node that should be isolated.
    :param test_if_applicable: If `True` then do not perform the splitting, only verify
        that it can be performed by returning either `True` or `False`.
    """

    # We can only isolate the nested SDFG if it is on global scope, for example
    #  inside a Map it is impossible to split the state.
    if state.scope_dict()[nsdfg_node] is not None:
        if test_if_applicable:
            return False
        raise ValueError(f'Cannot isolate NestedSDFG "{nsdfg_node}" because it is within a scope.')

    # These are the nodes that will be moved to the Pre State, they are found through
    #  a backwards search starting from the nodes that serves as input to the nested
    #  SDFG. It is important that these nodes, that serves as input to the nested
    #  SDFG are also belonging to this set. But they are only added if they needed.
    pre_nodes: Set[nodes.Node] = set()
    to_visit: List[nodes.Node] = []
    for iedge in state.in_edges(nsdfg_node):
        input_node: nodes.AccessNode = iedge.src
        assert isinstance(input_node, nodes.AccessNode)
        if state.in_degree(input_node) != 0:
            to_visit.append(input_node)
    visited: Set[nodes.Node] = set()
    while len(to_visit) > 0:
        node_to_process = to_visit.pop()
        if node_to_process in visited:
            continue
        pre_nodes.add(node_to_process)
        to_visit.extend(iedge.src for iedge in state.in_edges(node_to_process))

    # These are the nodes of the middle state. Which are all access nodes that serves
    #  as input to the nested SDFG and the nested SDFG itself.
    #  Note that the AccessNodes serving as input and output of the nested SDFG
    #  belonging to the pre and post set, respectively, as well.
    middle_nodes: Set[nodes.Node] = {nsdfg_node}
    for iedge in state.in_edges(nsdfg_node):
        if (not isinstance(iedge.src, nodes.AccessNode)) or isinstance(iedge.src.desc(state.sdfg), data.View):
            if test_if_applicable:
                return False
            raise ValueError("Can only split if the inputs to the nested SDFG are AccessNodes to non view data.")
        middle_nodes.add(iedge.src)
    for oedge in state.out_edges(nsdfg_node):
        # We require that there is only one incoming edge on an output node. This seems
        #  like a restriction but it is not, because the inliner for nested SDFGs
        #  requires that the whole array is mapped inside. So if there would be
        #  multiple incoming edges the original SDFG would be invalid, because the
        #  same memory would be written to multiple times.
        if ((not all(iedge.src is nsdfg_node for iedge in state.in_edges(oedge.dst)))
                or (not isinstance(oedge.dst, nodes.AccessNode)) or isinstance(oedge.dst, data.View)):
            if test_if_applicable:
                return False
            raise ValueError(
                "Can only split if the out to the nested SDFG are AccessNodes to non view data and the AccessNodes are only connected to the nested SDFG."
            )
        middle_nodes.add(oedge.dst)

    # These are the nodes that belongs to the Post State. There are two reasons why a
    #  node belongs to the set of post nodes.
    #  The first is that the node does not belong to any other set.
    post_nodes: Set[nodes.Node] = {
        node
        for node in state.nodes() if (node not in pre_nodes) and (node not in middle_nodes)
    }

    # The second reason, are read dependencies, for this we have to look at the incoming
    #  edges and add any node that we need.
    if len(post_nodes) != 0:
        for pnode in post_nodes.copy():
            for iedge in state.in_edges(pnode):
                node: nodes.Node = iedge.src
                if node not in post_nodes:
                    if (not isinstance(node, nodes.AccessNode)) or isinstance(node.desc(state.sdfg), data.View):
                        if test_if_applicable:
                            return False
                        raise ValueError("Can not replicate non non-View AccessNodes into the post state.")
                    post_nodes.add(node)

    if test_if_applicable:
        return True

    # Now we are creating the two new states.
    parent_graph = state.parent_graph
    pre_state: dace.SDFGState = parent_graph.add_state_before(state=state,
                                                              label=(state.label +
                                                                     "_pre_state" if state.label else None))
    post_state: dace.SDFGState = parent_graph.add_state_after(state=state,
                                                              label=(state.label +
                                                                     "_post_state" if state.label else None))

    # We will now populate the pre state.
    pre_old_to_new_map: Dict[nodes.Node, nodes.Node] = dict()
    for node in pre_nodes:
        new_node = copy.deepcopy(node) if node in middle_nodes else node
        pre_old_to_new_map[node] = new_node
        pre_state.add_node(new_node)

    # Now add the edges, we only have to inspect the incoming ones.
    for old_dst in pre_nodes:
        new_dst = pre_old_to_new_map[old_dst]
        for old_iedge in state.in_edges(old_dst):
            old_src = old_iedge.src
            if old_src in pre_nodes:
                new_src = pre_old_to_new_map[old_src]
                pre_state.add_edge(
                    new_src,
                    old_iedge.src_conn,
                    new_dst,
                    old_iedge.dst_conn,
                    copy.deepcopy(old_iedge.data),
                )

    # Now we will populate the post state.
    post_old_to_new_map: Dict[nodes.Node, nodes.Node] = dict()
    for node in post_nodes:
        new_node = copy.deepcopy(node) if (node in middle_nodes or node in pre_nodes) else node
        post_old_to_new_map[node] = new_node
        post_state.add_node(new_node)

    # Now make the connections inside the post node. For this we only have to look
    #  at the outgoing edges in the original state.
    for old_src in post_nodes:
        new_src = post_old_to_new_map[old_src]
        for old_oedge in state.out_edges(old_src):
            old_dst = old_oedge.dst
            if old_dst in post_nodes:
                new_dst = post_old_to_new_map[old_dst]
                post_state.add_edge(
                    new_src,
                    old_oedge.src_conn,
                    new_dst,
                    old_oedge.dst_conn,
                    copy.deepcopy(old_oedge.data),
                )

    # Remove all nodes from the middle state that are not classified as middle nodes,
    #  this will also remove all the edges that are no longer needed.
    for node in list(state.nodes()):
        if node not in middle_nodes:
            state.remove_node(node)

    return (pre_state, state, post_state)


def _get_internal_subset(internal_memlet: Memlet,
                         external_memlet: Memlet,
                         use_src_subset: bool = False,
                         use_dst_subset: bool = False) -> subsets.Subset:
    if (internal_memlet.data != external_memlet.data and internal_memlet.other_subset is not None):
        return internal_memlet.other_subset
    if not use_src_subset and not use_dst_subset:
        return internal_memlet.subset
    if use_src_subset and use_dst_subset:
        raise ValueError('Source and destination subsets cannot be specified at the same time')
    if use_src_subset and internal_memlet.src_subset is not None:
        return internal_memlet.src_subset
    if use_dst_subset and internal_memlet.dst_subset is not None:
        return internal_memlet.dst_subset
    return internal_memlet.subset


def unsqueeze_memlet(internal_memlet: Memlet,
                     external_memlet: Memlet,
                     preserve_minima: bool = False,
                     use_src_subset: bool = False,
                     use_dst_subset: bool = False,
                     internal_offset: Tuple[int] = None,
                     external_offset: Tuple[int] = None) -> Memlet:
    """ Unsqueezes and offsets a memlet, as per the semantics of nested
        SDFGs.
        :param internal_memlet: The internal memlet (inside nested SDFG) before modification.
        :param external_memlet: The external memlet before modification.
        :param preserve_minima: Do not change the subset's minimum elements.
        :param use_src_subset: If both sides of the memlet refer to same array, prefer source subset.
        :param use_dst_subset: If both sides of the memlet refer to same array, prefer destination subset.
        :param internal_offset: The internal memlet's data descriptor offset.
        :param external_offset: The external memlet's data descriptor offset.
        :return: Offset Memlet to set on the resulting graph.
    """
    internal_subset = _get_internal_subset(internal_memlet, external_memlet, use_src_subset, use_dst_subset)
    internal_offset = internal_offset or [0] * len(internal_subset)
    external_offset = external_offset or [0] * len(external_memlet.subset)
    internal_subset = internal_subset.offset_new(internal_offset, False)
    result = Memlet.from_memlet(internal_memlet)
    result.subset = internal_subset

    shape = external_memlet.subset.size()
    if len(internal_subset) < len(external_memlet.subset):
        ones = [i for i, d in enumerate(shape) if d == 1]
        # Special case: If internal memlet is one element and the top
        # memlet uses all its dimensions, ignore the internal element
        # TODO: There must be a better solution
        if (len(internal_subset) == 1 and ones == list(range(len(shape)))
                and (internal_subset[0] == (0, 0, 1) or internal_subset[0] == 0)):
            to_unsqueeze = ones[1:]
        else:
            to_unsqueeze = ones

        # NOTE: There can be an issue where a unitary dimension wasn't squeezed, e.g., when using the dataflow syntax.
        # In such cases, more ones than necessary will be detected.
        # TODO: Find a better solution
        if len(internal_subset) + len(to_unsqueeze) > len(external_memlet.subset):
            external_subset = external_memlet.subset.offset_new(external_offset, False)
            to_unsqueeze = [i for i, d in enumerate(shape) if d == 1 and external_subset[i] != (0, 0, 1)]
        if len(internal_subset) + len(to_unsqueeze) != len(external_memlet.subset):
            raise NotImplementedError

        result.subset.unsqueeze(to_unsqueeze)
        internal_offset = list(internal_offset)
        for axis in sorted(to_unsqueeze):
            internal_offset.insert(axis, external_offset[axis])
    elif len(internal_subset) > len(external_memlet.subset):
        # Try to squeeze internal memlet
        remaining = result.subset.squeeze()
        if len(result.subset) != len(external_memlet.subset):
            raise ValueError('Unexpected extra dimensions in internal memlet '
                             'while un-squeezing memlet.\nExternal memlet: %s\n'
                             'Internal memlet: %s' % (external_memlet, internal_memlet))
        internal_offset = [internal_offset[idx] for idx in range(len(internal_offset)) if idx in remaining]

    external_subset = external_memlet.subset.offset_new(external_offset, False)
    result.subset.offset(external_subset, False)
    result.subset.offset(external_offset, True)

    if preserve_minima:
        if len(result.subset) != len(external_memlet.subset):
            raise ValueError('Memlet specifies reshape that cannot be un-squeezed.\n'
                             'External memlet: %s\nInternal memlet: %s' % (external_memlet, internal_memlet))
        original_minima = external_memlet.subset.min_element()
        for i in set(range(len(original_minima))):
            rb, re, rs = result.subset.ranges[i]
            result.subset.ranges[i] = (original_minima[i], re, rs)
    # TODO: Offset rest of memlet according to other_subset
    if external_memlet.other_subset is not None:
        raise NotImplementedError

    # Actual result preserves 'other subset' and placement of subsets in memlet
    actual_result = Memlet.from_memlet(internal_memlet)
    actual_result.data = external_memlet.data
    if actual_result.other_subset:
        if internal_memlet.data == external_memlet.data:
            actual_result.subset = result.subset
        else:
            actual_result.other_subset = actual_result.subset
            actual_result.subset = result.subset
            actual_result._is_data_src = not actual_result._is_data_src
    else:
        actual_result.subset = result.subset

    return actual_result


def replicate_scope(sdfg: SDFG, state: SDFGState, scope: ScopeSubgraphView) -> ScopeSubgraphView:
    """
    Replicates a scope subgraph view within a state, reconnecting all external
    edges to the same nodes.

    :param sdfg: The SDFG in which the subgraph scope resides.
    :param state: The SDFG state in which the subgraph scope resides.
    :param scope: The scope subgraph to replicate.
    :return: A reconnected replica of the scope.
    """
    exit_node = state.exit_node(scope.entry)

    # Replicate internal graph
    new_nodes = []
    new_entry = None
    new_exit = None
    to_find_new_names: Set[nodes.AccessNode] = set()
    for node in scope.nodes():
        node_copy = copy.deepcopy(node)
        if node == scope.entry:
            new_entry = node_copy
        elif node == exit_node:
            new_exit = node_copy

        if (isinstance(node, nodes.AccessNode) and node.desc(sdfg).lifetime == dtypes.AllocationLifetime.Scope
                and node.desc(sdfg).transient):
            to_find_new_names.add(node_copy)
        state.add_node(node_copy)
        new_nodes.append(node_copy)

    for edge in scope.edges():
        src = scope.nodes().index(edge.src)
        dst = scope.nodes().index(edge.dst)
        state.add_edge(new_nodes[src], edge.src_conn, new_nodes[dst], edge.dst_conn, copy.deepcopy(edge.data))

    # Reconnect external scope nodes
    for edge in state.in_edges(scope.entry):
        state.add_edge(edge.src, edge.src_conn, new_entry, edge.dst_conn, copy.deepcopy(edge.data))
    for edge in state.out_edges(exit_node):
        state.add_edge(new_exit, edge.src_conn, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))

    # Set the exit node's map to match the entry node
    new_exit.map = new_entry.map

    # Replicate all temporary transients within scope
    for node in to_find_new_names:
        desc = node.desc(sdfg)
        new_name = sdfg.add_datadesc(node.data, copy.deepcopy(desc), find_new_name=True)
        node.data = new_name
        for edge in state.all_edges(node):
            for e in state.memlet_tree(edge):
                e.data.data = new_name

    return ScopeSubgraphView(state, new_nodes, new_entry)


def offset_map(state: SDFGState, entry: nodes.MapEntry, dim: int, offset: symbolic.SymbolicType, negative: bool = True):
    """
    Offsets a map parameter and its contents by a value.

    :param state: The state in which the map resides.
    :param entry: The map entry node.
    :param dim: The map dimension to offset.
    :param offset: The value to offset by.
    :param negative: If True, offsets by ``-offset``.
    """
    entry.map.range.offset(offset, negative, indices=[dim])
    param = entry.map.params[dim]
    subgraph = state.scope_subgraph(entry)
    # Offset map param by -offset, contents by +offset and vice versa
    if negative:
        subgraph.replace(param, f'({param} + {offset})')
    else:
        subgraph.replace(param, f'({param} - {offset})')


def split_interstate_edges(cfg: ControlFlowRegion) -> None:
    """
    Splits all inter-state edges into edges with conditions and edges with assignments.
    This procedure helps in nested loop detection.

    :param cfg: The control flow graph to split
    :note: Operates in-place on the graph.
    """
    for cfg in cfg.all_control_flow_regions():
        for e in cfg.edges():
            if e.data.assignments and not e.data.is_unconditional():
                tmpstate = cfg.add_state()
                cfg.add_edge(e.src, tmpstate, InterstateEdge(condition=e.data.condition))
                cfg.add_edge(tmpstate, e.dst, InterstateEdge(assignments=e.data.assignments))
                cfg.remove_edge(e)


def is_symbol_unused(sdfg: SDFG, sym: str) -> bool:
    """
    Checks for uses of symbol in an SDFG, and if there are none returns False.

    :param sdfg: The SDFG to search.
    :param sym: The symbol to test.
    :return: True if the symbol can be removed, False otherwise.
    """
    # TODO: Investigate if this function can be replaced by a call to `used_symbols()`.
    #   See https://github.com/spcl/dace/pull/2080#discussion_r2226418881
    for desc in sdfg.arrays.values():
        if sym in map(str, desc.free_symbols):
            return False
    for state in sdfg.states():
        if sym in state.free_symbols:
            return False
    for e in sdfg.all_interstate_edges():
        if sym in e.data.free_symbols:
            return False

    # Not found, symbol can be removed
    return True


def are_subsets_contiguous(subset_a: subsets.Subset, subset_b: subsets.Subset, dim: int = None) -> bool:

    if dim is not None:
        # A version that only checks for contiguity in certain
        # dimension (e.g., to prioritize stride-1 range)
        if (not isinstance(subset_a, subsets.Range) or not isinstance(subset_b, subsets.Range)):
            raise NotImplementedError('Contiguous subset check only implemented for ranges')

        # Other dimensions must be equal
        for i, (s1, s2) in enumerate(zip(subset_a.ranges, subset_b.ranges)):
            if i == dim:
                continue
            if s1[0] != s2[0] or s1[1] != s2[1] or s1[2] != s2[2]:
                return False

        # Set of conditions for contiguous dimension
        ab = (subset_a[dim][1] + 1) == subset_b[dim][0]
        a_overlap_b = subset_a[dim][1] >= subset_b[dim][0]
        ba = (subset_b[dim][1] + 1) == subset_a[dim][0]
        b_overlap_a = subset_b[dim][1] >= subset_a[dim][0]
        # NOTE: Must check with "==" due to sympy using special types
        return (ab == True or a_overlap_b == True or ba == True or b_overlap_a == True)

    # General case
    bbunion = subsets.bounding_box_union(subset_a, subset_b)
    try:
        if bbunion.num_elements() == (subset_a.num_elements() + subset_b.num_elements()):
            return True
    except TypeError:
        pass

    return False


def find_contiguous_subsets(subset_list: List[subsets.Subset], dim: int = None) -> Set[subsets.Subset]:
    """
    Finds the set of largest contiguous subsets in a list of subsets.

    :param subsets: Iterable of subset objects.
    :param dim: Check for contiguity only for the specified dimension.
    :return: A list of contiguous subsets.
    """
    # Currently O(n^3) worst case. TODO: improve
    subset_set = set(subsets.Range.from_indices(s) if isinstance(s, subsets.Indices) else s for s in subset_list)
    while True:
        for sa, sb in itertools.product(subset_set, subset_set):
            if sa is sb:
                continue
            if sa.covers(sb):
                subset_set.remove(sb)
                break
            elif sb.covers(sa):
                subset_set.remove(sa)
                break
            elif are_subsets_contiguous(sa, sb, dim):
                subset_set.remove(sa)
                subset_set.remove(sb)
                subset_set.add(subsets.bounding_box_union(sa, sb))
                break
        else:  # No modification performed
            break
    return subset_set


def constant_symbols(sdfg: SDFG) -> Set[str]:
    """
    Returns a set of symbols that will never change values throughout the course
    of the given SDFG. Specifically, these are the input symbols (i.e., not
    defined in a particular scope) that are never set by interstate edges.

    :param sdfg: The input SDFG.
    :return: A set of symbol names that remain constant throughout the SDFG.
    """
    interstate_symbols = {k for e in sdfg.all_interstate_edges() for k in e.data.assignments.keys()}
    return set(sdfg.symbols) - interstate_symbols


def simplify_state(state: SDFGState, remove_views: bool = False) -> MultiDiGraph:
    """
    Returns a networkx MultiDiGraph object that contains all the access nodes
    and corresponding edges of an SDFG state. The removed code nodes and map
    scopes are replaced by edges that connect their ancestor and succesor access
    nodes.

    :param state: The input SDFG state.
    :return: The MultiDiGraph object.
    """

    sdfg = state.parent

    # Copy the whole state
    G = MultiDiGraph()
    for n in state.nodes():
        G.add_node(n)
    for n in state.nodes():
        for e in state.all_edges(n):
            G.add_edge(e.src, e.dst)
    # Collapse all mappings and their scopes into one node
    scope_children = state.scope_children()
    for n in scope_children[None]:
        if isinstance(n, nodes.EntryNode):
            G.add_edges_from([(n, x) for (y, x) in G.out_edges(state.exit_node(n))])
            G.remove_nodes_from(scope_children[n])
    # Remove all nodes that are not AccessNodes or have incoming
    # wcr edges and connect their predecessors and successors
    for n in state.nodes():
        if n in G.nodes():
            if (not isinstance(n, nodes.AccessNode) or (remove_views and isinstance(sdfg.arrays[n.data], data.View))):
                for p in G.predecessors(n):
                    for c in G.successors(n):
                        G.add_edge(p, c)
                G.remove_node(n)
            else:
                for e in state.all_edges(n):
                    if e.data.wcr is not None:
                        for p in G.predecessors(n):
                            for s in G.successors(n):
                                G.add_edge(p, s)
                        G.remove_node(n)
                        break

    return G


def tile(sdfg: SDFG, map_entry: nodes.MapEntry, divides_evenly: bool, skew: bool, **tile_sizes: symbolic.SymbolicType):
    """
    Helper function that tiles a Map scope by the given sizes, in the
    given order.

    :param sdfg: The SDFG where the map resides.
    :param map_entry: The map entry node to tile.
    :param divides_evenly: If True, skips pre/postamble for cases
                           where the map dimension is not a multiplier
                           of the tile size.
    :param skew: If True, skews the tiled map to start from zero. Helps
                 compilers improve performance in certain cases.
    :param tile_sizes: An ordered dictionary of the map parameter names
                       to tile and their respective tile size (which can be
                       symbolic expressions).
    """
    # Avoid import loop
    from dace.transformation.dataflow import StripMining

    for k, v in tile_sizes.items():
        StripMining.apply_to(sdfg,
                             dict(dim_idx=map_entry.params.index(k),
                                  tile_size=str(v),
                                  divides_evenly=divides_evenly,
                                  skew=skew),
                             map_entry=map_entry)


def permute_map(map_entry: nodes.MapEntry, perm: List[int]):
    """ Permutes indices of a map according to a given list of integers. """
    map_entry.map.params = [map_entry.map.params[p] for p in perm]
    map_entry.map.range = [map_entry.map.range[p] for p in perm]


def extract_map_dims(sdfg: SDFG, map_entry: nodes.MapEntry, dims: List[int]) -> Tuple[nodes.MapEntry, nodes.MapEntry]:
    """
    Helper function that extracts specific map dimensions into an outer map.

    :param sdfg: The SDFG where the map resides.
    :param map_entry: Map entry node to extract.
    :param dims: A list of dimension indices to extract.
    :return: A 2-tuple containing the extracted map and the remainder map.
    """
    # Avoid import loop
    from dace.transformation.dataflow import MapCollapse, MapExpansion

    # Make extracted dimensions first
    permute_map(map_entry, dims + [i for i in range(len(map_entry.map.params)) if i not in dims])
    # Expand map
    if len(map_entry.map.params) > 1:
        entries = MapExpansion.apply_to(sdfg, map_entry=map_entry)

        # Collapse extracted maps
        extracted_map = entries[0]
        for idx in range(len(dims) - 1):
            extracted_map, _ = MapCollapse.apply_to(
                sdfg,
                outer_map_entry=extracted_map,
                inner_map_entry=entries[idx + 1],
                permissive=True,  # Since MapExpansion creates sequential maps
            )

        # Collapse remaining maps
        map_to_collapse = entries[len(dims)]
        for idx in range(len(dims), len(entries) - 1):
            map_to_collapse, _ = MapCollapse.apply_to(
                sdfg,
                outer_map_entry=map_to_collapse,
                inner_map_entry=entries[idx + 1],
                permissive=True,  # Since MapExpansion creates sequential maps
            )
    else:
        extracted_map = map_entry
        map_to_collapse = map_entry

    return extracted_map, map_to_collapse


def scope_tree_recursive(state: SDFGState, entry: Optional[nodes.EntryNode] = None) -> ScopeTree:
    """
    Returns a scope tree that includes scopes from nested SDFGs.

    :param state: The state that contains the root of the scope tree.
    :param entry: A scope entry node to set as root, otherwise the state is
                  the root if None is given.

    :note: This function adds a `state` attribute to the `ScopeTree` objects, it refers to
        the state to which the scope was found.
    """
    # We have to make a copy because we modify the `ScopeTree` objects that are returned and they
    #  are cached. We can not use `deepcopy()` here because this would also copy scope nodes,
    #  which we do not want. Thus we call `ScopeTree.copy()` which only copies the `ScopeTree`s
    #  inside `children` but everything else is just assigned.
    stree = copy.copy(state.scope_tree()[entry])
    stree.state = state  # Annotate state in tree

    # Add nested SDFGs as children
    def traverse(state: SDFGState, treenode: ScopeTree):
        snodes = copy.copy(state.scope_children()[treenode.entry])  # See above why.

        for node in snodes:
            if isinstance(node, nodes.NestedSDFG):
                for nstate in node.sdfg.states():
                    ntree = copy.copy(nstate.scope_tree()[None])  # See above why.
                    ntree.state = nstate
                    treenode.children.append(ntree)

        for child in treenode.children:
            if hasattr(child, 'state') and child.state != state:
                traverse(child.state, child)

    traverse(state, stree)
    return stree


def get_internal_scopes(state: SDFGState,
                        entry: nodes.EntryNode,
                        immediate: bool = False,
                        recursive_scope_tree: Optional[ScopeTree] = None) -> List[Tuple[SDFGState, nodes.EntryNode]]:
    """
    Returns all internal scopes within a given scope, including if they
    reside in nested SDFGs.

    :param state: State in which entry node resides.
    :param entry: The entry node to start from.
    :param immediate: If True, only returns the scopes that are immediately nested in the map.
    :param recursive_scope_tree: The recursive scope tree, see `scope_tree_recursive()` for more.
    """

    if recursive_scope_tree is None:
        recursive_scope_tree = scope_tree_recursive(state, entry)

    result = []

    def traverse(state: SDFGState, treenode: ScopeTree):
        for child in treenode.children:
            if child.entry is not None:
                result.append((state, child.entry))
                if not immediate:
                    traverse(state, child)
            else:  # Nested SDFG
                traverse(child.state, child)

    traverse(state, recursive_scope_tree)
    return result


def gpu_map_has_explicit_threadblocks(state: SDFGState, entry: nodes.EntryNode) -> bool:
    """
    Returns True if GPU_Device map has explicit thread-block maps nested within.
    """
    rstree = scope_tree_recursive(state, entry)
    internal_maps = get_internal_scopes(state, entry, recursive_scope_tree=rstree)
    if any(m.schedule in (dtypes.ScheduleType.GPU_ThreadBlock, dtypes.ScheduleType.GPU_ThreadBlock_Dynamic)
           for _, m in internal_maps):
        return True

    imm_maps = get_internal_scopes(state, entry, immediate=True, recursive_scope_tree=rstree)
    if any(m.schedule == dtypes.ScheduleType.Default for _, m in imm_maps):
        return True

    return False


def gpu_map_has_explicit_dyn_threadblocks(state: SDFGState, entry: nodes.EntryNode) -> bool:
    """
    Returns True if GPU_Device map has explicit thread-block maps nested within.
    """
    internal_maps = get_internal_scopes(state, entry)
    if any(m.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic for _, m in internal_maps):
        return True

    return False


def reconnect_edge_through_map(
        state: SDFGState, edge: graph.MultiConnectorEdge[Memlet], new_node: Union[nodes.EntryNode, nodes.ExitNode],
        keep_src: bool) -> Tuple[graph.MultiConnectorEdge[Memlet], graph.MultiConnectorEdge[Memlet]]:
    """
    Reconnects an edge through a map scope, removes old edge, and returns the
    two new edges.

    :param state: The state in which the edge and map reside.
    :param edge: The edge to reconnect and remove.
    :param new_node: The scope (map) entry or exit to reconnect through.
    :param keep_src: If True, keeps the source of the edge intact, otherwise
                     keeps destination of edge.
    :return: A 2-tuple of (incoming edge, outgoing edge).
    """
    if keep_src:
        result = state.add_edge_pair(new_node,
                                     edge.dst,
                                     edge.src,
                                     edge.data,
                                     internal_connector=edge.dst_conn,
                                     external_connector=edge.src_conn)
    else:
        result = state.add_edge_pair(new_node,
                                     edge.src,
                                     edge.dst,
                                     edge.data,
                                     internal_connector=edge.src_conn,
                                     external_connector=edge.dst_conn)
    state.remove_edge(edge)
    return result


def contained_in(state: SDFGState, node: nodes.Node, scope: nodes.EntryNode) -> bool:
    """
    Returns true if the specified node is contained within the scope opened
    by the given entry node (including through nested SDFGs).
    """
    # A node is contained within itself
    if node is scope:
        return True
    cursdfg = state.parent
    curstate = state
    curscope = state.entry_node(node)
    while cursdfg is not None:
        while curscope is not None:
            if curscope is scope:
                return True
            curscope = curstate.entry_node(curscope)
        curstate = cursdfg.parent
        curscope = cursdfg.parent_nsdfg_node
        cursdfg = cursdfg.parent_sdfg
    return False


def get_parent_map(state: SDFGState, node: Optional[nodes.Node] = None) -> Optional[Tuple[nodes.EntryNode, SDFGState]]:
    """
    Returns the map in which the state (and node) are contained in, or None if
    it is free.

    :param state: The state to test or parent of the node to test.
    :param node: The node to test (optional).
    :return: A tuple of (entry node, state) or None.
    """
    cursdfg = state.parent
    curstate = state
    curscope = node
    while cursdfg is not None:
        if curscope is not None:
            curscope = curstate.entry_node(curscope)
            if curscope is not None:
                return curscope, curstate
        curstate = cursdfg.parent
        curscope = cursdfg.parent_nsdfg_node
        cursdfg = cursdfg.parent_sdfg
    return None


def redirect_edge(state: SDFGState,
                  edge: graph.MultiConnectorEdge[Memlet],
                  new_src: Optional[nodes.Node] = None,
                  new_dst: Optional[nodes.Node] = None,
                  new_src_conn: Optional[str] = None,
                  new_dst_conn: Optional[str] = None,
                  new_data: Optional[str] = None,
                  new_memlet: Optional[Memlet] = None) -> graph.MultiConnectorEdge[Memlet]:
    """
    Redirects an edge in a state. Choose which elements to override by setting
    the keyword arguments.

    :param state: The SDFG state in which the edge resides.
    :param edge: The edge to redirect.
    :param new_src: If provided, redirects the source of the new edge.
    :param new_dst: If provided, redirects the destination of the new edge.
    :param new_src_conn: If provided, renames the source connector of the edge.
    :param new_dst_conn: If provided, renames the destination connector of the
                         edge.
    :param new_data: If provided, changes the data on the memlet of the edge,
                     and the entire associated memlet tree.
    :param new_memlet: If provided, changes only the memlet of the new edge.
    :return: The new, redirected edge.
    :note: ``new_data`` and ``new_memlet`` cannot be used at the same time.
    """
    if new_data is not None and new_memlet is not None:
        raise ValueError('new_data and new_memlet cannot both be given.')
    mtree = None
    if new_data is not None:
        mtree = state.memlet_tree(edge)
    state.remove_edge(edge)
    if new_data is not None:
        memlet = copy.deepcopy(edge.data)
        memlet.data = new_data
        # Rename on full memlet tree
        for e in mtree:
            e.data.data = new_data
    else:
        memlet = new_memlet or edge.data
    new_edge = state.add_edge(new_src or edge.src, new_src_conn or edge.src_conn, new_dst or edge.dst, new_dst_conn
                              or edge.dst_conn, memlet)
    return new_edge


def replace_code_to_code_edges(sdfg: SDFG):
    """
    Adds access nodes between all code->code edges in each state.

    :param sdfg: The SDFG to process.
    """
    for state in sdfg.states():
        for edge in state.edges():
            if not isinstance(edge.src, nodes.CodeNode) or not isinstance(edge.dst, nodes.CodeNode):
                continue
            # Add access nodes
            aname = state.add_access(edge.data.data)
            state.add_edge(edge.src, edge.src_conn, aname, None, edge.data)
            state.add_edge(aname, None, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))
            state.remove_edge(edge)


def can_run_state_on_fpga(state: SDFGState):
    """
    Checks if state can be executed on FPGA. Used by FPGATransformState
    and HbmTransform.
    """
    for node, graph in state.all_nodes_recursive():
        # Consume scopes are currently unsupported
        if isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
            return False

        # Streams have strict conditions due to code generator limitations
        if (isinstance(node, nodes.AccessNode) and isinstance(graph.sdfg.arrays[node.data], data.Stream)):
            nodedesc = graph.sdfg.arrays[node.data]
            sdict = graph.scope_dict()
            if nodedesc.storage in [
                    dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_Pinned, dtypes.StorageType.CPU_ThreadLocal
            ]:
                return False

            # Cannot allocate FIFO from CPU code
            if sdict[node] is None:
                return False

            # Arrays of streams cannot have symbolic size on FPGA
            if symbolic.issymbolic(nodedesc.total_size, graph.sdfg.constants):
                return False

            # Streams cannot be unbounded on FPGA
            if nodedesc.buffer_size < 1:
                return False

    return True


def make_map_internal_write_external(sdfg: SDFG, state: SDFGState, map_exit: nodes.MapExit, access: nodes.AccessNode,
                                     sink: nodes.AccessNode):
    """
    Any writes to the Access node `access` that occur inside the Map with exit node `map_exit` are redirected to the
    Access node `sink` that is outside the Map. This method will remove, if possible, `access` and replace it with a
    transient.

    :param sdfg: The SDFG in which the Access node resides.
    :param state: The State in which the Access node resides.
    :param map_exit: The exit node of the Map.
    :param access: The Access node being written inside the Map.
    :param sink: The Access node to be written outside the Map.
    """

    # Special case for scalars: if there is no write conflict resolution, then abort, since it is implied that the
    # scalar is thread-local.
    if isinstance(access.desc(sdfg), data.Scalar):
        if any(e.data.wcr is None for e in state.in_edges(access)):
            return
    # Ignore views
    if isinstance(access.desc(sdfg), data.View):
        return

    # Compute the union of the destination subsets of the edges that write to `access.`
    in_union = None
    map_dependency = False
    for e in state.in_edges(access):
        subset = e.data.get_dst_subset(e, state)
        if any(str(s) in map_exit.map.params for s in subset.free_symbols):
            map_dependency = True
        if in_union is None:
            in_union = subset
        else:
            in_union = in_union.union(subset)

    # If none of the input subsets depend on the map parameters, then abort, since the array is thread-local.
    if not map_dependency:
        return

    # Check if the union covers the output edges of `access.`
    covers_out = True
    if in_union is None:
        covers_out = False
    else:
        for e in state.out_edges(access):
            subset = e.data.get_src_subset(e, state)
            if not in_union.covers(subset):
                covers_out = False
                break

    # If the union covers the output edges of `access`, then we can remove `access` and replace it with a transient.
    if covers_out:
        shape = in_union.size()
        if shape == [1]:
            name, _ = sdfg.add_scalar(access.data, access.desc(sdfg).dtype, transient=True, find_new_name=True)
        else:
            name, _ = sdfg.add_array(access.data, shape, access.desc(sdfg).dtype, transient=True, find_new_name=True)
        new_n = state.add_access(name)
        visited = set()
        for e in state.in_edges(access):
            if e in visited:
                continue
            offset = e.data.get_dst_subset(e, state)
            # NOTE: There can be nested Maps. Therefore, we need to iterate over the MemletTree.
            for e2 in state.memlet_tree(e):
                if e2 in visited:
                    continue
                visited.add(e2)
                src_subset = e2.data.get_src_subset(e2, state)
                dst_subset = e2.data.get_dst_subset(e2, state)
                dst = new_n if e2.dst is access else e2.dst
                state.add_edge(
                    e2.src, e2.src_conn, dst, e2.dst_conn,
                    Memlet(data=name, subset=dst_subset.offset_new(offset, negative=True), other_subset=src_subset))
            src_subset = e.data.get_src_subset(e, state)
            dst_subset = e.data.get_dst_subset(e, state)
            state.add_memlet_path(new_n,
                                  map_exit,
                                  sink,
                                  memlet=Memlet(data=sink.data,
                                                subset=copy.deepcopy(dst_subset),
                                                other_subset=dst_subset.offset_new(dst_subset, negative=True)))
        for e in state.out_edges(access):
            if e in visited:
                continue
            offset = e.data.get_src_subset(e, state)
            # NOTE: There can be nested Maps. Therefore, we need to iterate over the MemletTree.
            for e2 in state.memlet_tree(e):
                if e2 in visited:
                    continue
                visited.add(e2)
                src_subset = e2.data.get_src_subset(e2, state)
                dst_subset = e2.data.get_dst_subset(e2, state)
                src = new_n if e2.src is access else e2.src
                state.add_edge(
                    src, e2.src_conn, e2.dst, e2.dst_conn,
                    Memlet(data=name, subset=src_subset.offset_new(offset, negative=True), other_subset=dst_subset))
        for e in visited:
            state.remove_edge(e)
        state.remove_node(access)
    # Otherwise, we only add a memlet path to the sink.
    else:
        for e in state.in_edges(access):
            subset = e.data.get_dst_subset(e, state)
            state.add_memlet_path(access,
                                  map_exit,
                                  sink,
                                  memlet=Memlet(data=sink.data,
                                                subset=copy.deepcopy(subset),
                                                other_subset=copy.deepcopy(subset)))


def all_isedges_between(src: ControlFlowBlock, dst: ControlFlowBlock) -> Iterable[Edge[InterstateEdge]]:
    """
    Helper function that generates an iterable of all edges potentially encountered between two control flow blocks.
    """
    if src.sdfg is not dst.sdfg:
        raise RuntimeError('Blocks reside in different SDFGs')

    if src.parent_graph is dst.parent_graph:
        # Simple case where both blocks reside in the same graph:
        edges = set()
        for p in src.parent_graph.all_simple_paths(src, dst, as_edges=True):
            for e in p:
                edges.add(e)
                if isinstance(e.dst, ControlFlowRegion):
                    edges.update(e.dst.all_interstate_edges())
        return edges
    else:
        # In the case where the two blocks are not in the same graph, we follow this procedure:
        # 1. Collect the list of control flow regions on the direct path between the source and destination:
        #   a) Determine the 'lowest common parent' region
        #   b) Determine the list of parents of the source before the common parent is reached
        #   c) Determine the list of parents of the destination before the common parent is reached.
        # 2. In each of the parents of the source, add all edges from the source or the next parent until the
        #    end(s) of each region to the result
        # 3. In each of the destination's parents, add all edges from the start block on until the destination
        #    or next parent to the result.
        # 4. In the lowest common parent region, find all edge paths between the next parent regions for both
        #    the source and destination.
        # Note that for each edge, if the destination is a control flow region, any edges inside of it may also
        # be on the path and consequently also need to be added.
        edges = set()

        # Step 1.a): Find the lowest common parent region.
        common_regions = set()
        pivot_graph = src.parent_graph
        all_parent_regions_src = [pivot_graph]
        while not isinstance(pivot_graph, SDFG):
            pivot_graph = pivot_graph.parent_graph
            all_parent_regions_src.append(pivot_graph)
        pivot_graph = dst.parent_graph
        all_parent_regions_dst = [pivot_graph]
        while not isinstance(pivot_graph, SDFG):
            pivot_graph = pivot_graph.parent_graph
            all_parent_regions_dst.append(pivot_graph)
            if pivot_graph in all_parent_regions_src:
                common_regions.add(pivot_graph)

        # Step 1.b) and 1.c): Determine the list of parents involved in the path for the source and destination.
        involved_src: List[ControlFlowRegion] = []
        involved_dst: List[ControlFlowRegion] = []
        common_parent: ControlFlowRegion = None
        for r in all_parent_regions_src:
            if r not in common_regions:
                involved_src.append(r)
            else:
                common_parent = r
                break
        for r in all_parent_regions_dst:
            if r not in common_regions:
                involved_dst.append(r)
            else:
                if r is not common_parent:
                    raise RuntimeError('No common parent found')
                break

        # Step 2
        src_pivot = src
        for r in involved_src:
            if isinstance(r, ConditionalBlock):
                src_pivot = r
                continue
            for sink in r.sink_nodes():
                for p in r.all_simple_paths(src_pivot, sink, as_edges=True):
                    for e in p:
                        edges.add(e)
                        if isinstance(e.dst, ControlFlowRegion):
                            edges.update(e.dst.all_interstate_edges())
            src_pivot = r
        # Step 3
        dst_pivot = dst
        for r in involved_dst:
            if isinstance(r, ConditionalBlock):
                dst_pivot = r
                continue
            for p in r.all_simple_paths(r.start_block, dst_pivot, as_edges=True):
                for e in p:
                    edges.add(e)
                    if isinstance(e.dst, ControlFlowRegion):
                        edges.update(e.dst.all_interstate_edges())
            dst_pivot = r

        # Step 4
        for p in common_parent.all_simple_paths(src_pivot, dst_pivot, as_edges=True):
            for e in p:
                edges.add(e)
                if isinstance(e.dst, ControlFlowRegion) and not e.dst is dst_pivot:
                    edges.update(e.dst.all_interstate_edges())

        return edges
