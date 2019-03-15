""" Data dependency analysis functionality, as well as functions to convert
    an AST-parsed data-centric Python program into an SDFG. """
import ast
from collections import deque, OrderedDict
from copy import deepcopy as dcpy
import sympy

from dace import data as dt, types, symbolic
from dace.graph import edges as ed
from dace.graph import nodes as nd
from dace import subsets as sbs
from dace import sdfg
from dace.memlet import EmptyMemlet, Memlet
from dace.frontend.python import astnodes, astutils
from dace.frontend.python.astparser import MemletRemover, ModuleInliner


def create_states_simple(pdp,
                         out_sdfg,
                         start_state=None,
                         end_state=None,
                         start_edge=None):
    """ Creates a state per primitive, with the knowledge that they can be 
        optimized later.
        @param pdp: A parsed dace program.
        @param out_sdfg: The output SDFG.
        @param start_state: The starting/parent state to connect from (for
                            recursive calls).
        @param end_state: The end/parent state to connect to (for
                          recursive calls).
        @return: A dictionary mapping between a state and the list of dace
                 primitives included in it.
    """
    state_to_primitives = OrderedDict()

    # Create starting state and edge
    if start_state is None:
        start_state = out_sdfg.add_state('start')
        state_to_primitives[start_state] = []
    if start_edge is None:
        start_edge = ed.InterstateEdge()

    previous_state = start_state
    previous_edge = start_edge

    for i, primitive in enumerate(pdp.children):
        state = out_sdfg.add_state(primitive.name)
        state_to_primitives[state] = []
        # Edge that can be created on entry to control flow children
        entry_edge = None

        #########################################
        # Cases depending on primitive type
        #########################################

        # Nothing special happens with a dataflow node (nested states are
        # handled with a separate call to create_states_simple)
        if isinstance(primitive, astnodes._DataFlowNode):
            out_sdfg.add_edge(previous_state, state, previous_edge)
            state_to_primitives[state] = [primitive]
            previous_state = state
            previous_edge = ed.InterstateEdge()

        # Control flow needs to traverse into children nodes
        elif isinstance(primitive, astnodes._ControlFlowNode):
            # Iteration has >=3 states - begin, loop[...], end; and connects the
            # loop states, as well as the begin to end directly if the condition
            # did not evaluate to true
            if isinstance(primitive, astnodes._IterateNode):

                condition = ast.parse(
                    '(%s %s %s)' % (primitive.params[0], '<'
                                    if primitive.range[0][2] >= 0 else '>',
                                    primitive.range[0][1] + 1)).body[0]
                condition_neg = astutils.negate_expr(condition)

                # Loop-start state
                lstart_state = out_sdfg.add_state(primitive.name + '_start')
                state_to_primitives[lstart_state] = []
                out_sdfg.add_edge(previous_state, lstart_state, previous_edge)
                out_sdfg.add_edge(
                    lstart_state,
                    state,
                    ed.InterstateEdge(
                        assignments={
                            primitive.params[0]: primitive.range[0][0]
                        }))

                # Loop-end state that jumps back to `state`
                loop_state = out_sdfg.add_state(primitive.name + '_end')
                state_to_primitives[loop_state] = []
                # Connect loop
                out_sdfg.add_edge(
                    loop_state,
                    state,
                    ed.InterstateEdge(
                        assignments={
                            primitive.params[0]:
                            symbolic.pystr_to_symbolic(primitive.params[0]) +
                            primitive.range[0][2]
                        }))

                # End connection
                previous_state = state
                previous_edge = ed.InterstateEdge(condition=condition_neg)

                # Create children states
                cmap = create_states_simple(
                    primitive,
                    out_sdfg,
                    state,
                    loop_state,
                    ed.InterstateEdge(condition=condition))
                state_to_primitives.update(cmap)

            # Loop is similar to iterate, but more general w.r.t. conditions
            elif isinstance(primitive, astnodes._LoopNode):
                loop_condition = primitive.condition

                # Entry
                out_sdfg.add_edge(previous_state, state, previous_edge)

                # Loop-end state that jumps back to `state`
                loop_state = out_sdfg.add_state(primitive.name + '_end')
                state_to_primitives[loop_state] = []

                # Loopback
                out_sdfg.add_edge(loop_state, state, ed.InterstateEdge())
                # End connection
                previous_state = state
                previous_edge = ed.InterstateEdge(
                    condition=astutils.negate_expr(loop_condition))
                entry_edge = ed.InterstateEdge(condition=loop_condition)

                # Create children states
                cmap = create_states_simple(primitive, out_sdfg, state,
                                            loop_state, entry_edge)
                state_to_primitives.update(cmap)

            elif isinstance(primitive, astnodes._IfNode):
                if_condition = primitive.condition
                # Check if we have an else node, otherwise add a skip condition
                # ourselves
                if (i + 1) < len(pdp.children) and isinstance(
                        pdp.children[i + 1], astnodes._ElseNode):
                    has_else = True
                    else_prim = pdp.children[i + 1]
                    else_condition = else_prim.condition
                else:
                    has_else = False
                    else_condition = astutils.negate_expr(primitive.condition)

                # End-of-branch state (converge to this)
                bend_state = out_sdfg.add_state(primitive.name + '_end')
                state_to_primitives[bend_state] = []

                # Entry
                out_sdfg.add_edge(previous_state, state, previous_edge)

                # Create children states
                cmap = create_states_simple(
                    primitive,
                    out_sdfg,
                    state,
                    bend_state,
                    ed.InterstateEdge(condition=if_condition))
                state_to_primitives.update(cmap)

                # Handle 'else' condition
                if not has_else:
                    out_sdfg.add_edge(
                        state,
                        bend_state,
                        ed.InterstateEdge(condition=else_condition))
                else:
                    # Recursively parse 'else' primitive's children
                    cmap = create_states_simple(
                        else_prim,
                        out_sdfg,
                        state,
                        bend_state,
                        ed.InterstateEdge(condition=else_condition))
                    state_to_primitives.update(cmap)

                # Exit
                previous_state = bend_state
                previous_edge = ed.InterstateEdge()

            elif isinstance(primitive, astnodes._ElseNode):
                if i - 1 < 0 or not isinstance(pdp.children[i - 1],
                                               astnodes._IfNode):
                    raise SyntaxError('Found else state without matching if')

                # If 'else' state is correct, we already processed it
                del state_to_primitives[state]
                out_sdfg.remove_node(state)

    # Connect to end_state (and create it if necessary)
    if end_state is None:
        end_state = out_sdfg.add_state('end')
        state_to_primitives[end_state] = []
    out_sdfg.add_edge(previous_state, end_state, previous_edge)

    return state_to_primitives


def _make_full_range(memlet: astnodes._Memlet):
    fullRange = sbs.Range([(0, s - 1, 1) for s in memlet.data.shape])
    fullMemlet = astnodes._Memlet(memlet.data,
                                  memlet.dataname, memlet.attribute,
                                  fullRange.num_elements(), None, None,
                                  fullRange, memlet.veclen, None, None, {})
    return fullMemlet


def _full_memlet_from_array(arrayname, array):
    fullRange = sbs.Range([(0, s - 1, 1) for s in array.shape])
    fullMemlet = astnodes._Memlet(array, arrayname, None,
                                  fullRange.num_elements(), None, None,
                                  fullRange, 1, None, None, {})
    return fullMemlet


def inherit_dependencies(prim):

    # Inject tasklets for map nodes and push down dependencies
    if (isinstance(prim, (astnodes._MapNode, astnodes._ConsumeNode))
            and len(prim.children) == 0):
        tasklet = astnodes._TaskletNode(prim.name, prim.ast)
        tasklet.parent = prim
        tasklet.inputs = OrderedDict(
            [(k, v) for k, v in prim.inputs.items() if '__DACEIN_' not in k])
        tasklet.outputs = OrderedDict(
            [(k, v) for k, v in prim.outputs.items() if '__DACEIN_' not in k])
        prim.inputs = OrderedDict(
            [(k, v) for k, v in prim.inputs.items() if '__DACEIN_' in k])
        prim.outputs = OrderedDict(
            [(k, v) for k, v in prim.outputs.items() if '__DACEIN_' in k])
        prim.children.append(tasklet)

    # The recursive dependencies of this node which we will return
    dependIn = OrderedDict()
    dependOut = OrderedDict()

    # Add own dependencies (input)
    inputQueue = deque(prim.inputs.items())
    while len(inputQueue) > 0:
        arrname, memlet = inputQueue.popleft()
        fullMemlet = _make_full_range(memlet)
        dependIn[fullMemlet.dataname] = fullMemlet
        # Additional dependencies (e.g., as a result of indirection)
        for aname, additional_arr in memlet.otherdeps.items():
            additional_astmemlet = _full_memlet_from_array(
                aname, additional_arr)
            dependIn[additional_astmemlet.dataname] = additional_astmemlet

    # Add own dependencies (output)
    outputQueue = deque(prim.outputs.items())
    while len(outputQueue) > 0:
        arrname, memlet = outputQueue.popleft()
        fullMemlet = _make_full_range(memlet)
        dependOut[fullMemlet.dataname] = fullMemlet
        if isinstance(memlet.subset, astnodes._Memlet):
            outputQueue.push(memlet.subset)

    # Add recursively inherited dependencies
    inheritIn = OrderedDict()
    inheritOut = OrderedDict()
    arrs = prim.transients.keys()
    for child in prim.children:
        childIn, childOut = inherit_dependencies(child)
        # Only inherit dependencies from arrays defined in this scope
        inheritIn.update(
            OrderedDict([(k, v) for k, v in childIn.items() if k not in arrs]))
        inheritOut.update(
            OrderedDict(
                [(k, v) for k, v in childOut.items() if k not in arrs]))

    # We should not overwrite an explicit dependency with an inherited one:
    # this is most likely a programming mistake
    for key in inheritIn.keys():
        if key in prim.inputs:
            raise ValueError("Inherited dependency from '" + child.name +
                             "' overwrites explicit dependency in '" +
                             prim.name + "' (" + str(prim.inputs[key]) + ")")
    for key in inheritOut.keys():
        if key in prim.outputs:
            raise ValueError("Inherited dependency from '" + child.name +
                             "' overwrites explicit dependency in '" +
                             prim.name + "' (" + str(prim.outputs[key]) + ")")
    prim.inputs.update(inheritIn)
    prim.outputs.update(inheritOut)
    dependIn.update(dcpy(inheritIn))
    dependOut.update(dcpy(inheritOut))

    if isinstance(prim, astnodes._ControlFlowNode):
        # Don't inherit dependencies across control flow boundaries
        return OrderedDict(), OrderedDict()
    else:
        return dependIn, dependOut


def _subset_has_indirection(subset):
    for dim in subset:
        if not isinstance(dim, tuple):
            dim = [dim]
        for r in dim:
            if symbolic.contains_sympy_functions(r):
                return True
    return False


def _add_astmemlet_edge(sdfg,
                        state,
                        src_node,
                        src_conn,
                        dst_node,
                        dst_conn,
                        ast_memlet,
                        data=None,
                        wcr=None,
                        wcr_identity=None):
    try:
        if src_node.data == dst_node.data:
            raise RuntimeError("Added edge connection data nodes "
                               "with same descriptor: {} to {}".format(
                                   src_node, dst_node))
    except AttributeError:
        pass
    if _subset_has_indirection(ast_memlet.subset):
        add_indirection_subgraph(sdfg, state, src_node, dst_node, ast_memlet)
        return

    if data is not None:
        raise NotImplementedError('This should never happen')

    memlet = Memlet(ast_memlet.dataname, ast_memlet.num_accesses,
                    ast_memlet.subset, ast_memlet.veclen, wcr, wcr_identity)
    state.add_edge(src_node, src_conn, dst_node, dst_conn, memlet)


def _get_input_symbols(inputs, freesyms):
    syminputs = set(
        str(i)[9:] for i in inputs.keys() if str(i).startswith('__DACEIN_'))
    return freesyms & syminputs


# TODO: The following two functions can be replaced with better dataflow
# generation procedures


def input_node_for_array(state, data: str):
    # If the node appears as one of the source nodes, return it first
    for n in state.source_nodes():
        if isinstance(n, nd.AccessNode):
            if n.data == data:
                return n
    # Otherwise, if the node is located elsewhere, return it
    for n in state.nodes():
        if isinstance(n, nd.AccessNode):
            if n.data == data:
                return n

    return nd.AccessNode(data)


def output_node_for_array(state, data: str):
    for n in state.sink_nodes():
        if isinstance(n, nd.AccessNode):
            if n.data == data:
                return n

    return nd.AccessNode(data)


def _build_dataflow_graph_recurse(sdfg, state, primitives, modules, superEntry,
                                  super_exit):
    # Array of pairs (exit node, memlet)
    exit_nodes = []

    if len(primitives) == 0:
        # Inject empty tasklets into empty states
        primitives = [astnodes._EmptyTaskletNode("Empty Tasklet", None)]

    for prim in primitives:
        label = prim.name

        # Expand node to get entry and exit points
        if isinstance(prim, astnodes._MapNode):
            if len(prim.children) == 0:
                raise ValueError("Map node expected to have children")
            mapNode = nd.Map(
                label, prim.params, prim.range, is_async=prim.is_async)
            # Add connectors for inputs that exist as array nodes
            entry = nd.MapEntry(
                mapNode,
                _get_input_symbols(prim.inputs, prim.range.free_symbols))
            exit = nd.MapExit(mapNode)
        elif isinstance(prim, astnodes._ConsumeNode):
            if len(prim.children) == 0:
                raise ValueError("Consume node expected to have children")
            consumeNode = nd.Consume(label, (prim.params[1], prim.num_pes),
                                     prim.condition)
            entry = nd.ConsumeEntry(consumeNode)
            exit = nd.ConsumeExit(consumeNode)
        elif isinstance(prim, astnodes._ReduceNode):
            rednode = nd.Reduce(prim.ast, prim.axes, prim.identity)
            state.add_node(rednode)
            entry = rednode
            exit = rednode
        elif isinstance(prim, astnodes._TaskletNode):
            if isinstance(prim, astnodes._EmptyTaskletNode):
                tasklet = nd.EmptyTasklet(prim.name)
            else:
                # Remove memlets from tasklet AST
                if prim.language == types.Language.Python:
                    clean_code = MemletRemover().visit(prim.ast)
                    clean_code = ModuleInliner(modules).visit(clean_code)
                else:  # Use external code from tasklet definition
                    if prim.extcode is None:
                        raise SyntaxError("Cannot define an intrinsic "
                                          "tasklet without an implementation")
                    clean_code = prim.extcode
                tasklet = nd.Tasklet(
                    prim.name,
                    set(prim.inputs.keys()),
                    set(prim.outputs.keys()),
                    code=clean_code,
                    language=prim.language,
                    code_global=prim.gcode)  # TODO: location=prim.location

            # Need to add the tasklet in case we're in an empty state, where no
            # edge will be drawn to it
            state.add_node(tasklet)
            entry = tasklet
            exit = tasklet

        elif isinstance(prim, astnodes._NestedSDFGNode):
            prim.sdfg.parent = state
            prim.sdfg._parent_sdfg = sdfg
            prim.sdfg.update_sdfg_list([])
            nsdfg = nd.NestedSDFG(prim.name, prim.sdfg,
                                  set(prim.inputs.keys()),
                                  set(prim.outputs.keys()))
            state.add_node(nsdfg)
            entry = nsdfg
            exit = nsdfg

        elif isinstance(prim, astnodes._ProgramNode):
            return
        elif isinstance(prim, astnodes._ControlFlowNode):
            continue
        else:
            raise TypeError("Node type not implemented: " +
                            str(prim.__class__))

        # Add incoming edges
        for varname, memlet in prim.inputs.items():
            arr = memlet.dataname
            if (prim.parent is not None
                    and memlet.dataname in prim.parent.transients.keys()):
                node = input_node_for_array(state, memlet.dataname)

                # Add incoming edge into transient as well
                # FIXME: A bit hacked?
                if arr in prim.parent.inputs:
                    astmem = prim.parent.inputs[arr]
                    _add_astmemlet_edge(sdfg, state, superEntry, None, node,
                                        None, astmem)

                    # Remove local name from incoming edge to parent
                    prim.parent.inputs[arr].local_name = None
            elif superEntry:
                node = superEntry
            else:
                node = input_node_for_array(state, memlet.dataname)

            # Destination connector inference
            # Connected to a tasklet or a nested SDFG
            dst_conn = (memlet.local_name
                        if isinstance(entry, nd.CodeNode) else None)
            # Connected to a scope as part of its range
            if str(varname).startswith('__DACEIN_'):
                dst_conn = str(varname)[9:]
            # Handle special case of consume input stream
            if (isinstance(entry, nd.ConsumeEntry)
                    and memlet.data == prim.stream):
                dst_conn = 'IN_stream'

            # If a memlet that covers this input already exists, skip
            # generating this one; otherwise replace memlet with ours
            skip_incoming_edge = False
            remove_edge = None
            for e in state.edges_between(node, entry):
                if e.data.data != memlet.dataname or dst_conn != e.dst_conn:
                    continue
                if e.data.subset.covers(memlet.subset):
                    skip_incoming_edge = True
                    break
                elif memlet.subset.covers(e.data.subset):
                    remove_edge = e
                    break
                else:
                    print('WARNING: Performing bounding-box union on',
                          memlet.subset, 'and', e.data.subset, '(in)')
                    e.data.subset = sbs.bounding_box_union(
                        e.data.subset, memlet.subset)
                    e.data.num_accesses += memlet.num_accesses
                    skip_incoming_edge = True
                    break

            if remove_edge is not None:
                state.remove_edge(remove_edge)

            if skip_incoming_edge == False:
                _add_astmemlet_edge(sdfg, state, node, None, entry, dst_conn,
                                    memlet)

        # If there are no inputs, generate a dummy edge
        if superEntry and len(prim.inputs) == 0:
            state.add_edge(superEntry, None, entry, None, EmptyMemlet())

        if len(prim.children) > 0:
            # Recurse
            inner_outputs = _build_dataflow_graph_recurse(
                sdfg, state, prim.children, modules, entry, exit)
            # Infer output node for each memlet
            for i, (out_src, mem) in enumerate(inner_outputs):
                # If there is no such array in this primitive's outputs,
                # it's an external array (e.g., a map in a map). In this case,
                # connect to the exit node
                if mem.dataname in prim.outputs:
                    inner_outputs[i] = (out_src, prim.outputs[mem.dataname])
                else:
                    inner_outputs[i] = (out_src, mem)
        else:
            inner_outputs = [(exit, mem) for mem in prim.outputs.values()]

        # Add outgoing edges
        for out_src, astmem in inner_outputs:

            data = astmem.data
            dataname = astmem.dataname

            # If WCR is not none, it needs to be handled in the code. Check for
            # this after, as we only expect it for one distinct case
            wcr_was_handled = astmem.wcr is None

            # TODO: This is convoluted. We should find a more readable
            # way of connecting the outgoing edges.

            if super_exit is None:

                # Assert that we're in a top-level node
                if ((not isinstance(prim.parent, astnodes._ProgramNode)) and
                    (not isinstance(prim.parent, astnodes._ControlFlowNode))):
                    raise RuntimeError("Expected to be at the top node")

                # Looks hacky
                src_conn = (astmem.local_name if isinstance(
                    out_src, (nd.Tasklet, nd.NestedSDFG)) else None)

                # Here we just need to connect memlets directly to their
                # respective data nodes
                out_tgt = output_node_for_array(state, astmem.dataname)

                # If a memlet that covers this outuput already exists, skip
                # generating this one; otherwise replace memlet with ours
                skip_outgoing_edge = False
                remove_edge = None
                for e in state.edges_between(out_src, out_tgt):
                    if e.data.data != astmem.dataname or src_conn != e.src_conn:
                        continue
                    if e.data.subset.covers(astmem.subset):
                        skip_outgoing_edge = True
                        break
                    elif astmem.subset.covers(e.data.subset):
                        remove_edge = e
                        break
                    else:
                        print('WARNING: Performing bounding-box union on',
                              astmem.subset, 'and', e.data.subset, '(out)')
                        e.data.subset = sbs.bounding_box_union(
                            e.data.subset, astmem.subset)
                        e.data.num_accesses += astmem.num_accesses
                        skip_outgoing_edge = True
                        break

                if skip_outgoing_edge == True:
                    continue
                if remove_edge is not None:
                    state.remove_edge(remove_edge)

                _add_astmemlet_edge(
                    sdfg,
                    state,
                    out_src,
                    src_conn,
                    out_tgt,
                    None,
                    astmem,
                    wcr=astmem.wcr,
                    wcr_identity=astmem.wcr_identity)
                wcr_was_handled = (True if astmem.wcr is not None else
                                   wcr_was_handled)

                # If the program defines another output, connect it too.
                # This refers to the case where we have streams, which
                # must define an input and output, and sometimes this output
                # is defined in pdp.outputs
                if (isinstance(out_tgt, nd.AccessNode)
                        and isinstance(out_tgt.desc(sdfg), dt.Stream)):
                    try:
                        stream_memlet = next(
                            v for k, v in prim.parent.outputs.items()
                            if k == out_tgt.data)
                        stream_output = output_node_for_array(
                            state, stream_memlet.dataname)
                        _add_astmemlet_edge(sdfg, state, out_tgt, None,
                                            stream_output, None, stream_memlet)
                    except StopIteration:  # Stream output not found, skip
                        pass

            else:  # We're in a nest

                if isinstance(prim, astnodes._ScopeNode):
                    # We're a map or a consume node, that needs to connect our
                    # exit to either an array or to the super_exit
                    if data.transient and dataname in prim.parent.transients:
                        # Connect the exit directly
                        out_tgt = output_node_for_array(state, data.dataname)
                        _add_astmemlet_edge(sdfg, state, out_src, None,
                                            out_tgt, None, astmem)
                    else:
                        # This is either a transient defined in an outer scope,
                        # or an I/O array, so redirect thruogh the exit node
                        _add_astmemlet_edge(sdfg, state, out_src, None,
                                            super_exit, None, astmem)
                        # Instruct outer recursion layer to continue the route
                        exit_nodes.append((super_exit, astmem))
                elif isinstance(
                        prim,
                    (astnodes._TaskletNode, astnodes._NestedSDFGNode)):
                    # We're a tasklet, and need to connect either to the exit
                    # if the array is I/O or is defined in a scope further out,
                    # or directly to the transient if it's defined locally
                    if dataname in prim.parent.transients:
                        # This is a local transient variable, so connect to it
                        # directly
                        out_tgt = output_node_for_array(state, data.dataname)
                        _add_astmemlet_edge(sdfg, state, out_src,
                                            astmem.local_name, out_tgt, None,
                                            astmem)
                    else:
                        # This is an I/O array, or an outer level transient, so
                        # redirect through the exit node
                        _add_astmemlet_edge(
                            sdfg,
                            state,
                            out_src,
                            astmem.local_name,
                            super_exit,
                            None,
                            astmem,
                            wcr=astmem.wcr,
                            wcr_identity=astmem.wcr_identity)
                        exit_nodes.append((super_exit, astmem))
                        if astmem.wcr is not None:
                            wcr_was_handled = True  # Sanity check
                else:
                    raise TypeError("Unexpected node type: {}".format(
                        type(out_src).__name__))

            if not wcr_was_handled and not isinstance(prim,
                                                      astnodes._ScopeNode):
                raise RuntimeError("Detected unhandled WCR for primitive '{}' "
                                   "of type {}. WCR is only expected for "
                                   "tasklets in a map/consume scope.".format(
                                       prim.name,
                                       type(prim).__name__))

    return exit_nodes


def build_dataflow_graph(sdfg, state, primitives, modules):
    _build_dataflow_graph_recurse(sdfg, state, primitives, modules, None, None)


def add_indirection_subgraph(sdfg, graph, src, dst, memlet):
    """ Replaces the specified edge in the specified graph with a subgraph that
        implements indirection without nested AST memlet objects. """
    if not isinstance(memlet, astnodes._Memlet):
        raise TypeError("Expected memlet to be astnodes._Memlet")

    indirect_inputs = set()
    indirect_outputs = set()

    # Scheme for multi-array indirection:
    # 1. look for all arrays and accesses, create set of arrays+indices
    #    from which the index memlets will be constructed from
    # 2. each separate array creates a memlet, of which num_accesses = len(set)
    # 3. one indirection tasklet receives them all + original array and
    #    produces the right output index/range memlet
    #########################
    # Step 1
    accesses = OrderedDict()
    newsubset = dcpy(memlet.subset)
    for dimidx, dim in enumerate(memlet.subset):
        # Range/Index disambiguation
        direct_assignment = False
        if not isinstance(dim, tuple):
            dim = [dim]
            direct_assignment = True

        for i, r in enumerate(dim):
            for expr in sympy.preorder_traversal(r):
                if symbolic.is_sympy_userfunction(expr):
                    fname = expr.func.__name__
                    if fname not in accesses:
                        accesses[fname] = []

                    # Replace function with symbol (memlet local name to-be)
                    if expr.args in accesses[fname]:
                        aindex = accesses[fname].index(expr.args)
                        toreplace = 'index_' + fname + '_' + str(aindex)
                    else:
                        accesses[fname].append(expr.args)
                        toreplace = 'index_' + fname + '_' + str(
                            len(accesses[fname]) - 1)

                    if direct_assignment:
                        newsubset[dimidx] = r.subs(expr, toreplace)
                    else:
                        newsubset[dimidx][i] = r.subs(expr, toreplace)
    #########################
    # Step 2
    ind_inputs = {'__ind_' + memlet.local_name}
    ind_outputs = {'lookup'}
    # Add accesses to inputs
    for arrname, arr_accesses in accesses.items():
        for i in range(len(arr_accesses)):
            ind_inputs.add('index_%s_%d' % (arrname, i))

    tasklet = nd.Tasklet("Indirection", ind_inputs, ind_outputs)

    input_index_memlets = []
    for arrname, arr_accesses in accesses.items():
        arr = memlet.otherdeps[arrname]
        for i, access in enumerate(arr_accesses):
            # Memlet to load the indirection index
            indexMemlet = Memlet(arrname, 1, sbs.Indices(list(access)), 1)
            input_index_memlets.append(indexMemlet)
            graph.add_edge(src, None, tasklet, "index_%s_%d" % (arrname, i),
                           indexMemlet)

    #########################
    # Step 3
    # Create new tasklet that will perform the indirection
    indirection_ast = ast.parse("lookup = {arr}[{index}]".format(
        arr='__ind_' + memlet.local_name,
        index=', '.join([symbolic.symstr(s) for s in newsubset])))
    # Conserve line number of original indirection code
    tasklet.code = ast.copy_location(indirection_ast.body[0], memlet.ast)

    # Create transient variable to trigger the indirected load
    if memlet.num_accesses == 1:
        storage = sdfg.add_scalar(
            '__' + memlet.local_name + '_value',
            memlet.data.dtype,
            transient=True)
    else:
        storage = sdfg.add_array(
            '__' + memlet.local_name + '_value',
            memlet.data.dtype,
            storage=types.StorageType.Default,
            transient=True,
            shape=memlet.bounding_box_size())
    indirectRange = sbs.Range([(0, s - 1, 1) for s in storage.shape])
    dataNode = nd.AccessNode('__' + memlet.local_name + '_value')

    # Create memlet that depends on the full array that we look up in
    fullRange = sbs.Range([(0, s - 1, 1) for s in memlet.data.shape])
    fullMemlet = Memlet(memlet.dataname, memlet.num_accesses, fullRange,
                        memlet.veclen)
    graph.add_edge(src, None, tasklet, '__ind_' + memlet.local_name,
                   fullMemlet)

    # Memlet to store the final value into the transient, and to load it into
    # the tasklet that needs it
    indirectMemlet = Memlet('__' + memlet.local_name + '_value',
                            memlet.num_accesses, indirectRange, memlet.veclen)
    graph.add_edge(tasklet, 'lookup', dataNode, None, indirectMemlet)

    valueMemlet = Memlet('__' + memlet.local_name + '_value',
                         memlet.num_accesses, indirectRange, memlet.veclen)
    graph.add_edge(dataNode, None, dst, memlet.local_name, valueMemlet)
