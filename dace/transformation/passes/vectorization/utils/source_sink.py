# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Boundary-node classification and reduction-lift helpers.

The four ``get_{scalar,array}_{source,sink}_nodes`` functions classify
SDFGState access nodes by transientness and shape. ``move_out_reduction``
and its pattern-detection helpers (``input_is_zero_and_transient_accumulator``,
``only_one_flop_after_source``) lift simple ``acc = 0; for: acc += x; use(acc)``
patterns out of a NestedSDFG so the accumulator can be vectorized.

Per the locked policy (mechanical-only + defensive checks stay), every
helper is moved verbatim. The post-S7 reductions redesign (R-1..R-5)
will replace these with a single ``recognize_reduction()`` + delete the
quad of source/sink classifiers entirely.
"""
import copy
from typing import List, Set, Tuple

import dace
import dace.sdfg.tasklet_utils as tutil


def get_scalar_source_nodes(
    sdfg: dace.SDFG, non_transient_only: bool,
    skip: Set[str] = set()) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns source nodes (in-degree 0 access nodes) for scalars (or shape-1 arrays) with no incoming edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient scalars.

    Returns:
        List of tuples (state, AccessNode).
    """

    source_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.in_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1, )):
                    if non_transient_only is False or arr.transient is False:
                        if node.data not in skip:
                            source_nodes.append((state, node))
    return source_nodes


def get_array_source_nodes(sdfg: dace.SDFG,
                           non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns source nodes for arrays with more than one element (shape != (1,)) and no incoming edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient arrays.

    Returns:
        List of tuples (state, AccessNode).
    """

    source_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.in_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if (isinstance(arr, dace.data.Array) and (arr.shape != (1, ) and arr.shape != [
                        1,
                ])):
                    if non_transient_only is False or arr.transient is False:
                        source_nodes.append((state, node))
    return source_nodes


def get_scalar_sink_nodes(sdfg: dace.SDFG, non_transient_only: bool,
                          skip: Set[str]) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns sink nodes for scalars (or shape-1 arrays) with no outgoing edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient scalars.

    Returns:
        List of tuples (state, AccessNode).
    """

    sink_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.out_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Scalar) or isinstance(arr, dace.data.Array) and arr.shape == (1, ):
                    if non_transient_only is False or arr.transient is False:
                        if node.data not in skip:
                            sink_nodes.append((state, node))
    return sink_nodes


def get_array_sink_nodes(sdfg: dace.SDFG,
                         non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """
    Returns sink nodes for arrays with shape > 1 and no outgoing edges.

    Args:
        sdfg: The SDFG to inspect.
        non_transient_only: If True, include only non-transient arrays.

    Returns:
        List of tuples (state, AccessNode).
    """
    sink_nodes = list()
    for state in sdfg.all_states():
        for node in state.nodes():
            if (isinstance(node, dace.nodes.AccessNode) and state.out_degree(node) == 0):
                arr = state.sdfg.arrays[node.data]
                if isinstance(arr, dace.data.Array) and arr.shape != (1, ):
                    if non_transient_only is False or arr.transient is False:
                        sink_nodes.append((state, node))
    return sink_nodes


def check_writes_to_scalar_sinks_happen_through_assign_tasklets(
        sdfg: dace.SDFG, scalar_sink_nodes: List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]):
    """
    Ensures all writes to scalar sink nodes occur through simple assignment tasklets.
    Assignments can also occur through AccessNode -Edge-> AccessNode where `other_subset` is not none.
    Auto-vectorization does not support that.

    Args:
        sdfg: The SDFG to check.
        scalar_sink_nodes: List of scalar sink nodes to validate.

    Raises:
        Exception if a scalar sink write is not via an assignment tasklet.
    """
    # ``is_assignment_tasklet`` migrates to ``utils.tasklets`` in S6;
    # imported lazily here to keep the call chain working through the
    # re-export shim until then.
    from dace.transformation.passes.vectorization.vectorization_utils import is_assignment_tasklet
    for state, sink_node in scalar_sink_nodes:
        in_edges = state.in_edges(sink_node)
        if len(in_edges) != "1":
            raise Exception("All scalar sink nodes should have at max 1 incoming edge")
        in_edge = in_edges[0]
        src = in_edge.src
        if not (isinstance(src, dace.nodes.Tasklet) and is_assignment_tasklet(src)):
            raise Exception("All write to scalar should happen through an assignment tasklet")


def only_one_flop_after_source(state: dace.SDFGState, node: dace.nodes.AccessNode):
    """
    Checks whether only one computational tasklet (non-assignment) occurs after a given source node.
    Does BFS starting from the access node.

    Args:
        state: The SDFG state containing the node.
        node: The source AccessNode.

    Returns:
        Tuple (bool, List of nodes) indicating if the condition holds and the nodes checked.
    """
    from dace.transformation.passes.vectorization.vectorization_utils import is_assignment_tasklet

    nodes_to_check = [node]
    tasklets_with_flops = 0
    checked_nodes = []

    while nodes_to_check:
        cur_node = nodes_to_check.pop(0)
        checked_nodes.append(cur_node)
        if isinstance(cur_node, dace.nodes.Tasklet) and not is_assignment_tasklet(cur_node):
            tasklets_with_flops += 1
        nodes_to_check += [e.dst for e in state.out_edges(cur_node)]
        if tasklets_with_flops > 1:
            return False, []

    return tasklets_with_flops <= 1, checked_nodes


def input_is_zero_and_transient_accumulator(state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG,
                                            inner_state: dace.SDFGState, source_node: dace.nodes.AccessNode,
                                            sink_node: dace.nodes.AccessNode):
    """
    Checks if a transient accumulator is initialized to zero and used in an in-place reduction pattern.
    `nsdfg` is the parent nsdfg node and the state is where the nsdfg resides in.

    It traverses the nsdfg node backwards using the find a zero-assignment to the accumulator.
    The accumulator is the `source_node.data`. For it to be an accumulator source and sink needs to be the
    same too.

    Args:
        state: The parent SDFG state.
        nsdfg: The NestedSDFG node.
        inner_state: Inner state of the NestedSDFG.
        source_node: Source access node feeding the accumulator.
        sink_node: Sink access node consuming the accumulator.

    Returns:
        Tuple (bool, accumulator_name) indicating if the pattern is valid and the accumulator's name.
    """

    # Make sure the data of in and out edges refer to the same name
    sink_data = sink_node.data
    source_data = source_node.data
    sink_connector = nsdfg.out_connectors[sink_data]
    source_connector = nsdfg.in_connectors[source_data]
    sink_edges = state.out_edges_by_connector(nsdfg, sink_data)
    source_edges = state.in_edges_by_connector(nsdfg, source_data)

    out_source_datas = {ie.data.data for ie in source_edges if ie.data is not None}
    out_sink_datas = {oe.data.data for oe in sink_edges if oe.data is not None}
    if len(out_sink_datas) != 1:
        return False, ""
    if len(out_source_datas) != 1:
        return False, ""
    out_sink_data = out_sink_datas.pop()
    out_source_data = out_source_datas.pop()

    if out_source_data != out_sink_data:
        return False, ""

    # Find the first access node of the source node outside
    source_edges = list(state.in_edges_by_connector(nsdfg, source_data))
    assert len(source_edges) == 1, f"{source_edges} for in connector {source_data} of {nsdfg}"
    source_edge = source_edges[0]
    mpath = state.memlet_path(source_edge)
    src_acc_node = mpath[0].src
    if not isinstance(src_acc_node, dace.nodes.AccessNode):
        return False, ""

    # Ensure the access node directly connects to a memset-0 tasklet
    if state.in_degree(src_acc_node) != 1:
        return False, ""

    in_tasklet = state.in_edges(src_acc_node)[0].src
    if not isinstance(in_tasklet, dace.nodes.Tasklet):
        return False, ""

    code_str = in_tasklet.code.as_string
    if len(in_tasklet.out_connectors) != 1:
        return False, ""
    out_conn = next(iter(in_tasklet.out_connectors))
    if not (code_str.strip() != f"{out_conn} = 0" or code_str.strip() != f"{out_conn} = 0;"):
        return False, ""

    # If all true return true and accumulator name
    return True, src_acc_node.data


def expand_assignment_tasklets(state: dace.SDFGState, name: str, vector_width: int):
    """
    Expands assignment tasklets writing to an array at a to be over the vector length
    over the unit stride dimension a[0] = ..., a[1] = ..., ...
    For assignment tasklets the dataname given as name.

    Args:
        state: The SDFG state to modify.
        name: The array being written.
        vector_width: Length of the vector to expand to.
    """
    for e in state.edges():
        if (isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == name and isinstance(e.src, dace.nodes.Tasklet)):
            code = e.src.code
            in_conns = e.src.in_connectors
            out_conns = e.src.out_connectors
            if len(in_conns) != 0:
                raise NotImplementedError(
                    f"expand_assignment_tasklets: non-assignment tasklet {e.src.label} feeds accumulator "
                    f"{name} with input connectors {in_conns}; only literal-init tasklets (no in-conns) "
                    f"are supported by the reduction-lift contract")
            assert len(out_conns) == 1, f"{out_conns}"
            out_conn = next(iter(out_conns))
            assert code.language == dace.dtypes.Language.Python
            assert code.as_string.startswith(f"{out_conn} =")
            rhs = code.as_string.split("=")[-1].strip()
            ncode_str = "\n".join([f"{out_conn}[{i}] = {rhs}" for i in range(vector_width)])
            e.src.code = dace.properties.CodeBlock(ncode_str)


def reduce_before_use(state: dace.SDFGState, name: str, vector_width: int, op: str):
    """
    Adds a reduction tasklet to reduce a vectorized array into a scalar before its use.

    Args:
        state: The SDFG state.
        name: Array to reduce.
        vector_width: Number of vector elements.
        op: Reduction operation (e.g., "+", "*").
    """
    # TODO: Reduction can be optimized (e.g. logarithmic depth or checking of vector templates have a reduction op)

    # Any time a tasklet reads name[0:vector_width] then we need to reduce it before
    # In a reduction tasklet
    for edge in state.edges():
        dst = edge.dst
        src = edge.src
        if isinstance(dst, dace.nodes.Tasklet) and edge.data is not None and edge.data.data == name:
            arr = state.sdfg.arrays[name]
            state.sdfg.add_scalar(name=name + "_scl",
                                  dtype=arr.dtype,
                                  storage=arr.storage,
                                  transient=True,
                                  lifetime=arr.lifetime)
            an = state.add_access(name + "_scl")
            an.setzero = True
            t = state.add_tasklet(name=f"scalarize_{name}",
                                  inputs={"_in"},
                                  outputs={"_out"},
                                  code="_out =" + f" {op} ".join([f"_in[{i}]" for i in range(vector_width)]))
            t.add_in_connector("_in")
            t.add_out_connector("_out")
            state.add_edge(src, None, t, "_in", copy.deepcopy(edge.data))
            state.add_edge(t, "_out", an, None, dace.memlet.Memlet(f"{name}_scl[0]"))
            state.add_edge(an, None, edge.dst, edge.dst_conn, dace.memlet.Memlet(f"{name}_scl[0]"))

            state.remove_edge(edge)


def move_out_reduction(scalar_source_nodes, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, inner_sdfg: dace.SDFG,
                       vector_width) -> Tuple[bool, str, str]:
    """
    Moves a reduction out of a NestedSDFG, vectorizing transient accumulators and adjusting tasklets.

    This function is typically used when a computation pattern consists of:
      1. A scalar source feeding a NestedSDFG,
      2. A transient accumulator initialized to zero (outside nested SDFG)
      3. A single computational tasklet updating the accumulator (e.g. acc = acc + some_var)
      4. A scalar sink at the end of the nsdfg for the accumulator.

    The transformation performs the following steps:
        1. Checks that there is at most one floating-point operation after the source. (For condition 3)
        2. Validates that the accumulator is a transient scalar initialized to zero. (For condition 1 and 2)
        3. Extracts the operation performed on the accumulator (e.g., addition, multiplication).
        4. Reshapes the source, sink, and accumulator arrays to a vectorized form of size `vector_width`.
        5. Updates all memlets accessing the accumulator to cover the full vector range.
        6. Expands assignment tasklets to operate on all vector elements.
        7. Inserts a reduction tasklet that combines the vector elements back to a scalar before use.

    Args:
        scalar_source_nodes: List of tuples `(state, node)` representing source scalar nodes feeding the NestedSDFG.
        state: Parent SDFGState containing the NestedSDFG node.
        nsdfg: NestedSDFG node where the reduction occurs.
        inner_sdfg: Inner SDFG of the NestedSDFG.
        vector_width: The width of vectorization for the accumulator.

    Notes:
        - Only supports simple reduction patterns with one operation and transient accumulators.
        - The function assumes that the scalar source and sink nodes are properly connected through the NestedSDFG.
        - The reduction operation is extracted automatically from the first tasklet after the source.

    """
    # ``replace_arrays_with_new_shape`` and ``replace_all_access_subsets``
    # still live in ``vectorization_utils.py`` (migrate in S6); imported
    # lazily to keep the call chain working through the re-export shim.
    from dace.transformation.passes.vectorization.vectorization_utils import (
        replace_all_access_subsets,
        replace_arrays_with_new_shape,
    )

    num_flops, node_path = only_one_flop_after_source(scalar_source_nodes[0][0], scalar_source_nodes[0][1])
    is_inout_accumulator, accumulator_name = input_is_zero_and_transient_accumulator(
        state, nsdfg, scalar_source_nodes[0][0], scalar_source_nodes[0][1], node_path[-1])
    op = tutil._extract_single_op(node_path[1].code.as_string)
    if num_flops <= 1 and is_inout_accumulator:
        source_data = scalar_source_nodes[0][1].data
        sink_data = node_path[-1].data
        replace_arrays_with_new_shape(inner_sdfg, {source_data, sink_data}, (vector_width, ), None)
        replace_arrays_with_new_shape(state.sdfg, {accumulator_name}, (vector_width, ), None)
        replace_all_access_subsets(state, accumulator_name, f"0:{vector_width}")
        expand_assignment_tasklets(state, accumulator_name, vector_width)
        reduce_before_use(state, accumulator_name, vector_width, op)

        return True, source_data, sink_data
    return False, source_data, sink_data
