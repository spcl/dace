# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Boundary-node classification and reduction-lift helpers.

The ``get_{scalar,array}_{source,sink}_nodes`` functions classify access
nodes by transientness and shape. ``move_out_reduction`` and its detection
helpers lift a simple ``acc = 0; for: acc += x; use(acc)`` pattern out of
a NestedSDFG so the accumulator can be vectorized.
"""
import copy
from typing import List, Set, Tuple

import dace
import dace.sdfg.tasklet_utils as tutil


def _is_scalar_or_shape_one(arr: dace.data.Data) -> bool:
    return isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array) and arr.shape == (1, ))


def _get_boundary_nodes(sdfg: dace.SDFG, *, side: str, kind: str, non_transient_only: bool,
                        skip: Set[str]) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """Single classifier backing the four ``get_*_{source,sink}_nodes`` helpers.

    :param sdfg: SDFG to inspect.
    :param side: ``"source"`` (in_degree == 0) or ``"sink"`` (out_degree == 0).
    :param kind: ``"scalar"`` (Scalar or shape-(1,) Array) or ``"array"``
        (Array with shape != (1,)).
    :param non_transient_only: If True, drop transient access nodes.
    :param skip: Data names to exclude.
    :returns: List of ``(state, access_node)`` pairs matching the criteria.
    """
    assert side in ("source", "sink"), side
    assert kind in ("scalar", "array"), kind
    result: List[Tuple[dace.SDFGState, dace.nodes.AccessNode]] = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, dace.nodes.AccessNode):
                continue
            degree = state.in_degree(node) if side == "source" else state.out_degree(node)
            if degree != 0:
                continue
            arr = state.sdfg.arrays[node.data]
            if kind == "scalar":
                if not _is_scalar_or_shape_one(arr):
                    continue
            else:
                if not (isinstance(arr, dace.data.Array) and arr.shape != (1, )):
                    continue
            if non_transient_only and arr.transient:
                continue
            if node.data in skip:
                continue
            result.append((state, node))
    return result


def get_scalar_source_nodes(
    sdfg: dace.SDFG, non_transient_only: bool,
    skip: Set[str] = set()) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """Return source nodes (in-degree 0) for scalars or shape-(1,) arrays.

    :param sdfg: SDFG to inspect.
    :param non_transient_only: If True, drop transient access nodes.
    :param skip: Data names to exclude.
    :returns: List of ``(state, access_node)`` pairs.
    """
    return _get_boundary_nodes(sdfg, side="source", kind="scalar", non_transient_only=non_transient_only, skip=skip)


def get_array_source_nodes(sdfg: dace.SDFG,
                           non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """Return source nodes (in-degree 0) for arrays with shape != (1,).

    :param sdfg: SDFG to inspect.
    :param non_transient_only: If True, drop transient access nodes.
    :returns: List of ``(state, access_node)`` pairs.
    """
    return _get_boundary_nodes(sdfg, side="source", kind="array", non_transient_only=non_transient_only, skip=set())


def get_scalar_sink_nodes(sdfg: dace.SDFG, non_transient_only: bool,
                          skip: Set[str]) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """Return sink nodes (out-degree 0) for scalars or shape-(1,) arrays.

    :param sdfg: SDFG to inspect.
    :param non_transient_only: If True, drop transient access nodes.
    :param skip: Data names to exclude.
    :returns: List of ``(state, access_node)`` pairs.
    """
    return _get_boundary_nodes(sdfg, side="sink", kind="scalar", non_transient_only=non_transient_only, skip=skip)


def get_array_sink_nodes(sdfg: dace.SDFG,
                         non_transient_only: bool) -> List[Tuple[dace.SDFGState, dace.nodes.AccessNode]]:
    """Return sink nodes (out-degree 0) for arrays with shape != (1,).

    :param sdfg: SDFG to inspect.
    :param non_transient_only: If True, drop transient access nodes.
    :returns: List of ``(state, access_node)`` pairs.
    """
    return _get_boundary_nodes(sdfg, side="sink", kind="array", non_transient_only=non_transient_only, skip=set())


def check_writes_to_scalar_sinks_happen_through_assign_tasklets(sdfg: dace.SDFG,
                                                                scalar_sink_nodes: List[Tuple[dace.SDFGState,
                                                                                              dace.nodes.AccessNode]]):
    """Ensure all writes to scalar sink nodes go through assignment tasklets.

    Auto-vectorization does not support a write via an AccessNode-to-AccessNode
    edge with a non-None ``other_subset``.

    :param sdfg: The SDFG to check.
    :param scalar_sink_nodes: List of ``(state, sink_node)`` pairs to validate.
    :raises Exception: if a scalar sink does not have exactly one in-edge, or
        if its write is not via an assignment tasklet.
    """
    # ``is_assignment_tasklet`` lives in ``utils.tasklets`` (S6b);
    # imported lazily here to avoid a circular import at module load
    # time (tasklets imports re from this module's siblings).
    from dace.transformation.passes.vectorization.utils.tasklets import is_assignment_tasklet
    for state, sink_node in scalar_sink_nodes:
        in_edges = state.in_edges(sink_node)
        if len(in_edges) != 1:
            raise Exception(f"All scalar sink nodes should have exactly 1 incoming edge, got {len(in_edges)} on "
                            f"{sink_node.data} in {state.label}")
        in_edge = in_edges[0]
        src = in_edge.src
        if not (isinstance(src, dace.nodes.Tasklet) and is_assignment_tasklet(src)):
            raise Exception("All write to scalar should happen through an assignment tasklet")


def only_one_flop_after_source(state: dace.SDFGState, node: dace.nodes.AccessNode):
    """Check whether at most one computational (non-assignment) tasklet follows a source.

    BFS forward from ``node``.

    :param state: The SDFG state containing the node.
    :param node: The source AccessNode.
    :returns: ``(holds, checked_nodes)`` — whether the condition holds and the
        nodes visited (empty list when more than one flop is found).
    """
    from dace.transformation.passes.vectorization.utils.tasklets import is_assignment_tasklet

    nodes_to_check = [node]
    visited = set()
    tasklets_with_flops = 0
    checked_nodes = []

    while nodes_to_check:
        cur_node = nodes_to_check.pop(0)
        if cur_node in visited:
            continue
        visited.add(cur_node)
        checked_nodes.append(cur_node)
        if isinstance(cur_node, dace.nodes.Tasklet) and not is_assignment_tasklet(cur_node):
            tasklets_with_flops += 1
        nodes_to_check += [e.dst for e in state.out_edges(cur_node)]
        if tasklets_with_flops > 1:
            return False, []

    return tasklets_with_flops <= 1, checked_nodes


def input_is_zero_and_transient_accumulator(state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG,
                                            source_node: dace.nodes.AccessNode, sink_node: dace.nodes.AccessNode):
    """Check for a transient accumulator zero-initialised before an in-place reduction.

    Traverses backward from the source access node in the parent state to find
    a zero-assignment to the accumulator. The source and sink data must be the
    same name for it to be an accumulator.

    :param state: The parent SDFG state.
    :param nsdfg: The NestedSDFG node.
    :param source_node: Source access node feeding the accumulator.
    :param sink_node: Sink access node consuming the accumulator.
    :returns: ``(is_valid, accumulator_name)``; ``accumulator_name`` is ``""``
        when the pattern does not match.
    """

    # Make sure the data of in and out edges refer to the same name
    sink_data = sink_node.data
    source_data = source_node.data
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
    body = code_str.strip().rstrip(";").rstrip()
    prefix = f"{out_conn} ="
    if not body.startswith(prefix):
        return False, ""
    rhs = body[len(prefix):].strip().rstrip("fFdDlL")
    try:
        if float(rhs) != 0.0:
            return False, ""
    except ValueError:
        return False, ""

    # If all true return true and accumulator name
    return True, src_acc_node.data


def expand_assignment_tasklets(state: dace.SDFGState, name: str, vector_width: int):
    """Expand literal-init assignment tasklets writing ``name`` to all vector lanes.

    Rewrites ``a = v`` into ``a[0] = v; ...; a[W-1] = v``.

    :param state: The SDFG state to modify.
    :param name: The array being written.
    :param vector_width: Length of the vector to expand to.
    :raises NotImplementedError: if a non-assignment tasklet (with input
        connectors) feeds the accumulator.
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


#: Reduction op token (from ``recognize_reduction`` / the legacy
#: ``_extract_single_op``) -> the ``Reduce`` node ``wcr`` lambda. Only
#: associative ops with a horizontal-reduce primitive; ``-`` / ``/`` and
#: anything else raise (a hand-rolled ``_in[0] - _in[1] - …`` chain was
#: numerically wrong for those anyway).
_OP_TO_WCR = {
    "+": "lambda a, b: a + b",
    "*": "lambda a, b: a * b",
    "max": "lambda a, b: max(a, b)",
    "min": "lambda a, b: min(a, b)",
    "&": "lambda a, b: a & b",
    "|": "lambda a, b: a | b",
    "^": "lambda a, b: a ^ b",
}


def reduce_before_use(state: dace.SDFGState, name: str, vector_width: int, op: str, tree: bool = False):
    """Scalarize a vectorized ``(W,)`` accumulator via a ``Reduce`` libnode.

    Inserts a ``dace.libraries.standard.Reduce`` node
    (``implementation='vectorized'``) over ``name[0:W]`` -> ``name_scl``
    before each reader and expands it inline, so the final W-fold is the
    registered ``horizontal_reduce_<op>`` kernel (RV-2). This replaces
    the former hand-rolled chain/tree scalarize tasklet.

    :param state: The SDFG state.
    :param name: The ``(vector_width,)`` accumulator array to scalarize.
    :param vector_width: Number of vector lanes.
    :param op: Reduction op token (``"+"``, ``"*"``, ``"max"``,
        ``"min"``, ``"&"``, ``"|"``, ``"^"``).
    :param tree: Unused (kept for call-site compatibility); the
        intrinsic vs. log-depth-tree choice is made inside the runtime
        ``horizontal_reduce_<op>`` primitive.
    :raises NotImplementedError: if ``op`` is not an associative
        reduction with a horizontal-reduce primitive.
    """
    del tree  # decision now lives in the runtime horizontal_reduce_<op>
    from dace.libraries.standard.nodes.reduce import Reduce
    # Ensure the 'vectorized' Reduce implementation is registered.
    from dace.transformation.passes.vectorization import reduce_expansion  # noqa: F401

    if op not in _OP_TO_WCR:
        raise NotImplementedError(
            f"reduce_before_use: unsupported reduction op {op!r}; supported "
            f"{sorted(_OP_TO_WCR)} (- / / and custom ops have no associative horizontal reduce)")

    for edge in list(state.edges()):
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
            red = Reduce(f"scalarize_{name}", wcr=_OP_TO_WCR[op], axes=[0])
            red.implementation = "vectorized"
            red.schedule = dace.dtypes.ScheduleType.Sequential
            state.add_node(red)
            state.add_edge(src, None, red, None, copy.deepcopy(edge.data))
            state.add_edge(red, None, an, None, dace.memlet.Memlet(f"{name}_scl[0]"))
            state.add_edge(an, None, edge.dst, edge.dst_conn, dace.memlet.Memlet(f"{name}_scl[0]"))
            state.remove_edge(edge)
            # Expand inline via the registered schedule-aware
            # ``ExpandReduceVectorized`` (Sequential -> the vectorized
            # horizontal_reduce_<op> kernel). New ``expand(state)``
            # interface uses ``node.implementation`` ('vectorized').
            red.expand(state)


def move_out_reduction(scalar_source_nodes, state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, inner_sdfg: dace.SDFG,
                       vector_width) -> Tuple[bool, str, str]:
    """Lift a simple zero-init reduction out of a NestedSDFG and vectorize it.

    Handles the pattern: a scalar source feeds the NSDFG, a transient
    accumulator is zero-initialised outside it, a single computational
    tasklet updates the accumulator, and a scalar sink consumes it. The
    accumulator is widened to ``vector_width``, its memlets and assignment
    tasklets are expanded, and a reduction tasklet scalarizes it before use.

    :param scalar_source_nodes: List of ``(state, node)`` source scalar nodes
        feeding the NestedSDFG.
    :param state: Parent SDFGState containing the NestedSDFG node.
    :param nsdfg: NestedSDFG node where the reduction occurs.
    :param inner_sdfg: Inner SDFG of the NestedSDFG.
    :param vector_width: Vectorization width for the accumulator.
    :returns: ``(lifted, source_data, sink_data)`` — whether the reduction
        was lifted, and the source / sink data names.
    """
    # ``replace_arrays_with_new_shape`` lives in ``utils.arrays`` (S6a)
    # and ``replace_all_access_subsets`` in ``utils.subsets`` (S6d-b);
    # imported lazily to avoid module-load-time cycles.
    from dace.transformation.passes.vectorization.utils.arrays import replace_arrays_with_new_shape
    from dace.transformation.passes.vectorization.utils.subsets import replace_all_access_subsets
    from dace.transformation.passes.vectorization.utils.reductions import recognize_reduction

    num_flops, node_path = only_one_flop_after_source(scalar_source_nodes[0][0], scalar_source_nodes[0][1])
    source_data = scalar_source_nodes[0][1].data
    # ``node_path`` is populated by ``only_one_flop_after_source``'s BFS;
    # if the source has no outgoing dataflow we'd have nothing to lift,
    # so bail out rather than IndexError on ``node_path[1]`` / [-1].
    if len(node_path) < 2:
        return False, source_data, ""
    is_inout_accumulator, accumulator_name = input_is_zero_and_transient_accumulator(
        state, nsdfg, scalar_source_nodes[0][1], node_path[-1])
    sink_data = node_path[-1].data
    # Reduction operator: prefer the robust RMW-shape recogniser
    # (handles a compound right-hand side — ``acc = acc + a*b`` — that
    # the single-op ``_extract_single_op`` mis-parses). Scan the path
    # for the first tasklet recognised as a read-modify-write reduction
    # and take its op; fall back to the legacy single-op extraction on
    # ``node_path[1]`` for shapes the recogniser does not cover, so the
    # behaviour is unchanged wherever it already worked.
    src_state = scalar_source_nodes[0][0]
    op = None
    for _n in node_path:
        if isinstance(_n, dace.nodes.Tasklet):
            _red = recognize_reduction(src_state, _n)
            if _red is not None:
                op = _red.op
                break
    if op is None:
        op = tutil._extract_single_op(node_path[1].code.as_string)
    if num_flops <= 1 and is_inout_accumulator:
        replace_arrays_with_new_shape(inner_sdfg, {source_data, sink_data}, (vector_width, ), None)
        replace_arrays_with_new_shape(state.sdfg, {accumulator_name}, (vector_width, ), None)
        replace_all_access_subsets(state, accumulator_name, f"0:{vector_width}")
        expand_assignment_tasklets(state, accumulator_name, vector_width)
        reduce_before_use(state, accumulator_name, vector_width, op)

        return True, source_data, sink_data
    return False, source_data, sink_data
