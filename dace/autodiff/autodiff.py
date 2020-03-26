import dace
from dace import Memlet
from dace.graph import dot, graph
import dace.graph.nodes as nd

import ast
import itertools
from collections import deque
from copy import deepcopy as dc
from typing import Iterator, Tuple, Deque

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from astunparse import unparse


def symbolically_diff_tasklet(tasklet: nd.Tasklet, x: str) -> nd.Tasklet:
    """Symbolically differentiate the value of an assignment tasklet with respect to a given
       in_connector.
    """

    if len(tasklet.code) != 1 or type(tasklet.code[0]) is not ast.Assign:
        raise ValueError(
            "Differentiating tasklets more complex than assign statements is not supported"
        )
    assign: ast.Assign = tasklet.code[0]
    if len(assign.targets) != 1:
        raise ValueError(
            "Expected assignment statement with at most one target")
    target = assign.targets[0].id

    input_vars = set(tasklet.in_connectors)
    if x not in input_vars:
        raise ValueError("Unknown connector {}".format(x))

    sp_expr: sp.Expr = parse_expr(unparse(assign.value))

    def symbs_to_strings(symbs):
        return {str(symb) for symb in symbs}

    free_symbs = symbs_to_strings(sp_expr.free_symbols)

    if target in free_symbs:
        raise ValueError("Inplace operations are currently not supported")

    if input_vars != free_symbs:
        print(input_vars)
        print(sp_expr.free_symbols)
        raise ValueError(
            "Expected all in_connectors to be used in the tasklet code")

    diff_expr = sp_expr.diff(sp.symbols(x))

    return nd.Tasklet(
        tasklet.label + "_backward",
        inputs=symbs_to_strings(diff_expr.free_symbols) | {target + "_grad"},
        outputs={x + "_grad"},
        code=x + "_grad = {}_grad * ({})".format(
            target, str(diff_expr))  # this should be valid python code
    )


def _check_volume_one(memlet: dace.Memlet):
    elems = memlet.subset.num_elements()

    if len(elems.free_symbols) > 0 or int(memlet.subset.num_elements()) != 1:
        raise ValueError(
            "Autodiff only supported for tasklets that operate on memlets with volume 1"
        )


def diff_mapped_tasklet(sdfg: dace.SDFG, forward_state: dace.SDFGState,
                        backward_state: dace.SDFGState, map_entry: nd.MapEntry,
                        map_exit: nd.MapExit, tasklet: nd.Tasklet, target: str,
                        X: str):

    in_arrays = set()
    out_arrays = set()

    # input checking
    ########################################

    X_conn = None
    for _, _, map_tasklet, conn, memlet in forward_state.out_edges(map_entry):
        if map_tasklet is not tasklet:
            raise ValueError(
                "Expected tasklet to be connected directly to the map exit and entry"
            )
        _check_volume_one(memlet)

        if memlet.data in in_arrays:
            raise ValueError(
                "Using the same array for multiple memlets is not supported")

        in_arrays.add(memlet.data)

        if memlet.data == X:
            X_conn = conn

    if len(forward_state.out_edges(tasklet)) != 1:
        raise ValueError(
            "Tasklets with more than one output are not supported")

    target_conn = None
    target_memlet: Memlet = None
    for map_tasklet, conn, _, _, memlet in forward_state.in_edges(map_exit):
        if map_tasklet is not tasklet:
            raise ValueError(
                "Expected tasklet to be connected directly to the map exit and entry"
            )
        _check_volume_one(memlet)

        if memlet.data in in_arrays:
            raise ValueError("Inplace operations are not supported")

        out_arrays.add(memlet.data)

        if memlet.data == target:
            target_conn = conn
            target_memlet = dc(memlet)

    if target_conn is None:
        raise ValueError(
            "target ({}) not found in tasklet outputs".format(target))

    if X_conn is None:
        raise ValueError("X_array ({}) not found in tasklet inputs".format(X))

    target_grad = sdfg.arrays[target + "_grad"]

    # allocate an array for the gradient of x
    X_arr = sdfg.arrays[X]
    if X_arr.dtype not in [dace.float16, dace.float32, dace.float64]:
        raise ValueError("Expected dtype of x to be float, got {}".format(
            X_arr.dtype.to_string()))

    # TODO @orausch grad arrays should be zero at the start of execution; how should that be handled?
    _, X_grad_arr = sdfg.add_array(X + "_grad",
                                   shape=X_arr.shape,
                                   dtype=X_arr.dtype)

    # HACK @orausch ask how to do this properly
    ndrange = dict(
        zip(map_entry.map.params,
            str(map_entry.map.range).split(", ")))

    diff_map_entry, diff_map_exit = backward_state.add_map(
        X + "_diff", ndrange)

    diff_tasklet = symbolically_diff_tasklet(tasklet, X_conn)
    backward_state.add_node(diff_tasklet)

    diff_map_entry.add_in_connector("IN_1")
    diff_map_entry.add_out_connector("OUT_1")

    target_memlet.data = target + "_grad"
    target_memlet.wcr = None
    # connect the target's gradient
    backward_state.add_edge(
        backward_state.add_read(target + "_grad"), None, diff_map_entry,
        "IN_1",
        Memlet.from_array(target + "_grad", sdfg.arrays[target + "_grad"]))
    backward_state.add_edge(diff_map_entry, "OUT_1", diff_tasklet,
                            target_conn + "_grad", target_memlet)

    i = 2
    X_memlet: Memlet = None

    # connect the inputs from the forward pass
    for _, _, _, conn, memlet in forward_state.in_edges(tasklet):

        if conn in diff_tasklet.in_connectors:
            access = backward_state.add_read(memlet.data)
            arr = sdfg.arrays[memlet.data]
            diff_map_entry.add_in_connector("IN_" + str(i))
            diff_map_entry.add_out_connector("OUT_" + str(i))
            backward_state.add_edge(access, None, diff_map_entry,
                                    "IN_" + str(i),
                                    Memlet.from_array(memlet.data, arr))
            backward_state.add_edge(diff_map_entry, "OUT_" + str(i),
                                    diff_tasklet, conn, dc(memlet))
            i += 1

        if conn == X_conn:
            X_memlet = dc(memlet)

    if X_memlet is None:
        raise ValueError("X_array ({}) not found in tasklet inputs".format(X))

    # TODO @orausch this isn't always necessary; maybe we can add a transform to remove this if possible?
    X_memlet.wcr = "lambda x, y: x + y"
    X_memlet.data = X + "_grad"

    diff_map_exit.add_in_connector("IN_1")
    diff_map_exit.add_out_connector("OUT_1")
    backward_state.add_edge(diff_tasklet, X_conn + "_grad", diff_map_exit,
                            "IN_1", X_memlet)
    backward_state.add_edge(
        diff_map_exit, "OUT_1", backward_state.add_write(X + "_grad"), None,
        Memlet.from_array(X + "_grad", X_grad_arr, wcr="lambda x, y: x + y"))


def add_backward_pass(sdfg: dace.SDFG, forward_state: dace.SDFGState,
                      target: dace.nodes.AccessNode):
    """Generate the backward pass wrt. target.
    """

    # This is currently a bit ugly due to the constrains on graph structure.

    target_arr = sdfg.arrays[target.data]
    if target_arr.dtype not in [dace.float16, dace.float32, dace.float64]:
        raise ValueError("Expected dtype of x to be float, got {}".format(
            target_arr.dtype.to_string()))

    if len(target_arr.total_size.free_symbols) > 0 or int(
            target_arr.total_size) != 1:
        raise ValueError("Targets with more than one element are unsupported")

    target_grad, target_grad_arr = sdfg.add_array(target.data + "_grad",
                                                  target_arr.shape,
                                                  target_arr.dtype)

    def getpred(state: dace.SDFGState, node: nd.Node):
        in_edges = state.in_edges(node)
        preds = {edge.src for edge in in_edges}
        if len(preds) > 1:
            raise ValueError("Unsupported graph structure")
        return next(iter(preds))

    # this is a breath first search along collapsed maps
    queue: Deque[Tuple[nd.AccessNode, dace.SDFGState]] = deque([target])
    prev_state = forward_state
    while queue:
        current = queue.popleft()

        if type(current) is not nd.AccessNode:
            raise ValueError("Unsupported graph structure")

        try:
            map_x = getpred(forward_state, current)
            task = getpred(forward_state, map_x)
            map_e = getpred(forward_state, task)

            if (type(map_e) is not nd.MapEntry or  #
                    type(task) is not nd.Tasklet or  #
                    type(map_x) is not nd.MapExit):
                raise ValueError("Unsupported graph structure")

            input_edges = forward_state.in_edges(map_e)
            inputs = list({edge.src for edge in input_edges})
            for inp in inputs:
                if type(inp) is not nd.AccessNode:
                    raise ValueError("Unsupported graph structure")
                # generate the backward pass for this map
                backward_state = sdfg.add_state()
                diff_mapped_tasklet(sdfg, forward_state, backward_state, map_e,
                                    map_x, task, current.data, inp.data)

                # add interstate edges
                sdfg.add_edge(prev_state, backward_state,
                              dace.InterstateEdge())
                prev_state = backward_state

                # recursively generate backward pass for inp
                if len(forward_state.in_edges(inp)) > 0:
                    queue.append(inp)

        except StopIteration:
            raise ValueError("Unsupported graph structure")
