import dace
from dace import Memlet
from dace.graph import dot, graph
import dace.graph.nodes as nd

import ast
import itertools
from collections import deque
from copy import deepcopy as dc
from typing import Iterator, Tuple, Deque, Dict, Set

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from astunparse import unparse


class AutoDiffException(Exception):
    pass


def _strings_to_symbols(strings: Set[str]) -> Set[sp.Symbol]:
    return {sp.symbols(string) for string in strings}


def _symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}


def code_to_exprs(code: str, inputs: Set[str],
                  outputs: Set[str]) -> Dict[str, sp.Expr]:
    """Convert a python string to a set of (simplified) symbolic sympy expressions. Currently, this
       supports only code consisting of assignment statements.
       :param code: the code to convert
       :param code: the inputs (i.e. the defined variables) for the code
       :param code: the outputs to generate simplified expressions for
       :return: map from outputs to symbolic expressions
    """

    inputs = list(inputs)
    outputs = list(outputs)

    if type(code) is str:
        # clean up the code
        cleaned_code = unparse(ast.parse(code))
    else:
        # should be an ast
        cleaned_code = unparse(code)

    code_fn = """
def symbolic_execution({}):
    from sympy import exp, log
    def log2(x):
        return log(x, 2)

    def log10(x):
        return log(x, 10)

    # mostly taken from cmath.h
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh

    from sympy import Pow as pow, sqrt

    from sympy import sign, floor, ceiling as ceil, Abs as abs, Abs as fabs
    from sympy import Max as max, Min as min
    from sympy import Max as fmax, Min as fmin
    {}
    return {}
    """
    code_fn = code_fn.format(
        ", ".join(inputs),
        "\n".join("    " + line.strip() for line in cleaned_code.split("\n")),
        ", ".join(outputs))

    try:
        exec(code_fn)

        # no idea why, but simply calling symbolic_execution doesn't work
        results = vars()['symbolic_execution'](
            *[sp.symbols(inp) for inp in inputs])

        if len(outputs) > 1:
            return dict(zip(outputs, results))
        else:
            return {outputs[0]: results}
    except Exception as e:
        raise AutoDiffException(
            "Exception occured while attempting to symbolically execute code:\n{}\n{}"
            .format(cleaned_code, code_fn)) from e


def symbolically_diff_tasklet(tasklet: nd.Tasklet, target: str,
                              x: str) -> nd.Tasklet:
    """Symbolically differentiate the value of an output of a tasklet (target) with respect to an
       input x.
    """

    if tasklet.language is not dace.dtypes.Language.Python:
        raise AutoDiffException(
            "Expected tasklet with language Python, got language {}".format(
                tasklet.language))

    code_str = tasklet.code
    if type(code_str) is ast.Module:
        # unparse the tree
        code_str = unparse(code_str)

    if x not in tasklet.in_connectors:
        raise AutoDiffException("Unknown connector {}".format(x))

    if target not in tasklet.out_connectors:
        raise AutoDiffException("Unknown connector {}".format(x))

    output_exprs = code_to_exprs(code_str, tasklet.in_connectors,
                                 tasklet.out_connectors)

    target_expr = output_exprs[target]

    free_symbs = _symbols_to_strings(target_expr.free_symbols)

    if target in free_symbs:
        raise AutoDiffException(
            "Inplace operations are currently not supported for autodiff")

    diff_expr = target_expr.diff(sp.symbols(x))

    if diff_expr.atoms(sp.Derivative):
        # the final result contains a call to sp.Derivative
        raise AutoDiffException(
            "Unable to differentiate expression: {}".format(diff_expr.expr))

    return nd.Tasklet(
        "_" + tasklet.label + "_backward_",
        inputs=_symbols_to_strings(diff_expr.free_symbols)
        | {target + "_grad"},
        outputs={x + "_grad"},
        code=x + "_grad = {}_grad * ({})".format(
            target, str(diff_expr))  # this should be valid python code
    )


def _check_volume_one(memlet: dace.Memlet):
    elems = memlet.subset.num_elements()

    if len(elems.free_symbols) > 0 or int(memlet.subset.num_elements()) != 1:
        raise AutoDiffException(
            "Autodiff only supported for scalar tasklets (tasklets with input memlets with volume 1)"
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
            raise AutoDiffException(
                "Expected tasklet to be connected directly to the map exit and entry"
            )
        _check_volume_one(memlet)

        if memlet.data in in_arrays:
            raise AutoDiffException(
                "Using the same array for multiple memlets is currently not supported in autodiff"
            )

        in_arrays.add(memlet.data)

        if memlet.data == X:
            X_conn = conn

    if len(forward_state.out_edges(tasklet)) != 1:
        raise AutoDiffException(
            "Tasklets with more than one output are not supported")

    target_conn = None
    target_memlet: Memlet = None
    for map_tasklet, conn, _, _, memlet in forward_state.in_edges(map_exit):
        if map_tasklet is not tasklet:
            raise AutoDiffException(
                "Expected tasklet to be connected directly to the map exit and entry"
            )
        _check_volume_one(memlet)

        if memlet.data in in_arrays:
            raise AutoDiffException(
                "Inplace operations are currently not supported in autodiff")

        out_arrays.add(memlet.data)

        if memlet.data == target:
            target_conn = conn
            target_memlet = dc(memlet)

    if target_conn is None:
        raise AutoDiffException(
            "target ({}) not found in tasklet outputs".format(target))

    if X_conn is None:
        raise AutoDiffException(
            "X_array ({}) not found in tasklet inputs".format(X))

    target_grad = sdfg.arrays[target + "_grad"]

    # allocate an array for the gradient of x
    X_arr = sdfg.arrays[X]
    if X_arr.dtype not in [dace.float16, dace.float32, dace.float64]:
        raise AutoDiffException(
            "Expected dtype of x to be float, got {}".format(
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

    diff_tasklet = symbolically_diff_tasklet(
        tasklet,
        target_conn,
        X_conn,
    )
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
        raise AutoDiffException(
            "X_array ({}) not found in tasklet inputs".format(X))

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
        raise AutoDiffException(
            "Expected dtype of x to be float, got {}".format(
                target_arr.dtype.to_string()))

    if len(target_arr.total_size.free_symbols) > 0 or int(
            target_arr.total_size) != 1:
        raise AutoDiffException(
            "Targets with more than one element are currently unsupported in autodiff"
        )

    target_grad, target_grad_arr = sdfg.add_array(target.data + "_grad",
                                                  target_arr.shape,
                                                  target_arr.dtype)

    def getpred(state: dace.SDFGState, node: nd.Node):
        in_edges = state.in_edges(node)
        preds = {edge.src for edge in in_edges}
        if len(preds) > 1:
            raise AutoDiffException("Unsupported graph structure for autodiff")
        return next(iter(preds))

    # this is a breath first search along collapsed maps
    queue: Deque[Tuple[nd.AccessNode, dace.SDFGState]] = deque([target])
    prev_state = forward_state
    while queue:
        current = queue.popleft()

        if type(current) is not nd.AccessNode:
            raise AutoDiffException("Unsupported graph structure for autodiff")

        try:
            map_x = getpred(forward_state, current)
            task = getpred(forward_state, map_x)
            map_e = getpred(forward_state, task)

            if (type(map_e) is not nd.MapEntry or  #
                    type(task) is not nd.Tasklet or  #
                    type(map_x) is not nd.MapExit):
                raise AutoDiffException(
                    "Unsupported graph structure for autodiff")

            input_edges = forward_state.in_edges(map_e)
            inputs = list({edge.src for edge in input_edges})
            for inp in inputs:
                if type(inp) is not nd.AccessNode:
                    raise AutoDiffException(
                        "Unsupported graph structure for autodiff")
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
            raise AutoDiffException("Unsupported graph structure")
