"""Automatic Differentiation of SDFGStates.

   This module exposes the add_backward_pass method that can be used to add a backward pass to an
   SDFGState.
"""
import dace
from dace import Memlet, SDFG, SDFGState
import dace.graph.nodes as nd
from dace.sdfg import ScopeSubgraphView
from dace.graph.nxutil import dfs_topological_sort
from dace.frontend.operations import detect_reduction_type

import ast
import itertools
from collections import deque, defaultdict
from copy import deepcopy as dc
from typing import Iterator, Tuple, Deque, Dict, Set, List

import aenum
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from astunparse import unparse


class AutoDiffException(Exception):
    """Base class for all exceptions related to automatic differentiation"""
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


def _check_one(value):
    if type(value) is int:
        return value == 1

    if len(value.free_symbols) > 0 or int(value) != 1:
        raise AutoDiffException("Expected value one, got {}".format(value))


def _invert_access(access: dace.AccessType) -> dace.AccessType:
    if access == dace.AccessType.ReadOnly:
        return dace.AccessType.WriteOnly
    elif access == dace.AccessType.WriteOnly:
        return dace.AccessType.ReadOnly
    return access


def _has_inplace_operation(state: dace.SDFGState,
                           target: nd.AccessNode) -> bool:
    """Returns true if state has any inplace operations before target"""

    seen_accesses: Set[str] = set()
    for src, _, dst, _, _ in state.bfs_edges(target):
        if type(src) is nd.AccessNode:
            if src.data in seen_accesses:
                return True
            seen_accesses.add(src.data)
        if type(dst) is nd.AccessNode:
            if dst.data in seen_accesses:
                return True
            seen_accesses.add(dst.data)
    return False


class BackwardPassGenerator:
    """Signatures:
       Generate the backward pass for the node.

       :param node: the node to generate a backward pass for.
       :param output_grads: the nodes that produce the gradients for the outputs. This is a map from
                            output_connector to Tuple[Node, input_connector].
       :param required_grads: the list of input connectors that gradients need to be generated
                              for. For each of these, there should be an output on the reverse node
                              with the suffix _grad.
       :return: the reversed node

    """
    def __init__(self, sdfg: SDFG, state: SDFGState, required_grads: Set[str],
                 target: nd.AccessNode):
        """Generate the backward pass for a state wrt. a `target` scalar.

           :param sdfg: the `SDFG` to differentiate
           :param forward_state: the `SDFGState` to differentiate
           :param required_grads: the names of all arrays that gradients should be calculated for
           :param target: the `AccessNode` in `state` to generate the backward pass for. This
                          should be an array with one element.
        """
        self.sdfg = sdfg
        self.state = state
        self.required_grads = required_grads
        self.target = target

    def backward(self):

        target_arr = self.sdfg.arrays[self.target.data]
        try:
            _check_one(target_arr.total_size)
        except AutoDiffException as e:
            raise AutoDiffException("Expected scalar target") from e

        self.reverse_map = {}
        if self.required_grads.difference(node.data
                                          for node in self.state.nodes()
                                          if type(node) is nd.AccessNode):
            raise AutoDiffException(
                "Could not find AccessNode nodes in {} with the following data {}"
                .format(
                    self.sdfg,
                    self.required_grads.difference(
                        node.data for node in self.state.nodes()
                        if type(node) is nd.AccessNode)))

        # check for inplace operations (i.e. duplicated access nodes)
        if _has_inplace_operation(self.state, self.target):
            raise AutoDiffException(
                "Inplace operations are currently not supported in autodiff")

        source_nodes = [
            node for node in self.state.source_nodes()
            if node.data in self.required_grads
        ]

        required_nodes: Set[nd.Node] = set()

        # determine which nodes we need to reverse
        required_nodes = set()
        for src, _, dst, _, _ in self.state.bfs_edges(source_nodes):
            required_nodes |= {src, dst}
            if dst is self.target:
                break
        else:
            raise AutoDiffException(
                "target node {} was not reachable from source nodes {}".format(
                    self.target, required_grads))

        self.grad_memlets: Dict[str, List[Memlet]] = defaultdict(list)
        self._reverse_subgraph(
            ScopeSubgraphView(self.state, list(required_nodes)))

    def append_grad(self, conn):
        if conn is None:
            return None
        else:
            return conn + "_grad"

    def _reverse_subgraph(self, subgraph: ScopeSubgraphView):

        nodes_to_skip = set()
        # all memlets that write to grads
        # a reversed topological sort is a topological sort on the reverse graph
        for node in reversed(
                list(
                    dfs_topological_sort(subgraph,
                                         subgraph.source_nodes(),
                                         target=self.target))):
            if type(node) is nd.MapExit:
                # TODO handle scopes
                raise AutoDiffException("TODO")
            else:
                output_grads = [edge for edge in subgraph.out_edges(node)]
                required_grads = [
                    edge.dst_conn for edge in subgraph.in_edges(node)
                ]

                rev: nd.Node = getattr(self, '_reverse_' +
                                       type(node).__name__)(node, output_grads,
                                                            required_grads)
                # add rev to the graph and hook up the edges
                self.reverse_map[node] = rev
                subgraph.graph.add_node(rev)

                # connect the gradients of the outputs (as inputs)
                for _, output_conn, dest_node, input_conn, memlet in output_grads:
                    dest_node = self.reverse_map[dest_node]
                    if detect_reduction_type(memlet.wcr) not in [
                            None, dace.dtypes.ReductionType.Sum
                    ]:
                        raise AutoDiffException(
                            "Unsupported reduction type {}".format(
                                detect_reduction_type(memlet.wcr)))

                    memlet = dc(memlet)

                    if memlet.data not in self.grad_memlets:
                        # this grad hasn't been written before: initialize it
                        array = self.state.parent.arrays[memlet.data]

                        # this can clearly fail if someone chooses annoying array names; let's
                        # ignore this for now
                        if type(array) is dace.data.Scalar:
                            self.sdfg.add_scalar(memlet.data + "_grad",
                                                 array.dtype,
                                                 storage=array.storage,
                                                 transient=array.transient,
                                                 toplevel=array.toplevel)
                        elif type(array) is dace.data.Array:
                            self.state.parent.add_array(
                                memlet.data + "_grad",
                                array.shape,
                                array.dtype,
                                storage=array.storage,
                                materialize_func=array.materialize_func,
                                transient=array.transient,
                                strides=array.strides,
                                offset=array.offset,
                                toplevel=array.toplevel,
                                allow_conflicts=array.allow_conflicts,
                                total_size=array.total_size)
                        else:
                            raise AutoDiffException(
                                "Unsupported data descriptor {}".format(array))

                    self.grad_memlets[memlet.data].append(memlet)
                    memlet.data = memlet.data + "_grad"
                    subgraph.graph.add_edge(dest_node,
                                            self.append_grad(input_conn), rev,
                                            self.append_grad(output_conn),
                                            memlet)

                if isinstance(node, nd.AccessNode):
                    # this means we are writing out a grad to an array. In this case, we need to set
                    # all incoming memlets to WCR Sum
                    # TODO @orausch there could be an intersection check here
                    for edge in subgraph.graph.in_edges(rev):
                        for path_edge in subgraph.graph.memlet_path(edge):
                            path_edge.data.wcr = "lambda x, y: x + y"

                # connect any required inputs from the forward pass
                required_inputs = rev.in_connectors.difference(
                    self.append_grad(edge.src_conn) for edge in output_grads)
                for src, src_conn, dst, dst_conn, memlet in subgraph.graph.in_edges(
                        node):
                    if dst_conn in required_inputs:
                        subgraph.graph.add_edge(src, src_conn, rev, dst_conn,
                                                memlet)

    def _reverse_AccessNode(self, node: nd.AccessNode,
                            output_grads: Dict[str, Tuple[nd.Node, str,
                                                          Memlet]],
                            required_grads: List[str]):

        return nd.AccessNode(node.data + "_grad",
                             access=_invert_access(node.access))

    def _reverse_Tasklet(self, tasklet: nd.Tasklet,
                         output_grads: Dict[str, Tuple[nd.Node, str, Memlet]],
                         required_grads: List[str]) -> nd.Tasklet:

        if tasklet.language is not dace.dtypes.Language.Python:
            raise AutoDiffException(
                "Expected tasklet with language Python, got language {}".
                format(tasklet.language))

        # tasklets should have scalar inputs
        for _, _, _, _, memlet in self.state.in_edges(tasklet):
            try:
                _check_one(memlet.subset.num_elements())
            except AutoDiffException as e:
                raise AutoDiffException(
                    "Autodiff only supported for scalar tasklets (tasklets with input memlets with volume 1)"
                ) from e

        code_str = tasklet.code
        if type(code_str) is ast.Module:
            # unparse the tree
            code_str = unparse(code_str)

        output_exprs = code_to_exprs(code_str, tasklet.in_connectors,
                                     tasklet.out_connectors)
        rev_code = []

        # the outputs of the reversed nodes are the grads of inputs of the original node
        rev_outputs = set()
        rev_inputs = set()

        for _, output_conn, _, input_conn, memlet in output_grads:
            # tasklets should have scalar outputs
            _check_one(memlet.subset.num_elements())
            # for each output_conn...
            for inp in required_grads:
                # ...add the code to generate {inp}_grad
                rev_outputs.add(inp + "_grad")

                output_expr = output_exprs[output_conn]

                # symbolically differentiate the output by inp
                diff_expr = output_expr.diff(sp.symbols(inp))

                if diff_expr.atoms(sp.Derivative):
                    # the final result contains a call to sp.Derivative
                    raise AutoDiffException(
                        "Unable to symbolically differentiate expression: {}".
                        format(diff_expr.expr))

                rev_inputs |= _symbols_to_strings(
                    diff_expr.free_symbols) | {output_conn + "_grad"}

                rev_code.append(
                    "{input}_grad = {output}_grad * ({diff_expr})".format(
                        input=inp,
                        output=output_conn,
                        diff_expr=str(diff_expr)))

        return nd.Tasklet("_" + tasklet.label + "_reverse_",
                          inputs=rev_inputs,
                          outputs=rev_outputs,
                          code="\n".join(rev_code))

    def _reverse_MapExit(self, node: nd.Tasklet,
                         output_grads: Dict[str, Tuple[nd.Node, str]],
                         required_grads: List[str]):
        pass
