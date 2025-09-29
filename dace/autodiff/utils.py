import collections
import numbers
import typing
import copy
import inspect
import ast
import re
import astunparse
import sympy as sp
from typing import List, Tuple, Set, Dict

# DaCe imports
import dace
from dace import dtypes,  data as dt
from dace.sdfg import SDFG, SDFGState, graph as dgraph, state as dstate, utils as dutils, nodes as nd
import dace.data as dt
from dace.sdfg.state import LoopRegion
from dace.frontend.python.parser import DaceProgram
import dace.util.utils as utils

# Autodiff imports
from dace.autodiff.base_abc import BackwardContext, BackwardResult



from dace.autodiff.base_abc import AutoDiffException
def forward_in_desc_with_name(forward_node: nd.Node, context: BackwardContext, name) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.

        :param forward_node: the node.
        :param context: the backward context.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return utils.in_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, name)


def forward_out_desc_with_name(forward_node: nd.Node, context: BackwardContext, name) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.

        :param forward_node: the node.
        :param context: the backward context.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return utils.out_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, name)


def add_backward_desc_for_connector(backward_sdfg: dace.SDFG, forward_node: nd.Node, context: BackwardContext,
                                    connector: str, input: bool) -> str:
    """ Adds the backward array for the connector of ``forward_node``.

        :param backward_sdfg: the sdfg to add to.
        :param forward_node: the forward node with the connector that we want to add a descriptor for
        :param connector: the connector on the forward node that we want to add the descriptor for
        :param input: ``True`` if the connector is an input, ``False`` otherwise
        :return: the name of the newly added array in ``backward_sdfg``.
    """

    if input:
        edge = utils.in_edge_with_name(forward_node, context.forward_state, connector)
    else:
        edge = utils.out_edge_with_name(forward_node, context.forward_state, connector)
    arr_name = edge.data.data

    forward_desc = context.forward_sdfg.arrays[arr_name]

    new_desc = copy.deepcopy(forward_desc)
    new_desc.transient = False
    return backward_sdfg.add_datadesc(arr_name + "_grad", new_desc, find_new_name=True)


def add_backward_desc(backward_sdfg: dace.SDFG, forward_sdfg: dace.SDFG, forward_desc: dt.Data,
                      forward_name: str) -> str:
    """ Adds the backward array for the given descriptor.

        :param backward_sdfg: the sdfg to add to.
        :param forward_sdfg: the forward sdfg.
        :param forward_desc: the data descriptor of the forward array from ``forward_sdfg``.
        :param forward_name: a name for the forward array (does not have to match it's actual name).
        :return: the name of the newly added array in ``backward_sdfg``.
    """
    backward_name = utils.find_str_not_in_set(forward_sdfg.arrays, forward_name + "_grad")
    new_desc = copy.deepcopy(forward_desc)
    new_desc.transient = False
    return backward_sdfg.add_datadesc(backward_name, new_desc)


def add_empty_sdfg_for_node(forward_node: nd.Node, required_descriptors: typing.List[str],
                            context: BackwardContext) -> typing.Tuple[nd.NestedSDFG, BackwardResult]:
    """ Given a node, return an SDFG that can be used as a nested SDFG expansion for that node.

        ``required_descriptors`` may contain:
        * Inputs/outputs of the forward node (these will be hooked up as required)
        * Inputs/outputs of the forward node with the ``_grad`` suffix. These will be hooked up
          as the gradients of the corresponding inputs/outputs.

        The descriptors will be initialized using the descriptors connected to edges of the
        forward node.

        :param forward_node: the node in the forward pass
        :param required_descriptors: A list of descriptors that should be added to the SDFG.
        :param context: the backward context
        :return: the nested SDFG and backward result for the forward node
    """

    nsdfg = dace.SDFG(forward_node.label + "_backward_expansion")

    def _get_fwd_descriptor(name):
        """Returns the descriptor and whether it is an input"""
        if name in forward_node.out_connectors:
            return forward_out_desc_with_name(forward_node, context, name), False
        elif name in forward_node.in_connectors:
            return forward_in_desc_with_name(forward_node, context, name), True

        raise ValueError(f"Could not find {name} in inputs or outputs of {forward_node}")

    outputs_to_connect_from_forward = []

    result = BackwardResult.empty()
    inputs = set()
    outputs = set()

    for name in required_descriptors:
        if name.endswith("_grad"):
            # hook this up as a gradient
            desc, is_input = _get_fwd_descriptor(name[:-5])
            if is_input:
                result.required_grad_names[name[:-5]] = name
            else:
                result.given_grad_names[name[:-5]] = name
            # input grads are outputs of the backward node
            if is_input:
                outputs.add(name)
            else:
                inputs.add(name)
        else:
            desc, is_input = _get_fwd_descriptor(name)
            if not is_input:
                outputs_to_connect_from_forward.append(name)
            inputs.add(name)
        ndesc = copy.deepcopy(desc)
        ndesc.transient = False
        nsdfg.add_datadesc(name, ndesc)

    bwd_node = context.backward_state.add_nested_sdfg(nsdfg, inputs, outputs)
    for output in outputs_to_connect_from_forward:
        connect_output_from_forward(forward_node, bwd_node, context, output)

    return bwd_node, result


def backward_program_for_node(program, context: BackwardContext,
                              forward_node: nd.Node) -> typing.Tuple[nd.Node, BackwardResult]:
    """ Expand a function to the backward function for a node.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.

        Gradient parameters should be the name of the forward parameter, appended with _grad. For these arguments the
        data descriptors will match the data descriptors of the inputs/outputs they correspond to.
    """

    input_names = set(inp.name for inp in forward_node.schema.inputs)
    output_names = set(outp.name for outp in forward_node.schema.outputs)

    if input_names.intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError("program_for_node cannot be applied on nodes of this type;"
                         " '{}' is both an input and an output".format(next(input_names.intersection(output_names))))

    def name_without_grad_in(name, collection):
        return name[-5:] == "_grad" and name[:-5] in collection

    params = inspect.signature(program).parameters

    backward_result = BackwardResult.empty()

    inputs = {}
    outputs = {}
    for name, param in params.items():
        if name in input_names:
            inputs[name] = copy.deepcopy(forward_in_desc_with_name(forward_node, context, name))

        elif name_without_grad_in(name, input_names):
            outputs[name] = copy.deepcopy(forward_in_desc_with_name(forward_node, context, name[:-5]))
            backward_result.required_grad_names[name[:-5]] = name

        elif name in output_names:
            inputs[name] = copy.deepcopy(forward_out_desc_with_name(forward_node, context, name))

        elif name_without_grad_in(name, output_names):
            inputs[name] = copy.deepcopy(forward_out_desc_with_name(forward_node, context, name[:-5]))
            backward_result.given_grad_names[name[:-5]] = name

        else:
            raise ValueError("'{}' was not found as an input or output for {}".format(name, forward_node.schema.name))

    program.__annotations__ = {**inputs, **outputs}

    sdfg = DaceProgram(program, (), {}, False, dace.DeviceType.CPU).to_sdfg()

    result_node = context.backward_state.add_nested_sdfg(sdfg, set(inputs), set(outputs))

    return result_node, backward_result


def connect_output_from_forward(forward_node: nd.Node, backward_node: nd.Node, context: BackwardContext,
                                output_connector_name: str):
    """ Connect an output of the forward node as an input to the backward node. This is done by forwarding the array
        from the forward pass.

        Conceptually, this is similar to pytorch's ctx.save_for_backward.

        :param forward_node: the node in the forward pass.
        :param backward_node: the node in the backward pass.
        :param context: the backward context.
        :param output_connector_name: the name of the connector on the backward pass. The output of that connector will
                                      be forwarded to the connector of the same name on the backward node.
    """
    output_edge = utils.out_edge_with_name(forward_node, context.forward_state, output_connector_name)

    # add the array of the output to backward_input_arrays that it will be forwarded by the autodiff engine
    output_arr_name = output_edge.data.data
    if output_arr_name not in context.backward_generator.backward_input_arrays:
        data_desc = copy.deepcopy(context.forward_sdfg.arrays[output_arr_name])
        context.backward_generator.backward_input_arrays[output_arr_name] = data_desc

        if context.backward_generator.separate_sdfgs:
            data_desc.transient = False
            context.backward_sdfg.add_datadesc(output_arr_name, data_desc)

        read = context.backward_state.add_read(output_arr_name)
    else:
        cand = [
            n for n, _ in context.backward_state.all_nodes_recursive()
            if isinstance(n, nd.AccessNode) and n.data == output_arr_name
        ]
        read = cand[0]
    context.backward_state.add_edge(read, None, backward_node, output_connector_name, copy.deepcopy(output_edge.data))


def cast_consts_to_type(code: str, dtype: dace.typeclass) -> str:
    """ Convert a piece of code so that constants are wrapped in casts to ``dtype``.

        For example:

            x * (3 / 2)

        becomes:

            x * (dace.float32(3) / dace.float32(2))

        This is only done when it is required due to a Div operator.

        :param code: the code string to convert.
        :param dtype: the dace typeclass to wrap cast to
        :return: a string of the converted code.
    """

    class CastConsts(ast.NodeTransformer):

        def __init__(self):
            self._in_div_stack = collections.deque()

        def visit_Num(self, node):
            if self._in_div_stack:
                return ast.copy_location(
                    ast.parse(f"dace.{dtype.to_string()}({astunparse.unparse(node)})").body[0].value, node)
            else:
                return self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp):
            if node.op.__class__.__name__ == "Pow":
                # within pow, we don't need to cast unless there is a new div
                old_stack = self._in_div_stack
                # reset the stack
                self._in_div_stack = collections.deque()
                node = self.generic_visit(node)
                self._in_div_stack = old_stack
                return node

            elif node.op.__class__.__name__ == "Div":
                self._in_div_stack.append(None)
                node = self.generic_visit(node)
                self._in_div_stack.popleft()
                return node
            else:
                return self.generic_visit(node)

        def visit_Constant(self, node):
            if self._in_div_stack:
                return ast.copy_location(
                    ast.parse(f"dace.{dtype.to_string()}({astunparse.unparse(node)})").body[0].value, node)
            else:
                return self.generic_visit(node)

    return astunparse.unparse(CastConsts().visit(ast.parse(code)))


def init_grad(data: str, sdfg: SDFG, current_state: SDFGState):
    """
    Add a state where `data` is initialized with zero.

    :param data: the data to initialize
    :param sdfg: the SDFG to add the state to
    :param current_state: the current state; the initialization will be done before this state
    """
    arr = sdfg.arrays[data]

    state = sdfg.add_state_before(current_state, label="init_" + data)

    scalar = 0
    if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, arr.storage):
        cuda = False
    elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, arr.storage):
        cuda = True
    else:
        raise ValueError(f"Unsupported storage {arr.storage}")

    if isinstance(arr, (dt.Array, dt.Scalar)):
        state.add_mapped_tasklet(
            "_init_" + data + "_", {
                "i{}".format(i): "0:{}".format(shape)
                for i, shape in enumerate(arr.shape)
            }, {},
            "__out = {}".format(scalar),
            {"__out": dace.Memlet.simple(data, ", ".join("i{}".format(i) for i in range(len(arr.shape))))},
            schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default,
            external_edges=True)
    elif type(arr) is dt.View:
        # not need to initialize: the viewed array will always be visited
        # (since a view can never be a required grad), and thus the viewed array will be initialized.
        pass
    else:
        raise AutoDiffException("Unsupported data descriptor {}".format(arr))


def symbols_to_strings(symbs: Set[sp.Symbol]) -> Set[str]:
    return {str(symb) for symb in symbs}


def extract_indices(expression: str) -> Dict[str, List[str]]:
    """
    Extracts indexed array names and their indices from a given string expression.
    """
    # Regular expression to match the array names and their indices
    pattern = r"(\w+)\[((?:\w+,?\s*)+)\]"

    # Find all matches in the given expression
    matches = re.findall(pattern, expression)

    # Create a dictionary to store the arrays and their indices
    index_map = {}
    for name, indices in matches:
        # Split indices by comma and remove any extra spaces
        index_list = [index.strip() for index in indices.split(',')]
        index_map[name] = index_list

    return index_map


def code_to_exprs(code: str, tasklet: nd.Tasklet,
                  symbols: List[str]) -> Tuple[Dict[str, sp.Expr], Dict[str, List[str]]]:
    """ Convert a python string to a set of (simplified) symbolic sympy expressions. Currently, this
        supports only code consisting of assignment statements.

        :param code: the code to convert
        :param inputs: the inputs (i.e. the defined variables) for the code
        :param outputs: the outputs to generate simplified expressions for
        :return: map from outputs to symbolic expressions
    """

    inputs: List[str] = list(tasklet.in_connectors)
    outputs: List[str] = list(tasklet.out_connectors)

    # Add the definition of global constant symbols that are presen in the code
    # Prepare the Symbol declaration code
    symbol_code = ""
    for symb in symbols:
        symbol_code += f"    {symb} = sp.symbols('{symb}')\n"

    # We prepare a map of indexed objects and their indices
    indexed_objects_map = extract_indices(code)

    # For now, make sure none of the outputs are indexed objects
    assert not any(out in indexed_objects_map for out in outputs)

    # Add the definition of indexed objects to the sympy code
    indexed_objects_code = ""
    for conn in inputs + outputs:
        if (conn in inputs and isinstance(tasklet.in_connectors[conn], dace.dtypes.pointer)
                or (conn in outputs and isinstance(tasklet.out_connectors[conn], dace.dtypes.pointer))):
            assert conn in indexed_objects_map
            indexed_objects_code += f"    {conn} = sp.IndexedBase('{conn}')\n"
            for idx in indexed_objects_map[conn]:
                indexed_objects_code += f"    {idx} = sp.symbols('{idx}', cls=sp.Idx)\n"

    code_fn = """
def symbolic_execution({}):
    # define functions from cmath.h
    from sympy import exp, log
    def log2(x):
        return log(x, 2)
    def log10(x):
        return log(x, 10)
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh
    from sympy import Pow as pow, sqrt
    from sympy import sign, floor, ceiling as ceil, Abs as abs, Abs as fabs
    from sympy import Max as max, Min as min
    from sympy import Max as fmax, Min as fmin
    from sympy import erf
    import sympy as sp
{}
{}
{}
    return {}
    """
    code_fn = code_fn.format(
        ", ".join(inputs),
        symbol_code,
        indexed_objects_code,
        "\n".join("    " + line.strip() for line in code.split("\n")),
        ", ".join(outputs),
    )

    # Clean out type coonversions from the code
    code_fn = re.sub(r"dace\.(float32|int32|float64|int64)\((.*?)\)", r"\2", code_fn)

    try:
        # need to have dace so things like `dace.float32(1)` work
        temp_globals = {'dace': dace}
        exec(code_fn, temp_globals)

        # no idea why, but simply calling symbolic_execution doesn't work
        results = temp_globals["symbolic_execution"](*[sp.symbols(inp) for inp in inputs])

        if len(outputs) > 1:
            # make sure that everything is a sympy expression
            for i, res in enumerate(results):
                if not isinstance(res, sp.Expr):
                    results[i] = sp.sympify(res)
            return dict(zip(outputs, results)), indexed_objects_map
        else:
            # make sure that everything is a sympy expression
            if not isinstance(results, sp.Expr):
                results = sp.sympify(results)
            return {outputs[0]: results}, indexed_objects_map
    except Exception as e:
        raise AutoDiffException(
            "Exception occured while attempting to symbolically execute code:\n{}".format(code)) from e


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_int_eq_value(value, target_value: int) -> bool:
    if isinstance(value, numbers.Integral):
        return value == target_value

    if len(value.free_symbols) > 0 or int(value) != target_value:
        return False

    return True


def invert_map_connector(conn):
    if conn.startswith("IN"):
        return "OUT" + conn[2:]
    elif conn.startswith("OUT"):
        return "IN" + conn[3:]
    else:
        raise AutoDiffException("Could not parse map connector '{}'".format(conn))


def path_src_node_in_subgraph(edge: dgraph.MultiConnectorEdge, subgraph: dstate.StateSubgraphView):
    path_src = subgraph.memlet_path(edge)[0].src
    return path_src in subgraph.nodes()


def get_read_only_arrays(sdfg: SDFG) -> Set[str]:
    """
    Get the arrays that are only read in SDFG
    """
    written_to_arrays = set()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nd.AccessNode):
            if parent.in_degree(node) > 0 and any(not e.data.is_empty() for e in parent.in_edges(node)):
                written_to_arrays.add(node.data)

    read_only_arrays = set(sdfg.arrays.keys()) - written_to_arrays
    return read_only_arrays


def get_state_topological_order(graph) -> List[SDFGState]:
    """
    Returns the SDFG states in topological order.
    """
    all_nodes = list(dutils.dfs_topological_sort(graph, graph.source_nodes()))
    state_order = []
    for node in all_nodes:
        if isinstance(node, SDFGState):
            state_order.append(node)
        elif isinstance(node, LoopRegion):
            loop_state_order = get_state_topological_order(node)
            state_order.extend(loop_state_order)
        else:
            raise AutoDiffException(
                f"Unsupported node type {node} at the highest level of the SDFG while getting the state order")

    # All states in the graph need to be present in the state order
    if isinstance(graph, SDFG) and set(state_order) != set(graph.states()):
        raise AutoDiffException("Could not find all states of the SDFG in the state order")
    return state_order

class SympyCleaner(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id == "pi":
            return ast.copy_location(ast.parse("(3.141592653589)").body[0], node)
        else:
            return self.generic_visit(node)