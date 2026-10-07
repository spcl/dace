# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
import inspect
import numbers
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import astunparse
import sympy as sp

# DaCe imports
import dace
import dace.sdfg.utils as utils
from dace import dtypes, symbolic
from dace import data as dt
from dace.frontend.python.parser import DaceProgram
from dace.sdfg import SDFG, SDFGState, graph as dgraph, nodes as nd, state as dstate
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.while_to_for_loop import condition_terms

# Autodiff imports
from dace.autodiff.base_abc import AutoDiffException, BackwardContext, BackwardResult

#: Global that the generated ``symbolic_execution`` source reads its pre-built symbols from, so
#: minting happens here -- where the dtypes are known -- instead of inside an exec'd string.
SYMBOL_TABLE_GLOBAL = "dace_ad_symbol_table"


def forward_in_desc_with_name(forward_node: nd.Node, context: BackwardContext, name: str) -> dt.Data:
    """Find the descriptor of the data that connects to input connector ``name``.

    :param forward_node: The node in the forward pass.
    :param context: The backward context containing forward SDFG and state information.
    :param name: The input connector name to find the descriptor for.
    :return: The data descriptor that connects to the specified connector.
    """
    return utils.in_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, name)


def forward_out_desc_with_name(forward_node: nd.Node, context: BackwardContext, name: str) -> dt.Data:
    """Find the descriptor of the data that connects to output connector ``name``.

    :param forward_node: The node in the forward pass.
    :param context: The backward context containing forward SDFG and state information.
    :param name: The output connector name to find the descriptor for.
    :return: The data descriptor that connects to the specified connector.
    """
    return utils.out_desc_with_name(forward_node, context.forward_state, context.forward_sdfg, name)


def add_backward_desc_for_connector(
    backward_sdfg: dace.SDFG, forward_node: nd.Node, context: BackwardContext, connector: str, input: bool
) -> str:
    """Adds the backward array for the connector of ``forward_node``.

    :param backward_sdfg: The SDFG to add the backward array descriptor to.
    :param forward_node: The forward node with the connector to create a descriptor for.
    :param context: The backward context containing forward SDFG and state information.
    :param connector: The connector name on the forward node.
    :param input: True if the connector is an input, False if it's an output.
    :return: The name of the newly added gradient array in ``backward_sdfg``.
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


def add_backward_desc(
    backward_sdfg: dace.SDFG, forward_sdfg: dace.SDFG, forward_desc: dt.Data, forward_name: str
) -> str:
    """Adds the backward array for the given descriptor.

    :param backward_sdfg: The SDFG to add the backward array descriptor to.
    :param forward_sdfg: The forward SDFG used for finding unique names.
    :param forward_desc: The data descriptor of the forward array.
    :param forward_name: A name for the forward array (doesn't have to match its actual name).
    :return: The name of the newly added gradient array in ``backward_sdfg``.
    """
    backward_name = dt.find_new_name(forward_name + "_grad", forward_sdfg.arrays)
    new_desc = copy.deepcopy(forward_desc)
    new_desc.transient = False
    return backward_sdfg.add_datadesc(backward_name, new_desc)


def connector_symbol(name: str, dtype: Union[dtypes.typeclass, None] = None) -> symbolic.symbol:
    """Mint the DaCe symbol standing for a tasklet connector or an SDFG symbol.

    Always a ``symbolic.symbol``, never a bare ``sympy.Symbol``: the two compare unequal while
    still substituting for one another, which is what makes ``diff()`` return a silent zero.
    ``integer=None`` where the dtype is unknown, so that ``symbol``'s int32 default does not
    assert an integer assumption the declaration never made.
    """
    if isinstance(dtype, dtypes.typeclass):
        return symbolic.symbol(name, dtype)
    return symbolic.symbol(name, integer=None)


def backward_symbol_mapping(nested_sdfg: SDFG, parent_state: SDFGState) -> Dict[str, symbolic.symbol]:
    """Symbol mapping for a backward nested SDFG, resolved against the scope it is placed in.

    Backward SDFGs are assembled from descriptors deep-copied out of the parent, so identity is the
    right mapping -- but it has to be spelled out. ``add_nested_sdfg`` installs its identity default
    only *after* its own missing-symbol check, which therefore can never fire, and the default maps
    every symbol to a bare name, dropping the dtype the parent declared for it. A symbol the parent
    does not declare is scope-local (a map parameter, say) and has no dtype to carry over.
    """
    parent_symbols = parent_state.sdfg.symbols
    return {name: connector_symbol(name, parent_symbols.get(name)) for name in sorted(nested_sdfg.free_symbols)}


def add_empty_sdfg_for_node(
    forward_node: nd.Node, required_descriptors: List[str], context: BackwardContext
) -> Tuple[nd.NestedSDFG, BackwardResult]:
    """Given a node, return an SDFG that can be used as a nested SDFG expansion for that node.

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

    bwd_node = context.backward_state.add_nested_sdfg(
        nsdfg, sorted(inputs), sorted(outputs), symbol_mapping=backward_symbol_mapping(nsdfg, context.backward_state)
    )
    for output in outputs_to_connect_from_forward:
        connect_output_from_forward(forward_node, bwd_node, context, output)

    return bwd_node, result


def backward_program_for_node(
    program, context: BackwardContext, forward_node: nd.Node
) -> Tuple[nd.Node, BackwardResult]:
    """Expand a function to the backward function for a node.

    The dtypes for the arguments will be extracted by matching the parameter names to edges.

    Gradient parameters should be the name of the forward parameter, appended with _grad. For these arguments the
    data descriptors will match the data descriptors of the inputs/outputs they correspond to.
    """

    input_names = set(inp.name for inp in forward_node.schema.inputs)
    output_names = set(outp.name for outp in forward_node.schema.outputs)

    if input_names.intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type; '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))
            )
        )

    def name_without_grad_in(name, collection):
        return name[-5:] == "_grad" and name[:-5] in collection

    params = inspect.signature(program).parameters

    backward_result = BackwardResult.empty()

    inputs = {}
    outputs = {}
    for name, _ in params.items():
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

    result_node = context.backward_state.add_nested_sdfg(
        sdfg, sorted(inputs), sorted(outputs), symbol_mapping=backward_symbol_mapping(sdfg, context.backward_state)
    )

    return result_node, backward_result


def connect_output_from_forward(
    forward_node: nd.Node, backward_node: nd.Node, context: BackwardContext, output_connector_name: str
):
    """Connect an output of the forward node as an input to the backward node. This is done by forwarding the array
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
            n
            for n, _ in context.backward_state.all_nodes_recursive()
            if isinstance(n, nd.AccessNode) and n.data == output_arr_name
        ]
        read = cand[0]
    context.backward_state.add_edge(read, None, backward_node, output_connector_name, copy.deepcopy(output_edge.data))


def cast_consts_to_type(code: str, dtype: dace.typeclass) -> str:
    """Convert a piece of code so that constants are wrapped in casts to ``dtype``.

    For example::

        x * (3 / 2)

    becomes::

        x * (dace.float32(3) / dace.float32(2))

    This is only done when it is required due to a Div operator to ensure proper
    type casting in mathematical expressions during automatic differentiation.

    :param code: The code string to convert.
    :param dtype: The DaCe typeclass to cast constants to.
    :return: A string of the converted code with properly typed constants.
    """

    class CastConsts(ast.NodeTransformer):
        def __init__(self):
            self._in_div_stack = collections.deque()

        def visit_Num(self, node):
            if self._in_div_stack:
                return ast.copy_location(
                    ast.parse(f"dace.{dtype.to_string()}({astunparse.unparse(node)})").body[0].value, node
                )
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
                    ast.parse(f"dace.{dtype.to_string()}({astunparse.unparse(node)})").body[0].value, node
                )
            else:
                return self.generic_visit(node)

    return astunparse.unparse(CastConsts().visit(ast.parse(code)))


def init_grad(data: str, sdfg: SDFG, current_state: SDFGState) -> None:
    """Add a state where ``data`` is initialized with zero.

    This function creates a new state before the current state that initializes
    the gradient array with zeros. It handles different storage types (CPU/GPU)
    and array types appropriately.

    :param data: The name of the data array to initialize.
    :param sdfg: The SDFG to add the initialization state to.
    :param current_state: The current state; initialization will be done before this state.
    :raises ValueError: If the storage type is not supported.
    :raises AutoDiffException: If the data descriptor type is not supported.
    """
    arr = sdfg.arrays[data]

    state = sdfg.add_state_before(current_state, label="init_" + data)

    scalar = 0
    if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, arr.storage):
        cuda = False
    elif dtypes.can_access(dtypes.ScheduleType.GPU_Device, arr.storage):
        cuda = True
    else:
        raise ValueError(f"Unsupported storage {arr.storage}")

    if isinstance(arr, (dt.Array, dt.Scalar)):
        state.add_mapped_tasklet(
            "_init_" + data + "_",
            {"i{}".format(i): "0:{}".format(shape) for i, shape in enumerate(arr.shape)},
            {},
            "__out = {}".format(scalar),
            {"__out": dace.Memlet.simple(data, ", ".join("i{}".format(i) for i in range(len(arr.shape))))},
            schedule=dtypes.ScheduleType.GPU_Device if cuda else dtypes.ScheduleType.Default,
            external_edges=True,
        )
    elif type(arr) is dt.View:
        # not need to initialize: the viewed array will always be visited
        # (since a view can never be a required grad), and thus the viewed array will be initialized.
        pass
    else:
        raise AutoDiffException("Unsupported data descriptor {}".format(arr))


def extract_indices(expression: str) -> Dict[str, List[str]]:
    """Extracts indexed array names and their indices from a given string expression.

    This function uses regular expressions to find patterns like "array[i, j, k]"
    and returns a dictionary mapping array names to their index lists.

    :param expression: The string expression to analyze.
    :return: A dictionary mapping array names to lists of their indices.

    Example::

        >>> extract_indices("a[i, j] + b[k]")
        {'a': ['i', 'j'], 'b': ['k']}
    """
    # Regular expression to match the array names and their indices
    pattern = r"(\w+)\[((?:\w+,?\s*)+)\]"

    # Find all matches in the given expression
    matches = re.findall(pattern, expression)

    # Create a dictionary to store the arrays and their indices
    index_map = {}
    for name, indices in matches:
        # Split indices by comma and remove any extra spaces
        index_list = [index.strip() for index in indices.split(",")]
        index_map[name] = index_list

    return index_map


class _CastRemover(ast.NodeTransformer):
    """
    Replaces casts with their argument: casts to DaCe types (``dace.float64(x)``) and floating-point casts with C type
    names, which the Python frontend emits for casts through variables that hold a DaCe type (``double(x)``).
    """

    _FLOAT_CASTS = ("float", "double")

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        is_dace_cast = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "dace"
            and isinstance(vars(dace).get(func.attr), dtypes.typeclass)
        )
        is_float_cast = isinstance(func, ast.Name) and func.id in self._FLOAT_CASTS
        if (is_dace_cast or is_float_cast) and len(node.args) == 1 and not node.keywords:
            return node.args[0]
        return node


def _strip_casts(code: str) -> str:
    """Removes casts to DaCe types from tasklet code, keeping the grouping of the expressions they wrap."""
    tree = _CastRemover().visit(ast.parse(code))
    return astunparse.unparse(tree).strip()


def code_to_exprs(
    code: str, tasklet: nd.Tasklet, symbols: Dict[str, dtypes.typeclass]
) -> Tuple[Dict[str, sp.Expr], Dict[str, List[str]]]:
    """Convert a python string to a set of (simplified) symbolic sympy expressions. Currently, this
    supports only code consisting of assignment statements.

    :param code: the code to convert
    :param tasklet: the tasklet the code belongs to; its connectors are the defined variables
    :param symbols: the SDFG symbols visible to the code, mapped to their dtypes
    :return: map from outputs to symbolic expressions, and the map of indexed objects to indices
    """

    inputs: List[str] = list(tasklet.in_connectors)
    outputs: List[str] = list(tasklet.out_connectors)

    # Symbols reach the generated source through this table instead of being minted inside it from a
    # bare name: minting here is the only place that still knows their dtypes.
    symbol_table: Dict[str, sp.Expr] = {}

    # Add the definition of global constant symbols that are presen in the code
    # Prepare the Symbol declaration code
    symbol_code = ""
    for symb, symb_dtype in symbols.items():
        symbol_table[symb] = connector_symbol(symb, symb_dtype)
        symbol_code += f"    {symb} = {SYMBOL_TABLE_GLOBAL}['{symb}']\n"

    # We prepare a map of indexed objects and their indices
    indexed_objects_map = extract_indices(code)

    # For now, make sure none of the outputs are indexed objects
    indexed_outputs = [out for out in outputs if out in indexed_objects_map]
    if indexed_outputs:
        raise AutoDiffException(f"Indexed outputs are not currently supported: {indexed_outputs}")

    # Add the definition of indexed objects to the sympy code
    indexed_objects_code = ""
    for conn in inputs + outputs:
        if (
            conn in inputs
            and isinstance(tasklet.in_connectors[conn], dace.dtypes.pointer)
            or (conn in outputs and isinstance(tasklet.out_connectors[conn], dace.dtypes.pointer))
        ):
            if conn not in indexed_objects_map:
                raise AutoDiffException(f"Expected connector '{conn}' to be in indexed objects map for pointer type")
            indexed_objects_code += f"    {conn} = sp.IndexedBase('{conn}')\n"
            for idx in indexed_objects_map[conn]:
                idx_sym = connector_symbol(idx, symbols.get(idx) or tasklet.in_connectors.get(idx))
                # An Idx label must be integral, so a non-integer declaration cannot supply one.
                symbol_table[idx] = sp.Idx(idx_sym if idx_sym.is_integer else symbolic.symbol(idx))
                indexed_objects_code += f"    {idx} = {SYMBOL_TABLE_GLOBAL}['{idx}']\n"

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
    # Clean out type conversions from the code (they do not change the value symbolically)
    code = _strip_casts(code)
    code_fn = code_fn.format(
        ", ".join(inputs),
        symbol_code,
        indexed_objects_code,
        "\n".join("    " + line.strip() for line in code.split("\n")),
        ", ".join(outputs),
    )

    try:
        # need to have dace so things like `dace.float32(1)` work
        temp_globals = {"dace": dace, SYMBOL_TABLE_GLOBAL: symbol_table}
        exec(code_fn, temp_globals)

        # no idea why, but simply calling symbolic_execution doesn't work
        results = temp_globals["symbolic_execution"](
            *[connector_symbol(inp, tasklet.in_connectors[inp]) for inp in inputs]
        )

        # pystr_to_symbolic rather than sympify: it keeps `//` as int_floor and mints DaCe symbols.
        if len(outputs) > 1:
            # make sure that everything is a sympy expression. A new list, not in place: a
            # multi-output symbolic_execution returns a tuple.
            normalized = [res if isinstance(res, sp.Expr) else symbolic.pystr_to_symbolic(res) for res in results]
            return dict(zip(outputs, normalized)), indexed_objects_map
        else:
            # make sure that everything is a sympy expression
            if not isinstance(results, sp.Expr):
                results = symbolic.pystr_to_symbolic(results)
            return {outputs[0]: results}, indexed_objects_map
    except Exception as e:
        raise AutoDiffException(
            "Exception occurred while attempting to symbolically execute code:\n{}".format(code)
        ) from e


def is_int_eq_value(value, target_value: int) -> bool:
    if isinstance(value, numbers.Integral):
        return value == target_value

    if len(value.free_symbols) > 0 or int(value) != target_value:
        return False

    return True


def invert_map_connector(conn: str) -> str:
    if conn.startswith("IN"):
        return "OUT" + conn[2:]
    elif conn.startswith("OUT"):
        return "IN" + conn[3:]
    else:
        raise AutoDiffException("Could not parse map connector '{}'".format(conn))


def path_src_node_in_subgraph(edge: dgraph.MultiConnectorEdge, subgraph: dstate.StateSubgraphView) -> bool:
    path_src = subgraph.memlet_path(edge)[0].src
    return path_src in subgraph.nodes()


def get_read_only_arrays(sdfg: SDFG) -> Set[str]:
    """Get the arrays that are only read in SDFG.

    This function identifies arrays that are never written to (only have outgoing
    edges with data or only empty memlets on incoming edges).

    :param sdfg: The SDFG to analyze.
    :return: A set of array names that are read-only in the SDFG.
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
    all_nodes = list(utils.dfs_topological_sort(graph, graph.source_nodes()))
    state_order = []
    for node in all_nodes:
        if isinstance(node, SDFGState):
            state_order.append(node)
        elif isinstance(node, LoopRegion):
            loop_state_order = get_state_topological_order(node)
            state_order.extend(loop_state_order)
        else:
            raise AutoDiffException(
                f"Unsupported node type {node} at the highest level of the SDFG while getting the state order"
            )

    # All states in the graph need to be present in the state order
    if isinstance(graph, SDFG) and set(state_order) != set(graph.states()):
        raise AutoDiffException("Could not find all states of the SDFG in the state order")
    return state_order


def shape_has_symbols_to_replace(sdfg: SDFG, shape: Union[str, sp.Symbol, sp.Expr]) -> bool:
    """
    Check if the shape dimension passed as a parameter has a symbol that needs to be replaced.
    We do not replace global SDFG symbols but rather the loop indices only.
    """
    defined_symbols = sdfg.free_symbols | set(sdfg.arg_names)
    if isinstance(shape, str):
        shape = dace.symbolic.pystr_to_symbolic(shape)
    return dace.symbolic.issymbolic(shape, defined_symbols)


def get_loop_end(start: str, end: str, loop: LoopRegion) -> str:
    """
    Get the smallest and largest index of a loop given the start and end values.
    This is an attempt at estimating the number of iterations of the loop.
    """
    start_sym = dace.symbolic.pystr_to_symbolic(start)
    end_sym = dace.symbolic.pystr_to_symbolic(end)
    if not dace.symbolic.issymbolic(start_sym) and not dace.symbolic.issymbolic(end_sym):
        int_start, int_end = int(start_sym), int(end_sym)
        if int_start < int_end:
            # Increasing loop
            largest_index = int_end
            smallest_index = int_start
        else:
            # Decreasing loop e.g., range(6, -1, -1)
            # Since the start will be the first index there are start+1 iterations
            largest_index = int_start + 1
            smallest_index = int_end
    else:
        # We check using the update statement
        change = analyze_loop_change(loop.update_statement.as_string, loop.loop_variable)
        if change == "increase":
            # Increasing loop
            largest_index = end
            smallest_index = start
        else:
            # Decreasing loop
            # Since the start will be the first index there are start+1 iterations
            largest_index = start + "+1"
            smallest_index = end

    return smallest_index, largest_index


def analyze_loop_change(code: str, loop_variable: str) -> str:
    """Analyze if the given loop variable in the provided code increases or decreases.

    :param code: The Python code to analyze.
    :param loop_variable: The name of the loop variable to analyze.
    :return: ``'increase'``, ``'decrease'``, or ``'unknown'``.
    """
    tree = ast.parse(code)
    change_type = "unknown"

    for node in ast.walk(tree):
        # Look for assignment statements
        if isinstance(node, ast.Assign):
            # Ensure the assignment targets the loop variable
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if target == loop_variable and isinstance(node.value, ast.BinOp):
                    # Check for `loop_variable = loop_variable + ...`
                    if isinstance(node.value.left, ast.Name) and node.value.left.id == loop_variable:
                        # Analyze the right-hand side for increase or decrease
                        rhs = node.value.right
                        if isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.USub):  # Unary negative
                            if isinstance(rhs.operand, ast.Constant) and isinstance(rhs.operand.value, (int, float)):
                                change_type = "decrease"
                        elif isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.UAdd):  # Unary positive
                            if isinstance(rhs.operand, ast.Constant) and isinstance(rhs.operand.value, (int, float)):
                                change_type = "increase"
                        elif isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float)):
                            change_type = "increase" if rhs.value > 0 else "decrease"
    if change_type == "unknown":
        raise AutoDiffException(f"Could not determine loop variable change in code: {code}")
    return change_type


def get_map_nest_information(
    edges_list: List[dstate.MultiConnectorEdge],
) -> Tuple[List, List[str], List, Dict[str, Tuple]]:
    """ """
    # First, get the shape of the new array
    shape_list = []

    # We will also need the starting range of the maps in the path
    start_range = []

    # And the names of the parameters of the maps in the path
    param_list = []

    for e in edges_list:
        edge_src = e.src
        if isinstance(edge_src, nd.MapEntry):
            for rng in edge_src.map.range.ranges:
                # the range contains the last index in the loop
                # while we want the size so we add 1
                shape_list.append(rng[1] + 1)
                start_range.append(rng[0])
            for par in edge_src.map.params:
                param_list.append(par)

    if not (len(param_list) == len(shape_list) == len(start_range)):
        raise AutoDiffException(
            f"Mismatched lengths: params={len(param_list)}, shapes={len(shape_list)}, ranges={len(start_range)}"
        )

    # Create a dictionary mapping parameters to their start and end ranges
    param_dict = {param: (start, end) for param, start, end in zip(param_list, start_range, shape_list)}
    return start_range, param_list, shape_list, param_dict


def get_all_path_edges(
    state: SDFGState, source: nd.Node, starting_edge: dgraph.MultiConnectorEdge
) -> List[dgraph.MultiConnectorEdge]:
    """
    We will start from the target node and go back until we reach the destination.
    Starting edge should be an in node
    """
    all_edges = []
    memlet_path = state.memlet_path(starting_edge)
    all_edges += memlet_path
    first_source = memlet_path[0].src
    if first_source == source:
        return all_edges

    # If there is only one edge coming to the first node
    if state.in_degree(first_source) == 1:
        edge = state.in_edges(first_source)[0]
        memlet_path = state.memlet_path(edge)
        all_edges += memlet_path
        first_source = memlet_path[0].src
        if first_source == source:
            return all_edges

    raise AutoDiffException("Can't easily find path. Upgrade function.")


def extract_conditional_expressions(tasklet_node: nd.Tasklet) -> Tuple[str, Optional[str], str]:
    """
    Given a conditional tasklet node, extract the if and else assignments and the condition. The else assignment is
    None if there is only an if statement. The supported formats are:

    1. ``out = expression_1 if condition else expression_2``, also wrapped in a call such as a cast
       (``out = dace.float32(expression_1 if condition else expression_2)``), which is applied to both branches.
    2. ``if condition: out = expression_1``

    The condition may be any expression of the tasklet's inputs (see :func:`conditional_connectors`).

    :return: A tuple ``(if assignment, else assignment or None, condition)``.
    """
    tasklet_code = tasklet_node.code.as_string
    try:
        body = ast.parse(tasklet_code).body
    except SyntaxError as ex:
        raise AutoDiffException(f"Could not parse conditional tasklet code: {tasklet_code}") from ex
    if len(body) != 1:
        raise AutoDiffException(f"Expected a single conditional statement in tasklet code: {tasklet_code}")
    stmt = body[0]

    if isinstance(stmt, ast.If):
        if stmt.orelse or len(stmt.body) != 1 or not isinstance(stmt.body[0], ast.Assign):
            raise AutoDiffException(f"Expected a single assignment in the if statement of tasklet code: {tasklet_code}")
        out_connector = _single_target(stmt.body[0], tasklet_code)
        if_value, else_value, condition = stmt.body[0].value, None, stmt.test
    elif isinstance(stmt, ast.Assign):
        out_connector = _single_target(stmt, tasklet_code)
        value, wrappers = stmt.value, []
        # Unwrap single-argument calls (e.g., casts) around the conditional expression
        while isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
            wrappers.append(value)
            value = value.args[0]
        if not isinstance(value, ast.IfExp):
            raise AutoDiffException(f"Could not find a conditional expression in tasklet code: {tasklet_code}")
        if_value, else_value, condition = value.body, value.orelse, value.test
        for wrapper in reversed(wrappers):
            if_value = ast.Call(func=wrapper.func, args=[if_value], keywords=[])
            else_value = ast.Call(func=wrapper.func, args=[else_value], keywords=[])
    else:
        raise AutoDiffException(f"Unsupported conditional tasklet code: {tasklet_code}")

    if out_connector not in tasklet_node.out_connectors:
        raise AutoDiffException(
            f"Output connector '{out_connector}' not found in tasklet output connectors: {list(tasklet_node.out_connectors.keys())}"
        )
    conditional = astunparse.unparse(condition).strip()
    conditional_connectors(conditional, tasklet_node)  # Validates the names the condition uses

    if_expression = f"{out_connector} = {astunparse.unparse(if_value).strip()}"
    else_expression = f"{out_connector} = {astunparse.unparse(else_value).strip()}" if else_value is not None else None
    return if_expression, else_expression, conditional


def _single_target(assign: ast.Assign, code: str) -> str:
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        raise AutoDiffException(f"Expected an assignment to a single output connector in tasklet code: {code}")
    return assign.targets[0].id


def conditional_connectors(conditional: str, tasklet_node: nd.Tasklet) -> Set[str]:
    """
    Returns the input connectors that the condition of a conditional tasklet reads (the backward tasklet evaluates the
    same condition, so it needs them as inputs).

    :raises AutoDiffException: If the condition reads names that are neither input connectors nor known functions.
    """
    names = {node.id for node in ast.walk(ast.parse(conditional)) if isinstance(node, ast.Name)}
    connectors = names & set(tasklet_node.in_connectors.keys())
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(conditional))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    unknown = names - connectors - calls - {"True", "False", "None", "dace", "math", "np", "numpy"}
    if unknown and not connectors and not calls:
        raise AutoDiffException(
            f"Conditional '{conditional}' reads {sorted(unknown)}, which are not tasklet input connectors: "
            f"{list(tasklet_node.in_connectors.keys())}"
        )
    return connectors


def check_edges_type_in_state(subgraph: dstate.StateSubgraphView) -> None:
    """
    Check if all the edges in this state are of type float, int, or boolean.
    """
    for edge, parent_subgraph in subgraph.all_edges_recursive():
        if isinstance(parent_subgraph, SDFGState):
            parent_sdfg = parent_subgraph.parent
        elif isinstance(parent_subgraph, dstate.StateSubgraphView):
            parent_sdfg = parent_subgraph.graph.parent
        elif isinstance(parent_subgraph, SDFG) or isinstance(parent_subgraph, LoopRegion):
            # if there are any fancy things on the interstate edges we should probably throw an error
            continue
        else:
            raise AutoDiffException("Unexpected subgraph structure")

        if edge.data.data:
            edge_type = parent_sdfg.arrays[edge.data.data].dtype
            if edge_type in [dace.string]:
                raise AutoDiffException(
                    f"Expected Subgraph to differentiate to only contain float, int, and bool edges, but data {edge.data}"
                    f" on edge {edge} has type {edge_type}"
                )


def state_within_loop(forward_state: SDFGState) -> Tuple[bool, LoopRegion]:
    """
    Check if this state will be executed several times within a loop.
    We check if any of the parents of this state is a loop region.
    """
    parent = forward_state.parent_graph
    while parent is not None:
        if isinstance(parent, LoopRegion):
            return True, parent
        parent = parent.parent_graph
    return False, None


class SympyCleaner(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "pi":
            return ast.copy_location(ast.parse("dace.math.pi").body[0].value, node)
        return self.generic_visit(node)


def loop_bound_condition(loop: LoopRegion) -> str:
    """
    The comparison of a loop's condition that bounds its loop variable: the condition itself, or the term that
    compares the loop variable in a conjunction (``i < end and <condition>``: a loop that may exit before ``end``).
    """
    terms = condition_terms(loop.loop_condition.as_string)
    for term in terms:
        if loop.loop_variable in {str(s) for s in symbolic.pystr_to_symbolic(term).free_symbols}:
            return term
    return loop.loop_condition.as_string


def may_exit_early(loop: LoopRegion) -> bool:
    """Whether a for loop's condition has terms besides the bound of its loop variable (e.g., that read data)."""
    return len(condition_terms(loop.loop_condition.as_string)) > 1


def loop_exit_symbol(loop: LoopRegion) -> Optional[str]:
    """The symbol that the edges after a loop assign the final value of its loop variable to, if any."""
    edges = loop.parent_graph.out_edges(loop)
    for name, value in edges[0].data.assignments.items() if edges else []:
        if value.strip() == loop.loop_variable and all(
            edge.data.assignments.get(name, "").strip() == loop.loop_variable for edge in edges
        ):
            return name
    return None


def extract_loop_region_info(loop: LoopRegion) -> Tuple[str, str]:
    """
    Use regular expression matching to extract the start and end of the loop region.
    We only treat regular for-loops with incrementation and decrementation updates.
    """

    # Extract the loop iterator
    it = loop.loop_variable

    # Extract the end of the loop from the conditional statement
    conditional = loop_bound_condition(loop)

    stride_sign = get_stride_sign(loop)

    # If the stride is positive
    if stride_sign > 0:
        conditional_expression = rf".*{it} < .*"
    else:
        # If the stride is negative
        conditional_expression = rf".*{it} > .*"

    # Match the conditional using regular expressions
    matches = re.search(conditional_expression, conditional)
    if not matches:
        raise AutoDiffException(f"Could not match conditional expression '{conditional_expression}' in '{conditional}'")
    expression = matches.group()
    matches_inner = re.search(conditional_expression[:-2], conditional)
    if not matches_inner:
        raise AutoDiffException(
            f"Could not match conditional pattern '{conditional_expression[:-2]}' in '{conditional}'"
        )
    expression_to_remove = matches_inner.group()
    end = expression.replace(expression_to_remove, "")

    # TODO: need more generalized solution for functions in the loop bounds
    if "floor" not in conditional:
        # There is no function call in the statement, remove parenthesis
        end = end.replace("(", "")
        end = end.replace(")", "")
        end = end.replace(" ", "")
    else:
        if expression_to_remove.startswith("(") and not expression_to_remove.endswith(")") and expression.endswith(")"):
            # Remove extra parenthesis
            end = end[:-1]

    # Get the start from the initialization code
    init_code = loop.init_statement.as_string
    matches = re.search(rf".*{it} = .*", init_code)
    if not matches:
        raise AutoDiffException(f"Could not find initialization pattern for loop variable '{it}' in '{init_code}'")
    expression = matches.group()
    matches = re.search(rf"{it} =", init_code)
    if not matches:
        raise AutoDiffException(f"Could not find assignment pattern for loop variable '{it}' in '{init_code}'")
    expression_to_remove = matches.group()
    start = expression.replace(expression_to_remove, "")

    # Remove parenthesis and space
    start = start.replace("(", "")
    start = start.replace(")", "")
    start = start.replace(" ", "")

    return start, end


def get_stride_sign(loop: LoopRegion) -> int:
    """Check if the stride for this loop is positive or negative.

    :param loop: The loop region to analyze.
    :return: ``1`` if the stride is positive, ``-1`` if negative.
    :raises AutoDiffException: If the loop has an unsupported structure.
    """
    if loop.update_statement is None:
        raise AutoDiffException("While loops are not yet supported in DaCe AD")
    update_statement = loop.update_statement.as_string
    if "-" in update_statement:
        return -1
    if "+" in update_statement:
        return 1

    # unsupported loop structure
    raise AutoDiffException(
        f"Expected the loop region {loop.label} to have a regular update statement. Instead got: {update_statement}"
    )


def strided_extent(shape, strides):
    """The number of elements a strided array spans."""
    return sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1


def reconstruct_view(sdfg: SDFG, target: SDFGState, name: str) -> nodes.AccessNode:
    """Adds the view ``name`` (and the views it views) to ``target``, as defined in another state of ``sdfg``."""
    for state in sdfg.states():
        if state is target:
            continue
        for node in state.data_nodes():
            if node.data != name:
                continue
            edge = utils.get_view_edge(state, node)
            if edge is None:
                continue
            viewed = edge.src if edge.dst is node else edge.dst
            if isinstance(sdfg.arrays[viewed.data], dt.View):
                source = reconstruct_view(sdfg, target, viewed.data)
            else:
                source = target.add_read(viewed.data)
            view = target.add_access(name)
            view.add_in_connector("views")
            target.add_edge(source, None, view, "views", copy.deepcopy(edge.data))
            return view
    raise AutoDiffException(f"Cannot find the definition of view {name}")


def view_as_array(desc: dt.View, transient: bool) -> dt.Array:
    """An array with the shape and layout (strides) of the given view."""
    return dt.Array(
        desc.dtype,
        desc.shape,
        storage=desc.storage,
        strides=desc.strides,
        total_size=strided_extent(desc.shape, desc.strides),
        transient=transient,
    )


def copy_view_at_exit(sdfg: SDFG, name: str) -> str:
    """
    Copies the view ``name`` into a new non-transient array with its layout, in a state appended to ``sdfg``.

    Views only exist in the states that define them, so their values are passed on to the backward pass through such
    a copy.

    :param sdfg: The SDFG that defines the view.
    :param name: The name of the view.
    :return: The name of the copy.
    """
    desc = sdfg.arrays[name]
    sinks = sdfg.sink_nodes()
    if len(sinks) != 1:
        raise AutoDiffException(f"Cannot forward view {name} from an SDFG with several sink blocks")
    state = sdfg.add_state_after(sinks[0], label=f"forward_{name}")
    copy_name = sdfg.add_datadesc(f"{name}_forwarded", view_as_array(desc, transient=False), find_new_name=True)
    subset = ", ".join(f"0:{s}" for s in desc.shape) or "0"
    state.add_nedge(
        reconstruct_view(sdfg, state, name),
        state.add_write(copy_name),
        dace.Memlet(data=name, subset=subset, other_subset=subset),
    )
    return copy_name


def view_definition(sdfg: SDFG, name: str) -> Optional[Tuple[nd.AccessNode, dgraph.MultiConnectorEdge]]:
    """Returns an access node of the view ``name`` in ``sdfg`` together with the edge that defines it, if any."""
    for state in sdfg.all_states():
        for node in state.data_nodes():
            if node.data == name:
                edge = utils.get_view_edge(state, node)
                if edge is not None:
                    return node, edge
    return None


def viewed_input(nsdfg_node: nd.NestedSDFG, name: str) -> Optional[str]:
    """
    Returns the read-only input of a nested SDFG that the view ``name`` inside it views, if there is one (e.g., the
    view a library node expansion makes of an input connector).
    """
    if not isinstance(nsdfg_node.sdfg.arrays.get(name), dt.View):
        return None
    definition = view_definition(nsdfg_node.sdfg, name)
    if definition is None:
        return None
    node, edge = definition
    viewed = edge.src if edge.dst is node else edge.dst
    if viewed.data in nsdfg_node.in_connectors and viewed.data not in nsdfg_node.out_connectors:
        return viewed.data
    return None


def redefine_view(forward: SDFG, backward: SDFG, name: str) -> None:
    """
    Defines the view ``name`` wherever ``backward`` reads it, the way ``forward`` defines it, so that the backward
    SDFG takes the data the view views instead.

    :param forward: The SDFG that defines the view.
    :param backward: The SDFG that reads the view.
    :param name: The name of the view.
    """
    node, edge = view_definition(forward, name)
    viewed = (edge.src if edge.dst is node else edge.dst).data
    if viewed not in backward.arrays:
        desc = copy.deepcopy(forward.arrays[viewed])
        desc.transient = False
        backward.add_datadesc(viewed, desc)
    backward.arrays[name].transient = True
    for state in backward.all_states():
        for view in state.data_nodes():
            if view.data != name or utils.get_view_edge(state, view) is not None:
                continue
            if state.in_degree(view) > 0:
                raise AutoDiffException(f"The backward pass writes the forward view {name}")
            view.add_in_connector("views")
            memlet = copy.deepcopy(edge.data)
            memlet.data = viewed
            state.add_edge(state.add_read(viewed), None, view, "views", memlet)
