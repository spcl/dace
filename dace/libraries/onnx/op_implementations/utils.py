# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import inspect
import copy
from typing import Dict, Tuple, Optional, Callable, Union, Any
import textwrap

import dace
from dace import SDFGState, SDFG, dtypes, nodes
from dace.frontend.python.parser import DaceProgram
from dace.registry import autoregister

from dace.libraries.onnx.nodes import onnx_op
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.nodes.node_utils import parse_variadic_param
from dace.sdfg.utils import in_desc_with_name, out_desc_with_name


def _resolve_connector_name(node: onnx_op.ONNXOp, param_name: str, is_input: bool, state: SDFGState) -> Optional[str]:
    """Return the actual connector name on ``node`` for the given schema parameter name.

    Implementation function parameters are named after ONNX schema parameters, but
    SDFG transformations (e.g. nested-SDFG inlining) may rename connectors to avoid
    name conflicts. This helper maps a schema parameter name back to the connector
    that currently represents it by using the node's schema-to-connector mapping.

    :param node: the ONNX node.
    :param param_name: the schema parameter name (may include a ``__i`` suffix for
                       variadic parameters).
    :param is_input: whether this is an input parameter.
    :param state: the state containing the node.
    :return: the connector name, or ``None`` if no matching connected connector exists.
    """
    if "__" in param_name:
        base_name, variadic_index = parse_variadic_param(param_name)
    else:
        base_name, variadic_index = param_name, 0

    edges = state.in_edges(node) if is_input else state.out_edges(node)
    for edge in edges:
        connector_name = edge.dst_conn if is_input else edge.src_conn
        param, number = node._get_schema_parameter(connector_name, is_input)
        if param is not None and param.name == base_name and number == variadic_index:
            return connector_name
    return None


def op_implementation(op, name):
    """A decorator that registers an op implementation.

    It should be used on classes that extend :class:`~dace.libraries.onnx.forward_implementation_abc.ONNXForward`.

    :param op: The ONNX name of the op to register for.
    :param name: The name of the implementation.
    """

    def dec(cls):
        if cls.__doc__ is not None:
            cls.__doc__ +=\
                """
                :Implementation name: ``"{}"``
                """.format(name)
        else:
            cls.__doc__ =\
                """
                :Implementation name: ``"{}"``
                """.format(name)

        return autoregister(cls, op=op, name=name)

    return dec


def program_for_node(program,
                     sdfg: SDFG,
                     state: SDFGState,
                     node: onnx_op.ONNXOp,
                     extra_vars: Optional[Dict[str, Any]] = None) -> SDFG:
    """Expand a function to a DaCe program.

    The dtypes for the arguments will be extracted by matching the parameter names to edges.

    All inputs that are not specified as parameters will be removed using
    constant_folding.remove_node_and_computation.

    :param program: The function to expand into a DaCe program.
    :param sdfg: The parent SDFG.
    :param state: The SDFG state containing the node.
    :param node: The ONNX node to create a program for.
    :param extra_vars: Optional extra variables to add to the program.
    :return: A new SDFG implementing the program.
    """

    from dace.transformation.onnx import constant_folding  # avoid import loop
    input_names = node.schema.non_variadic_inputs()
    variadic_input_names = node.schema.variadic_inputs()

    output_names = node.schema.non_variadic_outputs()
    variadic_output_names = node.schema.variadic_outputs()

    if set(input_names).intersection(output_names):
        # This is currently the case for only one ONNX op
        raise ValueError("program_for_node cannot be applied on nodes of this type;"
                         " '{}' are both an input and an output".format(set(input_names).intersection(output_names)))

    params = inspect.signature(program).parameters
    connectors_to_remove = set(input_names).difference(params)

    # Map from the function's parameter names (schema names) to the actual
    # connector names on the node. SDFG transformations such as nested-SDFG
    # inlining may rename connectors to avoid name conflicts, so the SDFG
    # generated from the implementation function must be rewritten to use the
    # actual connector names before it is attached to the parent node.
    param_to_connector = {}

    annotations = {}
    for name, param in params.items():
        if name in input_names or ("__" in name and parse_variadic_param(name)[0] in variadic_input_names):
            connector_name = _resolve_connector_name(node, name, True, state)
            if connector_name is None:
                raise ValueError("Could not find a connected input connector for parameter '{}' on node {}".format(
                    name, node))
            param_to_connector[name] = connector_name
            annotations[name] = in_desc_with_name(node, state, sdfg, connector_name)
        elif name in output_names or ("__" in name and parse_variadic_param(name)[0] in variadic_output_names):
            connector_name = _resolve_connector_name(node, name, False, state)
            if connector_name is None:
                raise ValueError("Could not find a connected output connector for parameter '{}' on node {}".format(
                    name, node))
            param_to_connector[name] = connector_name
            annotations[name] = out_desc_with_name(node, state, sdfg, connector_name)
        else:
            raise ValueError("'{}' was not found as an input or output for {}".format(name, node.schema.name))

    program.__annotations__ = annotations

    program.__name__ = node.label + "_expansion"
    result = DaceProgram(program, (), {}, False, dace.DeviceType.CPU)
    if extra_vars is not None:
        result.global_vars.update(extra_vars)

    for conn in connectors_to_remove:
        connector_name = _resolve_connector_name(node, conn, True, state)
        if connector_name is not None:
            constant_folding.remove_node_and_computation(sdfg, state, node, connector_name)

    sdfg = result.to_sdfg()

    # Rewrite the expansion SDFG so that its top-level array/connector names
    # match the actual connector names on the parent node.
    sdfg.replace_dict(param_to_connector, replace_keys=True)

    if node.schedule in dtypes.GPU_SCHEDULES:
        sdfg.apply_gpu_transformations()

    return sdfg


def empty_sdfg_for_node(
        sdfg: SDFG,
        state: SDFGState,
        node: onnx_op.ONNXOp,
        add_access_nodes=True) -> Tuple[SDFG, SDFGState, Dict[str, nodes.AccessNode], Dict[str, nodes.AccessNode]]:
    """Given a node, return an SDFG that can be used as a nested SDFG expansion for that node.

    The dtypes for the arguments will be extracted by matching the parameter names to edges.

    :param sdfg: The parent SDFG.
    :param state: The SDFG state containing the node.
    :param node: The ONNX node to create an SDFG for.
    :param add_access_nodes: Whether to add access nodes to the SDFG.
    :return: A tuple containing (nested SDFG, nested state, input nodes dict, output nodes dict).
    """
    nsdfg = SDFG(node.label + "_expansion")
    nstate = nsdfg.add_state()

    input_nodes = {}
    output_nodes = {}
    for edge, is_input in node.iter_edges(state, ignore_unknown=True):
        if is_input:
            conn_name = edge.dst_conn
            nsdfg.add_datadesc(conn_name, copy.deepcopy(in_desc_with_name(node, state, sdfg, conn_name)))
            if add_access_nodes:
                input_nodes[conn_name] = nstate.add_read(conn_name)
        else:
            conn_name = edge.src_conn
            nsdfg.add_datadesc(conn_name, copy.deepcopy(out_desc_with_name(node, state, sdfg, conn_name)))
            if add_access_nodes:
                output_nodes[conn_name] = nstate.add_write(conn_name)
        nsdfg.arrays[conn_name].transient = False

    return nsdfg, nstate, input_nodes, output_nodes


@dace.dtypes.paramdec
def python_pure_op_implementation(func, **compute: Dict[str, Callable]):
    """A decorator that registers a Python op implementation.

    The name of the function will be the name of the op that is being replaced.

    The compute parameter enables you to compute a variable given the node and
    its inputs/outputs. This variable will be namespaced when parsing the function.

    To use this, the argument names of the functions can be either:

    * ``node``, in which case the argument will be passed the node we are expanding,
    * or, the name of any connector of the node, in which case the argument will be
      the data descriptor for that connector

    For example, the following compute argument instantiation will make
    variables ``axis`` and ``shape`` available when the function is parsed.


    .. highlight:: python
    .. code-block:: python

        compute=dict(
            # Grabs the axis of a node
            axis=lambda node: node.axis
            # Grabs the shape of the connector with name 'data'
            shape=lambda data: data.shape
        )

    :param func: The function to register as an implementation
    :param compute: A dictionary of functions that compute variables.
    """

    @op_implementation(op=func.__name__, name="pure")
    class PureImpl(ONNXForward):

        @staticmethod
        def forward(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG) -> Union[nodes.Node, SDFG]:

            def compute_argument_resolver(arg: str):
                if arg == "node":
                    return node
                connector_name = _resolve_connector_name(node, arg, True, state)
                if connector_name is not None:
                    return in_desc_with_name(node, state, sdfg, connector_name)
                connector_name = _resolve_connector_name(node, arg, False, state)
                if connector_name is not None:
                    return out_desc_with_name(node, state, sdfg, connector_name)
                raise ValueError("Got unknown compute argument {}."
                                 " Arguments to compute can be either 'node',"
                                 " or the name of a connector of the node".format(arg))

            extra_vars = {}
            if compute is not None:
                for var_name, function in compute.items():

                    # Get the names of the lambda
                    argument_names = list(inspect.signature(function).parameters)

                    args = map(compute_argument_resolver, argument_names)
                    var_value = function(*args)

                    extra_vars[var_name] = var_value

            return program_for_node(func, sdfg, state, node, extra_vars=extra_vars)

    doc = \
    """
Pure implementation parsed with
:func:`~dace.libraries.onnx.op_implementations.utils.python_pure_op_implementation`.

.. code :: python

"""
    doc += textwrap.indent(inspect.getsource(func), prefix="    ")

    PureImpl.__module__ = func.__module__
    PureImpl.__name__ = func.__name__
    PureImpl.__qualname__ = func.__qualname__
    PureImpl.__doc__ = doc

    return PureImpl
