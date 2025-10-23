# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import itertools
from typing import Iterator, Tuple, List

import dace.sdfg.nodes as nd
from dace.sdfg import SDFG, SDFGState
from dace.properties import Property, make_properties
from dace.sdfg.graph import MultiConnectorEdge

from dace.libraries.onnx.nodes.node_utils import parse_variadic_param
from dace.libraries.onnx.schema import ONNXSchema, ONNXParameterType


def get_missing_arguments_message(function_name, missing_arguments, argument_type):
    names = list(map(lambda x: "'" + x + "'", missing_arguments))

    if len(missing_arguments) == 1:
        arglist = names[0]
    else:
        arglist = ", ".join(names[:-1]) + ", and " + names[-1]

    return "{function_name} missing {num_missing} required {argument_type}{s}: {arglist}".format(
        function_name=function_name,
        num_missing=len(missing_arguments),
        argument_type=argument_type,
        s='' if len(missing_arguments) == 1 else 's',
        arglist=arglist)


@make_properties
class ONNXOp(nd.LibraryNode):
    """ Abstract superclass for all ONNX ops. Do not use this class, use the concrete subclasses
        (e.g. :class:`~dace.libraries.onnx.nodes.onnx_op.ONNXConv`) instead.
    """

    # Global properties
    # these two are filled out in the generated constructor
    implementations = {}
    default_implementation = None
    default_backward_implementation = None

    # Object fields
    schema = Property(dtype=ONNXSchema, desc="The operator's ONNX OpSchema", allow_none=True)

    backward_implementation = Property(
        dtype=str,
        allow_none=True,
        desc="Which implementation this library node will expand into in the backward pass.")

    def iter_outputs_in_onnx_order(self, state: SDFGState) -> List[MultiConnectorEdge]:
        """ Iterate through the input edges in the same order as they would appear in an ONNX node proto.
            This assumes that the node has been validated!

            :param state: the state containing this node.
            :return: the out edges in the order as they would appear in the node proto.
        """
        return self._iter_params_in_onnx_order(state, inputs=False)

    def iter_inputs_in_onnx_order(self, state: SDFGState) -> List[MultiConnectorEdge]:
        """ Iterate through the output edges in the same order as they would appear in an ONNX node proto.
            This assumes that the node has been validated!

            :param state: the state containing this node.
            :return: the in edges in the order as they would appear in the node proto.
        """
        return self._iter_params_in_onnx_order(state, inputs=True)

    def _iter_params_in_onnx_order(self, state: SDFGState, inputs: bool = False) -> List[MultiConnectorEdge]:
        parameters = list(self.schema.inputs if inputs else self.schema.outputs)
        if len(parameters) == 0:
            return []
        if parameters[-1].param_type == ONNXParameterType.Variadic:
            name = parameters[-1].name
            parameters = itertools.chain([param.name for param in parameters[:-1]],
                                         (name + "__" + str(i) for i in itertools.count()))
        else:
            parameters = [param.name for param in parameters]

        edges = state.in_edges(self) if inputs else state.out_edges(self)
        parameters = list(itertools.islice(parameters, len(edges)))
        conn_to_edge = {edge.dst_conn if inputs else edge.src_conn: edge for edge in edges}

        return [conn_to_edge[name] for name in parameters]

    def iter_edges(
        self,
        state: SDFGState,
        ignore_unknown=False,
    ) -> Iterator[Tuple[MultiConnectorEdge, bool]]:
        """ Returns an iterator over tuples of an edge and a boolean that indicates whether that edge is an input,
            ordered by the order required by the schema.
            This method assumes that this node has been validated.

            :param state: the state containing this node.
            :param ignore_unknown: whether to ignore any edges that don't exist in the ONNX schema. Otherwise, an
                                   error will be thrown.
        """
        in_edges: List[MultiConnectorEdge] = state.in_edges(self)
        out_edges: List[MultiConnectorEdge] = state.out_edges(self)

        def get_idx(parameters, name):
            if '__' in name:
                name, number = parse_variadic_param(name)
            else:
                number = 0

            matched = [i for i, param in enumerate(parameters) if param.name == name]

            if len(matched) != 1:
                if ignore_unknown:
                    return None
                raise ValueError("Found {} connectors with name '{}', expected to find exactly one".format(
                    len(matched), name))

            parameter_idx = matched[0]

            # add on the variadic parameter index
            parameter_idx += number

            return parameter_idx

        if ignore_unknown:
            in_edges = [e for e in in_edges if get_idx(self.schema.inputs, e.dst_conn) is not None]
            out_edges = [e for e in out_edges if get_idx(self.schema.outputs, e.src_conn) is not None]

        sorted_in = sorted(in_edges, key=lambda edge: get_idx(self.schema.inputs, edge.dst_conn))
        sorted_out = sorted(out_edges, key=lambda edge: get_idx(self.schema.outputs, edge.src_conn))

        return itertools.chain(zip(sorted_in, itertools.repeat(True)), zip(sorted_out, itertools.repeat(False)))

    def validate(self, sdfg: SDFG, state: SDFGState):
        """ Validate this node.

            :param sdfg: the parent sdfg.
            :param state: the parent state.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)

        # check that we don't have connectors to None
        all_connectors = {edge.dst_conn for edge in in_edges}.union(edge.src_conn for edge in out_edges)
        if None in all_connectors:
            raise ValueError("Edges to ONNX Ops must not have connector None")

        # check that all edges have connectors
        ##########################################
        for edge, is_input in self.iter_edges(state):
            if is_input:
                conn_name = edge.dst_conn
                if conn_name not in self.in_connectors:
                    raise ValueError("Memlet {} leading to nonexistent input connector '{}'".format(
                        edge.data, conn_name))
            else:
                conn_name = edge.src_conn
                if conn_name not in self.out_connectors:
                    raise ValueError("Memlet {} leading to nonexistent output connector '{}'".format(
                        edge.data, conn_name))

        # check that we have all required in_edges
        ##########################################
        required_inputs = {inp.name for inp in self.schema.inputs if inp.param_type == ONNXParameterType.Single}
        passed_inputs = {inp.dst_conn
                         for inp in in_edges if '__' not in inp.dst_conn}  # we will test variadic inputs separately
        known_inputs = {inp.name for inp in self.schema.inputs}

        missing_inputs = required_inputs.difference(passed_inputs)
        if len(missing_inputs) > 0:
            raise ValueError(get_missing_arguments_message(self.schema.name, missing_inputs, "input"))

        # check that we have all required out_edges
        ##########################################
        required_outputs = {outp.name for outp in self.schema.outputs if outp.param_type == ONNXParameterType.Single}
        passed_outputs = {outp.src_conn
                          for outp in out_edges if '__' not in outp.src_conn}  # we will test variadic inputs separately
        known_outputs = {outp.name for outp in self.schema.outputs}

        missing_outputs = required_outputs.difference(passed_outputs)
        if len(missing_outputs) > 0:
            raise ValueError(get_missing_arguments_message(self.schema.name, missing_outputs, "output"))

        # check that we have no unknown in edges
        ##########################################
        unknown_inputs = passed_inputs.difference(known_inputs)
        if len(unknown_inputs) > 0:
            raise TypeError("Got an unexpected argument '{}'".format(list(unknown_inputs)[0]))

        # check that we have no unknown out edges
        ##########################################
        unknown_outputs = passed_outputs.difference(known_outputs)
        if len(unknown_outputs) > 0:
            raise TypeError("Got an unexpected argument '{}'".format(list(unknown_outputs)[0]))

        # check variadic params
        ##########################################
        variadic_inputs = {inp.name for inp in self.schema.inputs if inp.param_type == ONNXParameterType.Variadic}
        passed_variadic_inputs = {edge.dst_conn for edge in in_edges if '__' in edge.dst_conn}

        seen_variadic_numbers = set()
        for param in passed_variadic_inputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_inputs:
                raise ValueError("Got an unexpected variadic argument '{}'".format(param))
            if number in seen_variadic_numbers:
                raise ValueError("Got two variadic inputs with index {}, expected at most one".format(number))
            seen_variadic_numbers.add(number)

        # check that we have seen every number
        for i in range(len(seen_variadic_numbers)):
            if i not in seen_variadic_numbers:
                raise ValueError(
                    "Since {} variadic inputs were passed, expected variadic parameter with number {}".format(
                        len(seen_variadic_numbers), i))

        variadic_outputs = {outp.name for outp in self.schema.outputs if outp.param_type == ONNXParameterType.Variadic}
        passed_variadic_outputs = {edge.src_conn for edge in out_edges if '__' in edge.src_conn}
        seen_variadic_numbers = set()
        for param in passed_variadic_outputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_outputs:
                raise ValueError("Got an unexpected variadic argument '{}'".format(param))
            if number in seen_variadic_numbers:
                raise ValueError("Got two variadic outputs with index {}, expected at most one".format(number))
            seen_variadic_numbers.add(number)

        # check that we have seen every number
        for i in range(len(seen_variadic_numbers)):
            if i not in seen_variadic_numbers:
                raise ValueError(
                    "Since {} variadic outputs were passed, expected variadic parameter with number {}".format(
                        len(seen_variadic_numbers), i))

        # check that type params solve
        ##########################################

        assigned_params = {}
        for edge, is_input in self.iter_edges(state):
            conn_name = edge.dst_conn if is_input else edge.src_conn

            if '__' in conn_name:
                parsed_name, number = parse_variadic_param(conn_name)
            else:
                parsed_name = conn_name

            matching = [
                inp for inp in (self.schema.inputs if is_input else self.schema.outputs) if inp.name == parsed_name
            ]

            if len(matching) != 1:
                raise ValueError("Expected to find one {} parameter in schema with name '{}', but found {}".format(
                    "input" if is_input else "output", parsed_name, len(matching)))
            matched = matching[0]

            if '__' in conn_name and matched.param_type != ONNXParameterType.Variadic:
                raise ValueError("Got variadic argument '{}' for non-variadic parameter '{}'."
                                 " Ensure that non-variadic args do not contain '__'".format(conn_name, matched.name))

            if '__' not in conn_name and matched.param_type == ONNXParameterType.Variadic:
                raise ValueError(
                    "Expected variadic argument for variadic parameter '{}', got '{}'. Use '{}__i' as the connector"
                    " name, where i is the desired index of the variadic parameter.".format(
                        matched.name, conn_name, conn_name))

            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            # edge_dtype can be a vector type
            if matched.param_type == ONNXParameterType.Variadic and not matched.homogeneous:
                # non homogeneous parameters don't need to be consistent
                pass
            elif matched.type_str in assigned_params and (assigned_params[matched.type_str] != edge_dtype and
                                                          assigned_params[matched.type_str] != edge_dtype.base_type):
                raise ValueError(
                    "Could not solve type constraints;"
                    " excepted type '{expected}' for {param_type} '{conn_name}', got type '{actual}'".format(
                        expected=assigned_params[matched.type_str],
                        param_type="input" if is_input else "output",
                        conn_name=matched.name,
                        actual=edge_dtype))

            # otherwise, matched.type_str was not assigned a type yet: try to assign it
            cons = self.schema.type_constraints[matched.type_str]
            if edge_dtype not in cons.types and edge_dtype.base_type not in cons.types:
                raise ValueError(
                    "Expected type in '{possible}' for {param_type} '{conn_name}', got type '{actual}'".format(
                        possible=cons.types,
                        param_type="input" if is_input else "output",
                        conn_name=matched.name,
                        actual=edge_dtype))
            assigned_params[matched.type_str] = edge_dtype.base_type

        # check that we have all required attributes
        ##########################################
        required_attrs = {name for name, attr in self.schema.attributes.items() if attr.required}
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError("Expected value for required attribute '{}', got None".format(attr))
