import itertools
import logging
import collections
import types
from typing import Iterator, Tuple, List, Dict, Type

import dace
import dace.library
import dace.sdfg.nodes as nd
import dace.frontend.common.op_repository as dace_op_repo
from dace.frontend.python.newast import ProgramVisitor
import onnx
from dace.sdfg import SDFG, SDFGState
from dace.properties import Property, ListProperty, make_properties
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation.transformation import ExpandTransformation

from dace.libraries.onnx.environments import ONNXRuntime
from dace.libraries.onnx.nodes.node_utils import parse_variadic_param
from dace.libraries.onnx.schema import ONNXSchema, ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXParameterType, ONNXAttribute, ONNXParameter, ONNXTypeConstraint
from dace.libraries.onnx.nodes.node_codegen import expand_node

log = logging.getLogger(__name__)


def _get_typecons_docstring(cons: ONNXTypeConstraint) -> str:
    return "    * **{}** -- {}".format(
        cons.type_str,
        ", ".join(":class:`{}`".format(t.to_string()) for t in cons.types))


def _get_connector_docstring(param: ONNXParameter) -> str:
    return "    * **{}** ({}, {}) -- {}".format(param.name, param.type_str,
                                                param.param_type.name.lower(),
                                                param.description)


def _get_attr_docstring(attr: ONNXAttribute) -> str:
    param_doc = ":param {}: {}".format(attr.name, attr.description)

    if attr.attribute_type is ONNXAttributeType.Unsupported:
        return ""

    if attr.attribute_type is ONNXAttributeType.Tensor:
        type_string = "numpy.ndarray"
    else:
        type_string = _ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type].__name__

    type_string = ":class:`{}`".format(type_string)

    if attr.attribute_type in [
            ONNXAttributeType.Ints, ONNXAttributeType.Floats,
            ONNXAttributeType.Strings
    ]:
        type_string = ":class:`List` [{}]".format(type_string)

    if not attr.required:
        type_string = ":class:`Optional` [{}], default={}".format(
            type_string, repr(attr.default_value))

    param_type = ":type {}: {}".format(attr.name, type_string)

    return param_doc + "\n" + param_type


def get_missing_arguments_message(function_name, missing_arguments,
                                  argument_type):
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
    schema = Property(dtype=ONNXSchema,
                      desc="The operator's ONNX OpSchema",
                      allow_none=True)

    backward_implementation = Property(
        dtype=str,
        allow_none=True,
        desc=
        "Which implementation this library node will expand into in the backward pass."
    )

    def iter_outputs_in_onnx_order(
            self, state: SDFGState) -> List[MultiConnectorEdge]:
        """ Iterate through the input edges in the same order as they would appear in an ONNX node proto.
            This assumes that the node has been validated!

            :param state: the state containing this node.
            :return: the out edges in the order as they would appear in the node proto.
        """
        return self._iter_params_in_onnx_order(state, inputs=False)

    def iter_inputs_in_onnx_order(
            self, state: SDFGState) -> List[MultiConnectorEdge]:
        """ Iterate through the output edges in the same order as they would appear in an ONNX node proto.
            This assumes that the node has been validated!

            :param state: the state containing this node.
            :return: the in edges in the order as they would appear in the node proto.
        """
        return self._iter_params_in_onnx_order(state, inputs=True)

    def _iter_params_in_onnx_order(
            self,
            state: SDFGState,
            inputs: bool = False) -> List[MultiConnectorEdge]:
        parameters = list(
            self.schema.inputs if inputs else self.schema.outputs)
        if len(parameters) == 0:
            return []
        if parameters[-1].param_type == ONNXParameterType.Variadic:
            name = parameters[-1].name
            parameters = itertools.chain(
                [param.name for param in parameters[:-1]],
                (name + "__" + str(i) for i in itertools.count()))
        else:
            parameters = [param.name for param in parameters]

        edges = state.in_edges(self) if inputs else state.out_edges(self)
        parameters = list(itertools.islice(parameters, len(edges)))
        conn_to_edge = {
            edge.dst_conn if inputs else edge.src_conn: edge
            for edge in edges
        }

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

            matched = [
                i for i, param in enumerate(parameters) if param.name == name
            ]

            if len(matched) != 1:
                if ignore_unknown:
                    return None
                raise ValueError(
                    "Found {} connectors with name '{}', expected to find exactly one"
                    .format(len(matched), name))

            parameter_idx = matched[0]

            # add on the variadic parameter index
            parameter_idx += number

            return parameter_idx

        if ignore_unknown:
            in_edges = [
                e for e in in_edges
                if get_idx(self.schema.inputs, e.dst_conn) is not None
            ]
            out_edges = [
                e for e in out_edges
                if get_idx(self.schema.outputs, e.src_conn) is not None
            ]

        sorted_in = sorted(
            in_edges,
            key=lambda edge: get_idx(self.schema.inputs, edge.dst_conn))
        sorted_out = sorted(
            out_edges,
            key=lambda edge: get_idx(self.schema.outputs, edge.src_conn))

        return itertools.chain(zip(sorted_in, itertools.repeat(True)),
                               zip(sorted_out, itertools.repeat(False)))

    def validate(self, sdfg: SDFG, state: SDFGState):
        """ Validate this node.

            :param sdfg: the parent sdfg.
            :param state: the parent state.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)

        # check that we don't have connectors to None
        all_connectors = {edge.dst_conn
                          for edge in in_edges}.union(edge.src_conn
                                                      for edge in out_edges)
        if None in all_connectors:
            raise ValueError("Edges to ONNX Ops must not have connector None")

        # check that all edges have connectors
        ##########################################
        for edge, is_input in self.iter_edges(state):
            if is_input:
                conn_name = edge.dst_conn
                if conn_name not in self.in_connectors:
                    raise ValueError(
                        "Memlet {} leading to nonexistent input connector '{}'"
                        .format(edge.data, conn_name))
            else:
                conn_name = edge.src_conn
                if conn_name not in self.out_connectors:
                    raise ValueError(
                        "Memlet {} leading to nonexistent output connector '{}'"
                        .format(edge.data, conn_name))

        # check that we have all required in_edges
        ##########################################
        required_inputs = {
            inp.name
            for inp in self.schema.inputs
            if inp.param_type == ONNXParameterType.Single
        }
        passed_inputs = {
            inp.dst_conn
            for inp in in_edges if '__' not in inp.dst_conn
        }  # we will test variadic inputs separately
        known_inputs = {inp.name for inp in self.schema.inputs}

        missing_inputs = required_inputs.difference(passed_inputs)
        if len(missing_inputs) > 0:
            raise ValueError(
                get_missing_arguments_message(self.schema.name, missing_inputs,
                                              "input"))

        # check that we have all required out_edges
        ##########################################
        required_outputs = {
            outp.name
            for outp in self.schema.outputs
            if outp.param_type == ONNXParameterType.Single
        }
        passed_outputs = {
            outp.src_conn
            for outp in out_edges if '__' not in outp.src_conn
        }  # we will test variadic inputs separately
        known_outputs = {outp.name for outp in self.schema.outputs}

        missing_outputs = required_outputs.difference(passed_outputs)
        if len(missing_outputs) > 0:
            raise ValueError(
                get_missing_arguments_message(self.schema.name,
                                              missing_outputs, "output"))

        # check that we have no unknown in edges
        ##########################################
        unknown_inputs = passed_inputs.difference(known_inputs)
        if len(unknown_inputs) > 0:
            raise TypeError("Got an unexpected argument '{}'".format(
                list(unknown_inputs)[0]))

        # check that we have no unknown out edges
        ##########################################
        unknown_outputs = passed_outputs.difference(known_outputs)
        if len(unknown_outputs) > 0:
            raise TypeError("Got an unexpected argument '{}'".format(
                list(unknown_outputs)[0]))

        # check variadic params
        ##########################################
        variadic_inputs = {
            inp.name
            for inp in self.schema.inputs
            if inp.param_type == ONNXParameterType.Variadic
        }
        passed_variadic_inputs = {
            edge.dst_conn
            for edge in in_edges if '__' in edge.dst_conn
        }

        seen_variadic_numbers = set()
        for param in passed_variadic_inputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_inputs:
                raise ValueError(
                    "Got an unexpected variadic argument '{}'".format(param))
            if number in seen_variadic_numbers:
                raise ValueError(
                    "Got two variadic inputs with index {}, expected at most one"
                    .format(number))
            seen_variadic_numbers.add(number)

        # check that we have seen every number
        for i in range(len(seen_variadic_numbers)):
            if i not in seen_variadic_numbers:
                raise ValueError(
                    "Since {} variadic inputs were passed, expected variadic parameter with number {}"
                    .format(len(seen_variadic_numbers), i))

        variadic_outputs = {
            outp.name
            for outp in self.schema.outputs
            if outp.param_type == ONNXParameterType.Variadic
        }
        passed_variadic_outputs = {
            edge.src_conn
            for edge in out_edges if '__' in edge.src_conn
        }
        seen_variadic_numbers = set()
        for param in passed_variadic_outputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_outputs:
                raise ValueError(
                    "Got an unexpected variadic argument '{}'".format(param))
            if number in seen_variadic_numbers:
                raise ValueError(
                    "Got two variadic outputs with index {}, expected at most one"
                    .format(number))
            seen_variadic_numbers.add(number)

        # check that we have seen every number
        for i in range(len(seen_variadic_numbers)):
            if i not in seen_variadic_numbers:
                raise ValueError(
                    "Since {} variadic outputs were passed, expected variadic parameter with number {}"
                    .format(len(seen_variadic_numbers), i))

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
                inp for inp in (
                    self.schema.inputs if is_input else self.schema.outputs)
                if inp.name == parsed_name
            ]

            if len(matching) != 1:
                raise ValueError(
                    "Expected to find one {} parameter in schema with name '{}', but found {}"
                    .format("input" if is_input else "output", parsed_name,
                            len(matching)))
            matched = matching[0]

            if '__' in conn_name and matched.param_type != ONNXParameterType.Variadic:
                raise ValueError(
                    "Got variadic argument '{}' for non-variadic parameter '{}'."
                    " Ensure that non-variadic args do not contain '__'".
                    format(conn_name, matched.name))

            if '__' not in conn_name and matched.param_type == ONNXParameterType.Variadic:
                raise ValueError(
                    "Expected variadic argument for variadic parameter '{}', got '{}'. Use '{}__i' as the connector"
                    " name, where i is the desired index of the variadic parameter."
                    .format(matched.name, conn_name, conn_name))

            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            # edge_dtype can be a vector type
            if matched.param_type == ONNXParameterType.Variadic and not matched.homogeneous:
                # non homogeneous parameters don't need to be consistent
                pass
            elif matched.type_str in assigned_params and (
                    assigned_params[matched.type_str] != edge_dtype and
                    assigned_params[matched.type_str] != edge_dtype.base_type):
                raise ValueError(
                    "Could not solve type constraints;"
                    " excepted type '{expected}' for {param_type} '{conn_name}', got type '{actual}'"
                    .format(expected=assigned_params[matched.type_str],
                            param_type="input" if is_input else "output",
                            conn_name=matched.name,
                            actual=edge_dtype))

            # otherwise, matched.type_str was not assigned a type yet: try to assign it
            cons = self.schema.type_constraints[matched.type_str]
            if edge_dtype not in cons.types and edge_dtype.base_type not in cons.types:
                raise ValueError(
                    "Expected type in '{possible}' for {param_type} '{conn_name}', got type '{actual}'"
                    .format(possible=cons.types,
                            param_type="input" if is_input else "output",
                            conn_name=matched.name,
                            actual=edge_dtype))
            assigned_params[matched.type_str] = edge_dtype.base_type

        # check that we have all required attributes
        ##########################################
        required_attrs = {
            name
            for name, attr in dace_schema.attributes.items() if attr.required
        }
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(
                    "Expected value for required attribute '{}', got None".
                    format(attr))


def register_op_repo_replacement(cls: Type[ONNXOp], cls_name: str,
                                 dace_schema: ONNXSchema):
    @dace_op_repo.replaces("dace.libraries.onnx.{}".format(cls_name))
    def op_repo_replacement(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState,
                            **kwargs):
        attrs = {
            name: value
            for name, value in kwargs.items() if name in dace_schema.attributes
        }
        # remove used attrs
        kwargs = {k: v for k, v in kwargs.items() if k not in attrs}

        onnx_node = cls(name=cls_name, **attrs)
        state.add_node(onnx_node)

        input_names = dace_schema.non_variadic_inputs()
        variadic_inputs = dace_schema.variadic_inputs()

        output_names = dace_schema.non_variadic_outputs()
        variadic_outputs = dace_schema.variadic_outputs()

        inputs = {
            name: arr_name
            for name, arr_name in kwargs.items()
            if (name in input_names or
                # variadic params
                ("__" in name
                 and parse_variadic_param(name)[0] in variadic_inputs))
        }

        kwargs = {k: v for k, v in kwargs.items() if k not in inputs}

        outputs = {
            name: arr_name
            for name, arr_name in kwargs.items()
            if (name in output_names or
                # variadic params
                ("__" in name
                 and parse_variadic_param(name)[0] in variadic_outputs))
        }

        kwargs = {k: v for k, v in kwargs.items() if k not in outputs}

        if len(kwargs) > 0:
            raise TypeError(f"Unknown arguments {', '.join(kwargs)}")

        for inp, arr_name in inputs.items():
            read = state.add_read(arr_name)
            state.add_edge(read, None, onnx_node, inp,
                           sdfg.make_array_memlet(arr_name))
            onnx_node.add_in_connector(inp)

        for outp, arr_name in outputs.items():
            write = state.add_read(arr_name)
            state.add_edge(onnx_node, outp, write, None,
                           sdfg.make_array_memlet(arr_name))
            onnx_node.add_out_connector(outp)
        return []


def _get_schemas_from_version(version: int):
    name_to_schemas = collections.defaultdict(list)
    for schema in onnx.defs.get_all_schemas_with_history():
        name_to_schemas[schema.name].append(schema)

    all_schemas = []
    for name, schemas in name_to_schemas.items():
        schemas = sorted(schemas, key=lambda x: x.since_version)
        while schemas[-1].since_version > version:
            schemas.pop()

        all_schemas.append(schemas[-1])

    return all_schemas


_ONNX_OPS_BY_NAME = {}
# Generate all of the Op Nodes
for schema in _get_schemas_from_version(12):
    try:
        dace_schema = ONNXSchema.from_onnx_proto(schema)
        # if the schema has a parameter name that exists as both an input and an output, prepend "in_" and "out_"
        intersecting_names = set(i.name
                                 for i in dace_schema.inputs).intersection(
                                     o.name for o in dace_schema.outputs)
        for name in intersecting_names:
            in_cands = [i for i in dace_schema.inputs if i.name == name]
            out_cands = [i for i in dace_schema.outputs if i.name == name]
            assert len(in_cands) == len(out_cands) == 1
            in_cands[0].name = "in_" + name
            out_cands[0].name = "out_" + name

    except Exception as e:
        log.debug("Import of {} failed: {}".format(schema.name, e))
        continue

    attrs = {}
    # add properties for each op attribute
    for name, attr in dace_schema.attributes.items():
        if attr.attribute_type in [
                ONNXAttributeType.Int, ONNXAttributeType.String,
                ONNXAttributeType.Float, ONNXAttributeType.Tensor
        ]:
            attrs[name] = Property(
                dtype=_ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type],
                desc=attr.description,
                allow_none=True,
                default=None
                if attr.default_value is None else attr.default_value)
        elif attr.attribute_type in [
                ONNXAttributeType.Ints, ONNXAttributeType.Strings,
                ONNXAttributeType.Floats
        ]:
            attrs[name] = ListProperty(
                element_type=_ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type],
                desc=attr.description,
                allow_none=True,
                default=None
                if attr.default_value is None else attr.default_value)
        elif attr.required:
            raise NotImplementedError(
                "Required attribute '{}' has an unsupported type".format(
                    attr.name))

    required_attrs = {
        name
        for name, attr in dace_schema.attributes.items() if attr.required
    }

    def __init__(self, name, *args, location=None, **op_attributes):
        super(ONNXOp, self).__init__(
            name,
            location=location,
            # add required parameters as in/out connectors, without types for now
            inputs={
                inp.name
                for inp in self.schema.inputs
                if inp.param_type == ONNXParameterType.Single
            },
            outputs={
                out.name
                for out in self.schema.outputs
                if out.param_type == ONNXParameterType.Single
            })
        self.backward_implementation = None

        if len(args) > 0:
            raise TypeError(
                "__init__() takes 1 positional arguments but {} were given".
                format(1 + len(args)))

        missing_arguments = required_attrs.difference(op_attributes)
        if len(missing_arguments) > 0:

            raise TypeError(
                get_missing_arguments_message("__init__()", missing_arguments,
                                              "keyword-only argument"))

        unknown_attrs = set(op_attributes).difference(self.schema.attributes)
        if len(unknown_attrs) > 0:
            raise TypeError(
                "{}.__init__() got an unexpected keyword argument '{}'".format(
                    self.schema.name,
                    list(unknown_attrs)[0]))

        for name, attr in op_attributes.items():
            setattr(self, name, attr)

    input_connector_docstrings = "\n".join(
        _get_connector_docstring(param) for param in dace_schema.inputs)
    output_connector_docstrings = "\n".join(
        _get_connector_docstring(param) for param in dace_schema.outputs)

    cls_name = "ONNX" + dace_schema.name

    # the first line of the init docstring contains the signature of the method. This will be picked up by sphinx and
    # means that the generated sphinx docs have a proper signature, and not just *args, **kwargs.
    init_docstring = "__init__(name, *, {})\n".format(
        ", ".join(attr.name if attr.required else attr.name + "=" +
                  repr(attr.default_value)
                  for _, attr in dace_schema.attributes.items()))
    init_docstring += ":param name: the name of the node.\n" + "\n".join(
        _get_attr_docstring(attr)
        for _, attr in dace_schema.attributes.items())

    docstring = "\n" + dace_schema.doc
    type_docstrings = "\n".join(
        _get_typecons_docstring(cons)
        for _, cons in dace_schema.type_constraints.items())
    docstring += "\n\n"
    docstring += ":Node Inputs:" + input_connector_docstrings
    docstring += "\n\n"
    docstring += ":Node Outputs:" + output_connector_docstrings
    docstring += "\n\n"
    docstring += ":Type Constraints:" + type_docstrings

    attrs['__doc__'] = docstring + "\n"
    attrs['schema'] = dace_schema

    attrs['__init__'] = __init__

    cls = type(cls_name, (ONNXOp, ), attrs)
    cls = dace.library.node(cls)
    cls.__init__.__doc__ = "\n" + init_docstring

    # Register ORT implementation
    ##########################################

    @dace.library.expansion
    class Expansion(ExpandTransformation):
        environments = []

        @classmethod
        def expansion(cls, node, state: SDFGState, sdfg: SDFG):
            result = expand_node(node, state, sdfg)

            if not isinstance(result, SDFG):
                # when we return an SDFG the the environments will be determined recursively by codegen.
                cls.environments = map(dace.library.get_environment,
                                       result.environments)
            return result

    cls.register_implementation('onnxruntime', Expansion)

    # Register pure implementations
    ##########################################

    # avoid import loop
    from dace.libraries.onnx.forward_implementation_abc import ONNXForward

    registered = False
    for impl, args in ONNXForward.extensions().items():
        if "op" in args and args["op"] == schema.name:

            class Expansion(ExpandTransformation):
                environments = []
                forward_impl: ONNXForward = impl

                @classmethod
                def expansion(cls, node, state, sdfg, **kwargs):
                    # validate
                    node.validate(sdfg, state)

                    if cls.forward_impl.forward_can_be_applied(
                            node, state, sdfg):
                        result = cls.forward_impl.forward(
                            node, state, sdfg, **kwargs)
                        if hasattr(cls.forward_impl, "environments"):
                            cls.environments.extend(
                                cls.forward_impl.environments)
                        return result
                    else:
                        # fall back to ORT
                        log.info(
                            'Falling back to onnxruntime expansion for library node "{}". '
                            'Reason: forward_can_be_applied returned False'.
                            format(node.label))
                        result = expand_node(node, state, sdfg)
                        if not isinstance(result, SDFG):
                            # when we return an SDFG the the environments will be determined recursively by codegen.
                            cls.environments = map(
                                dace.library.get_environment,
                                result.environments)
                        return result

            implementation_name = args["name"]
            cls.register_implementation(implementation_name, Expansion)
            registered = True

    if not registered:
        cls.default_implementation = "onnxruntime"

    # register python frontend replacement
    #######################################
    register_op_repo_replacement(cls, cls_name, dace_schema)

    globals()[cls_name] = cls
    _ONNX_OPS_BY_NAME[cls_name] = cls

del cls


def has_onnx_node(name: str) -> bool:
    """ Check if an ONNX operator is supported.

        :param name: the operator name.
    """
    return ("ONNX" + name) in _ONNX_OPS_BY_NAME


def get_onnx_node(name: str) -> ONNXOp:
    """ Get the ONNX Operator node for an operator by name.

        :param name: the operator name
    """
    return _ONNX_OPS_BY_NAME["ONNX" + name]
