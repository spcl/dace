# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import logging
import collections
from typing import Iterator, Tuple, List, Dict, Type

import dace
import dace.library
import dace.sdfg.nodes as nd
import dace.frontend.common.op_repository as dace_op_repo
from dace.frontend.python.newast import ProgramVisitor
from dace import SDFG, SDFGState, dtypes, data
from dace.properties import Property, ListProperty, make_properties
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation.transformation import ExpandTransformation

from dace.libraries.onnx.nodes.node_utils import parse_variadic_param
from dace.libraries.onnx.schema import ONNXSchema, ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXParameterType, ONNXAttribute, ONNXParameter, ONNXTypeConstraint
from dace.libraries.onnx.forward_implementation_abc import ONNXForward

import dace.libraries.onnx.nodes.onnx_op as onnx_op
from dace.frontend.python.common import StringLiteral

import onnx

# This import is necessary, it registers implementations that will appear in ONNXForward.extensions()
import dace.libraries.onnx.op_implementations

log = logging.getLogger(__name__)


def _get_typecons_docstring(cons: ONNXTypeConstraint) -> str:
    """Generate documentation string for type constraints."""
    return "    * **{}** -- {}".format(cons.type_str,
                                       ", ".join(":class:`{}`".format(t.to_string()) for t in cons.types))


def _get_connector_docstring(param: ONNXParameter) -> str:
    """Generate documentation string for connectors."""
    return "    * **{}** ({}, {}) -- {}".format(param.name, param.type_str, param.param_type.name.lower(),
                                                param.description)


def _get_attr_docstring(attr: ONNXAttribute) -> str:
    """Generate documentation string for attributes."""
    param_doc = ":param {}: {}".format(attr.name, attr.description)

    if attr.attribute_type is ONNXAttributeType.Unsupported:
        return ""

    if attr.attribute_type is ONNXAttributeType.Tensor:
        type_string = "numpy.ndarray"
    else:
        type_string = _ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type].__name__

    type_string = ":class:`{}`".format(type_string)

    if attr.attribute_type in [ONNXAttributeType.Ints, ONNXAttributeType.Floats, ONNXAttributeType.Strings]:
        type_string = ":class:`List` [{}]".format(type_string)

    if not attr.required:
        type_string = ":class:`Optional` [{}], default={}".format(type_string, repr(attr.default_value))

    param_type = ":type {}: {}".format(attr.name, type_string)

    return param_doc + "\n" + param_type


def _get_all_schemas():
    """Get all ONNX schemas with version history."""
    name_to_schemas = collections.defaultdict(list)
    for schema in onnx.defs.get_all_schemas_with_history():
        name_to_schemas[schema.name].append(schema)

    all_schemas = []
    for name, schemas in name_to_schemas.items():
        all_schemas.extend(schemas)

    return all_schemas


def register_op_repo_replacement(cls: Type[onnx_op.ONNXOp], cls_name: str, dace_schema: ONNXSchema):
    """Register an op repository replacement for the given ONNX operation class."""

    @dace_op_repo.replaces("dace.libraries.onnx.{}".format(cls_name))
    def op_repo_replacement(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, **kwargs):
        attrs = {name: value for name, value in kwargs.items() if name in dace_schema.attributes}
        # Remove used attrs
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
                ("__" in name and parse_variadic_param(name)[0] in variadic_inputs))
        }

        kwargs = {k: v for k, v in kwargs.items() if k not in inputs}

        outputs = {
            name: arr_name
            for name, arr_name in kwargs.items()
            if (name in output_names or
                # variadic params
                ("__" in name and parse_variadic_param(name)[0] in variadic_outputs))
        }

        kwargs = {k: v for k, v in kwargs.items() if k not in outputs}

        if len(kwargs) > 0:
            raise TypeError(f"Unknown arguments {', '.join(kwargs)}")

        # Remove all non-string attributes
        # Sometimes constants are passed as inputs, but they do not require AccessNodes
        # so we add them first as attributes to the node
        for inp, arr_name in inputs.items():
            if not isinstance(arr_name, str):
                setattr(onnx_node, inp, arr_name)

        inputs = {inp: arr_name for inp, arr_name in inputs.items() if isinstance(arr_name, str)}

        for inp, arr_name in inputs.items():
            read = state.add_read(arr_name)
            state.add_edge(read, None, onnx_node, inp, sdfg.make_array_memlet(arr_name))
            onnx_node.add_in_connector(inp)

        for outp, arr_name in outputs.items():
            write = state.add_read(arr_name)
            state.add_edge(onnx_node, outp, write, None, sdfg.make_array_memlet(arr_name))
            onnx_node.add_out_connector(outp)
        return []


_ONNX_OPS = {}

# Generate all of the Op Nodes
for schema in _get_all_schemas():
    try:
        dace_schema = ONNXSchema.from_onnx_proto(schema)
        # If the schema has a parameter name that exists as both an input and an output, prepend "in_" and "out_"
        intersecting_names = set(i.name for i in dace_schema.inputs).intersection(o.name for o in dace_schema.outputs)
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
    # Add properties for each op attribute
    for name, attr in dace_schema.attributes.items():
        if attr.attribute_type in [
                ONNXAttributeType.Int, ONNXAttributeType.String, ONNXAttributeType.Float, ONNXAttributeType.Tensor
        ]:
            attrs[name] = Property(dtype=_ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type],
                                   desc=attr.description,
                                   allow_none=True,
                                   default=None if attr.default_value is None else attr.default_value)
        elif attr.attribute_type in [ONNXAttributeType.Ints, ONNXAttributeType.Strings, ONNXAttributeType.Floats]:
            attrs[name] = ListProperty(element_type=_ATTR_TYPE_TO_PYTHON_TYPE[attr.attribute_type],
                                       desc=attr.description,
                                       allow_none=True,
                                       default=None if attr.default_value is None else attr.default_value)
        elif attr.required:
            raise NotImplementedError("Required attribute '{}' has an unsupported type".format(attr.name))

    required_attrs = {name for name, attr in dace_schema.attributes.items() if attr.required}

    def __init__(self, name, *args, location=None, optional=set(), **op_attributes):
        super(onnx_op.ONNXOp, self).__init__(
            name,
            location=location,
            # Add required parameters as in/out connectors, without types for now
            inputs={
                inp.name
                for inp in self.schema.inputs if inp.param_type == ONNXParameterType.Single or (
                    inp.name in optional and inp.param_type == ONNXParameterType.Optional)
            },
            outputs={
                out.name
                for out in self.schema.outputs if out.param_type == ONNXParameterType.Single or (
                    out.name in optional and out.param_type == ONNXParameterType.Optional)
            })
        self.backward_implementation = None

        if len(args) > 0:
            raise TypeError("__init__() takes 1 positional arguments but {} were given".format(1 + len(args)))

        missing_arguments = required_attrs.difference(op_attributes)
        if len(missing_arguments) > 0:

            raise TypeError(
                onnx_op.get_missing_arguments_message("__init__()", missing_arguments, "keyword-only argument"))

        unknown_attrs = set(op_attributes).difference(self.schema.attributes)
        if len(unknown_attrs) > 0:
            raise TypeError("{}.__init__() got an unexpected keyword argument '{}'".format(
                self.schema.name,
                list(unknown_attrs)[0]))

        for name, attr in op_attributes.items():
            if isinstance(attr, StringLiteral):
                attr = attr.value
            setattr(self, name, attr)

    input_connector_docstrings = "\n".join(_get_connector_docstring(param) for param in dace_schema.inputs)
    output_connector_docstrings = "\n".join(_get_connector_docstring(param) for param in dace_schema.outputs)

    cls_name = "ONNX" + dace_schema.name

    # The first line of the init docstring contains the signature of the method. This will be picked up by sphinx and
    # means that the generated sphinx docs have a proper signature, and not just *args, **kwargs.
    init_docstring = "__init__(name, *, {})\n".format(", ".join(attr.name if attr.required else attr.name + "=" +
                                                                repr(attr.default_value)
                                                                for _, attr in dace_schema.attributes.items()))
    init_docstring += ":param name: The name of the node.\n" + "\n".join(
        _get_attr_docstring(attr) for _, attr in dace_schema.attributes.items())

    docstring = "\n" + dace_schema.doc
    type_docstrings = "\n".join(_get_typecons_docstring(cons) for _, cons in dace_schema.type_constraints.items())
    docstring += "\n\n"
    docstring += ":Node Inputs:" + input_connector_docstrings
    docstring += "\n\n"
    docstring += ":Node Outputs:" + output_connector_docstrings
    docstring += "\n\n"
    docstring += ":Type Constraints:" + type_docstrings

    attrs['__doc__'] = docstring + "\n"
    attrs['schema'] = dace_schema

    attrs['__init__'] = __init__

    cls_name_ver = cls_name + "_" + str(dace_schema.since_version)

    cls = type(cls_name_ver, (onnx_op.ONNXOp, ), attrs)
    cls = dace.library.node(cls)
    cls.__init__.__doc__ = "\n" + init_docstring

    # Register pure implementations
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

                    if cls.forward_impl.forward_can_be_applied(node, state, sdfg):
                        result = cls.forward_impl.forward(node, state, sdfg, **kwargs)
                        if hasattr(cls.forward_impl, "environments"):
                            cls.environments.extend(cls.forward_impl.environments)
                        return result

            implementation_name = args["name"]
            cls.register_implementation(implementation_name, Expansion)
            registered = True

    if not registered:
        # WARNING: No implementation found for this op
        cls.default_implementation = None

    version = schema.since_version

    if cls_name not in _ONNX_OPS:
        _ONNX_OPS[cls_name] = {}
    _ONNX_OPS[cls_name][version] = cls

for name, ver_to_cls in _ONNX_OPS.items():
    _ONNX_OPS[name] = dict(sorted(ver_to_cls.items()))
    for i, (version, cls) in enumerate(_ONNX_OPS[name].items()):
        if i == len(_ONNX_OPS[name]) - 1:
            # last version registered as the default
            globals()[name] = cls
            # register python frontend replacement
            register_op_repo_replacement(cls, name, cls.schema)
        # all other versions are registered with version as a suffix
        globals()[name + "_" + str(version)] = cls


def has_onnx_node(name: str) -> bool:
    """Check if an ONNX operator is supported.

    :param name: The operator name.
    :return: True if the operator is supported, False otherwise.
    """
    return ("ONNX" + name) in _ONNX_OPS


def get_onnx_node(name: str, version: int = -1) -> onnx_op.ONNXOp:
    """Get the ONNX Operator node for an operator by name.

    :param name: The operator name.
    :param version: The version of the operator (-1 for latest).
    :return: The ONNX operator node class.
    :raises ValueError: If no version of the operator is found for the given version.
    """
    name_to_versions = list(_ONNX_OPS["ONNX" + name].items())

    if version == -1:
        # Take the latest version
        return name_to_versions[-1][1]
    else:
        # Take the latest version which is less than or equal to the given version
        for ver, cls in reversed(name_to_versions):
            if ver <= version:
                return cls
        raise ValueError(f"No version of {name} found for version {version}")
