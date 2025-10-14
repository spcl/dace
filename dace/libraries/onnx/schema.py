# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
ONNX Schema System for DaCe.

This module provides a Python representation layer for ONNX protobuf schemas,
enabling type-safe interaction with ONNX operations in DaCe. It handles:

- Converting ONNX protobuf definitions to Python classes
- Type validation and constraint checking for ONNX operations
- Attribute and parameter schema definitions
- Automatic mapping between ONNX types and DaCe types

Key Components:
- onnx_representation: Decorator for creating Python representations of ONNX protobufs
- ONNXSchema: Complete schema for an ONNX operation
- ONNXAttribute: Attribute definitions (e.g., kernel_shape, strides)
- ONNXParameter: Input/output parameter specifications
- ONNXTypeConstraint: Type constraints for operation parameters
- Enums: ONNXAttributeType, ONNXParameterType for type classification

The schema system enables:
- Compile-time validation of ONNX operations
- Automatic property generation from schemas
- Type-safe conversion between ONNX and DaCe representations
- Integration with DaCe's property system

Example:
    @onnx_representation(onnx.TensorProto)
    class ONNXTensor:
        dims: List[int]
        data_type: int
"""

import logging
from itertools import chain
from typing import List

import aenum
import numpy as np
import onnx

import dace
from dace.dtypes import typeclass
from dace.libraries.onnx.converters import convert_onnx_proto, get_proto_attr, onnx_type_str_to_typeclass
from dace.properties import DictProperty, ListProperty, Property, make_properties

log = logging.getLogger(__name__)

#: Global registry of known ONNX protobuf types and their Python representations
_KNOWN_ONNX_PROTOS = {}


def onnx_representation(represents, **mapping):
    """ Decorator for python representations of ONNX protobufs.

        The decorator will monkey patch in the following methods:

        * ``__init__`` (a constructor based on the class properties)
        * ``construct_from_onnx_proto``
        * ``construct_from_json``

        :param represents: The onnx protobuf type that the decorated class represents

        :param mapping: a mapping from class property names to either:

                        a string ``s``
                            In this case, ``convert_onnx_attribute`` will be applied on the protobuf
                            attribute with the name ``s`` to get the property value.

                        a function ``f``
                            In this case, ``f`` will be called with the protobuf, and the property value
                            will be set to the return value of that call.

                        If a property name is not present in ``mapping``, the property name itself will be used to access
                        the protobuf attribute.
    """

    def decorator(cls):

        cls = make_properties(cls)

        # initialize the mapping with identity
        # this means that by default, we will read the property of the protobuf using the same name as the property name
        for name, _ in cls.__properties__.items():
            if name not in mapping:
                mapping[name] = name

        def __init__(self, *args, **kwargs):
            args = list(args)
            for name, prop in self.__properties__.items():
                if len(args) > 0:
                    # try to init all the positional args first
                    setattr(self, name, args.pop(0))
                else:
                    # then try kwargs
                    setattr(self, name, kwargs[name])
            self._represents = represents
            if hasattr(self, "validate"):
                self.validate()

        @classmethod
        def from_onnx_proto(cls, onnx_proto):

            if type(onnx_proto) is not represents:
                raise ValueError("Unexpected protobuf '{}' (type {}), expected protobuf of type {}".format(
                    onnx_proto, type(onnx_proto), represents))

            constructor_args = {}
            for name, _ in cls.__properties__.items():
                if type(mapping[name]) is str:
                    # if the value of the mapping for that property is a string, read the attribute with that name
                    constructor_args[name] = convert_onnx_proto(get_proto_attr(onnx_proto, mapping[name]))
                else:
                    # the value of the mapping should be a function, apply it to the onnx_proto
                    constructor_args[name] = mapping[name](onnx_proto)

            return cls(**constructor_args)

        @classmethod
        def from_json(cls, json, context=None):

            constructor_args = {
                name: prop.from_json(json[name] if name in json else prop.default)
                for name, prop in cls.__properties__.items()
            }
            return cls(**constructor_args)

        def to_json(self):
            serialized = dace.serialize.all_properties_to_json(self)
            serialized["type"] = cls.__name__
            return serialized

        cls.__init__ = __init__

        # the first line of the init docstring contains the signature of the method. This will be picked up by sphinx
        # and means that the generated sphinx docs have a proper signature, and not just *args, **kwargs.
        init_docstring = "__init__({})\n\n".format(", ".join(name + "=" + repr(prop._default)
                                                             for name, prop in cls.__properties__.items()))

        def get_prop_docstring(name, prop):
            return ":param {}: {}\n:type {}: ``{}``, default ``{}``".format(
                name, prop.__doc__, name,
                prop._dtype.__name__ if prop._dtype is not None else type(prop._default).__name__, repr(prop._default))

        init_docstring += "\n".join(get_prop_docstring(name, prop) for name, prop in cls.__properties__.items())

        cls.__init__.__doc__ = init_docstring

        cls.from_onnx_proto = from_onnx_proto
        cls.from_json = from_json
        cls.to_json = to_json
        from_onnx_proto.__func__.__doc__ = " Construct an object from an ONNX proto of type ``{}``. ".format(represents)
        from_json.__func__.__doc__ = " Construct an object json ".format(represents)
        to_json.__doc__ = " Serialize to json ".format(represents)

        # register so that we're able to load it
        _KNOWN_ONNX_PROTOS[represents] = cls

        return cls

    return decorator


class ONNXParameterType(aenum.AutoNumberEnum):
    Single = ()  #: single/required parameters
    Optional = ()  #: optional parameters
    Variadic = ()  #: variadic parameters


@onnx_representation(onnx.defs.OpSchema.FormalParameter,
                     type_str='type_str',
                     param_type='option',
                     homogeneous="is_homogeneous")
class ONNXParameter:
    """ Python representation of an ONNX parameter. """

    name = Property(dtype=str, desc="The parameter name")
    description = Property(dtype=str, desc="A description of the parameter")
    type_str = Property(dtype=str, desc="The type string of this parameter")
    param_type = Property(choices=ONNXParameterType,
                          desc="The type of the this parameter",
                          default=ONNXParameterType.Single)
    homogeneous = Property(dtype=bool, desc="Whether this parameter is homogeneous")

    def __repr__(self):
        return "{} ({})".format(self.name, str(self.param_type))


class ONNXAttributeType(aenum.AutoNumberEnum):
    Int = ()  #: Integer (python representation is ``int``)
    Float = ()  #: Float (python representation is ``float``)
    String = ()  #: String (python representation is ``str``)
    Ints = ()  #: Ints (python representation is ``List`` [``int``])
    Floats = ()  #: Floats (python representation is ``List`` [``float``])
    Strings = ()  #: Strings (python representation is ``List`` [``str``])
    Tensor = ()  #: Tensor (python representation is ``numpy.ndarray``)
    Unsupported = ()  #: Any unsupported attribute type


_ATTR_TYPE_TO_PYTHON_TYPE = {
    ONNXAttributeType.Int: int,
    ONNXAttributeType.Ints: int,
    ONNXAttributeType.Float: float,
    ONNXAttributeType.Floats: float,
    ONNXAttributeType.String: str,
    ONNXAttributeType.Strings: str,
    ONNXAttributeType.Tensor: np.ndarray
}


@onnx_representation(onnx.defs.OpSchema.Attribute, attribute_type='type')
class ONNXAttribute:
    """ Python representation of an ONNX attribute. """

    name = Property(dtype=str, desc="The attribute name")
    description = Property(dtype=str, desc="A description this attribute")
    required = Property(dtype=bool, desc="Whether this attribute is required")
    attribute_type = Property(choices=ONNXAttributeType,
                              desc="The type of this attribute",
                              default=ONNXAttributeType.Int)
    default_value = Property(dtype=None, desc="The default value of this attribute", default=None, allow_none=True)

    def validate(self):
        if self.required and self.attribute_type == ONNXAttributeType.Unsupported:
            raise NotImplementedError("Required attribute '{}' has an unsupported type".format(self.name))

    def __repr__(self):
        return self.name


@onnx_representation(
    onnx.defs.OpSchema.TypeConstraintParam,
    type_str='type_param_str',
    types=lambda proto: list(
        filter(lambda x: x is not None, map(onnx_type_str_to_typeclass, get_proto_attr(proto, "allowed_type_strs")))))
class ONNXTypeConstraint:
    """ Python representation of an ONNX type constraint. """

    type_str = Property(dtype=str, desc="The type parameter string")
    types = ListProperty(element_type=typeclass,
                         desc="The possible types. Note that only tensor types are currently supported.")

    def __repr__(self):
        return self.type_str


@onnx_representation(
    onnx.defs.OpSchema,
    inputs=lambda proto: list(map(convert_onnx_proto, get_proto_attr(proto, "inputs"))),
    outputs=lambda proto: list(map(convert_onnx_proto, get_proto_attr(proto, "outputs"))),
    attributes=lambda proto: {
        str(k): convert_onnx_proto(v)
        for k, v in get_proto_attr(proto, "attributes").items()
    },
    type_constraints=lambda proto:
    {str(cons.type_param_str): convert_onnx_proto(cons)
     for cons in get_proto_attr(proto, "type_constraints")})
class ONNXSchema:
    """Python representation of an ONNX schema"""

    name = Property(dtype=str, desc="The operator name")
    domain = Property(dtype=str, desc="The operator domain")
    doc = Property(dtype=str, desc="The operator's docstring")
    since_version = Property(dtype=int, desc="The version of the operator")
    attributes = DictProperty(key_type=str,
                              value_type=ONNXAttribute,
                              desc="The operator attributes. Keys should contain the name of the attribute, and values "
                              "should have type :class:`~dace.libraries.onnx.ONNXAttribute`.")
    type_constraints = DictProperty(
        key_type=str,
        value_type=ONNXTypeConstraint,
        desc="The type constraints for inputs and outputs. Keys should contain the type string of the constraint, "
        "values should have type :class:`~dace.libraries.onnx.ONNXTypeConstraint`.")
    inputs = ListProperty(element_type=ONNXParameter,
                          desc="The operator input parameter descriptors. Entries should have type"
                          " :class:`~dace.libraries.onnx.ONNXParameter`.")
    outputs = ListProperty(element_type=ONNXParameter,
                           desc="The operator output parameter descriptors. Entries should have type"
                           " :class:`~dace.libraries.onnx.ONNXParameter`.")

    def __repr__(self):
        return self.domain + "." + self.name

    def non_variadic_inputs(self) -> List[str]:
        return [i.name for i in self.inputs if i.param_type is not ONNXParameterType.Variadic]

    def variadic_inputs(self) -> List[str]:
        return [i.name for i in self.inputs if i.param_type is ONNXParameterType.Variadic]

    def non_variadic_outputs(self) -> List[str]:
        return [i.name for i in self.outputs if i.param_type is not ONNXParameterType.Variadic]

    def variadic_outputs(self) -> List[str]:
        return [i.name for i in self.outputs if i.param_type is ONNXParameterType.Variadic]

    def validate(self):
        # check all parameters with a type str have a entry in the type constraints
        for param in chain(self.inputs, self.outputs):
            if param.type_str not in self.type_constraints:
                # some operators put a type descriptor here. for those, we will try to insert a new type constraint
                cons_name = param.name + "_constraint"
                if cons_name in self.type_constraints:
                    raise ValueError(
                        "Attempted to insert new type constraint, but the name already existed. Please open an issue.")
                parsed_typeclass = onnx_type_str_to_typeclass(param.type_str)

                if parsed_typeclass is None:
                    log.debug("Could not parse typeStr '{}' for parameter '{}'".format(param.type_str, param.name))

                cons = ONNXTypeConstraint(cons_name, [parsed_typeclass] if parsed_typeclass is not None else [])
                self.type_constraints[cons_name] = cons
                param.type_str = cons_name

        # check for required parameters with no supported type
        for param in chain(self.inputs, self.outputs):
            if ((param.param_type == ONNXParameterType.Single or param.param_type == ONNXParameterType.Variadic)
                    and len(self.type_constraints[param.type_str].types) == 0):
                raise NotImplementedError("None of the types for parameter '{}' are supported".format(param.name))

        # check that all variadic parameter names do not contain "__"
        for param in chain(self.inputs, self.outputs):
            if param.param_type == ONNXParameterType.Variadic and "__" in param.name:
                raise ValueError(
                    "Unsupported parameter name '{}': variadic parameter names must not contain '__'".format(
                        param.name))

        # check that all inputs and outputs have unique names
        seen = set()
        for param in self.inputs:
            if param.name in seen:
                raise ValueError("Got duplicate input parameter name '{}'".format(param.name))
            seen.add(param.name)

        seen = set()
        for param in self.outputs:
            if param.name in seen:
                raise ValueError("Got duplicate output parameter name '{}'".format(param.name))
            seen.add(param.name)
