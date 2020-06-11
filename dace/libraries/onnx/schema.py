import onnx
import aenum

import dace
from dace.dtypes import typeclass
from dace.properties import make_properties, Property, DictProperty, ListProperty, SetProperty
from dace.libraries.onnx.converters import convert_onnx_attribute, onnx_type_str_onnx_type_str_to_dace_type

_KNOWN_ONNX_PROTOS = {}


def onnx_representation(represents, **mapping):
    """ Decorator for python representations of ONNX protobufs.

        The decorator will monkey patch in the following methods:
        * `__init__` (a constructor based on the class properties)
        * `construct_from_onnx_proto`
        * `construct_from_json`

        :param represents: The onnx protobuf type that the decorated class represents

        :param mapping: a mapping from class property names to either:

                        * a string `s`: In this case, `convert_onnx_attribute` will be applied on the protobuf attribute
                                        with the name `s` to get the property value.
                        * a function `f`: In this case, `f` will be called with the protobuf, and the property value
                                          will be set to the return value of that call.

                        If a property name is not present in `mapping`, the property name itself will be used to access
                         the protobuf attribute.
    """
    def decorator(cls):

        cls = make_properties(cls)

        for name, _ in cls.__properties__.items():
            if name not in mapping:
                mapping[name] = name

        def __init__(self, *args, **kwargs):
            """:param represents: the protobuf that this class represents"""
            for name, prop in self.__properties__.items():
                if len(args) > 0:
                    setattr(self, name, args.pop(0))
                else:
                    setattr(self, name, kwargs[name])
            self._represents = represents

        @classmethod
        def from_onnx_proto(cls, onnx_proto):
            if type(onnx_proto) is not represents:
                raise ValueError(
                    "Unexpected protobuf {}, expected protobuf of type {}".
                    format(onnx_proto, represents))

            constructor_args = {}
            for name, _ in cls.__properties__.items():
                if type(mapping[name]) is str:
                    constructor_args[name] = convert_onnx_attribute(
                        getattr(onnx_proto, mapping[name]))
                else:
                    mapping[name](onnx_proto)
                    constructor_args[name] = mapping[name](onnx_proto)

            return cls(**constructor_args)

        @classmethod
        def from_json(cls, json, parent=None):

            constructor_args = {
                name: prop.from_json(json[name])
                for name, prop in cls.__properties__.items()
            }
            return cls(**constructor_args)

        def to_json(self):
            return dace.serialize.all_properties_to_json(self)

        cls.__init__ = __init__
        cls.from_onnx_proto = from_onnx_proto
        cls.from_json = from_json
        cls.to_json = to_json

        # register so that we're able to load it
        _KNOWN_ONNX_PROTOS[represents] = cls

        return cls

    return decorator


@onnx_representation(onnx.defs.OpSchema.FormalParameter,
                     type_str='typeStr',
                     optional='option')
class ONNXParameter:
    """Python representation of an ONNX parameter"""

    name = Property(dtype=str, desc="The parameter name")
    description = Property(dtype=str, desc="A description of the parameter")
    type_str = Property(dtype=str, desc="The type string of this parameter")
    optional = Property(
        dtype=bool,
        desc=
        "Whether this parameter is optional. Note that variadic parameters are unsupported for now."
    )


class ONNXAttributeType(aenum.AutoNumberEnum):
    Int = ()
    Float = ()
    String = ()
    Ints = ()
    Floats = ()
    Strings = ()


@onnx_representation(onnx.defs.OpSchema.Attribute)
class ONNXAttribute:
    """Python representation of an ONNX attribute"""

    name = Property(dtype=str, desc="The attribute name")
    description = Property(dtype=str, desc="A description this attribute")
    required = Property(dtype=bool, desc="Whether this attribute is required")
    type = Property(choices=ONNXAttributeType,
                    desc="The type of this attribute",
                    default=ONNXAttributeType.Int)
    default_value = Property(dtype=None,
                             desc="The default value of this attribute",
                             default=None,
                             allow_none=True)


@onnx_representation(
    onnx.defs.OpSchema.TypeConstraintParam,
    type_str='type_param_str',
    types=lambda proto: list(
        map(onnx_type_str_onnx_type_str_to_dace_type, proto.allowed_type_strs)))
class ONNXTypeConstraint:

    type_str = Property(dtype=str, desc="The type parameter string")
    types = ListProperty(
        element_type=typeclass,
        desc=
        "The possible types. Note that only tensor types are currently supported."
    )


@onnx_representation(
    onnx.defs.OpSchema,
    inputs=lambda proto: list(map(convert_onnx_attribute, proto.inputs)),
    outputs=lambda proto: list(map(convert_onnx_attribute, proto.outputs)),
    attributes=lambda proto:
    {str(k): convert_onnx_attribute(v)
     for k, v in proto.attributes.items()},
    type_constraints=lambda proto: {
        str(cons.type_param_str): convert_onnx_attribute(cons)
        for cons in proto.type_constraints
    })
class ONNXSchema:
    """Python representation of an ONNX schema"""

    name = Property(dtype=str, desc="The operator name")
    domain = Property(dtype=str, desc="The operator domain")
    doc = Property(dtype=str, desc="The operator's docstring")
    since_version = Property(dtype=int, desc="The version of the operator")
    inputs = ListProperty(element_type=ONNXParameter,
                          desc="The operator input parameter descriptors")
    outputs = ListProperty(element_type=ONNXParameter,
                           desc="The operator output parameter descriptors")
    attributes = DictProperty(key_type=str,
                              value_type=ONNXAttribute,
                              desc="The operator attributes")
    type_constraints = DictProperty(
        key_type=str,
        value_type=ONNXTypeConstraint,
        desc="The type constraints for inputs and outputs")



