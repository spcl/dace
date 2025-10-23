# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Type conversion utilities for ONNX-DaCe integration.

This module provides conversion functions between ONNX and DaCe type systems:
- Converting ONNX protobuf types to DaCe types
- Converting DaCe types to ONNX representation
- Handling ONNX AttributeProto conversions
- Type validation and name sanitization

Key Functions:
- convert_onnx_proto: Convert ONNX protobuf objects to Python/DaCe types
- onnx_tensor_type_to_typeclass: Convert ONNX tensor types to DaCe typeclasses
- typeclass_to_onnx_str: Convert DaCe types to ONNX string representation
- clean_onnx_name: Sanitize ONNX names for valid DaCe identifiers
"""

import logging
import re
from typing import Union

import onnx
from dace import dtypes as dt
from dace.dtypes import typeclass
from onnx.numpy_helper import to_array

log = logging.getLogger(__name__)


def get_proto_attr(proto, name: str):
    """
    Safely access a protobuf attribute with encoding validation.

    This function provides defensive checks against encoding issues when accessing
    protobuf attributes. Python's getattr expects strings, but protobuf uses UTF-8.

    Args:
        proto: The protobuf object to access.
        name: The attribute name to retrieve (must be ASCII).

    Returns:
        The value of the requested attribute.

    Raises:
        ValueError: If the name is not ASCII-encodable.

    Note:
        HasField checks may break in proto3, but ONNX doesn't use proto3 yet.
    """

    def is_ascii(s: str) -> bool:
        """Check if a string is ASCII-encodable."""
        try:
            s.encode('ascii')
        except UnicodeEncodeError:
            return False
        else:
            return True

    if not is_ascii(name):
        raise ValueError(
            f"Attempted to access non-ASCII property name '{name}' on protobuf {proto} (type {type(proto)}). "
            "Please open an issue")

    return getattr(proto, name)


def convert_onnx_proto(attribute):
    from dace.libraries.onnx.schema import ONNXAttributeType, _KNOWN_ONNX_PROTOS, ONNXParameterType

    if type(attribute) in _KNOWN_ONNX_PROTOS:
        return _KNOWN_ONNX_PROTOS[type(attribute)].from_onnx_proto(attribute)

    if isinstance(attribute, (int, str, bool, float)):
        return attribute

    if type(attribute) is onnx.defs.OpSchema.FormalParameterOption:
        if attribute == onnx.defs.OpSchema.FormalParameterOption.Single:
            return ONNXParameterType.Single
        elif attribute == onnx.defs.OpSchema.FormalParameterOption.Optional:
            return ONNXParameterType.Optional
        elif attribute == onnx.defs.OpSchema.FormalParameterOption.Variadic:
            return ONNXParameterType.Variadic
        else:
            raise NotImplementedError(
                "Only single, optional and variadic formal parameters are supported, got".format(attribute))

    if type(attribute) is onnx.defs.OpSchema.AttrType:
        if attribute == onnx.defs.OpSchema.AttrType.FLOAT:
            return ONNXAttributeType.Float
        elif attribute == onnx.defs.OpSchema.AttrType.FLOATS:
            return ONNXAttributeType.Floats
        elif attribute == onnx.defs.OpSchema.AttrType.INT:
            return ONNXAttributeType.Int
        elif attribute == onnx.defs.OpSchema.AttrType.INTS:
            return ONNXAttributeType.Ints
        elif attribute == onnx.defs.OpSchema.AttrType.STRING:
            return ONNXAttributeType.String
        elif attribute == onnx.defs.OpSchema.AttrType.STRINGS:
            return ONNXAttributeType.Strings
        elif attribute == onnx.defs.OpSchema.AttrType.TENSOR:
            return ONNXAttributeType.Tensor
        else:
            log.debug("Got unsupported attribute type {}".format(attribute))
            return ONNXAttributeType.Unsupported

    if type(attribute) is onnx.AttributeProto:
        return convert_attribute_proto(attribute)

    raise NotImplementedError("No conversion implemented for {} (type {})".format(attribute, type(attribute)))


def convert_attribute_proto(proto):
    #  we cache the reverse map as an attribute of the method
    if hasattr(convert_attribute_proto, "inv_map"):
        inv_map = convert_attribute_proto.inv_map
    else:
        inv_map = {}
        for k, v in onnx.AttributeProto.AttributeType.items():
            if k == "FLOAT":
                inv_map[v] = lambda attr: get_proto_attr(attr, "f")
            elif k == "FLOATS":
                inv_map[v] = lambda attr: list(get_proto_attr(attr, "floats"))
            elif k == "INT":
                inv_map[v] = lambda attr: get_proto_attr(attr, "i")
            elif k == "INTS":
                inv_map[v] = lambda attr: list(get_proto_attr(attr, "ints"))
            elif k == "STRING":
                inv_map[v] = lambda attr: get_proto_attr(attr, "s").decode('utf-8')
            elif k == "STRINGS":
                inv_map[v] = lambda attr: list(map(lambda x: x.decode('utf-8'), get_proto_attr(attr, "strings")))
            elif k == "TENSOR":
                inv_map[v] = lambda attr: to_array(get_proto_attr(attr, "t"))

        convert_attribute_proto.inv_map = inv_map

    onnx_type = get_proto_attr(proto, "type")

    if onnx_type == 0:
        # in case of undefined return None
        return None

    if onnx_type not in inv_map:
        type_str = {v: k for k, v in onnx.AttributeProto.AttributeType.items()}[onnx_type]
        raise NotImplementedError(
            "Only FLOAT, FLOATS, INT, INTS, STRING, STRINGS and TENSOR attributes are supported, got attribute with type {}"
            .format(type_str))

    return inv_map[onnx_type](proto)


ONNX_DTYPES_TO_DACE_TYPE_CLASS = {
    'bool': dt.bool,
    'int8': dt.int8,
    'int16': dt.int16,
    'int32': dt.int32,
    'int64': dt.int64,
    'uint8': dt.uint8,
    'uint16': dt.uint16,
    'uint32': dt.uint32,
    'uint64': dt.uint64,
    'float16': dt.float16,
    'float': dt.float32,
    'double': dt.float64,
    'complex64': dt.complex64,
    'complex128': dt.complex128,
}


def typeclass_to_onnx_tensor_type_int(dtype: typeclass) -> int:
    #  we cache the reverse map as an attribute of the method
    if not hasattr(typeclass_to_onnx_tensor_type_int, "inv_map"):
        typeclass_to_onnx_tensor_type_int.inv_map = {
            v: getattr(onnx.TensorProto.DataType, k.upper())
            for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()
        }

    return typeclass_to_onnx_tensor_type_int.inv_map[dtype]


def onnx_tensor_type_to_typeclass(elem_type: int) -> typeclass:
    #  we cache the reverse map as an attribute of the method
    if hasattr(onnx_tensor_type_to_typeclass, "inv_map"):
        inv_map = onnx_tensor_type_to_typeclass.inv_map
    else:
        k: str
        v: int
        inv_map = {}
        for k, v in onnx.TensorProto.DataType.items():
            if k.lower() in ONNX_DTYPES_TO_DACE_TYPE_CLASS:
                inv_map[v] = ONNX_DTYPES_TO_DACE_TYPE_CLASS[k.lower()]

        onnx_tensor_type_to_typeclass.inv_map = inv_map

    if elem_type not in inv_map:
        raise ValueError("Got unsupported ONNX tensor type: {}".format({
            v: k
            for k, v in onnx.TensorProto.DataType.items()
        }[elem_type]))

    return inv_map[elem_type]


def typeclass_to_onnx_str(dtype: typeclass) -> str:
    #  we cache the reverse map as an attribute of the method
    if hasattr(typeclass_to_onnx_str, "inv_map"):
        inv_map = typeclass_to_onnx_str.inv_map
    else:
        inv_map = {v: k for k, v in ONNX_DTYPES_TO_DACE_TYPE_CLASS.items()}

    if dtype not in inv_map:
        raise ValueError("Attempted to convert unsupported dace type to ONNX type: {}".format(dtype))

    return inv_map[dtype]


def onnx_type_str_to_typeclass(onnx_str) -> Union[typeclass, None]:
    """Converts an onnx type string, like tensor(float16) to a dace typeclass"""

    results = re.findall(r"^tensor\((.+)\)", onnx_str)
    if len(results) != 1 or results[0] not in ONNX_DTYPES_TO_DACE_TYPE_CLASS:
        # we return None here, these types will be filtered out later
        return None

    return ONNX_DTYPES_TO_DACE_TYPE_CLASS[str(results[0])]


def clean_onnx_name(name: str) -> str:
    """
    Sanitize an ONNX name to make it a valid DaCe identifier.

    This function transforms ONNX names that may contain invalid characters
    or patterns into valid DaCe identifiers by:
    - Prefixing names starting with digits with "ONNX_"
    - Replacing special characters with their textual equivalents

    Args:
        name: The ONNX name to sanitize.

    Returns:
        A valid DaCe identifier based on the ONNX name.

    Examples:
        >>> clean_onnx_name("123_layer")
        'ONNX_123_layer'
        >>> clean_onnx_name("my.tensor:0")
        'myDOTtensorCOLON0'
    """
    # If the first character is a digit, add the ONNX_ prefix
    if re.match("^[0-9]", name):
        name = f"ONNX_{name}"

    # Replace special characters with their textual equivalents
    return (name.replace(".", "DOT").replace(":", "COLON").replace("/", "SLASH").replace("-", "DASH"))
