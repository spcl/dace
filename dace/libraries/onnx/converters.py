import re
from typing import Union

import onnx

from dace.dtypes import typeclass
from dace import dtypes as dt


def convert_onnx_attribute(attribute):
    from dace.libraries.onnx.schema import ONNXAttributeType, ONNXParameter, ONNXAttribute, _KNOWN_ONNX_PROTOS, ONNXParameterType

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
                "Only single, optional and variadic formal parameters are supported, got"
                .format(attribute))

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
        else:
            raise NotImplementedError(
                "Only FLOAT, FLOATS, INT, INTS, STRING, STRINGS attributes are supported, got {}"
                .format(attribute))

    if type(attribute) is onnx.AttributeProto:
        return convert_attribute_proto(attribute)

    raise NotImplementedError(
        "No conversion implemented for {}".format(attribute))


def convert_attribute_proto(proto):
    #  we cache the reverse map as an attribute of the method
    if hasattr(convert_attribute_proto, "inv_map"):
        inv_map = convert_attribute_proto.inv_map
    else:
        inv_map = {}
        for k, v in onnx.AttributeProto.AttributeType.items():
            if k == "FLOAT":
                inv_map[v] = lambda attr: attr.f
            elif k == "FLOATS":
                inv_map[v] = lambda attr: list(attr.floats)
            elif k == "INT":
                inv_map[v] = lambda attr: attr.i
            elif k == "INTS":
                inv_map[v] = lambda attr: list(attr.ints)
            elif k == "STRING":
                inv_map[v] = lambda attr: attr.s.decode('utf-8')
            elif k == "STRINGS":
                inv_map[v] = lambda attr: list(
                    map(lambda x: x.decode('utf-8'), attr.strings))

        convert_attribute_proto.inv_map = inv_map

    onnx_type = proto.type
    if onnx_type == 0:
        # in case of undefined return None
        return None
    if onnx_type not in inv_map:
        type_str = {v: k
                    for k, v in onnx.AttributeProto.AttributeType.items()
                    }[onnx_type]
        raise NotImplementedError(
            "Only FLOAT, FLOATS, INT, INTS, STRING, STRINGS attributes are supported, got attribute with type {}"
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
    'uint32': dt.int32,
    'uint64': dt.uint64,
    # TODO double check this vs bfloat16
    'float16': dt.float16,
    'float': dt.float32,
    'double': dt.float64,
    'complex64': dt.complex64,
    'complex128': dt.complex128,
}

def onnx_type_str_onnx_type_str_to_dace_type(
        onnx_str) -> Union[typeclass, None]:
    """Converts an onnx type string, like tensor(float16) to a dace typeclass"""

    results = re.findall(r"^tensor\((.+)\)", onnx_str)
    if len(results) != 1 or results[0] not in ONNX_DTYPES_TO_DACE_TYPE_CLASS:
        # we return None here, these types will be filtered out later
        return None

    return ONNX_DTYPES_TO_DACE_TYPE_CLASS[str(results[0])]
