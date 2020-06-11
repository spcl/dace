import re
import onnx

from dace.dtypes import typeclass


def convert_onnx_attribute(attribute):
    from dace.libraries.onnx.schema import ONNXAttributeType, ONNXParameter, ONNXAttribute, _KNOWN_ONNX_PROTOS

    if type(attribute) in _KNOWN_ONNX_PROTOS:
        return _KNOWN_ONNX_PROTOS[type(attribute)].from_onnx_proto(attribute)

    if isinstance(attribute, bytes):
        # not sure if this is right...
        return attribute.decode('utf-8')

    if isinstance(attribute, (int, str, bool)):
        return attribute

    if type(attribute) is onnx.defs.OpSchema.FormalParameterOption:
        if attribute == onnx.defs.OpSchema.FormalParameterOption.Single:
            # Single parameters are not optional
            return False
        elif attribute == onnx.defs.OpSchema.FormalParameterOption.Optional:
            return True
        else:
            raise NotImplementedError(
                "Only single and optional formal parameters are supported, got".
                format(attribute))

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
                inv_map[v] = lambda attr: attr.s
            elif k == "STRINGS":
                inv_map[v] = lambda attr: list(attr.strings)

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

    return convert_onnx_attribute(inv_map[onnx_type](proto))


ONNX_DTYPES_TO_NUMPY_STR = {
    'bool': 'bool',
    'int8': 'int8',
    'int16': 'int16',
    'int32': 'int32',
    'int64': 'int64',
    'uint8': 'uint8',
    'uint16': 'uint16',
    'uint32': 'uint32',
    'uint64': 'uint64',
    'float16': 'float16',
    'float': 'float32',
    'double': 'float64',
    'complex64': 'complex64',
    'complex128': 'complex128',
}


def onnx_type_str_onnx_type_str_to_dace_type(onnx_str) -> typeclass:
    """Converts an onnx type string, like tensor(float16) to a dace typeclass"""

    results = re.findall(r"^tensor\((.+)\)", onnx_str)
    if len(results) != 1 or results[0] not in ONNX_DTYPES_TO_NUMPY_STR:
        raise NotImplementedError(
            "Unable to parse ONNX type_str {}".format(onnx_str))

    return ONNX_DTYPES_TO_NUMPY_STR[str(results[0])]
