import onnx

from dace.libraries.onnx import schema


def test_read_parameter():
    onnx_schema = onnx.defs.get_schema("Conv")
    param0 = schema.ONNXParameter.from_onnx_proto(onnx_schema.inputs[0])
    param2 = schema.ONNXParameter.from_onnx_proto(onnx_schema.inputs[2])

    assert param0.name == 'X'
    assert param0.optional == False
    assert param0.type_str == 'T'

    assert param2.name == 'B'
    assert param2.optional == True
    assert param2.type_str == 'T'

    deserialized0 = schema.ONNXParameter.from_json(param0.to_json())
    assert deserialized0.name == 'X'
    assert deserialized0.optional == False
    assert deserialized0.type_str == 'T'


def test_read_attribute():
    onnx_schema = onnx.defs.get_schema("Conv")
    attr = schema.ONNXAttribute.from_onnx_proto(
        onnx_schema.attributes['auto_pad'])

    assert attr.default_value == 'NOTSET'
    assert attr.name == 'auto_pad'
    assert attr.required == False
    assert attr.type == schema.ONNXAttributeType.String

    deserialized = schema.ONNXAttribute.from_json(attr.to_json())

    assert deserialized.default_value == 'NOTSET'
    assert deserialized.name == 'auto_pad'
    assert deserialized.required == False
    assert deserialized.type == schema.ONNXAttributeType.String


def test_serialize_deserialize():
    onnx_schema = onnx.defs.get_schema("Conv")
    dace_schema = schema.ONNXSchema.from_onnx_proto(onnx_schema)

    assert dace_schema.name == "Conv"
    assert dace_schema.inputs[0].name == 'X'
    assert dace_schema.inputs[2].name == 'B'
    assert dace_schema.inputs[2].type_str == 'T'
    assert dace_schema.attributes['auto_pad'].default_value == 'NOTSET'


    deserialized = schema.ONNXSchema.from_json(dace_schema.to_json())

    assert deserialized.name == "Conv"
    assert deserialized.inputs[0].name == 'X'
    assert deserialized.inputs[2].name == 'B'
    assert deserialized.inputs[2].type_str == 'T'
    assert deserialized.attributes['auto_pad'].default_value == 'NOTSET'

if __name__ == '__main__':
    test_read_parameter()
    test_read_attribute()
    test_serialize_deserialize()
