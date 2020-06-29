from dace.libraries.onnx.nodes.onnx_op import ONNXConv
import pytest


def test_import():
    # just need to run the import statement
    node = ONNXConv("conv")


def test_errors():
    with pytest.raises(TypeError) as e:
        node = ONNXConv("conv", 2)
    assert "takes 2 positional arguments" in str(e.value)

    with pytest.raises(TypeError) as e:
        node = ONNXConv("conv", x=1)

    assert "unexpected" in str(e.value)


if __name__ == '__main__':
    test_import()
    test_errors()
