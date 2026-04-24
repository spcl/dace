import ast

import pytest

torch = pytest.importorskip('torch', reason='PyTorch not installed. Please install with: pip install dace[ml]')

import dace

from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.schedule_tree.type_inference import ScheduleTreeTypeInference
from dace.data.ml import ParameterArray

pytestmark = [pytest.mark.torch, pytest.mark.autodiff]


def _function_def(source: str) -> ast.FunctionDef:
    module = ast.parse(source)
    node = module.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_torch_autodiff_registry_entries_have_descriptor_inference():
    import dace.frontend.python.replacements.torch_autodiff  # noqa: F401

    assert oprepo.Replacements.get_descriptor_inference('torch.autograd.backward') is not None
    assert oprepo.Replacements.get_method_descriptor_inference('Array', 'requires_grad_') is not None
    assert oprepo.Replacements.get_method_self_descriptor_inference('Array', 'requires_grad_') is not None
    assert oprepo.Replacements.get_method_descriptor_inference('Array', 'backward') is not None
    assert oprepo.Replacements.get_attribute_descriptor_inference('ParameterArray', 'grad') is not None


def test_requires_grad_side_effect_enables_grad_attribute_inference():
    import dace.frontend.python.replacements.torch_autodiff  # noqa: F401

    program = _function_def('def prog(A):\n'
                            '    A.requires_grad_()\n'
                            '    grad = A.grad\n'
                            '    return grad\n')

    inferred = ScheduleTreeTypeInference({'torch': torch}, {'A': dace.data.Array(dace.float32, [4, 5])}).infer(program)

    grad_binding = inferred.get('grad')
    assert grad_binding is not None
    assert isinstance(grad_binding.descriptor, dace.data.Array)
    assert not isinstance(grad_binding.descriptor, ParameterArray)
    assert grad_binding.descriptor.dtype == dace.float32
    assert tuple(grad_binding.descriptor.shape) == (4, 5)


def test_torch_autodiff_descriptor_contracts():
    import dace.frontend.python.replacements.torch_autodiff  # noqa: F401

    infer_backward = oprepo.Replacements.get_descriptor_inference('torch.autograd.backward')
    infer_requires_grad = oprepo.Replacements.get_method_descriptor_inference('Array', 'requires_grad_')
    infer_requires_grad_self = oprepo.Replacements.get_method_self_descriptor_inference('Array', 'requires_grad_')
    infer_backward_method = oprepo.Replacements.get_method_descriptor_inference('Array', 'backward')
    infer_grad = oprepo.Replacements.get_attribute_descriptor_inference('ParameterArray', 'grad')

    array_desc = dace.data.Array(dace.float64, [3, 2], transient=True)

    assert infer_backward({'A': array_desc}, 'A') == ()
    assert infer_requires_grad(array_desc) == ()

    parameter_desc = infer_requires_grad_self(array_desc)
    assert isinstance(parameter_desc, ParameterArray)
    assert parameter_desc.dtype == dace.float64
    assert tuple(parameter_desc.shape) == (3, 2)

    grad_desc = infer_grad(parameter_desc)
    assert isinstance(grad_desc, dace.data.Array)
    assert not isinstance(grad_desc, ParameterArray)
    assert grad_desc.dtype == dace.float64
    assert tuple(grad_desc.shape) == (3, 2)

    assert infer_backward_method(array_desc) == ()


if __name__ == '__main__':
    pytest.main([__file__])
