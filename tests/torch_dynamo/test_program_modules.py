# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""torch.nn.Module objects used inside @dace.program, converted through the TorchDynamo capture."""
import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
from dace.frontend.ml.torch.dynamo import convertible  # noqa: E402

N = dace.symbol('N')

#: A global tensor read inside a module's forward (not visible to the program itself)
OFFSETS = torch.linspace(0, 1, 4)


class _Scaled(nn.Module):
    """A module whose output depends on Python attributes (guarded by Dynamo) and a buffer."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer('shift', torch.linspace(-1, 1, 4))
        self.factor = 2
        self.mode = 'add'

    def forward(self, x):
        y = self.fc(x) * self.factor
        if self.mode == 'add':
            return y + self.shift
        return y - self.shift


class _TwoOutputs(nn.Module):

    def forward(self, x, n: int):
        return torch.sin(x) + n, x.sum(dim=1)


class _ListDict(nn.Module):

    def forward(self, xs, d):
        return xs[0] + xs[1] * d['w']


class _UsesGlobal(nn.Module):

    def __init__(self):
        super().__init__()
        self.mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # A plain tensor attribute (not a parameter or buffer)

    def forward(self, x):
        return (x + OFFSETS) * self.mask


def _rand(*shape):
    return np.random.rand(*shape).astype(np.float32)


def _eager(module, *args):
    with torch.no_grad():
        result = module(*[torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in args])
    if isinstance(result, tuple):
        return tuple(r.numpy() for r in result)
    return result.numpy()


@pytest.mark.torch
def test_module_symbolic_sizes_compile_once():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()

    @dace.program
    def prog(x: dace.float32[N, 4]):
        return model(x) * 2

    captured = None
    for n in (5, 9, 2):
        x = _rand(n, 4)
        np.testing.assert_allclose(prog(x), _eager(model, x) * 2, rtol=1e-5, atol=1e-5)
        program = convertible.as_sdfg_convertible(model).program
        assert captured is None or program is captured, 'the module was captured again for another size'
        captured = program
    assert 'N' in captured.symbol_names.values()


@pytest.mark.torch
def test_parameters_by_reference():
    torch.manual_seed(1)
    model = nn.Linear(4, 4).eval()

    @dace.program
    def prog(x: dace.float32[N, 4]):
        return model(x)

    x = _rand(3, 4)
    np.testing.assert_allclose(prog(x), _eager(model, x), rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        model.weight.mul_(-1.0)
        model.bias.add_(3.0)
    np.testing.assert_allclose(prog(x), _eager(model, x), rtol=1e-5, atol=1e-5)


@pytest.mark.torch
def test_guards_reparse_on_module_state_change():
    torch.manual_seed(2)
    model = _Scaled().eval()

    @dace.program
    def prog(x: dace.float32[N, 4]):
        return model(x)

    x = _rand(5, 4)
    np.testing.assert_allclose(prog(x), _eager(model, x), rtol=1e-5, atol=1e-5)
    first = convertible.as_sdfg_convertible(model).program
    assert any(g.source is not None and g.source.qualname == 'mode' for g in first.guards)

    model.factor = 3  # Attribute values are constants of the captured graph, guarded by Dynamo
    np.testing.assert_allclose(prog(x), _eager(model, x), rtol=1e-5, atol=1e-5)
    model.mode = 'sub'
    np.testing.assert_allclose(prog(x), _eager(model, x), rtol=1e-5, atol=1e-5)
    assert convertible.as_sdfg_convertible(model).program is not first


@pytest.mark.torch
def test_integer_argument_and_tuple_return():
    module = _TwoOutputs()

    @dace.program
    def prog(x: dace.float32[N, 6], n: dace.int64):
        a, b = module(x, n)
        return a, b

    x = _rand(4, 6)
    a, b = prog(x, 3)
    ref_a, ref_b = _eager(module, x, 3)
    np.testing.assert_allclose(a, ref_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b, ref_b, rtol=1e-5, atol=1e-5)


@pytest.mark.torch
def test_submodule_call_without_annotations():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(6, 5), nn.Tanh())

    @dace.program
    def prog(x):
        return model[0](x) + 1

    x = _rand(3, 6)
    np.testing.assert_allclose(prog(x), _eager(model[0], x) + 1, rtol=1e-5, atol=1e-5)


@pytest.mark.torch
def test_list_and_dict_arguments():
    module = _ListDict()
    weights = _rand(6)

    @dace.program
    def prog(a: dace.float32[N, 6], b: dace.float32[N, 6]):
        return module([a, b], {'w': weights})

    a, b = _rand(3, 6), _rand(3, 6)
    np.testing.assert_allclose(prog(a, b), a + b * weights, rtol=1e-5, atol=1e-6)


@pytest.mark.torch
def test_global_and_attribute_tensors():
    """Tensors that only the module's code reads (globals, plain attributes) are passed and re-evaluated per call."""
    global OFFSETS
    module = _UsesGlobal()

    @dace.program
    def prog(x: dace.float32[N, 4]):
        return module(x)

    original = OFFSETS
    try:
        x = _rand(5, 4)
        np.testing.assert_allclose(prog(x), _eager(module, x), rtol=1e-5, atol=1e-6)
        OFFSETS = torch.full((4, ), 10.0)
        module.mask = torch.tensor([0.0, 2.0, 0.0, 2.0])
        np.testing.assert_allclose(prog(x), _eager(module, x), rtol=1e-5, atol=1e-6)
    finally:
        OFFSETS = original


if __name__ == '__main__':
    test_module_symbolic_sizes_compile_once()
    test_parameters_by_reference()
    test_guards_reparse_on_module_state_change()
    test_integer_argument_and_tuple_return()
    test_submodule_call_without_annotations()
    test_list_and_dict_arguments()
    test_global_and_attribute_tensors()
