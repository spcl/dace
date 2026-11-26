# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dynamic_shapes``: user control over which sizes are symbolic, and the names of the resulting DaCe symbols."""

import warnings

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.export import Dim  # noqa: E402

from dace.frontend.ml.torch import dynamo  # noqa: E402


def _run(compiled, fn, inputs_per_shape, **assert_kwargs):
    with torch.no_grad():
        for inputs in inputs_per_shape:
            ref = fn(*inputs)
            out = compiled(*inputs)
            torch.testing.assert_close(out, ref, **assert_kwargs)
    return compiled._dace_backend


def _symbols(backend):
    return set(backend.last_sdfg.free_symbols) | set(backend.last_sdfg.symbols)


@pytest.mark.torch
def test_all_symbolic_defeats_duck_shaping():

    def f(x, y):
        return (x * 2).sum() + y.sum(), x + 1

    compiled = dynamo.compile(f, dynamic_shapes="all")
    # Equal sizes in the first call would normally share one symbol and force a recompile for the second call
    backend = _run(compiled, f, [(torch.randn(4, 4), torch.randn(4, 4)), (torch.randn(5, 6), torch.randn(7, 8))])
    assert backend.compile_count == 1
    assert {"x_dim0", "x_dim1", "y_dim0", "y_dim1"} <= _symbols(backend)


@pytest.mark.torch
def test_named_dimensions_and_dim_objects():
    batch = Dim("batch", min=2, max=64)

    def f(x, w):
        return torch.relu(x @ w)

    compiled = dynamo.compile(f, dynamic_shapes={"x": {0: batch, 1: "features"}, "w": {0: "features", 1: "hidden"}})
    backend = _run(compiled, f, [(torch.randn(4, 5), torch.randn(5, 5)), (torch.randn(6, 7), torch.randn(7, 3))])
    assert backend.compile_count == 1
    syms = _symbols(backend)
    assert {"batch", "features", "hidden"} <= syms
    # x.shape[1] and w.shape[0] are tied by the matmul: one symbol, one name
    assert "features_2" not in syms


@pytest.mark.torch
def test_positional_specification():

    def f(x, y):
        return x + y.sum()

    compiled = dynamo.compile(f, dynamic_shapes=({0: "rows", 1: "cols"}, "all"))
    backend = _run(compiled, f, [(torch.randn(4, 5), torch.randn(4, 5)), (torch.randn(2, 3), torch.randn(6, 7))])
    assert backend.compile_count == 1
    assert {"rows", "cols", "y_dim0", "y_dim1"} <= _symbols(backend)


@pytest.mark.torch
def test_size_one_dims_stay_static_in_all_mode():

    def f(x, bias):
        return x + bias  # bias broadcasts over the leading dimension

    compiled = dynamo.compile(f, dynamic_shapes="all")
    backend = _run(compiled, f, [(torch.randn(4, 5), torch.randn(1, 5)), (torch.randn(3, 6), torch.randn(1, 6))])
    assert backend.compile_count == 1
    syms = _symbols(backend)
    assert "bias_dim0" not in syms
    # The broadcast ties bias.shape[1] to x.shape[1]: one symbol, named after the first argument
    assert {"x_dim0", "x_dim1"} <= syms and "bias_dim1" not in syms


@pytest.mark.torch
def test_module_stays_module_and_weights_static():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    compiled = dynamo.compile(model, dynamic_shapes="all")
    assert isinstance(compiled, nn.Module)
    assert compiled.training is False
    backend = _run(compiled, model, [(torch.randn(8, 8),), (torch.randn(3, 8),)], rtol=1e-4, atol=1e-5)
    assert backend.compile_count == 1
    assert "input_dim0" in _symbols(backend)
    # Parameters are static: some array has the concrete weight shape (6, 8)
    assert any(tuple(d.shape) == (6, 8) for d in backend.last_sdfg.arrays.values())


@pytest.mark.torch
def test_integer_argument_named():

    def f(x, k):
        return x * k + k

    compiled = dynamo.compile(f, dynamic_shapes={"x": "all", "k": "scale"})
    backend = _run(compiled, f, [(torch.randn(4, 5), 3), (torch.randn(2, 7), 5)])
    assert backend.compile_count == 1
    assert "scale" in _symbols(backend)


@pytest.mark.torch
def test_nested_arguments():

    def f(xs):
        return xs[0] * 2 + xs[1].sum()

    compiled = dynamo.compile(f, dynamic_shapes={"xs": [{0: "a", 1: "b"}, "all"]})
    backend = _run(compiled, f, [([torch.randn(4, 5), torch.randn(4, 5)],), ([torch.randn(2, 3), torch.randn(6, 7)],)])
    assert backend.compile_count == 1
    assert {"a", "b", "xs_1_dim0", "xs_1_dim1"} <= _symbols(backend)


@pytest.mark.torch
def test_duplicate_name_without_constraint_warns():

    def f(x, y):
        return x.sum() + y.sum()

    compiled = dynamo.compile(f, dynamic_shapes={"x": {0: "batch"}, "y": {0: "batch"}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = _run(compiled, f, [(torch.randn(4, 2), torch.randn(4, 3)), (torch.randn(5, 2), torch.randn(9, 3))])
    assert backend.compile_count == 1
    assert any("batch" in str(w.message) for w in caught)
    assert {"batch", "batch_2"} <= _symbols(backend)


@pytest.mark.torch
def test_unknown_argument_rejected():
    with pytest.raises(ValueError, match="unknown argument"):
        dynamo.compile(lambda x: x, dynamic_shapes={"z": "all"})


if __name__ == "__main__":
    for test in (
        test_all_symbolic_defeats_duck_shaping,
        test_named_dimensions_and_dim_objects,
        test_positional_specification,
        test_size_one_dims_stay_static_in_all_mode,
        test_module_stays_module_and_weights_static,
        test_integer_argument_named,
        test_nested_arguments,
        test_duplicate_name_without_constraint_warns,
        test_unknown_argument_rejected,
    ):
        torch._dynamo.reset()
        test()
        print(test.__name__, "ok")
