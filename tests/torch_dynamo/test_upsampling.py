# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Direct nearest-neighbor maps, including overloads, scaling, layout, and GPU storage."""

import math

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402
from torch.fx.experimental.proxy_tensor import make_fx  # noqa: E402

import dace  # noqa: E402
from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402
from dace.frontend.ml.torch.dynamo.decompositions import preserve_native_upsampling  # noqa: E402
from dace.frontend.ml.torch.dynamo.importer import GraphImporter  # noqa: E402
from dace.sdfg import infer_types  # noqa: E402


def _assert_direct(backend):
    assert not any("unsafe_index" in name or "arange" in name for name in backend.last_sdfg.arrays)
    assert sum(isinstance(n, dace.nodes.MapEntry) for n, _ in backend.last_sdfg.all_nodes_recursive()) == 1


@pytest.mark.torch
@pytest.mark.parametrize("nd", [1, 2, 3])
@pytest.mark.parametrize("overload", ["default", "vec"])
@pytest.mark.parametrize("case", ["size", "scales", "same", "double"])
def test_nearest_overloads(backend, nd, overload, case):
    spatial = (5, 6, 7)[:nd]
    if case == "size":
        scales = None
        size = (7, 9, 11)[:nd]
    else:
        scales = (1.7, 0.6, 1.25)[:nd] if case == "scales" else (1.1 if case == "same" else 2.1,) * nd
        size = tuple(math.floor(s * scale) for s, scale in zip(spatial, scales))
    op = getattr(getattr(torch.ops.aten, f"upsample_nearest{nd}d"), overload)

    def f(x):
        if overload == "vec":
            return op(x, list(size) if scales is None else None, list(scales) if scales is not None else None)
        return op(x, list(size), *(scales if scales is not None else [None] * nd))

    x = torch.arange(2 * 3 * math.prod(spatial), dtype=torch.float32).reshape((2, 3) + spatial)
    compiled = torch.compile(f, backend=backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), f(x), rtol=0, atol=0)
    _assert_direct(backend)


@pytest.mark.torch
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.uint8])
def test_nearest_dtype(backend, dtype):
    def f(x):
        return F.interpolate(x, size=(9, 11), mode="nearest")

    x = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).to(dtype)
    compiled = torch.compile(f, backend=backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), f(x), rtol=0, atol=0)
    _assert_direct(backend)


@pytest.mark.torch
@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("layout", ["channels_last", "strided", "singleton"])
def test_nearest_layout(backend, nd, layout):
    def f(x):
        return F.interpolate(x, scale_factor=1.7, mode="nearest", recompute_scale_factor=True)

    x = torch.randn((2, 3) + (5, 7, 9)[:nd])
    if layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last if nd == 2 else torch.channels_last_3d)
    elif layout == "strided":
        x = x.transpose(2, -1)[..., ::2]
    else:
        x = x[:1, :1, :1]
    compiled = torch.compile(f, backend=backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        actual, expected = compiled(x), f(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
    _assert_direct(backend)


@pytest.mark.torch
def test_nearest_gpu_graph():
    # Fake CUDA tensors check device-independent capture and GPU map inference
    # without allocating on a GPU or requiring a CUDA compiler.
    def f(x):
        return torch.ops.aten.upsample_nearest3d.vec(x, [7, 9, 11], None)

    with FakeTensorMode(), preserve_native_upsampling():
        x = torch.empty(2, 3, 4, 5, 6, device="cuda:0")
        gm = make_fx(f)(x)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert len(targets) == 1 and targets[0] in ("aten.upsample_nearest3d.default", "aten.upsample_nearest3d.vec")
    sdfg = GraphImporter(DaceBackend().options).import_graph(gm, [x], "nearest_gpu").sdfg
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.validate()
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert len(maps) == 1 and maps[0].schedule == dace.ScheduleType.GPU_Device
    assert all(desc.storage == dace.StorageType.GPU_Global for desc in sdfg.arrays.values())


@pytest.mark.torch
def test_nearest_preservation_restored_on_error():
    op = torch.ops.aten.upsample_nearest3d.vec
    original = dict(op.py_kernels)
    with pytest.raises(RuntimeError, match="capture failed"), preserve_native_upsampling():
        raise RuntimeError("capture failed")
    assert op.py_kernels == original


@pytest.mark.torch
def test_nearest_training_keeps_autograd_decomposition(backend):
    def f(x):
        return F.interpolate(x, scale_factor=2.0, mode="nearest")

    x = torch.randn(2, 3, 5, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_(True)
    actual = torch.compile(f, backend=backend, dynamic=False, fullgraph=True)(x)
    reference = f(reference_input)
    torch.testing.assert_close(actual, reference)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_input.grad)


@pytest.mark.gpu
@pytest.mark.torch
@pytest.mark.parametrize("nd", [1, 2, 3])
def test_nearest_cuda(backend, nd):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    def f(x):
        return F.interpolate(x, scale_factor=1.7, mode="nearest")

    x = torch.randn((2, 3) + (5, 7, 9)[:nd], device="cuda:0")
    compiled = torch.compile(f, backend=backend, dynamic=False, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), f(x), rtol=0, atol=0)
    _assert_direct(backend)
