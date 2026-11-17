# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Selection of vendor library-node implementations for GPU programs of the TorchDynamo frontend."""

import contextlib

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.ml  # noqa: E402  (registers the frontend; provides ``dace.ml.compile``)
from dace import dtypes  # noqa: E402
from dace.codegen.common import get_gpu_backend  # noqa: E402
from dace.frontend.ml.torch.dynamo.backend import select_implementations  # noqa: E402
from dace.libraries.torch import Convolution  # noqa: E402
from dace.transformation.auto.auto_optimize import find_fast_library  # noqa: E402


@contextlib.contextmanager
def _forced_gpu_backend(backend: str):
    """Forces the GPU backend to ``cuda`` or ``hip`` (``find_fast_library`` detects it through
    ``get_gpu_backend``, which caches its result, so the cache is cleared on both entry and exit)."""
    with dace.config.temporary_config():
        dace.Config.set("compiler", "cuda", "backend", value=backend)
        get_gpu_backend.cache_clear()
        try:
            yield
        finally:
            get_gpu_backend.cache_clear()


def _build_conv_sdfg(storage: dtypes.StorageType) -> tuple[dace.SDFG, Convolution]:
    """A one-node SDFG with an ATen convolution wired to arrays of the given storage."""
    n = 2
    sdfg = dace.SDFG("aten_conv")
    sdfg.add_array("input", [n, 3, 8, 8], dace.float32, storage=storage)
    sdfg.add_array("weight", [4, 3, 3, 3], dace.float32, storage=storage)
    sdfg.add_array("bias", [4], dace.float32, storage=storage)
    sdfg.add_array("output", [n, 4, 8, 8], dace.float32, storage=storage)
    state = sdfg.add_state()
    node = Convolution(
        "conv",
        target="aten.convolution.default",
        arguments={
            "stride": [1, 1],
            "padding": [1, 1],
            "dilation": [1, 1],
            "transposed": False,
            "output_padding": [0, 0],
            "groups": 1,
        },
    )
    state.add_node(node)
    for name in ("input", "weight", "bias"):
        node.add_in_connector(name)
        state.add_edge(state.add_read(name), None, node, name, dace.Memlet(name))
    node.add_out_connector("output")
    state.add_edge(node, "output", state.add_write("output"), None, dace.Memlet("output"))
    return sdfg, node


@pytest.mark.torch
def test_find_fast_library_knows_the_vendor_convolutions():
    """``find_fast_library`` offers cuDNN for cuda and MIOpen for hip (right after the BLAS library)."""
    with _forced_gpu_backend("cuda"):
        cuda = find_fast_library(dtypes.DeviceType.GPU)
    assert "cuDNN" in cuda
    assert cuda.index("cuDNN") == cuda.index("cuBLAS") + 1

    with _forced_gpu_backend("hip"):
        hip = find_fast_library(dtypes.DeviceType.GPU)
    assert "MIOpen" in hip
    assert hip.index("MIOpen") == hip.index("rocBLAS") + 1


@pytest.mark.torch
@pytest.mark.parametrize(
    "backend,impl,fn",
    [("cuda", "cuDNN", "cudnnConvolutionForward"), ("hip", "MIOpen", "miopenConvolutionForward")],
)
def test_select_implementations_picks_the_vendor_convolution(backend, impl, fn):
    """On a GPU SDFG, ``select_implementations`` chooses the vendor convolution and the code uses its API."""
    with _forced_gpu_backend(backend):
        sdfg, node = _build_conv_sdfg(dtypes.StorageType.GPU_Global)
        select_implementations(sdfg)
        assert node.implementation == impl
        try:
            sdfg.expand_library_nodes()
            code = sdfg.generate_code()
        except Exception as e:  # pragma: no cover - environment dependent
            pytest.skip(f"{impl} codegen needs a GPU toolchain that is absent: {e}")
        joined = "\n".join(c.clean_code for c in code)
        assert fn in joined


@pytest.mark.torch
def test_select_implementations_keeps_cpu_default():
    """On a CPU SDFG, ``select_implementations`` does nothing: the convolution keeps its ``pure`` default."""
    sdfg, node = _build_conv_sdfg(dtypes.StorageType.Default)
    select_implementations(sdfg)
    assert (node.implementation or node.default_implementation) == "pure"


@pytest.mark.gpu
@pytest.mark.torch
def test_conv2d_uses_cudnn_on_gpu():
    """End-to-end: a GPU convolution compiled through the frontend matches eager torch and uses cuDNN."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    torch.manual_seed(0)
    model = nn.Conv2d(3, 4, 3, padding=1).eval().cuda()
    compiled = dace.ml.compile(model)
    with torch.no_grad():
        for batch in (2, 5):
            x = torch.randn(batch, 3, 8, 8, device="cuda")
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)
    sdfg = compiled._dace_backend.last_sdfg
    conv = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Convolution))
    assert conv.implementation == "cuDNN"


if __name__ == "__main__":
    test_find_fast_library_knows_the_vendor_convolutions()
    test_select_implementations_picks_the_vendor_convolution("cuda", "cuDNN", "cudnnConvolutionForward")
    test_select_implementations_picks_the_vendor_convolution("hip", "MIOpen", "miopenConvolutionForward")
    test_select_implementations_keeps_cpu_default()
