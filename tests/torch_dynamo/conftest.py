# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo import DaceBackend  # noqa: E402


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def backend():
    """A fresh backend instance exposing ``compile_count`` and ``last_sdfg``."""
    return DaceBackend()


def compile_with(backend, fn, **kwargs):
    kwargs.setdefault("dynamic", True)
    return torch.compile(fn, backend=backend, **kwargs)
