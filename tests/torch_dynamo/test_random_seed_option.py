# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``random_seed`` option of the ahead-of-time routes: a fixed seed makes random operators deterministic."""

import warnings

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dace  # noqa: E402
import dace.ml  # noqa: E402
from dace.frontend.ml.torch.dynamo.context import RANDOM_SEED_SYMBOL  # noqa: E402

N = dace.symbol("N")


def _ml_available() -> bool:
    try:
        import dace.libraries.ml  # noqa: F401
    except ImportError:
        return False
    return True


needs_ml = pytest.mark.skipif(not _ml_available(), reason="dace.libraries.ml is not available")


@pytest.mark.torch
def test_to_sdfg_random_seed_specializes_symbol():
    drop = nn.Dropout(0.5).train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = dace.ml.to_sdfg(drop, dace.float32[N, 16])
        fixed = dace.ml.to_sdfg(drop, dace.float32[N, 16], random_seed=7)
    assert RANDOM_SEED_SYMBOL in default.free_symbols
    assert RANDOM_SEED_SYMBOL not in fixed.free_symbols and RANDOM_SEED_SYMBOL not in fixed.arglist()


@pytest.mark.torch
@needs_ml
def test_compile_random_seed_is_deterministic():
    drop = nn.Dropout(0.5).train()
    x = torch.rand(4, 512) + 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = dace.ml.compile(drop, dace.float32[N, 512], random_seed=7)
        fresh = dace.ml.compile(drop, dace.float32[N, 512])
    assert torch.equal(fixed(x), fixed(x)), "a fixed seed must reproduce the mask"
    assert not torch.equal(fresh(x), fresh(x)), "without a fixed seed every call draws a new mask"
    kept = fixed(x) != 0
    assert abs(kept.float().mean().item() - 0.5) < 0.05


if __name__ == "__main__":
    test_to_sdfg_random_seed_specializes_symbol()
    if _ml_available():
        test_compile_random_seed_is_deterministic()
