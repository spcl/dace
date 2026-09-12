# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for running the ML-frontend unit tests on GPU.

Every ML test is parametrized over :data:`DEVICES` so that the existing CPU behaviour is preserved and an
additional ``[gpu]`` variant is added that is only collected/run under ``pytest -m gpu``.  All GPU variants
drive the **experimental** CUDA code generator (``ExperimentalCUDACodeGen``) via :func:`experimental_cuda`,
since that is the code generator the ``new-gpu-codegen-dev`` line exists to exercise.
"""
import pytest

import dace

# Idiomatic device parametrization (mirrors tests/copynd_test.py): the "gpu" case carries the ``gpu`` marker
# so it is skipped unless the suite is invoked with ``-m gpu`` (and errors if no GPU is available).
DEVICES = ["cpu", pytest.param("gpu", marks=[pytest.mark.gpu])]


def is_gpu(device: str) -> bool:
    return device == "gpu"


def torch_device(device: str) -> str:
    """Map a test ``device`` string to the corresponding torch device string."""
    return "cuda" if is_gpu(device) else "cpu"


def experimental_cuda():
    """Context manager forcing the experimental CUDA code generator for the duration of compile + run.

    Use it around the call that triggers compilation (``sdfg(...)``, ``DaceModule(...)(...)``,
    ``ONNXModel(...)(...)``) so the experimental backend is active while code is generated.
    """
    return dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental')


def run_sdfg(sdfg: dace.SDFG, device: str, **inputs):
    """Run a hand-built SDFG on ``device``.

    On GPU the SDFG is lowered with :meth:`SDFG.apply_gpu_transformations` and executed under the
    experimental code generator.  Host (numpy) inputs are copied to/from the device by the generated
    copy-in/out states, so callers pass the same numpy arrays for both devices.

    Intended to replace a bare ``result = sdfg(**inputs)`` at the end of a raw-SDFG test.
    """
    if is_gpu(device):
        sdfg.apply_gpu_transformations()
        with experimental_cuda():
            return sdfg(**inputs)
    return sdfg(**inputs)
