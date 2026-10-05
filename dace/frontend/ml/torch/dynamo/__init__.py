# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
TorchDynamo frontend for DaCe.

Captures PyTorch programs through ``torch.compile`` (TorchDynamo + AOTAutograd), lowers the resulting ATen FX graph
into a DaCe schedule tree with native control flow, and compiles it once with symbolic shapes and strides.

Usage::

    import dace.ml
    compiled = dace.ml.compile(model)          # or torch.compile(model, backend='dace', dynamic=True)
    y = compiled(x)
"""
from .backend import DaceBackend, dace_backend
from .capture import CapturedProgram, capture
from .interface import compile
from .training import TrainingStep, training_step

__all__ = ['DaceBackend', 'dace_backend', 'compile', 'capture', 'CapturedProgram', 'TrainingStep', 'training_step']
