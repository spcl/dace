# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
EXPERIMENTAL: bytecode-level capture of data-dependent Python control flow for the ``dace`` TorchDynamo backend.

This package is a prototype and is not part of the supported frontend API; its entry point,
:class:`~dace.frontend.ml.torch.dynamo.cfg.goto_capture.CfgDaceBackend`, is a subclass of the production backend
that installs (process-wide, backend-gated) patches of Dynamo's conditional-jump handlers so that data-dependent
``if``/``while`` become ``torch.cond``/``torch.while_loop`` instead of graph breaks.
"""
from .goto_capture import CaptureEvent, CaptureState, CfgDaceBackend, ControlFlowCapture

__all__ = ['CfgDaceBackend', 'ControlFlowCapture', 'CaptureState', 'CaptureEvent']
