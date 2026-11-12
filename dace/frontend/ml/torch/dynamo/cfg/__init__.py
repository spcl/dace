# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
EXPERIMENTAL: capture of data-dependent Python control flow for the ``dace`` TorchDynamo backend.

:class:`~dace.frontend.ml.torch.dynamo.cfg.blocks.ControlFlowBackend` is a ``DaceBackend`` that, at a conditional jump
on tensor data, captures the rest of the frame as a control-flow graph of traced blocks (instead of graph-breaking) and
passes it to the DaCe importer through the opaque operator ``dace::cfg``.
"""

from .blocks import ControlFlowBackend

__all__ = ["ControlFlowBackend"]
