# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Multi-dim fixed-length tile-op library nodes.

These library nodes are the IR for the K-dim (``K \\in \\{1, 2, 3\\}``)
masked tile vectorization track. The 5 MVP nodes cover axpy / triad /
5-point stencil correctness gates; the post-MVP set adds ternary blends,
indirect accesses and reductions. Each node carries a ``widths`` tuple
(per-dim register-tile widths, innermost-last) and an optional ``_mask``
connector enabled via a ``has_mask`` constructor knob.

Layout mirrors :mod:`dace.libraries.standard`: ``nodes`` exports the
:class:`LibraryNode` subclasses; ``environments`` is reserved for
per-arch toolchain shims (empty in T1 — pure expansions only).
"""
from dace.library import register_library
from .nodes import *
from .environments import *

register_library(__name__, "tileops")
