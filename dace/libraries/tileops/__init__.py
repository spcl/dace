# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Library nodes for fixed-width tiles of one to three dims, the IR of the masked tile vectorization.

A node carries the ``widths`` of its tile, innermost last. Elementwise nodes read each operand as a ``Tile``, a
``Scalar`` or an inline ``Symbol`` (:mod:`~dace.libraries.tileops.kinds`), and the ones that can be gated take a
``_mask`` connector. Every node lowers as a loop over the lanes, and for K=1 most also as a call into the header of an
ISA backend; :mod:`~dace.libraries.tileops.dispatch` selects between them.

Layout mirrors :mod:`dace.libraries.standard`: ``nodes`` holds the library nodes and ``environments`` the toolchain
environments of the ISA backends.
"""
from dace.library import register_library
from .nodes import *
from .environments import *

register_library(__name__, "tileops")
