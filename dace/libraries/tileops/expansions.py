# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The two kinds of expansion of a tile node.

DaCe ties an expansion class to the one node it expands, so a node declares its own subclass of each backend; the
lowering itself is the node's ``pure_tasklet`` and ``isa_tasklet``.
"""

import dace
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


class ExpandTilePure(ExpandTransformation):
    """The per-lane C++ loop, which the compiler can still vectorize."""

    environments: list[type] = []

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return node.pure_tasklet(parent_state, parent_sdfg)


class ExpandTileIsa(ExpandTransformation):
    """A call into the header of one ISA backend; a subclass names the backend and its environment."""

    backend: str

    @classmethod
    def expansion(cls, node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return node.isa_tasklet(parent_state, parent_sdfg, cls.backend)
