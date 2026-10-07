# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The base of the tile library nodes."""

import dace
from dace import properties
from dace.libraries.tileops.dispatch import ISA
from dace.sdfg import nodes


@properties.make_properties
class TileOp(nodes.LibraryNode):
    """A library node over a tile of ``widths`` elements per dim, innermost last.

    The implementation is chosen from the vectorizer's ``target_isa`` (see :mod:`dace.libraries.tileops.dispatch`),
    not from the schedule, so device auto-selection must leave it alone. A node lowers either ``pure``, as a loop over
    the lanes, or, for K=1 and where :meth:`can_lower_to_isa` allows it, as a call into the header of a backend.
    """

    auto_select_implementation = False

    target_isa = properties.EnumProperty(
        dtype=ISA,
        default=ISA.SCALAR,
        desc="Target ISA the implementation is selected for, stamped by the vectorizer before expansion; AUTO picks "
        "the host's. Tiles of more than one dim are pure.",
    )
    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost last; one to three dims.",
    )

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        """Whether a backend header has a lowering for this configuration of the node."""
        return True

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(f"{type(self).__name__} defines no pure lowering")

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        raise NotImplementedError(f"{type(self).__name__} defines no ISA lowering")
