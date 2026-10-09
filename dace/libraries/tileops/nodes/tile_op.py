# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The base of the tile library nodes."""

import math
from typing import ClassVar

import dace
from dace import properties
from dace.libraries.tileops.dispatch import ISA, TileGroup
from dace.sdfg import nodes
from dace.sdfg.scope import is_devicelevel_gpu


@properties.make_properties
class TileOp(nodes.LibraryNode):
    """A library node over a tile of ``widths`` elements per dim, innermost last.

    The implementation is chosen from the vectorizer's ``target_isa`` (see :mod:`dace.libraries.tileops.dispatch`),
    not from the schedule, so device auto-selection must leave it alone. A node lowers either ``pure``, as a loop over
    the lanes, or, for K=1 and where :meth:`can_lower_to_isa` allows it, as a call into the header of a backend. A
    ``WARP`` or ``BLOCK`` node in a GPU kernel lowers ``block``: the lanes spread over the threads of its ``group``.
    """

    auto_select_implementation = False

    #: Whether every lane of the output is computed from that lane alone, so the lanes may run on different threads.
    #: A node that combines lanes (a reduction) or writes where its indices say (a scatter) runs on one thread.
    lanes_independent: ClassVar[bool] = False

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
    group = properties.EnumProperty(
        dtype=TileGroup,
        default=TileGroup.THREAD,
        desc="The threads that execute the node together. THREAD is one thread's register tile; WARP and BLOCK "
        "spread the lanes over a warp or a thread block in a GPU kernel and lower as THREAD on a CPU.",
    )
    num_warps = properties.Property(
        dtype=int,
        default=4,
        desc="Warps of the thread block a BLOCK node spreads its lanes over (Triton's num_warps).",
    )
    lanes_per_thread = properties.Property(
        dtype=int,
        default=1,
        desc="Consecutive lanes each thread of a WARP or BLOCK node takes at a time (Triton's sizePerThread); a node "
        "with a target ISA runs its ISA call on them, two fp16 lanes for one half2 instruction.",
    )

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        """Whether a backend header has a lowering for this configuration of the node."""
        return True

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(f"{type(self).__name__} defines no pure lowering")

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        raise NotImplementedError(f"{type(self).__name__} defines no ISA lowering")

    def reads_lane_wise(self, connector: str) -> bool:
        """Whether lane ``l`` of the output reads lane ``l`` of ``connector`` alone (an output connector: writes)."""
        return self.lanes_independent

    def output_elements(self) -> int:
        """The elements of the output tile, over which a ``block`` lowering spreads its threads."""
        return math.prod(self.widths)

    def group_threads(self) -> int:
        """The threads of the node's group."""
        return 32 if self.group is TileGroup.WARP else 32 * self.num_warps

    def expand(self, state_or_sdfg, state_or_impl=None, **kwargs) -> str:
        # Both interfaces of ``LibraryNode.expand``: ``(state, implementation)`` and ``(sdfg, state)``
        if isinstance(state_or_sdfg, dace.SDFGState):
            state, requested = state_or_sdfg, state_or_impl
        else:
            state, requested = state_or_impl, kwargs.get("implementation")
        if requested is None and self.group is not TileGroup.THREAD and is_devicelevel_gpu(state.sdfg, state, self):
            self.implementation = "block"
        return super().expand(state_or_sdfg, state_or_impl, **kwargs)
