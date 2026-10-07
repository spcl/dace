# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Typed knobs for the multi-dim tile-op vectorizer.

:class:`ISA` lives with the tile-op dispatch tables that consume it and is re-exported here with the other knobs.
"""

import enum

from dace.libraries.tileops.dispatch import ISA

__all__ = ["ISA", "RemainderStrategy", "BranchMode"]


class RemainderStrategy(enum.Enum):
    """How the tiler handles a map extent not divisible by the tile width."""

    FULL_MASK = enum.auto()  #: single W-strided map, mask every tile
    MASKED_TAIL = enum.auto()  #: mask-free interior + masked boundary
    SCALAR_POSTAMBLE = enum.auto()  #: divisible interior + step-1 scalar tail (K=1 only)
    BRANCHED_TAIL = enum.auto()  #: GPU K=1 only: one kernel, if(full-tile)=vector / else=scalar
    BRANCHED_MASKED_TAIL = enum.auto()  #: GPU K=1 DEFAULT: else arm masked, not scalar


class BranchMode(enum.Enum):
    """How a same-write-set ``if/else`` is lowered to a per-lane select."""

    MERGE = enum.auto()  #: per-lane ``TileITE`` blend
    FP_FACTOR = enum.auto()  #: ``c*t + (1-c)*e`` tile-binop arithmetic (K=1 only)
