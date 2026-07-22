# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Typed knobs for the multi-dim tile-op vectorizer.

Each optimization variant is a string-valued :class:`enum.Enum`, so a member both
reads nicely (``ISA.AVX512``) and compares/serialises as its string value
(``ISA.AVX512 == "AVX512"`` is ``True``). That lets a caller pass either the enum
member or the raw string; :func:`coerce` normalises a string to the member.
"""
import enum
from typing import Union


class ISA(str, enum.Enum):
    """Target instruction set for the K=1 tile-op backend."""
    AUTO = "AUTO"            #: resolve to the host's best ISA at expansion
    AVX512 = "AVX512"
    AVX2 = "AVX2"
    ARM_SVE = "ARM_SVE"
    ARM_NEON = "ARM_NEON"
    SCALAR = "SCALAR"        #: portable scalar reference
    CUDA = "CUDA"           #: GPU half2 (implies device=GPU)


class RemainderStrategy(str, enum.Enum):
    """How the tiler handles a map extent not divisible by the tile width."""
    FULL_MASK = "full_mask"                #: single W-strided map, mask every tile
    MASKED_TAIL = "masked_tail"            #: mask-free interior + masked boundary
    SCALAR_POSTAMBLE = "scalar_postamble"  #: divisible interior + step-1 scalar tail (K=1 only)
    BRANCHED_TAIL = "branched_tail"  #: GPU-only: ONE kernel with a control-flow branch,
    #: if(full-tile)=vectorized tile body / else=scalar tail, over the fused tile range. K=1 only.
    #: See :class:`~dace.transformation.passes.vectorization.fuse_branched_tail_remainder.FuseBranchedTailRemainder`.


class BranchMode(str, enum.Enum):
    """How a same-write-set ``if/else`` is lowered to a per-lane select."""
    MERGE = "merge"          #: per-lane ``TileITE`` blend
    FP_FACTOR = "fp_factor"  #: ``c*x + (1-c)*y`` tile-binop arithmetic (K=1 only)


def coerce_enum(enum_cls, value):
    """Return ``value`` as a member of ``enum_cls`` (accepts a member or its string)."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)  # raises ValueError on an unknown string


def coerce_isa(value: Union["ISA", str]) -> "ISA":
    return coerce_enum(ISA, value)


def coerce_remainder_strategy(value: Union["RemainderStrategy", str]) -> "RemainderStrategy":
    return coerce_enum(RemainderStrategy, value)


def coerce_branch_mode(value: Union["BranchMode", str]) -> "BranchMode":
    return coerce_enum(BranchMode, value)
