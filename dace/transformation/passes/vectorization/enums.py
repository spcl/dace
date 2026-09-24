# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Typed knobs for the multi-dim tile-op vectorizer.

Each variant is a string-valued :class:`enum.Enum`: a member reads nicely
(``ISA.AVX512``) and compares/serializes as its string (``ISA.AVX512 == "AVX512"``),
so callers may pass either. :func:`coerce_enum` normalizes a string to the member.
"""
import enum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=enum.Enum)


class ISA(str, enum.Enum):
    """Target instruction set for the K=1 tile-op backend."""
    AUTO = "AUTO"  #: resolve to the host's best ISA at expansion
    AVX512 = "AVX512"
    AVX2 = "AVX2"
    ARM_SVE = "ARM_SVE"
    ARM_NEON = "ARM_NEON"
    SCALAR = "SCALAR"  #: portable scalar reference
    CUDA = "CUDA"  #: GPU half2 (implies device=GPU)
    CUDA_WARP = "CUDA_WARP"  #: GPU warp-collective tile ops (implies device=GPU)


class RemainderStrategy(str, enum.Enum):
    """How the tiler handles a map extent not divisible by the tile width."""
    FULL_MASK = "full_mask"  #: single W-strided map, mask every tile
    MASKED_TAIL = "masked_tail"  #: mask-free interior + masked boundary
    SCALAR_POSTAMBLE = "scalar_postamble"  #: divisible interior + step-1 scalar tail (K=1 only)
    BRANCHED_TAIL = "branched_tail"  #: GPU K=1 only: one kernel, if(full-tile)=vector / else=scalar
    BRANCHED_MASKED_TAIL = "branched_masked_tail"  #: GPU K=1 DEFAULT: else arm masked, not scalar


class BranchMode(str, enum.Enum):
    """How a same-write-set ``if/else`` is lowered to a per-lane select."""
    MERGE = "merge"  #: per-lane ``TileITE`` blend
    FP_FACTOR = "fp_factor"  #: ``c*x + (1-c)*y`` tile-binop arithmetic (K=1 only)


def coerce_enum(enum_cls: type[EnumT], value: EnumT | str) -> EnumT:
    """Return ``value`` as a member of ``enum_cls`` (accepts a member or its string)."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)  # raises ValueError on an unknown string


def coerce_isa(value: ISA | str) -> ISA:
    return coerce_enum(ISA, value)


def coerce_remainder_strategy(value: RemainderStrategy | str) -> RemainderStrategy:
    return coerce_enum(RemainderStrategy, value)


def coerce_branch_mode(value: BranchMode | str) -> BranchMode:
    return coerce_enum(BranchMode, value)
