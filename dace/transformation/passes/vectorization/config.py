# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Overall configuration for the multi-dim tile-op vectorizer.

:class:`VectorizeConfig` bundles every vectorizer knob into one dataclass.
``__post_init__`` coerces string-valued variants to enum members, so a raw
string (``"AVX512"``) and the enum member (``ISA.AVX512``) are interchangeable.
"""
import dataclasses

from dace.dtypes import DeviceType
from dace.transformation.passes.vectorization.enums import (
    ISA,
    BranchMode,
    RemainderStrategy,
    coerce_branch_mode,
    coerce_isa,
    coerce_remainder_strategy,
)


@dataclasses.dataclass(slots=True)
class VectorizeConfig:
    """Every knob for :class:`VectorizeMultiDim`, grouped into one config object.

    :param widths: Per-dim tile widths, innermost-last (1..3 entries).
    :param target_isa: K=1 tile-op backend ISA.
    :param remainder_strategy: How a non-divisible map extent is tiled.
    :param branch_mode: How a same-write-set ``if/else`` lowers to a per-lane select.
    :param scalar_remainder_emit: ``"scalar"`` step-1 tail or ``"tile_k1"`` masked K=1 tile.
    :param loop_to_map_permissive: Pass ``permissive=True`` to the up-front ``LoopToMap``.
    :param expand_tile_nodes: Expand tile lib nodes to tasklets before returning.
        Default ``False``: left intact for inspection/further transformation, lowered
        later by the caller or ``compile()``.
    :param validate: Validate the SDFG once after the whole pipeline.
    :param validate_all: Also validate between every subpass.
    :param assume_even: Assume every tiled extent is divisible (skip the remainder).
    :param fuse_multiply_add: Fuse ``a*b + c`` into one FMA (native per ISA). Off by
        default: FMA rounds once vs. two roundings for separate ``*``/``+``, so results
        differ by up to 1 ULP from plain NumPy.
    :param device: Target device (CPU / GPU).
    """
    widths: tuple[int, ...]
    target_isa: ISA = ISA.AUTO
    remainder_strategy: RemainderStrategy = RemainderStrategy.MASKED_TAIL
    branch_mode: BranchMode = BranchMode.MERGE
    scalar_remainder_emit: str = "scalar"
    loop_to_map_permissive: bool = False
    expand_tile_nodes: bool = False
    validate: bool = True
    validate_all: bool = False
    assume_even: bool = False
    fuse_multiply_add: bool = False
    device: DeviceType = DeviceType.CPU

    def __post_init__(self) -> None:
        self.target_isa = coerce_isa(self.target_isa)
        self.remainder_strategy = coerce_remainder_strategy(self.remainder_strategy)
        self.branch_mode = coerce_branch_mode(self.branch_mode)
        self.widths = tuple(self.widths)
