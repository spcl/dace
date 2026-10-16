# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Overall configuration for the multi-dim tile-op vectorizer.

:class:`VectorizeConfig` bundles every vectorizer knob into one dataclass.
``__post_init__`` refuses a variant that is not a member of its enum.
"""

import dataclasses

from dace.dtypes import DeviceType
from dace.libraries.tileops.dispatch import TileGroup
from dace.transformation.passes.vectorization.enums import ISA, BranchMode, RemainderStrategy


@dataclasses.dataclass(slots=True)
class VectorizeConfig:
    """Every knob for :class:`VectorizeMultiDim`, grouped into one config object.

    :param widths: Per-dim tile widths, innermost-last (1..3 entries).
    :param target_isa: K=1 tile-op backend ISA.
    :param remainder_strategy: How a non-divisible map extent is tiled.
    :param branch_mode: How a same-write-set ``if/else`` lowers to a per-lane select.
    :param scalar_remainder_emit: ``"scalar"`` step-1 tail or ``"tile_k1"`` masked K=1 tile.
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
    :param tile_group: The threads a GPU tile node runs on. ``BLOCK`` (the GPU default) makes the tile the block's:
        ``widths`` are then the lanes each thread takes, and the tile is ``32 * num_warps`` times wider. ``THREAD``
        is one thread's register tile of ``widths`` lanes. A CPU tile is one core's either way.
    :param num_warps: The warps of the thread block a ``BLOCK`` tile spreads over.
    """

    widths: tuple[int, ...]
    target_isa: ISA = ISA.AUTO
    remainder_strategy: RemainderStrategy = RemainderStrategy.MASKED_TAIL
    branch_mode: BranchMode = BranchMode.MERGE
    scalar_remainder_emit: str = "scalar"
    expand_tile_nodes: bool = False
    validate: bool = True
    validate_all: bool = False
    assume_even: bool = False
    fuse_multiply_add: bool = False
    device: DeviceType = DeviceType.CPU
    tile_group: TileGroup = TileGroup.BLOCK
    num_warps: int = 4

    def __post_init__(self) -> None:
        for name, enum_cls in (
            ("target_isa", ISA),
            ("remainder_strategy", RemainderStrategy),
            ("branch_mode", BranchMode),
            ("tile_group", TileGroup),
        ):
            if not isinstance(getattr(self, name), enum_cls):
                raise TypeError(
                    f"VectorizeConfig.{name} must be a {enum_cls.__name__} member, got {getattr(self, name)!r}"
                )
        self.widths = tuple(self.widths)
