# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MaskedCopyLibraryNode``: a copy between a global array and a register tile that a per-lane mask gates."""

from dataclasses import dataclass
from typing import Any

import dace
from dace import dtypes, library, properties
from dace.libraries.standard.helper import collapse_shape_and_strides
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.tileops.alignment import align_template_arg
from dace.libraries.tileops.environments import (
    TileOpsAVX2,
    TileOpsAVX512,
    TileOpsCUDA,
    TileOpsNeon,
    TileOpsScalar,
    TileOpsSVE,
)
from dace.libraries.tileops.expansions import ExpandTileIsa, ExpandTilePure
from dace.libraries.tileops.isa import require_k1
from dace.libraries.tileops.lanes import nested_loops, tile_offset
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.validation import validate_mask_descriptor_lock
from dace.optionals import required
from dace.sdfg import nodes
from dace.symbolic import symstr

MASK_CONNECTOR_NAME = "_mask"

#: The ``(source, destination)`` storages of a masked load, which fills the tile and zeroes its inactive lanes without
#: reading their source.
LOAD_STORAGES = frozenset(
    {
        (dtypes.StorageType.GPU_Global, dtypes.StorageType.Register),
        (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared),
        (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register),
        (dtypes.StorageType.Default, dtypes.StorageType.Register),
        (dtypes.StorageType.Default, dtypes.StorageType.Default),
        # A later pass may give an array of registers (a reduction buffer) the storage of its tile.
        (dtypes.StorageType.Register, dtypes.StorageType.Register),
    }
)
#: The ``(source, destination)`` storages of a masked store, which writes the active lanes only.
STORE_STORAGES = frozenset({(destination, source) for source, destination in LOAD_STORAGES})


def holds_a_tile(desc: dace.data.Data) -> bool:
    """Whether a data container can be the tile of a masked copy: a transient that owns its elements, so no view."""
    return desc.transient and not isinstance(desc, dace.data.View)


def is_load(source: dace.data.Data, destination: dace.data.Data, label: str) -> bool:
    """Whether a masked copy from ``source`` to ``destination`` loads a tile (else it stores one).

    The storages tell the tile from the array. The same storage on both sides (``Default``, or ``Register``) does not,
    so the descriptors do: the tile is the transient side, which owns its elements (a view of an array is no tile). A
    pair that is neither, or that both rules read as the other direction, is refused instead of guessed.

    :raises NotImplementedError: If the storages are no masked load or store, or the direction is ambiguous.
    """
    pair = (source.storage, destination.storage)
    loads, stores = pair in LOAD_STORAGES, pair in STORE_STORAGES
    if loads and stores:
        if holds_a_tile(destination) and not holds_a_tile(source):
            return True
        if holds_a_tile(source) and not holds_a_tile(destination):
            return False
        raise NotImplementedError(
            f"{label}: {pair[0]} to {pair[1]} does not say whether it loads or stores a tile; "
            f"exactly one of the two data containers must be the transient tile."
        )
    if not loads and not stores:
        raise NotImplementedError(
            f"{label}: no masked copy from {pair[0]} to {pair[1]}; loads are "
            f"{sorted(map(str, LOAD_STORAGES))} and stores the mirrored pairs."
        )
    return loads


@dataclass(frozen=True, slots=True)
class Lanes:
    """How the lanes of a masked copy address the two windows.

    The extents and strides are those of the windows with their singleton dims dropped, so the step of a memlet is part
    of the stride and a window of a lower-rank tile on a higher-rank array lines up with it.
    """

    loads: bool
    ctype: str
    extents: list
    in_strides: list
    out_strides: list


@library.expansion
class ExpandMaskedCopyPure(ExpandTilePure):
    pass


@library.expansion
class ExpandMaskedCopyScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandMaskedCopyAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandMaskedCopyAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandMaskedCopyNeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandMaskedCopySVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandMaskedCopyCUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class MaskedCopyLibraryNode(CopyLibraryNode, TileOp):
    """A :class:`~dace.libraries.standard.nodes.copy.node.CopyLibraryNode` of a tile that ``_mask`` gates per lane.

    The tile is a register (or shared) array of ``widths`` and the other side a window of a global array, which the
    memlet gives its offset, extents and steps. A load fills the tile with the window and zeroes the lanes the mask
    switches off, without reading their source: a masked tail of the window may lie past the end of the array. A store
    writes the active lanes and leaves the others of the destination as they are. The storages of the two sides tell
    the two apart (:data:`LOAD_STORAGES`); the same storage on both (``Default``, or ``Register``) is read off the
    descriptors, the transient one that is no view being the tile, and an ambiguous pair raises.

    Without ``has_mask`` there is no ``_mask`` connector and every lane is active. The node does not copy a
    window onto one of other extents (a transposed tile) or of another dtype.
    """

    lanes_independent = True
    implementations = {
        "pure": ExpandMaskedCopyPure,
        "scalar": ExpandMaskedCopyScalar,
        "avx512": ExpandMaskedCopyAVX512,
        "avx2": ExpandMaskedCopyAVX2,
        "neon": ExpandMaskedCopyNeon,
        "sve": ExpandMaskedCopySVE,
        "cuda": ExpandMaskedCopyCUDA,
    }
    default_implementation = "pure"

    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=True,
        desc="Whether the ``_mask`` input connector gates the lanes.",
    )

    def __init__(self, name: str, widths: tuple[int, ...], has_mask: bool = True, **kwargs: Any):
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"MaskedCopyLibraryNode: widths must have length in {{1, 2, 3}}, got {widths!r}")
        super().__init__(name, **kwargs)
        self.widths = list(widths)
        self.has_mask = has_mask
        if has_mask:
            self.add_in_connector(MASK_CONNECTOR_NAME)

    def validate(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        allow_cross_storage: bool = True,
    ) -> tuple[str, dace.data.Data, dace.subsets.Subset, str, dace.data.Data, dace.subsets.Subset]:
        """Resolve the two data edges like the copy does, and check the mask and the pair of storages.

        :returns: ``(inp_name, inp, in_subset, out_name, out, out_subset)``
        :raises ValueError: If the mask or a data edge is not wired, or the windows differ in extent or dtype.
        :raises NotImplementedError: If the storages are no masked load or store.
        """
        out_edges = [edge for edge in state.out_edges(self) if edge.src_conn == OUTPUT_CONNECTOR_NAME]
        in_edges = [edge for edge in state.in_edges(self) if edge.dst_conn == INPUT_CONNECTOR_NAME]
        mask_edges = [edge for edge in state.in_edges(self) if edge.dst_conn == MASK_CONNECTOR_NAME]
        if len(out_edges) != 1 or len(in_edges) != 1:
            raise ValueError(
                f"{self.label}: expects exactly one {INPUT_CONNECTOR_NAME!r} and one {OUTPUT_CONNECTOR_NAME!r} edge."
            )
        if len(mask_edges) != int(self.has_mask):
            raise ValueError(
                f"{self.label}: has_mask={self.has_mask} but {len(mask_edges)} "
                f"{MASK_CONNECTOR_NAME!r} edges are connected."
            )
        inp, out = sdfg.arrays[required(in_edges[0].data.data)], sdfg.arrays[required(out_edges[0].data.data)]
        in_subset, out_subset = in_edges[0].data.subset, out_edges[0].data.subset
        if inp.dtype != out.dtype:
            raise ValueError(f"{self.label}: a masked copy does not convert dtypes (got {inp.dtype} to {out.dtype}).")
        in_extents = collapse_shape_and_strides(in_subset, inp.strides)[0]
        out_extents = collapse_shape_and_strides(out_subset, out.strides)[0]
        if tuple(in_extents) != tuple(out_extents):
            raise ValueError(
                f"{self.label}: the windows have different extents, {tuple(in_extents)} and "
                f"{tuple(out_extents)}; a masked copy does not transpose or reshape."
            )
        is_load(inp, out, self.label)
        if self.has_mask:
            validate_mask_descriptor_lock(
                self.label, MASK_CONNECTOR_NAME, sdfg.arrays[required(mask_edges[0].data.data)], tuple(self.widths)
            )
        return INPUT_CONNECTOR_NAME, inp, in_subset, OUTPUT_CONNECTOR_NAME, out, out_subset

    def stores(self, state: dace.SDFGState) -> bool:
        """Whether this copy stores a tile into an array, as the storages and descriptors of its sides say."""
        sdfg = state.sdfg
        inp = sdfg.arrays[next(state.in_edges_by_connector(self, INPUT_CONNECTOR_NAME)).data.data]
        out = sdfg.arrays[next(state.out_edges_by_connector(self, OUTPUT_CONNECTOR_NAME)).data.data]
        return not is_load(inp, out, self.label)

    def lanes(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Lanes:
        validated = self.validate(sdfg, state)
        inp, in_subset, out, out_subset = validated[1], validated[2], validated[4], validated[5]
        extents, in_strides = collapse_shape_and_strides(in_subset, inp.strides)
        out_strides = collapse_shape_and_strides(out_subset, out.strides)[1]
        return Lanes(is_load(inp, out, self.label), inp.dtype.ctype, extents, in_strides, out_strides)

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        lanes = self.lanes(sdfg, state)
        lane_names = [f"__l{dim}" for dim in range(len(lanes.extents))]
        source = (
            " + ".join(
                f"({lane} * ({symstr(stride)}))" for lane, stride in zip(lane_names, lanes.in_strides, strict=True)
            )
            or "0"
        )
        destination = (
            " + ".join(
                f"({lane} * ({symstr(stride)}))" for lane, stride in zip(lane_names, lanes.out_strides, strict=True)
            )
            or "0"
        )
        mask = f"{MASK_CONNECTOR_NAME}[{tile_offset(lanes.extents, lane_names)}]"
        read = f"{INPUT_CONNECTOR_NAME}[{source}]"
        write = f"{OUTPUT_CONNECTOR_NAME}[{destination}]"
        if not self.has_mask:
            body = f"{write} = {read};"
        elif lanes.loads:
            body = f"{write} = {mask} ? {read} : {lanes.ctype}(0);"
        else:
            body = f"if ({mask}) {{ {write} = {read}; }}"
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(self.input_names()),
            outputs={OUTPUT_CONNECTOR_NAME: None},
            code=nested_loops(lanes.extents, body),
            language=dace.dtypes.Language.CPP,
        )

    def input_names(self) -> list[str]:
        return [INPUT_CONNECTOR_NAME, *([MASK_CONNECTOR_NAME] if self.has_mask else [])]

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        # The headers move one lane dim: a window of several non-singleton dims is the pure loop.
        return len(self.lanes(sdfg, state).extents) == 1

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        lanes = self.lanes(sdfg, state)
        vlen = require_k1(self)
        in_edge = next(state.in_edges_by_connector(self, INPUT_CONNECTOR_NAME))
        out_edge = next(state.out_edges_by_connector(self, OUTPUT_CONNECTOR_NAME))
        array_edge = in_edge if lanes.loads else out_edge
        stride = (lanes.in_strides if lanes.loads else lanes.out_strides)[0]
        masked = "true" if self.has_mask else "false"
        mask_argument = MASK_CONNECTOR_NAME if self.has_mask else "nullptr"
        align = align_template_arg(self, state, sdfg, array_edge, backend, vlen, allow_shift=lanes.loads)
        function = "tile_load" if lanes.loads else "tile_store"
        code = (
            f"dace::tileops::{function}<{lanes.ctype}, {vlen}, {masked}{align}>"
            f"({OUTPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME}, {mask_argument}, {symstr(stride)});"
        )
        return nodes.Tasklet(
            label=f"{self.label}_{backend}",
            inputs=dict.fromkeys(self.input_names()),
            outputs={OUTPUT_CONNECTOR_NAME: None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
