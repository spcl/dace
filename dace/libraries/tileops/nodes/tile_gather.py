# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileGather`` — indirect read from a K-dim source into a K-dim tile.

Each output lane ``(l_0, ..., l_{K-1})`` of ``_dst`` is filled from
``_src[_idx_0[l_*], _idx_1[l_*], ...]`` where ``_idx_<k>`` is a
tile-shaped integer tile carrying the per-source-dim index. For a
1D source this is a single index tile ``_idx_0``; for a 2D source two
index tiles ``_idx_0`` and ``_idx_1`` (matching the cuTile pattern
``ct.gather(array, (idx_0, idx_1), mask=...)``).

When ``has_mask=True`` an inactive lane writes ``0`` into the
destination — matches the AVX-512 ``maskz_*`` / cuTile
``padding_value=0`` semantics.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, tile_offset


@library.expansion
class ExpandTileGatherPure(ExpandTransformation):
    """Correctness-only CPP tasklet: per-lane indirect read from ``_src``."""

    environments = []

    @staticmethod
    def expansion(node: "TileGather", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet walking the tile lane-by-lane, looking
        up each lane's flat source offset via the per-dim index tiles
        and the source's own per-dim strides.

        :param node: The ``TileGather`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
        src_arr = parent_sdfg.arrays[src_edge.data.data]
        src_ndim = node.source_ndim
        src_strides = [symstr(s) for s in src_arr.strides[-src_ndim:]]
        off = tile_offset(widths)
        flat_offset = " + ".join(
            f"((std::ptrdiff_t)_idx_{k}[{off}] * ({src_strides[k]}))" for k in range(src_ndim)
        )
        dst_dtype = parent_sdfg.arrays[
            next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst").data.data
        ].dtype.ctype
        if node.has_mask:
            body = f"_dst[{off}] = _mask[{off}] ? _src[{flat_offset}] : {dst_dtype}(0);"
        else:
            body = f"_dst[{off}] = _src[{flat_offset}];"
        code = nested_loops(widths, body)
        inputs = {"_src"} | {f"_idx_{k}" for k in range(src_ndim)}
        if node.has_mask:
            inputs.add("_mask")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileGatherCute(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileGather`.

    Emits ``ct.gather(__src, (__idx_0, __idx_1, ...), mask=__mask,
    padding_value=0)`` — the per-source-dim index-tile tuple form used
    by the reference cuTile kernels. For a 1D source the call becomes
    ``ct.gather(__src, __idx_0, ...)`` (single index tile, not a tuple
    — matches the ``manual_cutile_simple`` example).
    """

    environments = []

    @staticmethod
    def expansion(node: "TileGather", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``ct.gather``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with a ``ct.gather`` body.
        """
        src_ndim = node.source_ndim
        if src_ndim == 1:
            idx_arg = "__idx_0"
        else:
            idx_tuple = ", ".join(f"__idx_{k}" for k in range(src_ndim))
            idx_arg = f"({idx_tuple})"
        mask_arg = ", mask=__mask, padding_value=0" if node.has_mask else ""
        body = f"__output = ct.gather(__src, {idx_arg}{mask_arg})"
        inputs = {"__src"} | {f"__idx_{k}" for k in range(src_ndim)}
        if node.has_mask:
            inputs.add("__mask")
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.node
class TileGather(nodes.LibraryNode):
    """Indirect read from a K-dim source array into a tile transient.

    Connectors:

    * ``_src`` — the source array (full memlet); shape is determined
      by the surrounding scope.
    * ``_idx_<k>`` (for ``k`` in ``0..source_ndim-1``) — tile-shaped
      integer index tile providing per-source-dim indices.
    * ``_mask`` (optional, when ``has_mask=True``) — tile-shaped
      boolean mask gating each lane's write.
    * ``_dst`` — the output tile transient (``widths``-shaped).

    The per-lane flat offset into ``_src`` is
    ``sum_k _idx_<k>[lane] * src_strides[k]`` so the source's own
    strides are honored (works for C-layout, Fortran-layout, and
    arbitrary stride arrays).
    """

    implementations = {"pure": ExpandTileGatherPure, "cute": ExpandTileGatherCute}
    default_implementation = "pure"

    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim widths, innermost-last.",
    )
    source_ndim = properties.Property(
        dtype=int,
        allow_none=False,
        default=1,
        desc="Number of dims of the source array; ``source_ndim`` index "
        "connectors (``_idx_0`` .. ``_idx_{source_ndim-1}``) are declared.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, the ``_mask`` input connector is required.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 source_ndim: int = 1,
                 has_mask: bool = False,
                 location: Optional[str] = None):
        """Construct a ``TileGather`` node.

        :param name: Node label.
        :param widths: Per-tile-dim widths, innermost-last (length 1..3).
        :param source_ndim: Number of dims of the source array
            (defaults to 1 for 1D-source gather).
        :param has_mask: When True, declare the ``_mask`` input.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty or longer than 3, or
            if ``source_ndim`` < 1.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileGather: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if source_ndim < 1:
            raise ValueError(f"TileGather: source_ndim must be >= 1, got {source_ndim}")
        inputs = {"_src"} | {f"_idx_{k}" for k in range(source_ndim)}
        if has_mask:
            inputs.add("_mask")
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.source_ndim = int(source_ndim)
        self.has_mask = has_mask

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Check connectors.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_src" not in in_e:
            raise ValueError(f"{self.label}: required input '_src' not connected")
        for k in range(self.source_ndim):
            conn = f"_idx_{k}"
            if conn not in in_e:
                raise ValueError(f"{self.label}: required input {conn!r} not connected")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
