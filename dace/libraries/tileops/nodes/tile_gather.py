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
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE


def _strided_lane(stride: int, off: str) -> str:
    """Per-lane index into a (possibly strided) index tile.

    :param stride: The lane stride ``c`` into the index tile.
    :param off: The contiguous lane-offset expression (e.g. ``__l0``).
    :returns: ``off`` when ``stride == 1`` (contiguous index tile), else
        ``(c) * (off)`` — lane ``l`` reads ``_idx[c*l]`` from a
        ``c``-strided index window.
    """
    return off if stride == 1 else f"({stride}) * ({off})"


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
        idx_strides = node.index_strides or [1] * src_ndim
        flat_offset = " + ".join(
            f"((std::ptrdiff_t)_idx_{k}[{_strided_lane(idx_strides[k], off)}] * ({src_strides[k]}))"
            for k in range(src_ndim))
        dst_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_dst").data.data].dtype.ctype
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
            inputs={c: None
                    for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileGatherCute(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileGather`.

    Emits ``ct.gather(__src, (__idx_0, __idx_1, ...), mask=__mask,
    padding_value=<pad_value>)`` — the per-source-dim index-tile tuple
    form used by the reference cuTile kernels. For a 1D source the call
    becomes ``ct.gather(__src, __idx_0, ...)`` (single index tile, not a
    tuple — matches the ``manual_cutile_simple`` example). ``ct.gather``'s
    ``padding_value`` is an arbitrary scalar (unlike ``ct.load``'s enum),
    so :attr:`TileGather.pad_value` can install any reduction identity
    (``0`` / ``1`` / ``±inf``) for the OOB / masked lanes.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileGather", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``ct.gather``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with a ``ct.gather`` body.
        :raises NotImplementedError: If any ``index_strides`` entry is
            non-unit — ``ct.gather`` indexes the index tile directly and
            has no per-lane stride concept, so a ``c``-strided index
            window cannot be expressed (the strided window must be
            pre-gathered into a contiguous index tile upstream).
        """
        src_ndim = node.source_ndim
        # ct.gather has no per-lane stride: a non-unit index_strides cannot be
        # lowered to a single ct.gather over the index tile (the pure / ISA
        # paths read __idx[c*lane], which cuTile's gather does not express).
        if any(s != 1 for s in node.index_strides):
            raise NotImplementedError(f"{node.label}: TileGather cute lowering cannot express non-unit "
                                      f"index_strides={tuple(node.index_strides)!r}; ct.gather indexes the "
                                      f"index tile directly (no per-lane stride). Pre-gather the strided "
                                      f"index window into a contiguous index tile before this node.")
        if src_ndim == 1:
            idx_arg = "__idx_0"
        else:
            idx_tuple = ", ".join(f"__idx_{k}" for k in range(src_ndim))
            idx_arg = f"({idx_tuple})"
        pad_arg = f"padding_value={node.pad_value}"
        mask_arg = f", mask=__mask, {pad_arg}" if node.has_mask else f", {pad_arg}"
        body = f"__output = ct.gather(__src, {idx_arg}{mask_arg})"
        inputs = {"__src"} | {f"__idx_{k}" for k in range(src_ndim)}
        if node.has_mask:
            inputs.add("__mask")
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.expansion
class ExpandTileGatherScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_gather_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileGatherAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_gather_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileGatherAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_gather_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileGatherNeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_gather_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileGatherSVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_gather_tasklet(node, parent_state, parent_sdfg, "sve")


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

    implementations = {
        "pure": ExpandTileGatherPure,
        "cute": ExpandTileGatherCute,
        "scalar": ExpandTileGatherScalar,
        "avx512": ExpandTileGatherAVX512,
        "avx2": ExpandTileGatherAVX2,
        "neon": ExpandTileGatherNeon,
        "sve": ExpandTileGatherSVE
    }
    default_implementation = "pure"

    target_isa = properties.Property(
        dtype=str,
        allow_none=False,
        default="SCALAR",
        desc="CPU target ISA the Auto-dispatch lowers to for K==1 "
        "(SCALAR | AVX512 | AVX2 | ARM_SVE | ARM_NEON | CUTILE); K>=2 is pure. "
        "Stamped by the VectorizeCPUMultiDim orchestrator before expansion.",
    )

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
    index_strides = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-source-dim lane stride into each ``_idx_<k>`` tile. Empty "
        "(or all-1) means contiguous index tiles (``_idx_<k>[lane]``). A "
        "stride ``c>1`` means ``_idx_<k>`` is a contiguous bounding window of "
        "``c*(W-1)+1`` elements and lane ``l`` reads ``_idx_<k>[c*l]`` — the "
        "tile gather of a ``c``-strided index, e.g. ``b[idx[c*i]]``.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, the ``_mask`` input connector is required.",
    )
    pad_value = properties.Property(
        dtype=int,
        allow_none=False,
        default=0,
        desc="cuTile ``ct.gather`` ``padding_value`` for OOB / masked-out "
        "lanes; only the ``cute`` expansion reads it. Defaults to ``0`` (the "
        "``+`` reduction identity); set ``1`` for ``prod`` (cuTile's gather "
        "padding is an arbitrary scalar, unlike load's fixed enum).",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 source_ndim: int = 1,
                 has_mask: bool = False,
                 index_strides: Tuple[int, ...] = (),
                 pad_value: int = 0,
                 location: Optional[str] = None):
        """Construct a ``TileGather`` node.

        :param name: Node label.
        :param widths: Per-tile-dim widths, innermost-last (length 1..3).
        :param source_ndim: Number of dims of the source array
            (defaults to 1 for 1D-source gather).
        :param has_mask: When True, declare the ``_mask`` input.
        :param index_strides: Per-source-dim lane stride into each index
            tile (defaults to all-1: contiguous index tiles).
        :param pad_value: cuTile ``ct.gather`` ``padding_value`` for OOB /
            masked-out lanes (default ``0``); only the ``cute`` expansion
            uses it.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty or longer than 3, or
            if ``source_ndim`` < 1, or if ``index_strides`` length mismatches
            ``source_ndim``.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileGather: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if source_ndim < 1:
            raise ValueError(f"TileGather: source_ndim must be >= 1, got {source_ndim}")
        if index_strides and len(index_strides) != source_ndim:
            raise ValueError(f"TileGather: index_strides {index_strides!r} must have source_ndim={source_ndim} "
                             f"entries (or be empty for all-1)")
        inputs = {"_src"} | {f"_idx_{k}" for k in range(source_ndim)}
        if has_mask:
            inputs.add("_mask")
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.source_ndim = int(source_ndim)
        self.has_mask = has_mask
        self.index_strides = list(index_strides) if index_strides else [1] * int(source_ndim)
        self.pad_value = int(pad_value)

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
