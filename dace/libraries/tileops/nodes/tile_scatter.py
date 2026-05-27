# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileScatter`` — indirect write from a K-dim tile into a K-dim destination.

Each input lane ``(l_0, ..., l_{K-1})`` of ``_src`` is written to
``_dst[_idx_0[l_*], _idx_1[l_*], ...]`` where ``_idx_<k>`` is a
tile-shaped integer tile carrying the per-destination-dim index. The
shape mirrors :class:`TileGather`; only the data-flow direction
differs (src tile → indirect-indexed dst array slots).

When ``has_mask=True`` an inactive lane does NOT write — matches the
cuTile ``ct.scatter(dst, idx, value, mask=...)`` semantic.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, tile_offset
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE


@library.expansion
class ExpandTileScatterPure(ExpandTransformation):
    """Correctness-only CPP tasklet: per-lane indirect write to ``_dst``."""

    environments = []

    @staticmethod
    def expansion(node: "TileScatter", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet walking the tile lane-by-lane, computing
        each lane's flat destination offset via the per-dim index tiles
        and the destination array's own per-dim strides.

        :param node: The ``TileScatter`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
        dst_arr = parent_sdfg.arrays[dst_edge.data.data]
        dst_ndim = node.dest_ndim
        dst_strides = [symstr(s) for s in dst_arr.strides[-dst_ndim:]]
        off = tile_offset(widths)
        flat_offset = " + ".join(f"((std::ptrdiff_t)_idx_{k}[{off}] * ({dst_strides[k]}))" for k in range(dst_ndim))
        if node.has_mask:
            body = f"if (_mask[{off}]) {{ _dst[{flat_offset}] = _src[{off}]; }}"
        else:
            body = f"_dst[{flat_offset}] = _src[{off}];"
        code = nested_loops(widths, body)
        inputs = {"_src"} | {f"_idx_{k}" for k in range(dst_ndim)}
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
class ExpandTileScatterCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileScatter`.

    Emits ``ct.scatter(__output, (__idx_0, ...), __src, mask=__mask)``
    — the cuTile pattern from ``manual_cutile_masked.py``.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileScatter", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``ct.scatter``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet with a ``ct.scatter`` body.
        """
        dst_ndim = node.dest_ndim
        if dst_ndim == 1:
            idx_arg = "__idx_0"
        else:
            idx_tuple = ", ".join(f"__idx_{k}" for k in range(dst_ndim))
            idx_arg = f"({idx_tuple})"
        mask_arg = ", mask=__mask" if node.has_mask else ""
        body = f"ct.scatter(__output, {idx_arg}, __src{mask_arg})"
        inputs = {"__src"} | {f"__idx_{k}" for k in range(dst_ndim)}
        if node.has_mask:
            inputs.add("__mask")
        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
        )


@library.expansion
class ExpandTileScatterScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_scatter_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileScatterAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_scatter_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileScatterAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_scatter_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileScatterNeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_scatter_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileScatterSVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_scatter_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileScatter(nodes.LibraryNode):
    """Indirect write from a K-dim tile into a destination array.

    Connectors:

    * ``_src`` — the source tile (``widths``-shaped).
    * ``_idx_<k>`` (for ``k`` in ``0..dest_ndim-1``) — tile-shaped
      integer index tile providing per-destination-dim indices.
    * ``_mask`` (optional) — tile-shaped boolean mask gating each
      lane's write.
    * ``_dst`` — the destination array (full memlet).

    Per-lane flat offset into ``_dst`` is
    ``sum_k _idx_<k>[lane] * dst_strides[k]``.
    """

    implementations = {
        "pure": ExpandTileScatterPure,
        "cutile": ExpandTileScatterCutile,
        "scalar": ExpandTileScatterScalar,
        "avx512": ExpandTileScatterAVX512,
        "avx2": ExpandTileScatterAVX2,
        "neon": ExpandTileScatterNeon,
        "sve": ExpandTileScatterSVE
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
    dest_ndim = properties.Property(
        dtype=int,
        allow_none=False,
        default=1,
        desc="Number of dims of the destination array; ``dest_ndim`` index "
        "connectors (``_idx_0`` .. ``_idx_{dest_ndim-1}``) are declared.",
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
                 dest_ndim: int = 1,
                 has_mask: bool = False,
                 location: Optional[str] = None):
        """Construct a ``TileScatter`` node.

        :param name: Node label.
        :param widths: Per-tile-dim widths, innermost-last (length 1..3).
        :param dest_ndim: Number of dims of the destination array.
        :param has_mask: When True, declare the ``_mask`` input.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3, or
            ``dest_ndim`` < 1.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileScatter: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dest_ndim < 1:
            raise ValueError(f"TileScatter: dest_ndim must be >= 1, got {dest_ndim}")
        inputs = {"_src"} | {f"_idx_{k}" for k in range(dest_ndim)}
        if has_mask:
            inputs.add("_mask")
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dest_ndim = int(dest_ndim)
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
        for k in range(self.dest_ndim):
            conn = f"_idx_{k}"
            if conn not in in_e:
                raise ValueError(f"{self.label}: required input {conn!r} not connected")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
