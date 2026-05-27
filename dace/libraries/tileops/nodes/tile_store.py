# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileStore`` — write a K-dim tile back into a global array.

Symmetric to :class:`TileLoad`; the pure expansion emits a CPP tasklet
that walks the K-fold nested index space.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, offset_via_strides, tile_offset
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE


@library.expansion
class ExpandTileStorePure(ExpandTransformation):
    """Correctness-only CPP tasklet copying ``_src`` into the tile region of ``_dst``."""

    environments = []

    @staticmethod
    def expansion(node: "TileStore", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that copies ``_src`` into the tile
        region of the destination, optionally gated by ``_mask``.

        Destination offsets use the destination array's per-dim strides
        (read from the connector descriptor at expansion time) scaled
        by an optional :attr:`dim_strides` coefficient (defaulting to 1).

        :param node: The ``TileStore`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        K = len(widths)
        dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
        dst_arr = parent_sdfg.arrays[dst_edge.data.data]
        ndim = len(dst_arr.strides)
        # Step along the array dim each tile dim maps to (``dst_dims``);
        # default to the last K dims in order (a plain row-major tile).
        dims = list(node.dst_dims) if node.dst_dims else list(range(ndim - K, ndim))
        dst_strides = [symstr(dst_arr.strides[d]) for d in dims]
        coeff = list(node.dim_strides) if node.dim_strides else [1] * K
        dst_off = offset_via_strides(coeff, dst_strides)
        src_off = tile_offset(widths)
        if node.has_mask:
            body = f"if (_mask[{src_off}]) {{ _dst[{dst_off}] = _src[{src_off}]; }}"
        else:
            body = f"_dst[{dst_off}] = _src[{src_off}];"
        code = nested_loops(widths, body)
        inputs = {"_src"} | ({"_mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileStoreCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileStore`.

    Two emission shapes, matching the reference cuTile kernels:

    * Unmasked: ``ct.store(__output, index=(__pid0, ...), tile=__src)``
      — contiguous block-tile store.
    * Masked: ``ct.scatter(__output, (idx_0, ...), __src, mask=__mask)``
      with per-lane indices ``idx_k = ct.arange(W_k) + __pid_k * W_k``,
      so OOB lanes at the tile tail are skipped per the iteration mask.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileStore", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``ct.store`` or ``ct.scatter``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet whose body either calls
            ``ct.store`` (unmasked) or ``ct.scatter`` (masked).
        """
        widths = list(node.widths)
        K = len(widths)
        lines = [f"__pid{k} = ct.bid({k})" for k in range(K)]
        if node.has_mask:
            for k, w in enumerate(widths):
                lines.append(f"__idx{k} = ct.arange({w}, dtype=ct.int32) + __pid{k} * {w}")
            idx_tuple = ", ".join(f"__idx{k}" for k in range(K))
            lines.append(f"ct.scatter(__output, ({idx_tuple},), __src, mask=__mask)")
        else:
            index_tuple = ", ".join(f"__pid{k}" for k in range(K))
            shape_tuple = ", ".join(str(w) for w in widths)
            lines.append(f"ct.store(__output, index=({index_tuple},), tile=__src)")
        inputs = {"__src"} | ({"__mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code="\n".join(lines),
            language=dace.dtypes.Language.Python,
        )


@library.expansion
class ExpandTileStoreScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileStoreAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileStoreAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileStoreNeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileStoreSVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileStore(nodes.LibraryNode):
    """Store a K-dim tile back into a global array.

    ``_src`` is the tile transient (``widths``-shaped); ``_dst`` carries
    the full memlet of the destination array with the out-edge's subset
    selecting the tile region. ``dim_strides`` records per-tile-dim
    strides into the destination view.
    """

    implementations = {
        "pure": ExpandTileStorePure,
        "cutile": ExpandTileStoreCutile,
        "scalar": ExpandTileStoreScalar,
        "avx512": ExpandTileStoreAVX512,
        "avx2": ExpandTileStoreAVX2,
        "neon": ExpandTileStoreNeon,
        "sve": ExpandTileStoreSVE
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
        desc="Per-dim tile widths, innermost-last.",
    )
    dim_strides = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim index coefficient; all 1s ⇒ unit step along each tile dim.",
    )
    dst_dims = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim destination-array dimension the tile dim maps to "
        "(innermost-last). Empty ⇒ the last K dims in order; a transposed / "
        "non-last mapping lists the actual array dims so the store steps along "
        "the correct axis.",
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
                 dim_strides: Optional[Tuple[int, ...]] = None,
                 dst_dims: Optional[Tuple[int, ...]] = None,
                 has_mask: bool = False,
                 location: Optional[str] = None):
        """Construct a ``TileStore`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param dim_strides: Per-tile-dim stride coefficients; defaults
            to all 1s (contiguous).
        :param has_mask: When True, declare the ``_mask`` input.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3 or
            if ``dim_strides`` length disagrees with ``widths``.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileStore: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(f"TileStore: dim_strides length {len(dim_strides)} != widths length {len(widths)}")
        inputs = {"_src"} | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.dst_dims = list(dst_dims) if dst_dims else []
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
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
