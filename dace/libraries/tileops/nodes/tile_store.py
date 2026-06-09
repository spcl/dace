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

from .._pure_codegen import (gather_lane_offset, nested_loops, offset_via_strides, resolve_gather_deps, tile_offset)
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
        coeff = list(node.dim_strides) if node.dim_strides else [1] * K
        gather_set = set(node.gather_dims)
        if not gather_set:
            # Structured store: per-tile-dim affine path.
            dst_strides_tile = [symstr(dst_arr.strides[d]) for d in dims]
            dst_off = offset_via_strides(coeff, dst_strides_tile)
        else:
            # Dest-dim addressing (design section 9.3). Per DEST dim k in range(ndim):
            #   k in gather_dims -> `_idx_<k>[<flat lane>] * dst.strides[k]`
            #   k mapped to tile dim d via dst_dims -> `coeff[d] * dst.strides[k] * __l<d>`
            #   otherwise (untouched by the tile) -> outer `_dst` memlet carries the base offset.
            gather_idx_ref = {}
            for k in node.gather_dims:
                conn = f"_idx_{k}"
                edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == conn)
                idx_shape = tuple(parent_sdfg.arrays[edge.data.data].shape)
                deps_d = resolve_gather_deps(idx_shape, widths)
                if deps_d is None:
                    raise ValueError(f"{node.label}: cannot resolve deps for '{conn}' shape "
                                     f"{idx_shape} against widths {tuple(widths)}")
                gather_idx_ref[k] = gather_lane_offset(deps_d, widths, conn)
            dst_to_tile = {dims[d]: d for d in range(K)}
            parts = []
            for k in range(ndim):
                s = symstr(dst_arr.strides[k])
                if k in gather_set:
                    parts.append(f"(({gather_idx_ref[k]}) * ({s}))")
                elif k in dst_to_tile:
                    d = dst_to_tile[k]
                    parts.append(f"({coeff[d]} * ({s}) * __l{d})")
                # else: dest dim k untouched; outer base pointer covers it.
            dst_off = " + ".join(parts) if parts else "0"
        src_off = tile_offset(widths)
        # Resolve the per-lane source reference for each ``src_kind``:
        #   * ``Tile`` — the existing per-lane tile read.
        #   * ``Symbol`` — the literal / expression broadcast to every lane,
        #     cast to the destination dtype so a typed store resolves.
        #   * ``Scalar`` — a length-1 array read, broadcast to every lane.
        out_dtype = dst_arr.dtype.ctype
        if node.src_kind == "Symbol":
            src_ref = f"({out_dtype})({node.src_expr})"
        elif node.src_kind == "Scalar":
            src_ref = f"({out_dtype})(_src[0])"
        else:
            src_ref = f"_src[{src_off}]"
        if node.has_mask:
            body = f"if (_mask[{src_off}]) {{ _dst[{dst_off}] = {src_ref}; }}"
        else:
            body = f"_dst[{dst_off}] = {src_ref};"
        code = nested_loops(widths, body)
        inputs = (set() if node.src_kind == "Symbol" else {"_src"}) | ({"_mask"} if node.has_mask else set())
        inputs |= {f"_idx_{d}" for d in node.gather_dims}
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
        raise NotImplementedError(
            "ExpandTileStoreCutile: cuTile expansion stubbed out during G3 step 3 migration; the unified `TileLoad` / `TileStore` (with `gather_dims`) cuTile path will be reinstated after the per-source-dim gather contract lands per design "
            "section 6.4. Pin a `pure` expansion via `sdfg.expand_library_nodes(implementation='pure')` to lower this node for now."
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
    src_kind = properties.Property(
        dtype=str,
        allow_none=False,
        default="Tile",
        desc="Source operand kind. 'Tile' (default) reads a ``widths``-shaped "
        "tile transient via ``_src``. 'Symbol' broadcasts ``src_expr`` (a "
        "symbolic expression / numeric literal) to every lane and omits the "
        "``_src`` connector. 'Scalar' broadcasts a length-1 array value read "
        "via ``_src``.",
    )
    src_expr = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when ``src_kind=='Symbol'``; "
        "ignored otherwise.",
    )
    wcr = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Optional write-conflict-resolution lambda (e.g. ``lambda a, b: a + b``) "
        "applied per dst element. Required when any ``dim_strides`` entry is 0 -- "
        "the broadcast / collapse-out semantic where multiple lanes write to the "
        "same destination address would race without WCR. Lowered to an atomic / "
        "reduction store by the per-arch expansion.",
    )
    gather_dims = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Sorted DEST-array dim indices that SCATTER. Mirror of :attr:`TileLoad.gather_dims` "
        "(source-array dim indexing) -- ``len(widths) == K_tile`` and ``max(gather_dims) < dst_ndim`` "
        "(``dst_ndim`` read from the wired ``_dst`` edge at ``validate()`` time). Each ``d`` declares "
        "an ``_idx_<d>`` input connector whose descriptor shape is a Cartesian product of widths over "
        "the tile dims its scatter expression depends on (design section 9.2). Empty = structured "
        "store.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 dim_strides: Optional[Tuple[int, ...]] = None,
                 dst_dims: Optional[Tuple[int, ...]] = None,
                 has_mask: bool = False,
                 src_kind: str = "Tile",
                 src_expr: Optional[str] = None,
                 wcr: Optional[str] = None,
                 gather_dims: Optional[Tuple[int, ...]] = None,
                 location: Optional[str] = None):
        """Construct a ``TileStore`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param dim_strides: Per-tile-dim stride coefficients; defaults
            to all 1s (contiguous).
        :param has_mask: When True, declare the ``_mask`` input.
        :param src_kind: Source operand shape — ``"Tile"`` (default),
            ``"Symbol"`` (broadcast ``src_expr`` to every lane; ``_src``
            omitted), or ``"Scalar"`` (broadcast a length-1 array read
            via ``_src``).
        :param src_expr: Required when ``src_kind == 'Symbol'``.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3, if
            ``dim_strides`` length disagrees with ``widths``, or if
            ``src_kind`` is unsupported.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileStore: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(f"TileStore: dim_strides length {len(dim_strides)} != widths length {len(widths)}")
        if src_kind not in ("Tile", "Symbol", "Scalar"):
            raise ValueError(f"TileStore: src_kind must be one of {{'Tile', 'Symbol', 'Scalar'}}, got {src_kind!r}")
        if src_kind == "Symbol" and not src_expr:
            raise ValueError("TileStore: src_kind='Symbol' requires a non-empty src_expr")
        resolved_dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        if any(s == 0 for s in resolved_dim_strides) and not wcr:
            raise ValueError(f"TileStore: dim_strides {resolved_dim_strides!r} contains a 0 (collapse-out / "
                             "broadcast write); WCR is required to avoid races. Pass ``wcr='lambda a, b: a + b'`` "
                             "(or another reduction lambda) when collapsing tile dims to a shared destination.")
        # Validate gather_dims: sorted, unique, non-negative dest-dim indices.
        # The upper bound (max(gather_dims) < dst_ndim) is checked at validate() time since
        # ``dst_ndim`` depends on the wired ``_dst`` connector descriptor (design section 9.3).
        g = tuple(gather_dims) if gather_dims else ()
        if g != tuple(sorted(g)) or len(set(g)) != len(g) or any(d < 0 for d in g):
            raise ValueError(f"TileStore: gather_dims must be a sorted tuple of unique non-negative "
                             f"dest-dim indices; got {g!r}")
        # ``Symbol`` source has no ``_src`` connector — the literal is
        # embedded inline at expansion time. ``Tile`` and ``Scalar`` both
        # read through ``_src``.
        inputs = (set() if src_kind == "Symbol" else {"_src"}) | ({"_mask"} if has_mask else set())
        inputs |= {f"_idx_{d}" for d in g}
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = resolved_dim_strides
        self.dst_dims = list(dst_dims) if dst_dims else []
        self.has_mask = has_mask
        self.src_kind = src_kind
        self.src_expr = src_expr
        self.wcr = wcr
        self.gather_dims = list(g)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Check connectors + index-tile shape contract (design section 9.4).

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected, an
            index tile's descriptor shape is not a Cartesian product of
            widths, or its dtype is not in ``{int32, int64}``.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if self.src_kind != "Symbol" and "_src" not in in_e:
            raise ValueError(f"{self.label}: required input '_src' not connected (src_kind={self.src_kind!r})")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        if self.has_mask:
            from .._pure_codegen import validate_mask_descriptor_lock
            mask_arr = sdfg.arrays[in_e["_mask"].data.data]
            validate_mask_descriptor_lock(self.label, "_mask", mask_arr, tuple(self.widths))
        # Packed-layout lock (design section 2.3): refuse non-C non-Fortran dest strides.
        from .._pure_codegen import validate_packed_layout
        dst_arr = sdfg.arrays[out_e["_dst"].data.data]
        validate_packed_layout(self.label, "_dst", dst_arr)
        # gather_dims dest-dim upper bound + per-dim index-tile shape contract (design section 9.4).
        widths = tuple(self.widths)
        allowed_dtypes = {dace.int32, dace.int64}
        if self.gather_dims:
            dst_arr = sdfg.arrays[out_e["_dst"].data.data]
            dst_ndim = len(dst_arr.shape)
            if any(d >= dst_ndim for d in self.gather_dims):
                raise ValueError(f"{self.label}: gather_dims {tuple(self.gather_dims)} contains an index >= "
                                 f"dest ndim {dst_ndim} (dest '{out_e['_dst'].data.data}' shape "
                                 f"{tuple(dst_arr.shape)})")
        for d in self.gather_dims:
            conn = f"_idx_{d}"
            if conn not in in_e:
                raise ValueError(f"{self.label}: gather_dims includes {d} but '{conn}' is not connected")
            desc = sdfg.arrays[in_e[conn].data.data]
            shape = tuple(desc.shape)
            if resolve_gather_deps(shape, widths) is None:
                raise ValueError(f"{self.label}: '_idx_{d}' descriptor shape {shape} is not a Cartesian "
                                 f"product of widths {widths} for any sorted subset of tile dims "
                                 f"(design section 9.2)")
            if desc.dtype not in allowed_dtypes:
                raise ValueError(f"{self.label}: '_idx_{d}' dtype {desc.dtype} not in "
                                 f"{{int32, int64}} (design section 10.4)")
