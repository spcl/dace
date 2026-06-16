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


def _stride_dim_may_scatter(p: int, dst_dims: Optional[Tuple[int, ...]], gather_dims: Tuple[int, ...]) -> bool:
    """Whether tile dim ``p`` may legitimately carry a zero ``dim_strides`` entry.

    A zero stride on tile dim ``p`` means lane ``__l<p>`` does not advance the dest address. That
    is legal when ``p`` SCATTERS -- its dest dim is in ``gather_dims`` and the per-lane address
    comes from ``_idx_<d>`` (symmetric to ``TileLoad`` gather, which never rejects zero strides).
    On a non-scatter dim a zero stride collapses all ``W_p`` lanes onto one address and races
    without WCR.

    ``dst_dims=None`` selects the innermost-K default binding whose exact dest-dim indices need
    ``dst_ndim`` (not known at construction time), so the precise per-dim check defers to
    :meth:`TileStore.validate`; here we report ``True`` whenever any scatter dim exists.
    """
    if not gather_dims:
        return False
    if dst_dims is None:
        return True
    return dst_dims[p] in gather_dims


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
        # K=1 ISA backends (scalar / avx512 / avx2 / neon / sve): a call into
        # dace/tile_ops/<backend>.h -- same call, the backend's env pulls in the
        # matching header. Built by the shared factory (selector routes K>=2 to
        # ``pure``).
        **_isa_codegen.make_isa_expansions("Store", _isa_codegen.make_store_tasklet, globals()),
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
        # ``pystr_to_symbolic`` accepts both int and symbolic (e.g. ``ssym``)
        # values, so ``a[i * ssym]`` AFFINE patterns can preserve the symbolic
        # stride through serialization. Codegen uses string interpolation on
        # each element, so a symbolic value inlines correctly as a C++ var.
        element_type=dace.symbolic.pystr_to_symbolic,
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
        # Validate gather_dims: sorted, unique, non-negative dest-dim indices.
        # The upper bound (max(gather_dims) < dst_ndim) is checked at validate() time since
        # ``dst_ndim`` depends on the wired ``_dst`` connector descriptor (design section 9.3).
        g = tuple(gather_dims) if gather_dims else ()
        if g != tuple(sorted(g)) or len(set(g)) != len(g) or any(d < 0 for d in g):
            raise ValueError(f"TileStore: gather_dims must be a sorted tuple of unique non-negative "
                             f"dest-dim indices; got {g!r}")
        # Zero-stride collapse guard, narrowed to exempt SCATTER tile dims (see
        # :func:`_stride_dim_may_scatter`). A zero on a scatter dim addresses per-lane via
        # ``_idx_<d>`` (legal, symmetric to ``TileLoad``); a zero on a non-scatter dim collapses
        # ``W_p`` lanes onto one address and races without ``wcr``. The exact per-dim mapping when
        # ``dst_dims is None`` defers to ``validate()`` (needs ``dst_ndim``).
        if not wcr and any(s == 0 and not _stride_dim_may_scatter(p, dst_dims, g)
                           for p, s in enumerate(resolved_dim_strides)):
            raise ValueError(f"TileStore: dim_strides {resolved_dim_strides!r} has a 0 on a non-scatter tile "
                             "dim (collapse-out / broadcast write); WCR is required to avoid races. Pass "
                             "``wcr='lambda a, b: a + b'`` (or another reduction lambda) when collapsing tile "
                             "dims to a shared destination, or wire the dim as a scatter (gather_dims + _idx).")
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
        # Zero-stride collapse guard (precise; design section 3.5 + 5.1). A zero ``dim_strides[p]``
        # is legal only when tile dim ``p`` scatters -- its dest dim is in ``gather_dims`` so the
        # per-lane address comes from ``_idx_<d>``. On any other dim a zero stride collapses all
        # ``W_p`` lanes onto one dest address and races without ``wcr``. ``dst_dims`` defaults to the
        # innermost K dest dims; ``dst_ndim`` is read from the wired ``_dst`` edge here.
        if not self.wcr:
            K = len(widths)
            resolved_dst = (list(self.dst_dims) if self.dst_dims else list(
                range(len(dst_arr.shape) - K, len(dst_arr.shape))))
            g_set = set(self.gather_dims)
            collapsed = [p for p, s in enumerate(self.dim_strides) if s == 0 and resolved_dst[p] not in g_set]
            if collapsed:
                raise ValueError(
                    f"{self.label}: dim_strides {tuple(self.dim_strides)} has a 0 on non-scatter tile "
                    f"dim(s) {collapsed} (dest dims {[resolved_dst[p] for p in collapsed]} not in gather_dims "
                    f"{tuple(self.gather_dims)}); a collapse-out / broadcast write races without WCR. Provide "
                    f"``wcr`` or wire the dim as a scatter (gather_dims + _idx).")
        # Full-tile write contract (per user direction 2026-06-09): the destination memlet's
        # per-dim subset extents must match ``widths`` exactly under the ``dst_dims`` permutation.
        # Anything else -- partial-tile writes, single-element writes, scalar writes to global --
        # raises NotImplementedError so the orchestrator surfaces the gap loudly. Reductions
        # (scalar transient -> single-element global write) will be lowered via a dedicated
        # reduction path; non-full structured writes will land via a per-lane Python tasklet or
        # a single-element tile load once the design is final. Skip the check when
        # ``gather_dims`` is non-empty (scatter mode: the dest memlet covers the full dest array
        # and the per-lane addressing comes from the ``_idx_<k>`` connectors instead).
        if not self.gather_dims:
            dst_subset = out_e["_dst"].data.subset
            K = len(widths)
            # The dst memlet's subset spans the full dest array; extract its per-dim size and
            # compare to widths under the ``dst_dims`` permutation. ``dst_dims`` defaults to the
            # innermost K dims in order.
            dims = list(self.dst_dims) if self.dst_dims else list(range(len(dst_arr.shape) - K, len(dst_arr.shape)))
            try:
                subset_sizes = tuple(dst_subset.size())
            except Exception:  # noqa: BLE001 -- symbolic / non-Range subsets currently allowed (refused at codegen)
                subset_sizes = None
            if subset_sizes is not None:
                expected = tuple(widths[i] for i in range(K))
                actual = tuple(subset_sizes[d] for d in dims) if max(dims, default=-1) < len(subset_sizes) else None
                if actual is None or any(bool(dace.symbolic.simplify(a - e)) != 0 for a, e in zip(actual, expected)):
                    raise NotImplementedError(
                        f"{self.label}: non-full-tile structured store -- dest memlet "
                        f"subset sizes {subset_sizes} on dims {dims} != widths {expected}. Per user "
                        f"direction (design section 6.7 phasing): scalar / partial-tile / single-element "
                        f"writes to a global array raise NotImplementedError until the reduction "
                        f"(scalar transient -> single element) and single-element tile-load paths "
                        f"are designed. Use a scalar transient + TileReduce for accumulator stores; "
                        f"single-element writes are deferred.")
