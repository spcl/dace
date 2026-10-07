# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileScatter``: the stores of a tile that a masked copy cannot express, as a loop over the lanes.

Symmetric to :class:`~dace.libraries.tileops.nodes.tile_gather.TileGather`.
"""

import sympy

import dace
from dace import library, properties
from dace.codegen.cppunparse import pyexpr2cpp
from dace.sdfg import nodes

from dace.libraries.tileops.kinds import SCALAR, SYMBOL, TILE, VALID_KINDS
from dace.libraries.tileops.expansions import ExpandTilePure
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.lanes import (
    GATHER_INDEX_DTYPES,
    gather_lane_offset,
    nested_loops,
    offset_via_strides,
    resolve_gather_deps,
    tile_offset,
)
from dace.libraries.tileops.operands import scalar_operand_ref
from dace.libraries.tileops.validation import validate_mask_descriptor_lock, validate_packed_layout
from dace.optionals import required
from dace.sdfg.narrowing import as_range


@library.expansion
class ExpandTileScatterPure(ExpandTilePure):
    pass


def stride_dim_may_scatter(p: int, dst_dims: tuple[int, ...] | None, gather_dims: tuple[int, ...]) -> bool:
    """Whether tile dim ``p`` may legitimately carry a zero ``dim_strides`` entry.

    A zero stride on tile dim ``p`` means lane ``__l<p>`` does not advance the dest address. That
    is legal when ``p`` SCATTERS -- its dest dim is in ``gather_dims`` and the per-lane address
    comes from ``_idx_<d>`` (symmetric to ``TileGather`` gather, which never rejects zero strides).
    On a non-scatter dim a zero stride collapses all ``W_p`` lanes onto one address and races
    without WCR.

    ``dst_dims=None`` selects the innermost-K default binding whose exact dest-dim indices need
    ``dst_ndim`` (not known at construction time), so the precise per-dim check defers to
    :meth:`TileScatter.validate`; here we report ``True`` whenever any scatter dim exists.
    """
    if not gather_dims:
        return False
    if dst_dims is None:
        return True
    return dst_dims[p] in gather_dims


@library.node
class TileScatter(TileOp):
    """Store a K-dim tile into a global array in a way a masked copy cannot.

    ``_src`` is the tile and ``_dst`` carries the memlet of the destination array, which selects the tile region. The
    lanes address it through ``dim_strides`` (``0`` collapses a dim and needs ``wcr``), ``dst_dims`` (a transposed
    tile) or ``_idx_<d>`` index tiles (``gather_dims``); ``src_kind`` broadcasts a scalar or a symbol to every lane
    instead. A window the tile copies lane for lane is
    :class:`~dace.libraries.tileops.nodes.masked_copy.MaskedCopyLibraryNode`. The only lowering is the loop over the
    lanes.
    """

    implementations = {"pure": ExpandTileScatterPure}
    default_implementation = "pure"

    INPUT_CONNECTOR_NAME = "_src"
    OUTPUT_CONNECTOR_NAME = "_dst"

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
        default=TILE,
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
        desc="Symbolic expression embedded inline when ``src_kind=='Symbol'``; ignored otherwise.",
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
        desc="Sorted DEST-array dim indices that SCATTER. Mirror of :attr:`TileGather.gather_dims` "
        "(source-array dim indexing) -- ``len(widths) == K_tile`` and ``max(gather_dims) < dst_ndim`` "
        "(``dst_ndim`` read from the wired ``_dst`` edge at ``validate()`` time). Each ``d`` declares "
        "an ``_idx_<d>`` input connector whose descriptor shape is a Cartesian product of widths over "
        "the tile dims its scatter expression depends on (design section 9.2). Empty = structured "
        "store.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        dim_strides: tuple[int, ...] | None = None,
        dst_dims: tuple[int, ...] | None = None,
        has_mask: bool = False,
        src_kind: str = TILE,
        src_expr: str | None = None,
        wcr: str | None = None,
        gather_dims: tuple[int, ...] | None = None,
        location: str | None = None,
    ):
        """Construct a ``TileScatter`` node.

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
            raise ValueError(f"TileScatter: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(f"TileScatter: dim_strides length {len(dim_strides)} != widths length {len(widths)}")
        if src_kind not in VALID_KINDS:
            raise ValueError(f"TileScatter: src_kind must be one of {{'Tile', 'Symbol', 'Scalar'}}, got {src_kind!r}")
        if src_kind == SYMBOL and not src_expr:
            raise ValueError("TileScatter: src_kind='Symbol' requires a non-empty src_expr")
        resolved_dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        # Validate gather_dims: sorted, unique, non-negative dest-dim indices.
        # The upper bound (max(gather_dims) < dst_ndim) is checked at validate() time since
        # ``dst_ndim`` depends on the wired ``_dst`` connector descriptor (design section 9.3).
        g = tuple(gather_dims) if gather_dims else ()
        if g != tuple(sorted(g)) or len(set(g)) != len(g) or any(d < 0 for d in g):
            raise ValueError(
                f"TileScatter: gather_dims must be a sorted tuple of unique non-negative dest-dim indices; got {g!r}"
            )
        # Zero-stride collapse guard, narrowed to exempt SCATTER tile dims (see
        # :func:`stride_dim_may_scatter`). A zero on a scatter dim addresses per-lane via
        # ``_idx_<d>`` (legal, symmetric to ``TileGather``); a zero on a non-scatter dim collapses
        # ``W_p`` lanes onto one address and races without ``wcr``. The exact per-dim mapping when
        # ``dst_dims is None`` defers to ``validate()`` (needs ``dst_ndim``).
        if not wcr and any(
            s == 0 and not stride_dim_may_scatter(p, dst_dims, g) for p, s in enumerate(resolved_dim_strides)
        ):
            raise ValueError(
                f"TileScatter: dim_strides {resolved_dim_strides!r} has a 0 on a non-scatter tile "
                "dim (collapse-out / broadcast write); WCR is required to avoid races. Pass "
                "``wcr='lambda a, b: a + b'`` (or another reduction lambda) when collapsing tile "
                "dims to a shared destination, or wire the dim as a scatter (gather_dims + _idx)."
            )
        # ``Symbol`` source has no ``_src`` connector — the literal is
        # embedded inline at expansion time. ``Tile`` and ``Scalar`` both
        # read through ``_src``.
        inputs = (set() if src_kind == SYMBOL else {"_src"}) | ({"_mask"} if has_mask else set())
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
            widths, or its dtype is not one of ``GATHER_INDEX_DTYPES``.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if self.src_kind != SYMBOL and "_src" not in in_e:
            raise ValueError(f"{self.label}: required input '_src' not connected (src_kind={self.src_kind!r})")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
        if self.has_mask:
            mask_arr = sdfg.arrays[required(in_e["_mask"].data.data)]
            validate_mask_descriptor_lock(self.label, "_mask", mask_arr, tuple(self.widths))
        # Packed-layout lock (design section 2.3): refuse non-C non-Fortran dest strides.
        dst_arr = sdfg.arrays[required(out_e["_dst"].data.data)]
        validate_packed_layout(self.label, "_dst", dst_arr)
        # gather_dims dest-dim upper bound + per-dim index-tile shape contract (design section 9.4).
        widths = tuple(self.widths)
        if self.gather_dims:
            dst_arr = sdfg.arrays[required(out_e["_dst"].data.data)]
            dst_ndim = len(dst_arr.shape)
            if any(d >= dst_ndim for d in self.gather_dims):
                raise ValueError(
                    f"{self.label}: gather_dims {tuple(self.gather_dims)} contains an index >= "
                    f"dest ndim {dst_ndim} (dest '{out_e['_dst'].data.data}' shape "
                    f"{tuple(dst_arr.shape)})"
                )
        for d in self.gather_dims:
            conn = f"_idx_{d}"
            if conn not in in_e:
                raise ValueError(f"{self.label}: gather_dims includes {d} but '{conn}' is not connected")
            desc = sdfg.arrays[required(in_e[conn].data.data)]
            shape = tuple(desc.shape)
            if resolve_gather_deps(shape, widths) is None:
                raise ValueError(
                    f"{self.label}: '_idx_{d}' descriptor shape {shape} is not a Cartesian "
                    f"product of widths {widths} for any sorted subset of tile dims "
                    f"(design section 9.2)"
                )
            if desc.dtype not in GATHER_INDEX_DTYPES:
                raise ValueError(
                    f"{self.label}: '_idx_{d}' dtype {desc.dtype} not in {GATHER_INDEX_DTYPES} (design section 10.4)"
                )
        # Zero-stride collapse guard (precise; design section 3.5 + 5.1). A zero ``dim_strides[p]``
        # is legal only when tile dim ``p`` scatters -- its dest dim is in ``gather_dims`` so the
        # per-lane address comes from ``_idx_<d>``. On any other dim a zero stride collapses all
        # ``W_p`` lanes onto one dest address and races without ``wcr``. ``dst_dims`` defaults to the
        # innermost K dest dims; ``dst_ndim`` is read from the wired ``_dst`` edge here.
        if not self.wcr:
            K = len(widths)
            resolved_dst = (
                list(self.dst_dims) if self.dst_dims else list(range(len(dst_arr.shape) - K, len(dst_arr.shape)))
            )
            g_set = set(self.gather_dims)
            collapsed = [p for p, s in enumerate(self.dim_strides) if s == 0 and resolved_dst[p] not in g_set]
            if collapsed:
                raise ValueError(
                    f"{self.label}: dim_strides {tuple(self.dim_strides)} has a 0 on non-scatter tile "
                    f"dim(s) {collapsed} (dest dims {[resolved_dst[p] for p in collapsed]} not in gather_dims "
                    f"{tuple(self.gather_dims)}); a collapse-out / broadcast write races without WCR. Provide "
                    f"``wcr`` or wire the dim as a scatter (gather_dims + _idx)."
                )
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
                subset_sizes = tuple(as_range(dst_subset).size())
            except Exception:
                subset_sizes = None
            if subset_sizes is not None:
                expected = tuple(widths[i] for i in range(K))
                actual = tuple(subset_sizes[d] for d in dims) if max(dims, default=-1) < len(subset_sizes) else None
                # Compare each per-dim extent to its width with symbol-name reconciliation:
                # a full-tile size arrives as ``end - begin + W`` whose ``begin``/``end`` are the
                # SAME iterator under DIFFERENT assumption objects, so it does NOT self-cancel under
                # a plain ``simplify`` (``i - i`` stays). ``inequal_symbols`` equalizes same-name
                # symbols first, so a genuine full tile reads equal and only a real partial-tile size
                # trips the guard.
                if actual is None or any(
                    dace.symbolic.inequal_symbols(sympy.sympify(a), sympy.sympify(e)) for a, e in zip(actual, expected)
                ):
                    raise NotImplementedError(
                        f"{self.label}: non-full-tile structured store -- dest memlet "
                        f"subset sizes {subset_sizes} on dims {dims} != widths {expected}. Per user "
                        f"direction (design section 6.7 phasing): scalar / partial-tile / single-element "
                        f"writes to a global array raise NotImplementedError until the reduction "
                        f"(scalar transient -> single element) and single-element tile-load paths "
                        f"are designed. Use a scalar transient + TileReduce for accumulator stores; "
                        f"single-element writes are deferred."
                    )

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.symbolic import symstr

        widths = list(self.widths)
        K = len(widths)
        dst_edge = next(e for e in state.out_edges(self) if e.src_conn == "_dst")
        dst_arr = sdfg.arrays[required(dst_edge.data.data)]
        ndim = len(dst_arr.strides)
        # Step along the array dim each tile dim maps to (``dst_dims``);
        # default to the last K dims in order (a plain row-major tile).
        dims = list(self.dst_dims) if self.dst_dims else list(range(ndim - K, ndim))
        coeff = list(self.dim_strides) if self.dim_strides else [1] * K
        gather_set = set(self.gather_dims)
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
            for k in self.gather_dims:
                conn = f"_idx_{k}"
                edge = next(e for e in state.in_edges(self) if e.dst_conn == conn)
                idx_shape = tuple(required(sdfg.arrays[required(edge.data.data)]).shape)
                deps_d = resolve_gather_deps(idx_shape, widths)
                if deps_d is None:
                    raise ValueError(
                        f"{self.label}: cannot resolve deps for '{conn}' shape "
                        f"{idx_shape} against widths {tuple(widths)}"
                    )
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
        #   * ``Scalar`` — a volume-1 source passed by value, broadcast to every
        #     lane (or a tile-shape source widened upstream, read per lane).
        out_dtype = dst_arr.dtype.ctype
        if self.src_kind == SYMBOL:
            src_ref = f"({out_dtype})({pyexpr2cpp(self.src_expr)})"
        elif self.src_kind == SCALAR:
            # A volume-1 source is passed by value (bare ``_src``); a tile-shape
            # source widened upstream is a pointer read per lane (``_src[off]``).
            # ``[0]`` is a memlet concern, not a tasklet-body one.
            src_desc = sdfg.arrays[required(next(e for e in state.in_edges(self) if e.dst_conn == "_src").data.data)]
            ref, broadcast = scalar_operand_ref(src_desc, "_src", widths, src_off)
            src_ref = f"({out_dtype})({ref})" if broadcast else ref
        else:
            src_ref = f"_src[{src_off}]"
        if self.has_mask:
            body = f"if (_mask[{src_off}]) {{ _dst[{dst_off}] = {src_ref}; }}"
        else:
            body = f"_dst[{dst_off}] = {src_ref};"
        code = nested_loops(widths, body)
        inputs = (set() if self.src_kind == SYMBOL else {"_src"}) | ({"_mask"} if self.has_mask else set())
        inputs |= {f"_idx_{d}" for d in self.gather_dims}
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(inputs),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
