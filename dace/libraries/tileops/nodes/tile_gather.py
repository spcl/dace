# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileGather``: the loads of a tile that a masked copy cannot express.

A load through index tiles, a replicated, transposed or broadcast one. The pure expansion is a loop over the lanes,
which addresses the source with its strides; only the gather of a unit-stride 1-D array has an ISA lowering.
"""
from collections.abc import Sequence

import sympy

import dace
from dace import library, properties
from dace.codegen.cppunparse import pyexpr2cpp
from dace.sdfg import graph, nodes

from dace.libraries.tileops.kinds import SCALAR, SYMBOL, TILE, VALID_KINDS
from dace.libraries.tileops.environments import TileOpsAVX2, TileOpsAVX512, TileOpsCUDA, TileOpsNeon, TileOpsScalar, TileOpsSVE
from dace.libraries.tileops.expansions import ExpandTileIsa, ExpandTilePure
from dace.libraries.tileops.isa import require_k1
from dace.libraries.tileops.operands import connected_edges, edge_ctype, output_edge
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.libraries.tileops.lanes import GATHER_INDEX_DTYPES, gather_lane_offset, nested_loops, offset_via_strides, resolve_gather_deps, tile_offset
from dace.libraries.tileops.operands import scalar_operand_ref
from dace.libraries.tileops.validation import validate_mask_descriptor_lock, validate_packed_layout
from dace.optionals import required
from dace.sdfg.narrowing import as_range


def enclosing_map_params(parent_state: dace.SDFGState, node: nodes.Node) -> list[str]:
    """All map iter-var names enclosing ``node``, across nested-SDFG levels.

    The tile body is nested one (or more) levels below the tile map
    (``NestInnermostMapBodyIntoNSDFG``), so the map iter-var is a *free symbol*
    of the inner SDFG, not a scope entry of ``parent_state``. Walk up: collect
    map params in the current state's scope, then ascend through the owning
    SDFG's ``parent_nsdfg_node`` into the outer state and repeat.

    :param parent_state: State directly owning ``node``.
    :param node: The node whose enclosing maps are sought.
    :returns: Map param names from innermost to outermost (names as they appear
        in the SDFG; identity-preserved by the body-nesting pass).
    """
    params: list[str] = []
    cur_state = parent_state
    cur_node = node
    while cur_state is not None:
        sd = cur_state.scope_dict()
        p = sd.get(cur_node)
        while p is not None:
            if isinstance(p, nodes.MapEntry):
                params.extend(p.map.params)
            p = sd.get(p)
        owning_sdfg = cur_state.sdfg
        cur_node = owning_sdfg.parent_nsdfg_node
        cur_state = owning_sdfg.parent  # outer state holding the nested-SDFG node
    return params


def phase_aware_lane_exprs(node: "TileGather", parent_state: dace.SDFGState,
                           src_edge: graph.MultiConnectorEdge[dace.Memlet], dims: list[int],
                           replicate: Sequence[int | sympy.Basic]) -> list[str]:
    """Per-tile-dim per-lane source offset for non-dividing REPLICATE dims.

    For a REPLICATE dim whose factor ``D`` does not (provably) divide the tile
    width ``W`` -- a non-dividing static ``c[i // 3]`` (``W % 3 != 0``) or a
    symbolic divisor ``c[i // DV]`` -- the contracted-box broadcast
    ``_src[__l/D]`` over the base ``&src[(c*iter + c0)/D]`` is wrong unless every
    tile starts on a phase boundary (``W % D == 0`` ⇒ ``iter % D == 0``). This
    returns the phase-aware element offset RELATIVE to that base for lane
    ``__l<d>``::

        (c*iter + c0 + c*__l<d>) / D  -  (c*iter + c0) / D

    which reduces to the box ``__l<d> / D`` exactly when ``iter % D == 0``. The
    dividend ``c*iter + c0`` and divisor ``D`` are read from the source memlet's
    begin (an ``int_floor`` node); the iter-var symbol is resolved against the
    enclosing map scope so the rendered expression always uses the CURRENT
    (post-rename) name. Dims that don't need it get ``""`` (standard box /
    linear addressing). Integer ``/`` is floor for the non-negative index
    operands (the canonicalization non-negativity assumption).

    :param node: The ``TileGather`` being expanded.
    :param parent_state: State owning the node (for the map scope walk).
    :param src_edge: The ``_src`` in-edge (carries the source memlet).
    :param dims: Per-tile-dim source-array dim basis (``node.src_dims`` resolved).
    :param replicate: Per-tile-dim replicate factors (may be int or symbolic).
    :returns: A length-K list of per-lane offset C++ expressions; ``""`` where
        the standard box addressing applies.
    :raises NotImplementedError: On a non-``int_floor`` begin (e.g. ``int_ceil``)
        or a dividend that does not contain exactly one enclosing map iter-var.
    """
    widths = list(node.widths)
    K = len(widths)
    exprs = [""] * K
    # Collect enclosing map params -- the replicate dim's iter-var is one of them.
    map_params = enclosing_map_params(parent_state, node)
    for d in range(K):
        Dfac = replicate[d] if d < len(replicate) else 1
        try:
            Di = int(Dfac)
            if Di <= 1 or (int(widths[d]) % Di) == 0:
                continue  # no replicate, or D divides W -> the box path is correct
        except (TypeError, ValueError):
            if Dfac is None:
                continue  # no replicate
            # symbolic divisor -> can't prove W % D == 0 -> phase-aware
        begin = as_range(src_edge.data.subset).ranges[dims[d]][0]
        fname = type(begin).__name__
        if fname not in ("int_floor", "__int_floor"):
            raise NotImplementedError(f"{node.label}: non-dividing REPLICATE dim {d} expected an int_floor "
                                      f"begin in the source memlet, got {begin!r} ({fname}); int_ceil / "
                                      f"non-floor replicate-with-remainder is not yet supported.")
        dividend, divisor = begin.args
        div_syms = {str(s) for s in dividend.free_symbols}
        cand = [p for p in map_params if p in div_syms]
        if len(cand) != 1:
            raise NotImplementedError(f"{node.label}: non-dividing REPLICATE dim {d} dividend {dividend!r} must "
                                      f"contain exactly one enclosing map iter-var (found {cand} among {map_params}).")
        psym = next(s for s in dividend.free_symbols if str(s) == cand[0])
        from dace.symbolic import symstr
        dividend_lane = dividend.subs(psym, psym + sympy.Symbol(f"__l{d}"))
        div_str = symstr(divisor)
        exprs[d] = (f"(({symstr(dividend_lane)}) / ({div_str})) - "
                    f"(({symstr(dividend)}) / ({div_str}))")
    return exprs


@library.expansion
class ExpandTileGatherPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileGatherScalar(ExpandTileIsa):
    environments = [TileOpsScalar]
    backend = "scalar"


@library.expansion
class ExpandTileGatherAVX512(ExpandTileIsa):
    environments = [TileOpsAVX512]
    backend = "avx512"


@library.expansion
class ExpandTileGatherAVX2(ExpandTileIsa):
    environments = [TileOpsAVX2]
    backend = "avx2"


@library.expansion
class ExpandTileGatherNeon(ExpandTileIsa):
    environments = [TileOpsNeon]
    backend = "neon"


@library.expansion
class ExpandTileGatherSVE(ExpandTileIsa):
    environments = [TileOpsSVE]
    backend = "sve"


@library.expansion
class ExpandTileGatherCUDA(ExpandTileIsa):
    environments = [TileOpsCUDA]
    backend = "cuda"


@library.node
class TileGather(TileOp):
    """Load a K-dim tile out of a global array in a way a masked copy cannot.

    ``_src`` carries the memlet of the source array, which selects the tile region, and ``_dst`` is the tile. The
    lanes address the source through ``dim_strides`` (``0`` broadcasts a dim), ``replicate_factor_per_dim`` (lanes
    sharing a source element), ``src_dims`` (a transposed tile) or ``_idx_<d>`` index tiles (``gather_dims``);
    ``src_kind`` broadcasts a scalar or a symbol to every lane instead. A window the tile copies lane for lane is
    :class:`~dace.libraries.tileops.nodes.masked_copy.MaskedCopyLibraryNode`.
    """

    INPUT_CONNECTOR_NAME = "_src"
    OUTPUT_CONNECTOR_NAME = "_dst"

    implementations = {
        "pure": ExpandTileGatherPure,
        "scalar": ExpandTileGatherScalar,
        "avx512": ExpandTileGatherAVX512,
        "avx2": ExpandTileGatherAVX2,
        "neon": ExpandTileGatherNeon,
        "sve": ExpandTileGatherSVE,
        "cuda": ExpandTileGatherCUDA,
    }
    default_implementation = "pure"

    dim_strides = properties.ListProperty(
        # ``pystr_to_symbolic`` accepts both int and symbolic (e.g. ``ssym``)
        # values, so ``a[i * ssym]`` AFFINE patterns can preserve the symbolic
        # stride through serialization. Codegen uses string interpolation on
        # each element, so a symbolic value inlines correctly as a C++ var.
        element_type=dace.symbolic.pystr_to_symbolic,
        default=[],
        desc="Per-tile-dim index coefficient; all 1s ⇒ unit step along each tile dim.",
    )
    src_dims = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim source-array dimension the tile dim maps to "
        "(innermost-last). Empty ⇒ the last K dims in order; a transposed / "
        "non-last mapping lists the actual array dims so the load steps along "
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
        desc="Source kind. 'Tile' (default) reads the per-lane indexed element from a "
        "tile transient / strided view via ``_src``. 'Symbol' broadcasts ``src_expr`` "
        "(a numeric literal or in-scope symbolic expression) to every lane, omitting "
        "the ``_src`` connector. 'Scalar' broadcasts a length-1 array / "
        "``dace.data.Scalar`` value read via ``_src``.",
    )
    src_expr = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Literal / symbolic expression for ``src_kind='Symbol'``; ignored otherwise.",
    )
    replicate_factor_per_dim = properties.ListProperty(
        # ``pystr_to_symbolic`` accepts both int and symbolic (e.g. ``DV``
        # in ``c[i // DV]``) values, mirroring the symbolic-stride fix in
        # commit 3e1dc18c0. The pure expansion uses string interpolation on
        # each element, so a symbolic value inlines correctly as a C++ var.
        element_type=dace.symbolic.pystr_to_symbolic,
        default=[],
        desc="Per-tile-dim replicate factor (lanes-per-distinct-value within "
        "the dim). ``1`` (or empty) = no replication, the contiguous endpoint "
        "of the spectrum. ``k > 1`` = ``int_floor`` / ``int_ceil`` regime: load "
        "``W_d / k`` elements on this dim and group-broadcast each ``k`` times "
        "across consecutive lanes. The codegen template covers all three "
        "regimes (factor=1 contiguous, 1<k<W grouped, k=W full broadcast) "
        "uniformly.",
    )
    gather_dims = properties.ListProperty(
        element_type=int,
        default=[],
        desc=
        "Sorted SOURCE-array dim indices that GATHER. For each ``d in gather_dims`` an ``_idx_<d>`` input connector is "
        "declared; the connector's descriptor shape is the Cartesian product of widths over the tile "
        "dims the gather expression depends on (lane-dependency rule). Lane geometry "
        "(``widths``) and source addressing (``gather_dims``) are orthogonal: ``len(widths) == K_tile`` "
        "and ``max(gather_dims) < src_ndim`` (checked at ``validate()`` time since ``src_ndim`` "
        "is read from the wired ``_src`` edge). Empty list = no gather (structured load). "
        "ICON-shape example: ``B[idx[i, k], j, idb[i, k]]`` vec(i, j) -> ``widths=(W_i, W_j)``, "
        "``gather_dims=(0, 2)``, ``_idx_0`` shape ``(W_i,)``, ``_idx_2`` shape ``(W_i,)``.",
    )

    def __init__(self,
                 name: str,
                 widths: tuple[int, ...],
                 dim_strides: tuple[int, ...] | None = None,
                 src_dims: tuple[int, ...] | None = None,
                 has_mask: bool = False,
                 src_kind: str = TILE,
                 src_expr: str | None = None,
                 replicate_factor_per_dim: tuple[int, ...] | None = None,
                 gather_dims: tuple[int, ...] | None = None,
                 location: str | None = None):
        """Construct a ``TileGather`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param dim_strides: Per-tile-dim stride coefficients; defaults
            to all 1s (contiguous).
        :param src_dims: Per-tile-dim source-array dim mapping (empty ⇒
            last K dims in order).
        :param has_mask: When True, declare the ``_mask`` input.
        :param src_kind: ``"Tile"`` (default; per-lane indexed read of a
            tile-shape ``_src``), ``"Scalar"`` (broadcast a length-1 array
            / ``dace.data.Scalar`` value read via ``_src``), or ``"Symbol"``
            (broadcast ``src_expr`` to every lane; ``_src`` connector is
            omitted).
        :param src_expr: Required when ``src_kind="Symbol"`` — the literal
            / symbolic expression broadcast to every lane; ignored
            otherwise.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3, if
            ``dim_strides`` length disagrees with ``widths``, if
            ``src_kind`` is unknown, or if ``src_kind="Symbol"`` is
            given without ``src_expr``.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileGather: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(f"TileGather: dim_strides length {len(dim_strides)} != widths length {len(widths)}")
        if src_kind not in VALID_KINDS:
            raise ValueError(f"TileGather: src_kind must be one of 'Tile' | 'Symbol' | 'Scalar', got {src_kind!r}")
        if src_kind == SYMBOL and not src_expr:
            raise ValueError("TileGather: src_kind='Symbol' requires a non-empty src_expr")
        if replicate_factor_per_dim is not None:
            if len(replicate_factor_per_dim) != len(widths):
                raise ValueError(f"TileGather: replicate_factor_per_dim length "
                                 f"{len(replicate_factor_per_dim)} != widths length {len(widths)}")
            for d, (w, k) in enumerate(zip(widths, replicate_factor_per_dim)):
                # The factor only needs to be a positive integer. Divisibility
                # ``W % k == 0`` is NOT required: when ``k`` divides ``W`` the
                # pure expansion emits the contiguous box ``__l/k``; otherwise
                # (a non-dividing static ``k``, or a symbolic divisor that can't
                # be proven to divide) it emits the phase-aware per-lane offset
                # ``(c*iter + c0 + c*__l)/k - (c*iter + c0)/k`` instead (see
                # :func:`phase_aware_lane_exprs`). Both are correct.
                try:
                    k_int = int(k)
                except (TypeError, ValueError):
                    continue  # symbolic -- the phase-aware expansion handles it
                if k_int < 1:
                    raise ValueError(f"TileGather: replicate_factor_per_dim[{d}] = {k_int} must be >= 1")
        # Validate gather_dims: sorted, unique, non-negative source-dim indices.
        # The upper bound (max(gather_dims) < src_ndim) is checked at validate() time since
        # ``src_ndim`` depends on the wired ``_src`` connector descriptor (design section 9.3).
        g = tuple(gather_dims) if gather_dims else ()
        if g != tuple(sorted(g)) or len(set(g)) != len(g) or any(d < 0 for d in g):
            raise ValueError(f"TileGather: gather_dims must be a sorted tuple of unique non-negative "
                             f"source-dim indices; got {g!r}")
        # ``Symbol`` source has no ``_src`` connector — the literal is embedded
        # inline at expansion time.
        inputs = (set() if src_kind == SYMBOL else {"_src"}) | ({"_mask"} if has_mask else set())
        inputs |= {f"_idx_{d}" for d in g}
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.src_dims = list(src_dims) if src_dims else []
        self.has_mask = has_mask
        self.src_kind = src_kind
        self.src_expr = src_expr
        self.gather_dims = list(g)
        self.replicate_factor_per_dim = (list(replicate_factor_per_dim) if replicate_factor_per_dim else [1] *
                                         len(widths))

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Check connectors + index-tile shape contract (design section 9.4).

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected, an index
            tile's descriptor shape is not a Cartesian product of widths, or
            the dtype is not one of ``GATHER_INDEX_DTYPES``.
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
        # Packed-layout lock (design section 2.3): refuse non-C non-Fortran source strides.
        if self.src_kind == TILE:
            src_arr = sdfg.arrays[required(in_e["_src"].data.data)]
            validate_packed_layout(self.label, "_src", src_arr)
        # gather_dims source-dim upper bound + per-dim index-tile shape contract (design section 9.4).
        widths = tuple(self.widths)
        if self.gather_dims and self.src_kind == TILE:
            src_arr = sdfg.arrays[required(in_e["_src"].data.data)]
            src_ndim = len(src_arr.shape)
            if any(d >= src_ndim for d in self.gather_dims):
                raise ValueError(f"{self.label}: gather_dims {tuple(self.gather_dims)} contains an index >= "
                                 f"source ndim {src_ndim} (source '{in_e['_src'].data.data}' shape "
                                 f"{tuple(src_arr.shape)})")
        for d in self.gather_dims:
            conn = f"_idx_{d}"
            if conn not in in_e:
                raise ValueError(f"{self.label}: gather_dims includes {d} but '{conn}' is not connected")
            desc = sdfg.arrays[required(in_e[conn].data.data)]
            shape = tuple(desc.shape)
            if resolve_gather_deps(shape, widths) is None:
                raise ValueError(f"{self.label}: '_idx_{d}' descriptor shape {shape} is not a Cartesian "
                                 f"product of widths {widths} for any sorted subset of tile dims "
                                 f"(design section 9.2)")
            if desc.dtype not in GATHER_INDEX_DTYPES:
                raise ValueError(f"{self.label}: '_idx_{d}' dtype {desc.dtype} not in "
                                 f"{GATHER_INDEX_DTYPES} (design section 10.4)")

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.symbolic import symstr
        widths = list(self.widths)
        K = len(widths)
        dst_off = tile_offset(widths)
        dst_dtype = required(sdfg.arrays[required(
            next(e for e in state.out_edges(self) if e.src_conn == "_dst").data.data)]).dtype.ctype
        if self.src_kind == SYMBOL:
            src_ref = f"({dst_dtype})({pyexpr2cpp(self.src_expr)})"
        elif self.src_kind == SCALAR:
            # A volume-1 source (a true ``dace.data.Scalar``, a length-1 Array,
            # or a single-element access) is passed by value (``T _src``) and
            # referenced bare; a tile-shape source widened upstream is a pointer
            # read per lane (``_src[off]``). ``[0]`` is a memlet concern, never a
            # tasklet-body one (a by-value connector is not a pointer).
            src_edge = next(e for e in state.in_edges(self) if e.dst_conn == "_src")
            desc = sdfg.arrays[required(src_edge.data.data)]
            ref, broadcast = scalar_operand_ref(desc, "_src", widths, dst_off)
            src_ref = f"({dst_dtype})({ref})" if broadcast else ref
        else:
            src_edge = next(e for e in state.in_edges(self) if e.dst_conn == "_src")
            src_arr = sdfg.arrays[required(src_edge.data.data)]
            ndim = len(src_arr.strides)
            # Step along the array dim each tile dim maps to (``src_dims``);
            # default to the last K dims in order (a plain row-major tile).
            dims = list(self.src_dims) if self.src_dims else list(range(ndim - K, ndim))
            coeff = list(self.dim_strides) if self.dim_strides else [1] * K
            replicate = list(self.replicate_factor_per_dim) if self.replicate_factor_per_dim else [1] * K
            gather_set = set(self.gather_dims)
            if not gather_set:
                # Structured path: per-tile-dim affine contributions only.
                src_strides_tile = [symstr(src_arr.strides[d]) for d in dims]
                # Non-dividing REPLICATE (``W % D != 0`` or symbolic ``D``) gets a
                # phase-aware per-lane offset; dividing dims keep the contiguous
                # ``__l/D`` box (empty entry).
                lane_exprs = phase_aware_lane_exprs(self, state, src_edge, dims, replicate)
                src_off = offset_via_strides(coeff, src_strides_tile, replicate, lane_exprs)
            else:
                # Source-dim addressing (design section 9.2 / 9.3): per SOURCE dim k in range(ndim),
                # if k in gather_dims contribute `_idx_<k>[<flat lane>] * src.strides[k]`; otherwise,
                # if k is the tile-mapped source dim for some tile dim d (via src_dims), contribute
                # the affine `coeff[d] * src.strides[k] * (__l<d> / replicate[d])`; remaining source
                # dims fall outside the tile's reach -- their per-iteration index lives in the outer
                # `_src` memlet subset offset (the lib node addresses through ``_src[<offset>]`` and
                # the codegen-supplied base pointer carries everything not contributed here).
                gather_idx_ref = {}
                for k in self.gather_dims:
                    conn = f"_idx_{k}"
                    edge = next(e for e in state.in_edges(self) if e.dst_conn == conn)
                    idx_shape = tuple(required(sdfg.arrays[required(edge.data.data)]).shape)
                    deps_d = resolve_gather_deps(idx_shape, widths)
                    if deps_d is None:
                        raise ValueError(f"{self.label}: cannot resolve deps for '{conn}' shape "
                                         f"{idx_shape} against widths {tuple(widths)}")
                    gather_idx_ref[k] = gather_lane_offset(deps_d, widths, conn)
                src_to_tile = {dims[d]: d for d in range(K)}
                parts = []
                for k in range(ndim):
                    s = symstr(src_arr.strides[k])
                    if k in gather_set:
                        parts.append(f"(({gather_idx_ref[k]}) * ({s}))")
                    elif k in src_to_tile:
                        d = src_to_tile[k]
                        lane = f"__l{d}"
                        # Replicate factor: the box ``__l/D`` is correct only when
                        # ``D`` divides ``W`` (phase-0). A non-dividing / symbolic
                        # factor mixed with a gather dim would need the phase-aware
                        # offset the structured path emits, but the gather branch's
                        # base addressing differs -- refuse loudly rather than emit
                        # the phase-0-only box (no silent miscompile).
                        try:
                            Di = int(replicate[d])
                            if Di > 1 and (int(widths[d]) % Di) != 0:
                                raise NotImplementedError(
                                    f"{self.label}: non-dividing REPLICATE factor {Di} on tile dim {d} "
                                    f"(width {widths[d]}) mixed with a gather access is not supported "
                                    f"(phase-aware replicate-with-remainder is only wired on the "
                                    f"structured load path).")
                            emit_div = Di > 1
                        except (TypeError, ValueError):
                            raise NotImplementedError(
                                f"{self.label}: symbolic REPLICATE factor {replicate[d]!r} on tile dim {d} "
                                f"mixed with a gather access is not supported (cannot prove it divides "
                                f"width {widths[d]}; phase-aware replicate-with-remainder is only wired "
                                f"on the structured load path).")
                        if emit_div:
                            lane = f"({lane} / {replicate[d]})"
                        parts.append(f"({coeff[d]} * ({s}) * {lane})")
                    # else: source dim k has no per-lane contribution; outer base pointer covers it.
                src_off = " + ".join(parts) if parts else "0"
            src_ref = f"_src[{src_off}]"
        if self.has_mask:
            body = f"_dst[{dst_off}] = _mask[{dst_off}] ? {src_ref} : {dst_dtype}(0);"
        else:
            body = f"_dst[{dst_off}] = {src_ref};"
        code = nested_loops(widths, body)
        inputs = (set() if self.src_kind == SYMBOL else {"_src"}) | ({"_mask"} if self.has_mask else set())
        inputs |= {f"_idx_{d}" for d in self.gather_dims}
        tasklet = nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(inputs),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet

    def replicated(self) -> bool:
        """Whether lanes may share a source element, which the ``tile_gather`` header call cannot express.

        A symbolic factor counts: it cannot be shown to be 1.
        """
        for factor in self.replicate_factor_per_dim or []:
            try:
                if int(factor) > 1:
                    return True
            except (TypeError, ValueError):
                return True
        return False

    def is_unit_stride_gather(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        """Whether this is the gather ``a[idx[i]]`` the ``tile_gather`` header call covers.

        K=1, one gather dim on a 1-D source of unit stride, an index tile of one lane dim, and no replication. The
        source offset of lane ``l`` is then exactly ``_idx_0[l]``.
        """
        if len(self.widths) != 1 or len(self.gather_dims) != 1 or self.replicated() or int(self.gather_dims[0]) != 0:
            return False
        in_edges = connected_edges(state, self)
        source = sdfg.arrays[required(in_edges["_src"].data.data)]
        if len(source.shape) != 1 or not bool(dace.symbolic.simplify(source.strides[0] == 1)):
            return False
        index_edge = in_edges.get("_idx_0")
        if index_edge is None:
            return False
        try:
            index_shape = tuple(int(extent) for extent in required(sdfg.arrays[required(index_edge.data.data)]).shape)
        except (TypeError, ValueError):
            return False
        return index_shape == (int(self.widths[0]), )

    def can_lower_to_isa(self, state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
        return self.src_kind == TILE and self.is_unit_stride_gather(state, sdfg)

    def isa_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG, backend: str) -> nodes.Tasklet:
        in_edges = connected_edges(state, self)
        vlen = require_k1(self)
        destination_ctype = edge_ctype(sdfg, output_edge(state, self, "_dst"))
        index_ctype = edge_ctype(sdfg, in_edges["_idx_0"])
        masked = "true" if self.has_mask else "false"
        mask_argument = "_mask" if self.has_mask else "nullptr"
        code = (f"dace::tileops::tile_gather<{destination_ctype}, {index_ctype}, {vlen}, {masked}>"
                f"(_dst, _src, _idx_0, {mask_argument});")
        return nodes.Tasklet(
            label=f"{self.label}_{backend}",
            inputs=dict.fromkeys(["_src", *(["_mask"] if self.has_mask else []), "_idx_0"]),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
