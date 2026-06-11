# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileLoad`` — copy a K-dim tile out of a global array.

The pure expansion emits a CPP tasklet whose body walks the K-fold
nested index space using the source array's strides (which DaCe
codegen passes via ``__<arr>_strides`` from the surrounding scope).
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import (gather_lane_offset, nested_loops, offset_via_strides, resolve_gather_deps, tile_offset)
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE

#: Map the :attr:`TileLoad.pad_mode` property values to the cuTile
#: ``ct.PaddingMode`` enum members. cuTile's padding enum offers ``+inf``
#: (good for a downstream ``min`` reduction) but has **no** ``-inf`` / ``1``
#: member (so ``max`` / ``prod`` partial-tile identities cannot be installed
#: by load padding alone — they are routed to the reduction's pre-select;
#: see the L-pad-identity note in ``CUTILE_EXPANSION_DESIGN.md``).
_PAD_MODE_CUTE = {
    "ZERO": "ct.PaddingMode.ZERO",
    "NAN": "ct.PaddingMode.NAN",
    "POS_INF": "ct.PaddingMode.POSITIVE_INFINITY",
    "NEG_ZERO": "ct.PaddingMode.NEGATIVE_ZERO",
    "UNDETERMINED": "ct.PaddingMode.UNDETERMINED",
}


@library.expansion
class ExpandTileLoadPure(ExpandTransformation):
    """Correctness-only CPP tasklet copying the tile region into ``_dst``."""

    environments = []

    @staticmethod
    def expansion(node: "TileLoad", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that copies the tile region into the
        destination tile, optionally gated by ``_mask``.

        Source offsets use the source array's per-dim strides (read
        from the connector descriptor at expansion time) scaled by an
        optional :attr:`dim_strides` coefficient (defaulting to 1).

        Three source kinds (mirrors :class:`TileStore`):

        * ``src_kind="Tile"`` (default): ``_src`` is a tile-shape transient
          / strided view; standard per-lane indexed read.
        * ``src_kind="Scalar"``: ``_src`` is a length-1 array; every lane
          reads ``_src[0]`` (broadcast).
        * ``src_kind="Symbol"``: no ``_src`` connector; every lane writes
          the cast of :attr:`src_expr` (broadcast literal / symbolic).

        :param node: The ``TileLoad`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        K = len(widths)
        dst_off = tile_offset(widths)
        dst_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_dst").data.data].dtype.ctype
        if node.src_kind == "Symbol":
            src_ref = f"({dst_dtype})({node.src_expr})"
        elif node.src_kind == "Scalar":
            # DaCe passes a tasklet connector by value (``T _src``) for a true
            # ``dace.data.Scalar`` and for a single-element access into a
            # larger array (``a[j]``); only a genuine length-1 *array*
            # connector is a pointer (``T* _src``) needing ``_src[0]``.
            src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
            desc = parent_sdfg.arrays[src_edge.data.data]
            is_len1_array = (isinstance(desc, dace.data.Array)
                             and all(bool(dace.symbolic.simplify(s == 1)) for s in desc.shape))
            ref = "_src[0]" if is_len1_array else "_src"
            src_ref = f"({dst_dtype})({ref})"
        else:
            src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
            src_arr = parent_sdfg.arrays[src_edge.data.data]
            ndim = len(src_arr.strides)
            # Step along the array dim each tile dim maps to (``src_dims``);
            # default to the last K dims in order (a plain row-major tile).
            dims = list(node.src_dims) if node.src_dims else list(range(ndim - K, ndim))
            coeff = list(node.dim_strides) if node.dim_strides else [1] * K
            replicate = list(node.replicate_factor_per_dim) if node.replicate_factor_per_dim else [1] * K
            gather_set = set(node.gather_dims)
            if not gather_set:
                # Structured path: per-tile-dim affine contributions only.
                src_strides_tile = [symstr(src_arr.strides[d]) for d in dims]
                src_off = offset_via_strides(coeff, src_strides_tile, replicate)
            else:
                # Source-dim addressing (design section 9.2 / 9.3): per SOURCE dim k in range(ndim),
                # if k in gather_dims contribute `_idx_<k>[<flat lane>] * src.strides[k]`; otherwise,
                # if k is the tile-mapped source dim for some tile dim d (via src_dims), contribute
                # the affine `coeff[d] * src.strides[k] * (__l<d> / replicate[d])`; remaining source
                # dims fall outside the tile's reach -- their per-iteration index lives in the outer
                # `_src` memlet subset offset (the lib node addresses through ``_src[<offset>]`` and
                # the codegen-supplied base pointer carries everything not contributed here).
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
                src_to_tile = {dims[d]: d for d in range(K)}
                parts = []
                for k in range(ndim):
                    s = symstr(src_arr.strides[k])
                    if k in gather_set:
                        parts.append(f"(({gather_idx_ref[k]}) * ({s}))")
                    elif k in src_to_tile:
                        d = src_to_tile[k]
                        lane = f"__l{d}"
                        # Symbolic replicate factor: emit runtime division.
                        try:
                            emit_div = int(replicate[d]) > 1
                        except (TypeError, ValueError):
                            emit_div = True
                        if emit_div:
                            lane = f"({lane} / {replicate[d]})"
                        parts.append(f"({coeff[d]} * ({s}) * {lane})")
                    # else: source dim k has no per-lane contribution; outer base pointer covers it.
                src_off = " + ".join(parts) if parts else "0"
            src_ref = f"_src[{src_off}]"
        if node.has_mask:
            body = f"_dst[{dst_off}] = _mask[{dst_off}] ? {src_ref} : {dst_dtype}(0);"
        else:
            body = f"_dst[{dst_off}] = {src_ref};"
        code = nested_loops(widths, body)
        # Per user direction 2026-06-10: ``For SYM we can add a runtime check
        # and check for error ensuring W % SYM``. Emit a runtime ``W % SYM == 0``
        # assertion for each tile dim whose ``replicate_factor`` is symbolic.
        # The codegen formula ``__l / SYM`` is correct only when ``SYM`` divides
        # the tile width; statically-known factors are validated in
        # ``TileLoad.__init__``. The runtime check is added once (per
        # expansion), prepended to the per-lane loops so a mis-sized divisor
        # fails loudly with a descriptive abort message.
        if node.replicate_factor_per_dim:
            runtime_checks = []
            for d, k in enumerate(node.replicate_factor_per_dim):
                try:
                    int(k)
                    continue  # static -- already validated at construction
                except (TypeError, ValueError):
                    pass
                if k is None:
                    continue
                w_d = int(widths[d])
                runtime_checks.append(
                    f'if (({w_d}) % ({k}) != 0) {{ '
                    f'fprintf(stderr, "TileLoad runtime check failed: width[{d}]=%d not divisible by '
                    f'replicate_factor=%lld (formula __l/factor requires W %% factor == 0)\\n", '
                    f'{w_d}, (long long)({k})); '
                    f'std::abort(); }}'
                )
            if runtime_checks:
                code = "\n".join(runtime_checks) + "\n" + code
        inputs = (set() if node.src_kind == "Symbol" else {"_src"}) | ({"_mask"} if node.has_mask else set())
        inputs |= {f"_idx_{d}" for d in node.gather_dims}
        tasklet = nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@library.expansion
class ExpandTileLoadCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileLoad`.

    Emits ``ct.load(__src, index=(__pid0, ...), shape=(W_0, ...),
    padding_mode=...)`` — the contiguous block-tile read used by the
    reference cuTile kernels. ``ct.load`` has no ``mask=`` parameter
    (L-load-nomask), so mask gating is applied at the store side
    (:class:`TileStore` cutile via ``ct.scatter``) and ``has_mask`` does
    **not** add a ``__mask`` input here (the load body never reads it).
    The padding mode is selectable via :attr:`TileLoad.pad_mode` so the
    OOB tail of the last tile reads as the right identity for the
    downstream consumer (e.g. ``+inf`` ahead of a ``min`` reduction).
    """

    environments = []

    @staticmethod
    def expansion(node: "TileLoad", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(
            "ExpandTileLoadCutile: cuTile expansion stubbed out during G3 step 3 migration; the unified `TileLoad` / `TileStore` (with `gather_dims`) cuTile path will be reinstated after the per-source-dim gather contract lands per design "
            "section 6.4. Pin a `pure` expansion via `sdfg.expand_library_nodes(implementation='pure')` to lower this node for now."
        )


@library.expansion
class ExpandTileLoadScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_load_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileLoadAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_load_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileLoadAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_load_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileLoadNeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_load_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileLoadSVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_load_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileLoad(nodes.LibraryNode):
    """Load a K-dim tile out of a global array.

    ``_src`` carries the full memlet of the source array; the in-edge's
    subset selects the tile region. ``_dst`` is the tile transient
    (``widths``-shaped). ``dim_strides`` records the per-tile-dim stride
    coefficient applied to the source view, defaulting to all 1s
    (contiguous).
    """

    implementations = {
        "pure": ExpandTileLoadPure,
        "cutile": ExpandTileLoadCutile,
        "scalar": ExpandTileLoadScalar,
        "avx512": ExpandTileLoadAVX512,
        "avx2": ExpandTileLoadAVX2,
        "neon": ExpandTileLoadNeon,
        "sve": ExpandTileLoadSVE
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
    pad_mode = properties.Property(
        dtype=str,
        allow_none=False,
        default="ZERO",
        desc="cuTile OOB padding mode for the partial last tile, one of "
        "``ZERO | NAN | POS_INF | NEG_ZERO | UNDETERMINED`` mapping to the "
        "``ct.PaddingMode`` enum. Only the ``cutile`` expansion reads it. The "
        "orchestrator fusing a load into a reduction sets the right identity "
        "(``+`` → ZERO, ``min`` → POS_INF); ``max`` / ``prod`` have no padding "
        "identity in cuTile and rely on the reduction's pre-select instead.",
    )
    src_kind = properties.Property(
        dtype=str,
        allow_none=False,
        default="Tile",
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
        desc="Sorted SOURCE-array dim indices that GATHER (per TILIFICATION_TRANSFORMATION_DESIGN.md "
        "section 5 + section 9). For each ``d in gather_dims`` an ``_idx_<d>`` input connector is "
        "declared; the connector's descriptor shape is the Cartesian product of widths over the tile "
        "dims the gather expression depends on (section 9.2 lane-dependency rule). Lane geometry "
        "(``widths``) and source addressing (``gather_dims``) are orthogonal: ``len(widths) == K_tile`` "
        "and ``max(gather_dims) < src_ndim`` (checked at ``validate()`` time since ``src_ndim`` "
        "is read from the wired ``_src`` edge). Empty list = no gather (structured load). "
        "ICON-shape example: ``B[idx[i, k], j, idb[i, k]]`` vec(i, j) -> ``widths=(W_i, W_j)``, "
        "``gather_dims=(0, 2)``, ``_idx_0`` shape ``(W_i,)``, ``_idx_2`` shape ``(W_i,)``.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 dim_strides: Optional[Tuple[int, ...]] = None,
                 src_dims: Optional[Tuple[int, ...]] = None,
                 has_mask: bool = False,
                 pad_mode: str = "ZERO",
                 src_kind: str = "Tile",
                 src_expr: Optional[str] = None,
                 replicate_factor_per_dim: Optional[Tuple[int, ...]] = None,
                 gather_dims: Optional[Tuple[int, ...]] = None,
                 location: Optional[str] = None):
        """Construct a ``TileLoad`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param dim_strides: Per-tile-dim stride coefficients; defaults
            to all 1s (contiguous).
        :param src_dims: Per-tile-dim source-array dim mapping (empty ⇒
            last K dims in order).
        :param has_mask: When True, declare the ``_mask`` input.
        :param pad_mode: cuTile OOB padding mode (``ZERO | NAN | POS_INF
            | NEG_ZERO | UNDETERMINED``); only the ``cutile`` expansion uses
            it.
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
            raise ValueError(f"TileLoad: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(f"TileLoad: dim_strides length {len(dim_strides)} != widths length {len(widths)}")
        if src_kind not in ("Tile", "Symbol", "Scalar"):
            raise ValueError(f"TileLoad: src_kind must be one of 'Tile' | 'Symbol' | 'Scalar', got {src_kind!r}")
        if src_kind == "Symbol" and not src_expr:
            raise ValueError("TileLoad: src_kind='Symbol' requires a non-empty src_expr")
        if replicate_factor_per_dim is not None:
            if len(replicate_factor_per_dim) != len(widths):
                raise ValueError(f"TileLoad: replicate_factor_per_dim length "
                                 f"{len(replicate_factor_per_dim)} != widths length {len(widths)}")
            for d, (w, k) in enumerate(zip(widths, replicate_factor_per_dim)):
                # Per user direction 2026-06-10: ``For SYM we can add a runtime
                # check and check for error ensuring W % SYM``. The codegen
                # formula ``__l / k`` is correct ONLY when ``W % k == 0``.
                # * Static factor: validate ``W % k == 0`` here; refuse
                #   construction on violation.
                # * Symbolic factor (e.g. ``DV`` in ``c[i // DV]``): cannot
                #   statically verify; defer to a runtime check emitted in
                #   the pure expansion (see ExpandTileLoadPure).
                try:
                    k_int = int(k)
                except (TypeError, ValueError):
                    continue  # symbolic -- runtime check handles it
                if k_int < 1:
                    raise ValueError(f"TileLoad: replicate_factor_per_dim[{d}] = {k_int} must be >= 1")
                if k_int > 1 and w % k_int != 0:
                    raise ValueError(f"TileLoad: replicate_factor_per_dim[{d}] = {k_int} must divide "
                                     f"widths[{d}] = {w} (contracted-box load is W/k elements)")
        # Validate gather_dims: sorted, unique, non-negative source-dim indices.
        # The upper bound (max(gather_dims) < src_ndim) is checked at validate() time since
        # ``src_ndim`` depends on the wired ``_src`` connector descriptor (design section 9.3).
        g = tuple(gather_dims) if gather_dims else ()
        if g != tuple(sorted(g)) or len(set(g)) != len(g) or any(d < 0 for d in g):
            raise ValueError(f"TileLoad: gather_dims must be a sorted tuple of unique non-negative "
                             f"source-dim indices; got {g!r}")
        # ``Symbol`` source has no ``_src`` connector — the literal is embedded
        # inline at expansion time.
        inputs = (set() if src_kind == "Symbol" else {"_src"}) | ({"_mask"} if has_mask else set())
        inputs |= {f"_idx_{d}" for d in g}
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.src_dims = list(src_dims) if src_dims else []
        self.has_mask = has_mask
        self.pad_mode = pad_mode
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
            the dtype is not in ``{int32, int64}``.
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
        # Packed-layout lock (design section 2.3): refuse non-C non-Fortran source strides.
        if self.src_kind == "Tile":
            from .._pure_codegen import validate_packed_layout
            src_arr = sdfg.arrays[in_e["_src"].data.data]
            validate_packed_layout(self.label, "_src", src_arr)
        # gather_dims source-dim upper bound + per-dim index-tile shape contract (design section 9.4).
        widths = tuple(self.widths)
        allowed_dtypes = {dace.int32, dace.int64}
        if self.gather_dims and self.src_kind == "Tile":
            src_arr = sdfg.arrays[in_e["_src"].data.data]
            src_ndim = len(src_arr.shape)
            if any(d >= src_ndim for d in self.gather_dims):
                raise ValueError(f"{self.label}: gather_dims {tuple(self.gather_dims)} contains an index >= "
                                 f"source ndim {src_ndim} (source '{in_e['_src'].data.data}' shape "
                                 f"{tuple(src_arr.shape)})")
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
