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

from .._pure_codegen import nested_loops, offset_via_strides, tile_offset
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
            src_strides = [symstr(src_arr.strides[d]) for d in dims]
            coeff = list(node.dim_strides) if node.dim_strides else [1] * K
            # Per-dim replicate factor: ``__l<d> / k`` indexes a contracted
            # box of W/k elements when ``k > 1``, broadcasting each loaded
            # value to ``k`` consecutive lanes (the int_floor / int_ceil
            # regime). Default all-1 = no replication.
            replicate = list(node.replicate_factor_per_dim) if node.replicate_factor_per_dim else [1] * K
            src_off = offset_via_strides(coeff, src_strides, replicate)
            src_ref = f"_src[{src_off}]"
        if node.has_mask:
            body = f"_dst[{dst_off}] = _mask[{dst_off}] ? {src_ref} : {dst_dtype}(0);"
        else:
            body = f"_dst[{dst_off}] = {src_ref};"
        code = nested_loops(widths, body)
        inputs = (set() if node.src_kind == "Symbol" else {"_src"}) | ({"_mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


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
        """Return a Python tasklet emitting ``ct.load``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet whose body calls
            ``ct.load`` with the :attr:`TileLoad.pad_mode` padding mode.
        :raises ValueError: If :attr:`TileLoad.pad_mode` is not a
            recognised cuTile padding-mode name.
        """
        widths = list(node.widths)
        K = len(widths)
        if node.pad_mode not in _PAD_MODE_CUTE:
            raise ValueError(f"{node.label}: unknown pad_mode {node.pad_mode!r}; "
                             f"allowed: {sorted(_PAD_MODE_CUTE)}")
        pad_mode = _PAD_MODE_CUTE[node.pad_mode]
        shape_tuple = ", ".join(str(w) for w in widths)
        index_tuple = ", ".join(f"__pid{k}" for k in range(K))
        lines = [f"__pid{k} = ct.bid({k})" for k in range(K)]
        lines.append(f"__output = ct.load(__src, index=({index_tuple},), shape=({shape_tuple},),"
                     f" padding_mode={pad_mode})")
        # L-load-nomask: the load never reads a per-lane mask, so even with
        # has_mask=True we must NOT declare a dangling __mask input — the
        # mask is consumed downstream at the store/scatter.
        inputs = {"__src"}
        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code="\n".join(lines),
            language=dace.dtypes.Language.Python,
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
        element_type=int,
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
        element_type=int,
        default=[],
        desc="Per-tile-dim replicate factor (lanes-per-distinct-value within "
        "the dim). ``1`` (or empty) = no replication, the contiguous endpoint "
        "of the spectrum. ``k > 1`` = ``int_floor`` / ``int_ceil`` regime: load "
        "``W_d / k`` elements on this dim and group-broadcast each ``k`` times "
        "across consecutive lanes. The codegen template covers all three "
        "regimes (factor=1 contiguous, 1<k<W grouped, k=W full broadcast) "
        "uniformly.",
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
                if k < 1:
                    raise ValueError(f"TileLoad: replicate_factor_per_dim[{d}] = {k} must be >= 1")
                if k > 1 and w % k != 0:
                    raise ValueError(f"TileLoad: replicate_factor_per_dim[{d}] = {k} must divide "
                                     f"widths[{d}] = {w} (contracted-box load is W/k elements)")
        # ``Symbol`` source has no ``_src`` connector — the literal is embedded
        # inline at expansion time.
        inputs = (set() if src_kind == "Symbol" else {"_src"}) | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.src_dims = list(src_dims) if src_dims else []
        self.has_mask = has_mask
        self.pad_mode = pad_mode
        self.src_kind = src_kind
        self.src_expr = src_expr
        self.replicate_factor_per_dim = (list(replicate_factor_per_dim)
                                         if replicate_factor_per_dim else [1] * len(widths))

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Check connectors.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if self.src_kind != "Symbol" and "_src" not in in_e:
            raise ValueError(f"{self.label}: required input '_src' not connected (src_kind={self.src_kind!r})")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
