# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileITE`` per-lane select on K-dim register tiles.

The lib node lowers a branch-normalized ``merge(cond, then, else)``
tasklet (see :mod:`dace.runtime.include.dace.merge`) to a per-lane
blend ``_o[l] = _mask[l] ? _t[l] : _e[l]``. It is the K-dim tile
counterpart of the 1D ``vector_select`` blend the merge tasklet would
otherwise lower to.

The three inputs ``_mask`` / ``_t`` / ``_e`` are all tiles; ``_mask`` may
carry any dtype (the lifted condition is stored as ``0.0`` / ``1.0`` when
the branch normalization typed it after a float operand, or ``bool``),
while ``_t`` / ``_e`` / ``_o`` share the output element type.

Per user direction 2026-06-12 (``Unifiy mask connectors in the multi
dim pass globally``), the predicate connector is named ``_mask`` -- the
same name every other tile lib node uses for its gating predicate. No
separate iter-mask gate connector (per user: ``ITE write to global array
should be further masked so it should be fine``); the surrounding
:class:`TileStore` discards inactive iter-mask lanes.

The pure expansion returns a CPP tasklet whose body is a single
``for``-loop over the flattened tile (correctness-only).
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, tile_offset
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE
from .tile_binop import _TILE, _SYMBOL, _SCALAR, _VALID_KINDS, _is_tile_shape

# Capability probe for ``ct.where`` (cuTile's select). The cuTile runtime is
# never installed on CI, so this resolves to ``None`` there (meaning "assume
# present" — emit the richest ``ct.where`` form as the documented default).
# A unit test can override it to exercise the arithmetic-blend fallback / the
# non-finite-float raise. L-where-unconfirmed: no ``cuda.tile.where`` page
# exists in the online cuTile-Python API docs (only a Tile-IR ``select(cond,
# x, y)`` op), so its presence in the installed package stays unverified.
try:  # pragma: no cover - cuTile is not installed on CI
    import cuda.tile as ct  # type: ignore  # noqa: F401
    _CT_HAS_WHERE = hasattr(ct, "where")
except Exception:  # pragma: no cover - the CI path (no cuTile install)
    _CT_HAS_WHERE = None


@library.expansion
class ExpandTileITEPure(ExpandTransformation):
    """Correctness-only CPP tasklet lowering of ``TileITE``."""

    environments = []

    @staticmethod
    def expansion(node: "TileITE", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a single CPP tasklet that walks the flattened tile.

        :param node: The ``TileITE`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)
        off = tile_offset(widths)
        in_e = {e.dst_conn: e for e in parent_state.in_edges(node) if e.dst_conn is not None}
        out_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_o").data.data].dtype.ctype

        def _ref(kind, conn, expr, cast):
            """Per-lane C++ reference for one operand (a select arm or the cond).

            A ``Symbol`` operand embeds its loop-invariant expression inline,
            broadcast across every lane (user direction 2026-06-15). A ``Scalar``
            operand reads a length-1 / ``dace.data.Scalar`` source (``conn[0]`` for
            a length-1 Array pointer connector, bare ``conn`` for a by-value
            Scalar) and broadcasts it. A ``Tile`` operand indexes per lane. When
            ``cast`` is given (the arms, cast to the output dtype) it is applied;
            the cond is used in a boolean ``? :`` context so ``cast`` is ``None``.
            """
            if kind == _SYMBOL:
                return f"({cast})({expr})" if cast else f"({expr})"
            if kind == _TILE:
                return f"{conn}[{off}]"
            desc = parent_sdfg.arrays[in_e[conn].data.data]
            is_len1_array = (isinstance(desc, dace.data.Array)
                             and all(bool(dace.symbolic.simplify(s == 1)) for s in desc.shape))
            ref = f"{conn}[0]" if is_len1_array else conn
            return f"({cast})({ref})" if cast else f"({ref})"

        t_ref = _ref(node.kind_t, "_t", node.expr_t, out_dtype)
        e_ref = _ref(node.kind_e, "_e", node.expr_e, out_dtype)
        # Unified mask connector contract (user direction 2026-06-12: ``Unifiy
        # mask connectors in the multi dim pass globally``). ``_mask`` is the
        # select predicate for ITE (which lane gets ``_t`` vs ``_e``); a
        # loop-invariant cond may instead be inline (kind_mask='Symbol' /
        # 'Scalar', user direction 2026-06-15). The downstream global TileStore
        # handles iter-mask gating; no separate execution-gate connector.
        mask_ref = _ref(node.kind_mask, "_mask", node.expr_mask, None)
        body = f"_o[{off}] = ({mask_ref} ? {t_ref} : {e_ref});"
        code = nested_loops(widths, body)
        inputs = set()
        if node.kind_mask in (_TILE, _SCALAR):
            inputs.add("_mask")
        if node.kind_t in (_TILE, _SCALAR):
            inputs.add("_t")
        if node.kind_e in (_TILE, _SCALAR):
            inputs.add("_e")
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_o": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileITECutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileITE`.

    Primary (CI default): ``__output = ct.where(__mask, __then,
    __else)`` — the cuTile select primitive. The surrounding iteration
    mask is applied at the downstream ``ct.scatter`` store, not at the
    select (matching the reference cuTile kernels).

    Fallback (``ct.where`` known absent): an arithmetic blend
    ``__m = __mask.astype(__then.dtype); __output = __m * __then +
    (1.0 - __m) * __else``. This is exact for the ``0.0`` / ``1.0`` (or
    ``bool``) condition encoding, but ``0.0 * inf = NaN`` would leak a
    non-finite *unselected* lane into the result. So the fallback is
    emitted only for an **integer** output dtype; a float output with
    possibly-non-finite branches raises ``NotImplementedError`` because
    cuTile offers no other confirmed safe select.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileITE", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(
            "ExpandTileITECutile: cuTile expansion stubbed out during G3 step 3 migration; the unified `TileLoad` / `TileStore` (with `gather_dims`) cuTile path will be reinstated after the per-source-dim gather contract lands per design "
            "section 6.4. Pin a `pure` expansion via `sdfg.expand_library_nodes(implementation='pure')` to lower this node for now."
        )


@library.expansion
class ExpandTileITEScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_ite_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileITEAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_ite_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileITEAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_ite_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileITENeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_ite_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileITESVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_ite_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileITE(nodes.LibraryNode):
    """Per-lane select ``_o = _mask ? _t : _e`` on K-dim register tiles.

    Lowers the ``merge(cond, then, else)`` tasklet that branch
    normalization emits for a same-write-set ``if/else``. Per user
    direction 2026-06-12 (``Unifiy mask connectors in the multi dim pass
    globally``), the predicate connector is ``_mask`` -- the same name
    every other tile lib node uses. No separate iter-mask gate connector
    (per user: ``ITE write to global array should be further masked so
    it should be fine``); the downstream global :class:`TileStore` discards
    inactive iter-mask lanes.

    :cvar implementations: Per-target expansions; ``"pure"`` is the
        flattened CPP-loop correctness fallback. ``"cutile"`` emits the
        :mod:`cuda.tile`-Python ``ct.where`` equivalent (opt-in).
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {
        "pure": ExpandTileITEPure,
        "cutile": ExpandTileITECutile,
        "scalar": ExpandTileITEScalar,
        "avx512": ExpandTileITEAVX512,
        "avx2": ExpandTileITEAVX2,
        "neon": ExpandTileITENeon,
        "sve": ExpandTileITESVE
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
        desc="Per-dim tile widths, innermost-last; length in {1, 2, 3}.",
    )
    kind_t = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Then-arm kind: 'Tile' (read via _t), 'Scalar' (length-1 via _t), or 'Symbol' (inline expr_t).",
    )
    kind_e = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Else-arm kind: 'Tile' (read via _e), 'Scalar' (length-1 via _e), or 'Symbol' (inline expr_e).",
    )
    expr_t = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when kind_t == 'Symbol'; ignored otherwise.",
    )
    expr_e = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when kind_e == 'Symbol'; ignored otherwise.",
    )
    kind_mask = properties.Property(
        dtype=str,
        allow_none=False,
        default=_TILE,
        desc="Condition kind: 'Tile' (per-lane bool tile via _mask), 'Scalar' (length-1 bool via _mask), "
        "or 'Symbol' (loop-invariant predicate embedded inline via expr_mask, no _mask connector).",
    )
    expr_mask = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Loop-invariant predicate expression embedded inline when kind_mask == 'Symbol'; ignored otherwise.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 kind_t: str = _TILE,
                 kind_e: str = _TILE,
                 expr_t: Optional[str] = None,
                 expr_e: Optional[str] = None,
                 kind_mask: str = _TILE,
                 expr_mask: Optional[str] = None,
                 location: Optional[str] = None):
        """Construct a ``TileITE`` node.

        Every operand -- the condition (``_mask``) and both select arms (``_t`` /
        ``_e``) -- may be a Tile (per-lane, read via its connector), a Scalar
        (length-1 / ``dace.data.Scalar`` broadcast via its connector), or a
        Symbol (a loop-invariant expression / literal embedded inline and
        broadcast to every lane at expansion). Per user direction 2026-06-15
        (``any tile op accepts symbolic / scalar inputs ... for TileITE as
        well``); a Symbol operand declares NO connector. A loop-invariant
        condition (``kind_mask='Symbol'``) thus omits ``_mask`` entirely.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param kind_t: Then-arm kind -- ``"Tile"`` / ``"Scalar"`` / ``"Symbol"``.
        :param kind_e: Else-arm kind -- ``"Tile"`` / ``"Scalar"`` / ``"Symbol"``.
        :param expr_t: Required when ``kind_t == "Symbol"``.
        :param expr_e: Required when ``kind_e == "Symbol"``.
        :param kind_mask: Condition kind -- ``"Tile"`` / ``"Scalar"`` / ``"Symbol"``.
        :param expr_mask: Required when ``kind_mask == "Symbol"``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``widths`` length / kind, or a missing
            expression for a Symbol operand.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileITE: widths must have length in {{1, 2, 3}}, got {widths!r}")
        for label, kind in (("kind_t", kind_t), ("kind_e", kind_e), ("kind_mask", kind_mask)):
            if kind not in _VALID_KINDS:
                raise ValueError(f"TileITE: {label} must be one of {_VALID_KINDS}, got {kind!r}")
        if kind_t == _SYMBOL and not expr_t:
            raise ValueError("TileITE: kind_t='Symbol' requires expr_t")
        if kind_e == _SYMBOL and not expr_e:
            raise ValueError("TileITE: kind_e='Symbol' requires expr_e")
        if kind_mask == _SYMBOL and not expr_mask:
            raise ValueError("TileITE: kind_mask='Symbol' requires expr_mask")
        inputs = set()
        if kind_mask in (_TILE, _SCALAR):
            inputs.add("_mask")
        if kind_t in (_TILE, _SCALAR):
            inputs.add("_t")
        if kind_e in (_TILE, _SCALAR):
            inputs.add("_e")
        super().__init__(name, location=location, inputs=inputs, outputs={"_o"})
        self.widths = list(widths)
        self.kind_t = kind_t
        self.kind_e = kind_e
        self.expr_t = expr_t
        self.expr_e = expr_e
        self.kind_mask = kind_mask
        self.expr_mask = expr_mask

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Validate connector counts and the then/else/out dtype agreement.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If the output is not tile-shape or the
            materialised (Tile / Scalar) arm dtypes disagree with ``_o``
            (a cross-dtype select needs an explicit cast first). Symbol arms
            are cast to ``_o``'s dtype inline at expansion, so they are exempt.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_o" not in out_e:
            raise ValueError(f"{self.label}: required output '_o' not connected")
        # An operand connector is required only when its kind reads through a
        # connector (Tile / Scalar); a Symbol operand is inline (no connector).
        required = []
        for conn, kind in (("_mask", self.kind_mask), ("_t", self.kind_t), ("_e", self.kind_e)):
            if kind in (_TILE, _SCALAR):
                required.append(conn)
        for conn in required:
            if conn not in in_e:
                raise ValueError(f"{self.label}: required input {conn!r} not connected")
        o_arr = sdfg.arrays[out_e["_o"].data.data]
        # Output-kind rule (design 6.2): when ANY operand is a Tile, ``_o`` must be
        # tile-shape. (In the tile body at least the cond or one arm is a Tile; an
        # all-Symbol/Scalar ITE is loop-invariant and may keep a scalar output.)
        any_tile_input = _TILE in (self.kind_mask, self.kind_t, self.kind_e)
        if any_tile_input and not _is_tile_shape(o_arr, tuple(self.widths)):
            raise NotImplementedError(
                f"{self.label}: output-kind rule violated -- a Tile input is present but "
                f"'_o' descriptor is not tile-shape {tuple(self.widths)!r}. Per design section 6.2: "
                f"any Tile input -> Tile output.")
        arm_dtypes = {o_arr.dtype}
        for conn, kind in (("_t", self.kind_t), ("_e", self.kind_e)):
            if kind in (_TILE, _SCALAR):
                arm_dtypes.add(sdfg.arrays[in_e[conn].data.data].dtype)
        if arm_dtypes != {o_arr.dtype}:
            raise NotImplementedError(f"{self.label}: TileITE requires uniform dtype across _t, _e and _o "
                                      f"(got {sorted(str(d) for d in arm_dtypes)}); cast via separate tasklet first.")
