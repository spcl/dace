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
        out_dtype = parent_sdfg.arrays[next(e for e in parent_state.out_edges(node)
                                            if e.src_conn == "_o").data.data].dtype.ctype
        # Unified mask connector contract (user direction 2026-06-12: ``Unifiy
        # mask connectors in the multi dim pass globally``). ``_mask`` is the
        # select-arm predicate for ITE (which lane gets ``_t`` vs ``_e``).
        # The downstream global TileStore handles iter-mask gating; no
        # separate execution-gate connector is needed.
        body = f"_o[{off}] = (_mask[{off}] ? _t[{off}] : _e[{off}]);"
        code = nested_loops(widths, body)
        inputs = {"_mask", "_t", "_e"}
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
    def __init__(self, name: str, widths: Tuple[int, ...], location: Optional[str] = None):
        """Construct a ``TileITE`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``widths`` length.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileITE: widths must have length in {{1, 2, 3}}, got {widths!r}")
        super().__init__(name, location=location, inputs={"_mask", "_t", "_e"}, outputs={"_o"})
        self.widths = list(widths)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Validate connector counts and the then/else/out dtype agreement.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        :raises NotImplementedError: If ``_t`` / ``_e`` / ``_o`` dtypes
            disagree (a cross-dtype select needs an explicit cast first).
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_o" not in out_e:
            raise ValueError(f"{self.label}: required output '_o' not connected")
        for conn in ("_mask", "_t", "_e"):
            if conn not in in_e:
                raise ValueError(f"{self.label}: required input {conn!r} not connected")
        o_arr = sdfg.arrays[out_e["_o"].data.data]
        t_arr = sdfg.arrays[in_e["_t"].data.data]
        # Output-kind rule (design 6.2): TileITE inputs are implicitly Tile (no kind
        # properties), so the output must be tile-shape.
        from .tile_binop import _is_tile_shape
        if not _is_tile_shape(o_arr, tuple(self.widths)):
            raise NotImplementedError(
                f"{self.label}: output-kind rule violated -- TileITE inputs are implicitly Tile but "
                f"'_o' descriptor is not tile-shape {tuple(self.widths)!r}. Per design section 6.2: "
                f"any Tile input -> Tile output.")
        e_arr = sdfg.arrays[in_e["_e"].data.data]
        if {o_arr.dtype, t_arr.dtype, e_arr.dtype} != {o_arr.dtype}:
            raise NotImplementedError(
                f"{self.label}: TileITE requires uniform dtype across _t, _e and _o "
                f"(got {o_arr.dtype}, {t_arr.dtype}, {e_arr.dtype}); cast via separate tasklet first.")
