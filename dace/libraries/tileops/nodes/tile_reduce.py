# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileReduce`` — intra-tile reduction along one axis (or all axes).

Collapses a K-dim register tile to a (K-1)-dim tile (single-axis
reduction) or to a 1-element scalar (full reduction). Cross-tile
accumulation is the caller's job (typically a WCR memlet on the
output edge).

``op`` ∈ ``{+, *, min, max}`` with identities ``0``, ``1``, ``+inf``,
``-inf`` respectively.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops


def _identity_literal(op: str, ctype: str) -> str:
    """Return the C++ literal for ``op``'s identity at type ``ctype``.

    :param op: One of ``+``, ``*``, ``min``, ``max``.
    :param ctype: C++ scalar type name (e.g. ``double``).
    :returns: A C++ expression suitable as the initialiser.
    """
    if op == "+":
        return f"{ctype}(0)"
    if op == "*":
        return f"{ctype}(1)"
    if op == "min":
        return f"std::numeric_limits<{ctype}>::max()"
    if op == "max":
        return f"std::numeric_limits<{ctype}>::lowest()"
    raise ValueError(f"unknown op {op!r}")


def _combine_expr(op: str, acc: str, val: str) -> str:
    """Return the C++ expression that combines ``acc`` and ``val`` per ``op``.

    :param op: Reduction op.
    :param acc: Accumulator variable.
    :param val: New value variable.
    :returns: A C++ expression (no trailing semicolon).
    """
    if op == "+":
        return f"{acc} + {val}"
    if op == "*":
        return f"{acc} * {val}"
    if op == "min":
        return f"std::min({acc}, {val})"
    if op == "max":
        return f"std::max({acc}, {val})"
    raise ValueError(f"unknown op {op!r}")


_OP_CUTE = {"+": "ct.sum", "*": "ct.prod", "min": "ct.min", "max": "ct.max"}
_VALID_OPS = ("+", "*", "min", "max")

#: cuTile literal for each reduction op's identity, pre-selected into masked
#: lanes before the (mask-less, L-reduce-nomask) reduction. ``min`` / ``max``
#: need ``±inf`` which only ``ct.where`` can safely inject (the arithmetic
#: blend hits the ``inf * 0 = NaN`` hazard — see the L-reduce-nomask note).
_OP_IDENTITY_CUTE = {
    "+": "0",
    "*": "1",
    "min": "float('inf')",
    "max": "float('-inf')",
}

# Capability probe for ``ct.where`` (cuTile's select). The cuTile runtime is
# never installed on CI, so this resolves to ``None`` there (meaning "assume
# present" — emit the richest ``ct.where`` form as the documented default).
# A unit test can override it to exercise the no-``where`` fallback / raise
# paths. L-where-unconfirmed: no ``cuda.tile.where`` page exists in the
# online cuTile-Python API docs (only a Tile-IR ``select(cond, x, y)`` op),
# so its presence in the installed package stays unverified.
try:  # pragma: no cover - cuTile is not installed on CI
    import cuda.tile as ct  # type: ignore  # noqa: F401
    _CT_HAS_WHERE = hasattr(ct, "where")
except Exception:  # pragma: no cover - the CI path (no cuTile install)
    _CT_HAS_WHERE = None


@library.expansion
class ExpandTileReducePure(ExpandTransformation):
    """Correctness-only CPP tasklet — sequential per-lane reduction."""

    environments = []

    @staticmethod
    def expansion(node: "TileReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that walks the input tile lane-by-lane
        and accumulates into ``_dst``.

        For full reduction (``axis is None``) the output is a length-1
        scalar. For single-axis reduction, the output is a (K-1)-dim
        tile and each kept-dim slot is initialised then combined with
        every lane that maps to it. Masked lanes contribute identity
        (no-op via an ``if (_mask[k])`` gate).

        :param node: The ``TileReduce`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        widths = list(node.widths)
        K = len(widths)
        op = node.op
        src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
        ctype = parent_sdfg.arrays[src_edge.data.data].dtype.ctype
        init = _identity_literal(op, ctype)
        # Row-major flat offset into the K-dim input tile (matches the
        # tile transient's contiguous storage layout).
        src_off_terms = []
        stride = 1
        for d in reversed(range(K)):
            src_off_terms.append(f"__l{d}" if stride == 1 else f"(__l{d} * {stride})")
            stride *= widths[d]
        src_off = " + ".join(reversed(src_off_terms)) if K else "0"
        combine_acc = _combine_expr(op, "__acc", f"_src[{src_off}]")
        lane_gate = f"if (_mask[{src_off}]) " if node.has_mask else ""

        if node.axis is None:
            out_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
            scalar_dst = out_edge.data.subset is None or out_edge.data.subset.num_elements() == 1
            writeback = "_dst = __acc;" if scalar_dst else "_dst[0] = __acc;"
            body = f"{lane_gate}__acc = {combine_acc};"
            code = (f"{ctype} __acc = {init};\n"
                    f"{nested_loops(widths, body)}\n"
                    f"{writeback}")
        else:
            ax = node.axis
            if not (0 <= ax < K):
                raise ValueError(f"TileReduce: axis {ax} out of range for K={K}")
            # Kept-dims flat offset.
            # The reduce loop iterates over ALL widths with ``nested_loops``,
            # so its loop variables are ``__l0..__l{K-1}`` matching the
            # original tile-dim indices. ``reduce_kept_off`` uses those.
            kept_widths = [(d, w) for d, w in enumerate(widths) if d != ax]
            reduce_kept_terms = []
            kept_stride = 1
            for d, w in reversed(kept_widths):
                reduce_kept_terms.append(f"__l{d}" if kept_stride == 1 else f"(__l{d} * {kept_stride})")
                kept_stride *= w
            reduce_kept_off = " + ".join(reversed(reduce_kept_terms)) if kept_widths else "0"
            # The init loop is wrapped separately by ``nested_loops`` over
            # ``init_widths = [w for _, w in kept_widths]``, which renumbers
            # the loop variables sequentially as ``__l0..__l{Kept-1}`` — the
            # original dim indices don't survive. ``init_kept_off`` must use
            # the sequential names; reaching for ``__l{d}`` with the original
            # index would name an undeclared variable (e.g. ``__l1`` when
            # the only loop is ``__l0``).
            init_kept_terms = []
            kept_stride = 1
            for new_idx, w in reversed(list(enumerate(w for _, w in kept_widths))):
                init_kept_terms.append(f"__l{new_idx}" if kept_stride == 1 else f"(__l{new_idx} * {kept_stride})")
                kept_stride *= w
            init_kept_off = " + ".join(reversed(init_kept_terms)) if kept_widths else "0"
            init_body = f"_dst[{init_kept_off}] = {init};"
            combine_dst = _combine_expr(op, f"_dst[{reduce_kept_off}]", f"_src[{src_off}]")
            reduce_body = f"{lane_gate}_dst[{reduce_kept_off}] = {combine_dst};"
            init_widths = [w for _, w in kept_widths]
            code = (f"{nested_loops(init_widths, init_body)}\n"
                    f"{nested_loops(widths, reduce_body)}")
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
class ExpandTileReduceCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileReduce`.

    Unmasked (``has_mask=False``): ``__output = ct.sum(__src, axis=...)``
    (or ``ct.prod`` / ``ct.min`` / ``ct.max``).

    Masked (``has_mask=True``): cuTile reductions take no ``mask=`` /
    valid-region argument (L-reduce-nomask), so the inactive lanes must
    be pre-set to the op's identity (``+`` → 0, ``*`` → 1, ``min`` →
    ``+inf``, ``max`` → ``-inf``) before reducing. Primary form uses
    ``ct.where(__mask, __src, IDENT)``; the ``__mask`` input is genuinely
    consumed (fixing the prior dead-connector bug). When ``ct.where`` is
    known absent, ``+`` / ``*`` fall back to an arithmetic blend, while
    masked ``min`` / ``max`` raise ``NotImplementedError`` (the
    ``inf * 0 = NaN`` hazard makes the blend unsafe).
    """

    environments = []

    @staticmethod
    def expansion(node: "TileReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        raise NotImplementedError(
            "ExpandTileReduceCutile: cuTile expansion stubbed out during G3 step 3 migration; the unified `TileLoad` / `TileStore` (with `gather_dims`) cuTile path will be reinstated after the per-source-dim gather contract lands per design "
            "section 6.4. Pin a `pure` expansion via `sdfg.expand_library_nodes(implementation='pure')` to lower this node for now."
        )


@library.node
class TileReduce(nodes.LibraryNode):
    """Reduce a K-dim register tile along ``axis`` (or fully to a scalar).

    Connectors:

    * ``_src`` — the tile transient to reduce (``widths``-shaped).
    * ``_mask`` (optional) — tile-shaped boolean mask; inactive lanes
      contribute identity.
    * ``_dst`` — the reduced output. For ``axis is None`` this is a
      length-1 scalar; for single-axis reduction it has the kept-dim
      shape.
    """

    implementations = {"pure": ExpandTileReducePure, "cutile": ExpandTileReduceCutile}
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
    op = properties.Property(
        dtype=str,
        allow_none=False,
        default="+",
        desc="Reduction op (one of: + * min max).",
    )
    axis = properties.Property(
        dtype=int,
        allow_none=True,
        default=None,
        desc="Single tile-dim to reduce along; ``None`` ⇒ full reduction to scalar.",
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
                 op: str = "+",
                 axis: Optional[int] = None,
                 has_mask: bool = False,
                 location: Optional[str] = None):
        """Construct a ``TileReduce`` node.

        :param name: Node label.
        :param widths: Per-tile-dim widths, innermost-last (length 1..3).
        :param op: One of ``+``, ``*``, ``min``, ``max``.
        :param axis: Single dim index to reduce along; ``None`` ⇒ full
            reduction.
        :param has_mask: When True, declare the ``_mask`` input.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``op``, ``widths`` length, or
            out-of-range ``axis``.
        """
        if op not in _VALID_OPS:
            raise ValueError(f"TileReduce: unknown op {op!r}; allowed: {_VALID_OPS}")
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileReduce: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if axis is not None and not (0 <= axis < len(widths)):
            raise ValueError(f"TileReduce: axis {axis} out of range for K={len(widths)}")
        inputs = {"_src"} | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.op = op
        self.axis = axis
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
