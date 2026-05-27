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
            # Kept-dims flat offset (row-major over the dims != ax).
            kept_widths = [(d, w) for d, w in enumerate(widths) if d != ax]
            kept_off_terms = []
            kept_stride = 1
            for d, w in reversed(kept_widths):
                kept_off_terms.append(f"__l{d}" if kept_stride == 1 else f"(__l{d} * {kept_stride})")
                kept_stride *= w
            kept_off = " + ".join(reversed(kept_off_terms)) if kept_widths else "0"
            init_body = f"_dst[{kept_off}] = {init};"
            combine_dst = _combine_expr(op, f"_dst[{kept_off}]", f"_src[{src_off}]")
            reduce_body = f"{lane_gate}_dst[{kept_off}] = {combine_dst};"
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
class ExpandTileReduceCute(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileReduce`.

    Emits ``__output = ct.sum(__src, axis=...)`` (or ``ct.prod`` /
    ``ct.min`` / ``ct.max``). Mask gating is the caller's
    responsibility — cuTile reductions do not take a ``mask=`` argument
    so the upstream :class:`TileBinop` must pre-zero inactive lanes.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileReduce", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``ct.<op>(__src, axis=...)``.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet.
        """
        fn = _OP_CUTE[node.op]
        if node.axis is None:
            body = f"__output = {fn}(__src)"
        else:
            body = f"__output = {fn}(__src, axis={node.axis})"
        inputs = {"__src"} | ({"__mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_cute",
            inputs={c: None
                    for c in inputs},
            outputs={"__output": None},
            code=body,
            language=dace.dtypes.Language.Python,
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

    implementations = {"pure": ExpandTileReducePure, "cute": ExpandTileReduceCute}
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
