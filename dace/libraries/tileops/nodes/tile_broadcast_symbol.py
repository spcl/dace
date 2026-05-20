# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileBroadcastSymbol`` — splat a free-symbol expression to a K-dim tile.

The expression is evaluated once in the surrounding scope and copied to
every lane; this replaces the legacy ``_laneid_<i>`` per-lane scalar
fan-out for outer-scope symbol reads.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileBroadcastSymbolPure(ExpandTransformation):
    """K-fold nested-Map splat of the symbolic expression to every lane."""

    environments = []

    @staticmethod
    def expansion(node: "TileBroadcastSymbol", parent_state: dace.SDFGState,
                  parent_sdfg: dace.SDFG) -> dace.SDFG:
        """Build a nested SDFG whose body writes ``_c[l_*] = expr`` for
        every lane combination.

        :param node: The ``TileBroadcastSymbol`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: Nested SDFG that replaces the lib node in place.
        """
        out_arr, = node.validate(parent_sdfg, parent_state)
        widths = list(node.widths)

        sdfg = dace.SDFG(f"{node.label}_pure")
        sdfg.add_array("_c", widths, out_arr.dtype, transient=False)

        state = sdfg.add_state(f"{node.label}_state")
        K = len(widths)
        map_params = [f"__l{k}" for k in range(K)]
        map_rng = {p: f"0:{w}" for p, w in zip(map_params, widths)}
        access = ", ".join(map_params)

        body = f"__output = ({node.expr})"

        state.add_mapped_tasklet(
            f"{node.label}_tasklet",
            map_rng,
            {},
            body,
            {"__output": dace.Memlet(f"_c[{access}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class TileBroadcastSymbol(nodes.LibraryNode):
    """Splat a scalar symbol / expression to every lane of a K-dim tile.

    Replaces the legacy ``<base>_laneid_<i>`` per-lane scalar fan-out
    for outer-scope symbol reads: the K-dim path emits one
    ``TileBroadcastSymbol`` that lowers (post-T6) to a single
    arch-specific splat intrinsic.
    """

    implementations = {"pure": ExpandTileBroadcastSymbolPure}
    default_implementation = "pure"

    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost-last.",
    )
    expr = properties.Property(
        dtype=str,
        allow_none=False,
        default="0",
        desc="Symbolic expression evaluated in the surrounding scope and splatted to every lane.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 expr: str,
                 location: Optional[str] = None):
        """Construct a ``TileBroadcastSymbol`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param expr: Symbolic expression to splat; symbols resolve in
            the surrounding scope.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3 or
            ``expr`` is empty.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileBroadcastSymbol: widths length {len(widths)} not in {{1, 2, 3}}")
        if not expr:
            raise ValueError("TileBroadcastSymbol: expr must be non-empty")
        super().__init__(name, location=location, inputs=set(), outputs={"_c"})
        self.widths = list(widths)
        self.expr = expr

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[dace.data.Data]:
        """Confirm ``_c`` is connected.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :returns: ``(out_arr,)``.
        :raises ValueError: If ``_c`` is not connected.
        """
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_c" not in out_e:
            raise ValueError(f"{self.label}: required output '_c' not connected")
        out_arr = sdfg.arrays[out_e["_c"].data.data]
        return (out_arr,)
