# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileIota``: a per-lane affine or indirect fill of an integer tile."""

from collections.abc import Sequence

import dace
from dace import library, properties
from dace.libraries.tileops.expansions import ExpandTilePure
from dace.libraries.tileops.lanes import nested_loops, tile_offset
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandTileIotaPure(ExpandTilePure):
    pass


@library.expansion
class ExpandTileIotaCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileIota`.

    Emits per-lane index arrays via ``ct.arange()`` for each dimension,
    broadcast to the full tile shape for K>=2, then evaluates the
    node's ``expr`` as a Python expression over those lane arrays.
    Extra inputs (``_idx``, ``_src``, ...) are tile/array connectors that
    broadcast element-wise in the cuTile expression.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileIota", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet that fills the tile via cuTile ops.

        :param node: The ``TileIota`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet replacing the lib node.
        """
        widths = list(node.widths)
        K = len(widths)
        inputs = {c: None for c in node.extra_inputs}

        # Derive the lane-array dtype from the ``_dst`` descriptor so the
        # declared descriptor dtype and the runtime tile dtype agree (the
        # lane-id tiles are declared int64; ct.arange defaults would yield
        # int32 tiles). Falls back to int32 when the descriptor is unknown.
        ct_dtype = "ct.int32"
        for out_edge in parent_state.out_edges(node):
            if out_edge.src_conn != "_dst" or out_edge.data is None or out_edge.data.data is None:
                continue
            dst_desc = parent_sdfg.arrays.get(out_edge.data.data)
            if dst_desc is not None:
                ct_dtype = f"ct.{dst_desc.dtype.to_string()}"
            break

        # Degenerate single-lane case: all widths are 1.
        if all(w == 1 for w in widths):
            expr = node.expr
            # Safe: K is capped at 3 in __init__, so __l{p} names never overlap.
            for p in range(K):
                expr = expr.replace(f"__l{p}", "0")
            for conn in node.extra_inputs:
                expr = expr.replace(f"{conn}[0]", conn)
            body = f"_dst = {expr}"
            return nodes.Tasklet(
                label=f"{node.label}_cutile",
                inputs=inputs,
                outputs={"_dst": None},
                code=body,
                language=dace.dtypes.Language.Python,
            )

        shape_tuple = ", ".join(str(w) for w in widths)
        lines = []

        # Per-dim lane-index arrays, broadcast to full tile shape for K>=2.
        if K == 1:
            lines.append(f"__l0 = ct.arange({widths[0]}, dtype={ct_dtype})")
        else:
            for k in range(K):
                slc = ["None"] * K
                slc[k] = ":"
                slc_str = "[" + ", ".join(slc) + "]"
                lines.append(
                    f"__l{k} = ct.broadcast_to(ct.arange({widths[k]}, dtype={ct_dtype}){slc_str}, ({shape_tuple}))"
                )

        # The expression uses __l0..__l{K-1} which are now cuTile arrays.
        lines.append(f"_dst = {node.expr}")

        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs=inputs,
            outputs={"_dst": None},
            code="\n".join(lines),
            language=dace.dtypes.Language.Python,
        )


@library.node
class TileIota(TileOp):
    """``_dst[l_0, ..., l_{K-1}] = expr`` over a tile, the lane indices being spelled ``__l0, ..., __l{K-1}``.

    * An affine index tile: ``i + __l0`` is contiguous along dim 0, ``i + 2 * __l0`` strided.
    * An indirect index tile: ``extra_inputs = ("_idx", )`` and ``expr = "_idx[__l0]"``.
    * A multi-dim indirect one: ``extra_inputs = ("_src", )`` and ``expr = "_src[<flat offset of the lanes>]"``.
    """

    lanes_independent = True
    implementations = {"cutile": ExpandTileIotaCutile, "pure": ExpandTileIotaPure}
    default_implementation = "pure"

    expr = properties.Property(
        dtype=str,
        default="",
        desc="The per-lane value assigned to ``_dst``, over the lane indices ``__l0, ...`` and the extra inputs.",
    )
    extra_inputs = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Input connectors the expression reads, such as ``_idx`` for a tile lookup or ``_src`` for a view of an "
        "outer array.",
    )

    def __init__(
        self,
        name: str,
        widths: tuple[int, ...],
        expr: str,
        extra_inputs: Sequence[str] = (),
        location: str | None = None,
    ):
        if not 1 <= len(widths) <= 3:
            raise ValueError(f"TileIota: widths length {len(widths)} not in {{1, 2, 3}}")
        if not expr:
            raise ValueError("TileIota: expr is required (non-empty per-lane body)")
        super().__init__(name, location=location, inputs=dict.fromkeys(extra_inputs), outputs={"_dst"})
        self.widths = list(widths)
        self.expr = expr
        self.extra_inputs = list(extra_inputs)

    @property
    def free_symbols(self) -> set[str]:
        ignored = set(self.extra_inputs) | {f"__l{dim}" for dim in range(len(self.widths))}
        try:
            symbols = properties.CodeBlock(self.expr).get_free_symbols(ignored)
        except SyntaxError:
            # C++-only bodies are handled after expansion, like C++ tasklets.
            symbols = set()
        return super().free_symbols | symbols

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        in_conns = {edge.dst_conn for edge in state.in_edges(self)}
        if "_dst" not in {edge.src_conn for edge in state.out_edges(self)}:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        for conn in self.extra_inputs:
            if conn not in in_conns:
                raise ValueError(f"{self.label}: declared extra input {conn!r} not connected")

    def pure_tasklet(self, state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        widths = list(self.widths)
        if all(width == 1 for width in widths):
            # DaCe collapses a ``Register`` ``Array(shape=(1,))`` to a plain scalar, which ``_dst[__l0]`` would index
            # into: the single lane has index 0 and the extra inputs are bare scalars as well.
            expr = self.expr
            for dim in range(len(widths)):
                expr = expr.replace(f"__l{dim}", "0")
            for conn in self.extra_inputs:
                expr = expr.replace(f"{conn}[0]", conn)
            code = f"_dst = {expr};"
        else:
            code = nested_loops(widths, f"_dst[{tile_offset(widths)}] = {self.expr};")
        return nodes.Tasklet(
            label=f"{self.label}_pure",
            inputs=dict.fromkeys(self.extra_inputs),
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )
