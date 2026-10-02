# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The operands of an elementwise tile node and how its ``pure`` expansion reads them lane by lane."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import dace
from dace.codegen.cppunparse import pyexpr2cpp
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge

from dace.libraries.tileops.kinds import SCALAR, SYMBOL, TILE, VALID_KINDS
from dace.libraries.tileops.lanes import half_disambiguated, lane_invariant_assign, nested_loops, tile_offset
from dace.libraries.tileops.validation import edge_moves_a_tile, edge_moves_one_element, is_tile_shape, promotion_ok


@dataclass(frozen=True, slots=True)
class Operand:
    """One input of an elementwise tile node: the connector it reads, of which kind, and a symbol's expression."""
    conn: str
    kind: str
    expr: str | None = None

    @property
    def reads_connector(self) -> bool:
        return self.kind != SYMBOL


def check_operands(label: str, operands: Sequence[Operand]) -> None:
    """Refuse an operand of an unknown kind and a symbol without its expression.

    The properties of an operand ``_a`` are ``kind_a`` and ``expr_a``, which the messages name.
    """
    for operand in operands:
        if operand.kind not in VALID_KINDS:
            raise ValueError(f"{label}: kind{operand.conn} must be one of {VALID_KINDS}, got {operand.kind!r}")
        if operand.kind == SYMBOL and not operand.expr:
            raise ValueError(f"{label}: kind{operand.conn}='Symbol' requires expr{operand.conn}")


def input_connectors(operands: Sequence[Operand], has_mask: bool) -> list[str]:
    """The connectors the expansion of a node reads: its non-symbol operands, then the mask."""
    return [operand.conn for operand in operands if operand.reads_connector] + (["_mask"] if has_mask else [])


def connected_edges(state: dace.SDFGState, node: nodes.LibraryNode) -> dict[str, MultiConnectorEdge]:
    return {edge.dst_conn: edge for edge in state.in_edges(node) if edge.dst_conn is not None}


def output_edge(state: dace.SDFGState, node: nodes.LibraryNode, conn: str) -> MultiConnectorEdge:
    return next(edge for edge in state.out_edges(node) if edge.src_conn == conn)


def edge_ctype(sdfg: dace.SDFG, edge: MultiConnectorEdge) -> str:
    """The C++ element type of the array an edge moves."""
    return sdfg.arrays[edge.data.data].dtype.ctype


def scalar_operand_ref(desc: dace.data.Data, conn: str, widths: Sequence[int], offset: str) -> tuple[str, bool]:
    """The per-lane reference of a ``Scalar`` operand, and whether it is a broadcast.

    A tile-shaped array an upstream tile op widened reaches the expansion as a pointer carrying one value per lane, so
    it is read ``conn[offset]`` like a ``Tile`` operand. Any single-element source (a ``Scalar``, a length-1 array, a
    one-element access) is passed by value, so the bare ``conn`` is the broadcast; ``[0]`` belongs to the memlet.
    """
    if isinstance(desc, dace.data.Array) and is_tile_shape(desc, tuple(widths)):
        return f"{conn}[{offset}]", False
    return conn, True


def shared_ctype(operands: Sequence[Operand], in_edges: dict[str, MultiConnectorEdge], sdfg: dace.SDFG,
                 fallback: str) -> str:
    """The C++ type the value operands of a node share.

    It is the dtype of the first array operand, else that of the first symbol an inline expression names. The output
    type is not it: a comparison has ``bool`` output over ``double`` operands, and casting its symbol operand to the
    output type would turn ``1e-12`` into ``1``.

    :param fallback: The type when neither an array nor a typed symbol is there (the output type).
    """
    for operand in operands:
        if operand.reads_connector and operand.conn in in_edges:
            return edge_ctype(sdfg, in_edges[operand.conn])
    for operand in operands:
        if not operand.expr:
            continue
        try:
            names = dace.symbolic.symlist(dace.symbolic.pystr_to_symbolic(operand.expr))
        except (ValueError, TypeError, SyntaxError, AttributeError):
            continue
        for name in names:
            if str(name) in sdfg.symbols:
                return sdfg.symbols[str(name)].ctype
    return fallback


def operands_share_output_type(node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG,
                               operands: Sequence[Operand], out_conn: str) -> bool:
    """Whether the operands and the output have one type, which is all the ISA headers' single ``T`` carries.

    An op over operands of another type than its output (a comparison of ``double`` operands into ``bool``) would need
    its operands cast to the output type, and a C cast of a value truncates it.
    """
    out_ctype = edge_ctype(sdfg, output_edge(state, node, out_conn))
    return shared_ctype(operands, connected_edges(state, node), sdfg, out_ctype) == out_ctype


def has_lane_invariant_output(operands: Sequence[Operand], out_edge: MultiConnectorEdge) -> bool:
    """Whether every lane computes one value into a one-element output: no tile operand, no tile to write.

    Codegen binds a one-element connector by value, so such a node assigns its output once and never walks it.
    """
    return all(operand.kind != TILE for operand in operands) and edge_moves_one_element(out_edge)


@dataclass(slots=True)
class LaneOperands:
    """The operands of one node as the ``pure`` expansion reads them, one lane at a time."""
    sdfg: dace.SDFG
    in_edges: dict[str, MultiConnectorEdge]
    widths: list[int]
    offset: str
    shared: str
    cast: str

    @classmethod
    def of(cls, node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG, operands: Sequence[Operand],
           out_ctype: str) -> "LaneOperands":
        in_edges = connected_edges(state, node)
        shared = shared_ctype(operands, in_edges, sdfg, out_ctype)
        # A logical op's operands are bool tiles already, and casting a value to bool truncates it; the cast exists
        # only to settle type-strict overloads such as ``std::min(int, double)``.
        cast = "" if shared == dace.bool_.ctype else f"({shared})"
        widths = list(node.widths)
        return cls(sdfg, in_edges, widths, tile_offset(widths), shared, cast)

    def ctype(self, operand: Operand) -> str:
        """The C++ type the operand is read as, after the cast of a broadcast."""
        if operand.kind == SYMBOL:
            return self.shared
        desc = self.sdfg.arrays[self.in_edges[operand.conn].data.data]
        if operand.kind == SCALAR and scalar_operand_ref(desc, operand.conn, self.widths, self.offset)[1]:
            return self.shared
        return desc.dtype.ctype

    def reference(self, operand: Operand, others: Sequence[Operand]) -> str:
        """The per-lane C++ expression of one operand next to the ``others`` of the same op.

        A symbol and a broadcast scalar are cast to the shared type. A tile read keeps its own type, except
        ``dace::float16`` beside an operand of another type: ``__half`` converts implicitly to several types, so a
        mixed-type infix operator or an overloaded function cannot pick one, and ``half_disambiguated`` takes the one
        explicit lossless hop through ``float``.
        """
        if operand.kind == SYMBOL:
            return f"{self.cast}({pyexpr2cpp(operand.expr)})"
        desc = self.sdfg.arrays[self.in_edges[operand.conn].data.data]
        if operand.kind == TILE:
            reference = f"{operand.conn}[{self.offset}]"
        else:
            reference, broadcast = scalar_operand_ref(desc, operand.conn, self.widths, self.offset)
            if broadcast:
                return f"{self.cast}({reference})"
        meets = dace.float16.ctype if all(self.ctype(other) == dace.float16.ctype
                                          for other in others) else self.shared + "?mixed"
        return half_disambiguated(reference, desc.dtype.ctype, meets)


def elementwise_tasklet(node: nodes.LibraryNode, state: dace.SDFGState, operands: Sequence[Operand], out_conn: str,
                        rhs: str, out_ctype: str, in_edges: dict[str, MultiConnectorEdge]) -> nodes.Tasklet:
    """The ``pure`` tasklet of an elementwise node: ``out = rhs`` over the lanes, zero where the mask is off."""
    widths = list(node.widths)
    offset = tile_offset(widths)
    if has_lane_invariant_output(operands, output_edge(state, node, out_conn)):
        mask_elements = in_edges["_mask"].data.subset.num_elements() if node.has_mask else None
        code = lane_invariant_assign(out_conn, rhs, out_ctype, widths, mask_elements)
    else:
        if node.has_mask:
            body = f"{out_conn}[{offset}] = _mask[{offset}] ? ({rhs}) : {out_ctype}(0);"
        else:
            body = f"{out_conn}[{offset}] = {rhs};"
        code = nested_loops(widths, body)
    return nodes.Tasklet(
        label=f"{node.label}_pure",
        inputs=dict.fromkeys(input_connectors(operands, node.has_mask)),
        outputs={out_conn: None},
        code=code,
        language=dace.dtypes.Language.CPP,
    )


def validate_elementwise(
    node: nodes.LibraryNode,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    operands: Sequence[Operand],
    out_conn: str,
    has_mask: bool,
    promotion: Callable[[dace.typeclass, dace.typeclass], bool] | None = promotion_ok,
    unpromoted: Sequence[str] = ()
) -> None:
    """Check the wiring of an elementwise node and the dtypes it lowers.

    Every connector an operand or the mask reads must be connected. A tile operand makes the output a tile, whether
    the descriptor is tile-shaped or a memlet moves a tile-shaped window of a larger array. A tile operand is promoted
    to the output dtype before the op, so a narrowing is refused; ``promotion`` is the check, ``None`` skips it, and
    ``unpromoted`` lists the connectors the check does not apply to.

    :raises ValueError: If a required connector is not connected.
    :raises NotImplementedError: If a tile operand has an output that is not a tile or would narrow.
    """
    in_edges = connected_edges(state, node)
    out_edges = {edge.src_conn: edge for edge in state.out_edges(node) if edge.src_conn is not None}
    if out_conn not in out_edges:
        raise ValueError(f"{node.label}: required output {out_conn!r} not connected")
    if has_mask and "_mask" not in in_edges:
        raise ValueError(f"{node.label}: has_mask=True but '_mask' not connected")
    for operand in operands:
        if operand.reads_connector and operand.conn not in in_edges:
            raise ValueError(f"{node.label}: kind={operand.kind!r} but {operand.conn!r} not connected")
    out_desc = sdfg.arrays[out_edges[out_conn].data.data]
    widths = tuple(node.widths)
    if any(operand.kind == TILE for operand in operands) and not (is_tile_shape(out_desc, widths)
                                                                  or edge_moves_a_tile(out_edges[out_conn], widths)):
        raise NotImplementedError(f"{node.label}: output-kind rule violated -- a Tile input is present but "
                                  f"{out_conn!r} descriptor is not tile-shape {widths!r}. "
                                  f"Per design section 6.2: any Tile input -> Tile output.")
    if promotion is None:
        return
    for operand in operands:
        if operand.kind == TILE and operand.conn not in unpromoted:
            src = sdfg.arrays[in_edges[operand.conn].data.data].dtype
            if not promotion(src, out_desc.dtype):
                raise NotImplementedError(
                    f"{node.label}: Tile operand {operand.conn!r} dtype {src} cannot be promoted to output "
                    f"dtype {out_desc.dtype} (narrowing conversion); cast explicitly via a separate tasklet.")
