# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the ISA expansions of the tile nodes share: staging operands for a call of a per-signature tile function
(:mod:`dace.tile_functions`)."""

import contextlib
import contextvars
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import dace
from dace import tile_functions
from dace.codegen.cppunparse import pyexpr2cpp
from dace.libraries.tileops.kinds import SYMBOL, TILE
from dace.libraries.tileops.operands import Operand, connected_edges, edge_ctype, input_connectors, output_edge
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.subsets import Subset

_chunk: contextvars.ContextVar[int | None] = contextvars.ContextVar("tile_isa_chunk", default=None)

#: The first lane of the chunk an ISA call covers within a ``block`` lowering.
CHUNK_OFFSET = "__c"


@contextlib.contextmanager
def isa_chunk(lanes: int) -> Iterator[None]:
    """Within the block, an ISA call covers ``lanes`` lanes of its tile, one thread's share of a ``block`` lowering."""
    token = _chunk.set(lanes)
    try:
        yield
    finally:
        _chunk.reset(token)


def require_k1(node: nodes.LibraryNode) -> int:
    """The lanes one ISA call of a node whose backend takes one dim covers: its tile width, or the chunk of
    :func:`isa_chunk`. A node of more dims lowers ``pure``."""
    if len(node.widths) != 1:
        raise NotImplementedError(f"{node.label}: ISA tile-op backend is K=1 only; K>=2 lowers to 'pure'")
    chunk = _chunk.get()
    return node.widths[0] if chunk is None else chunk


def lane_start(start: str) -> str:
    """The position of the first lane an ISA call covers, for a tile whose lane 0 sits at ``start``."""
    return start if _chunk.get() is None else f"({start}) + {CHUNK_OFFSET}"


def broadcast_ref(conn: str, subset: Subset) -> str:
    """The reference to the value of a connector that carries one element.

    Codegen passes a one-element access by value, whether it reads a ``Scalar``, one element of an array or a length-1
    array at ``[0]``; only a connector of several elements is a pointer. The element count of the access decides, not
    the shape of the descriptor.
    """
    return conn if subset.num_elements() == 1 else f"{conn}[0]"


def pointer(conn: str, edge: MultiConnectorEdge[dace.Memlet]) -> str:
    """A connector as the pointer a tile function takes: codegen binds a one-element access by value."""
    return f"&{conn}" if edge.data.subset is not None and edge.data.subset.num_elements() == 1 else conn


def template_flag(flag: str) -> str:
    """A broadcast flag as the ``dace::tileops`` templates take it."""
    return "true" if flag == "b" else "false"


def call_text(function: str, arguments: Sequence[str]) -> str:
    return f"{function}({', '.join(arguments)});"


def element_code(ctype: str) -> str:
    """The type code a tile function name spells ``ctype`` with."""
    element = tile_functions.ELEMENT_TYPES.get(ctype)
    if element is None:
        raise NotImplementedError(f"no CPU tile function takes {ctype} elements")
    return element.code


@dataclass(slots=True)
class IsaCall:
    """The statements staging the operands of one tile function call, and the arguments they yield.

    A function takes one element type for the operands and the output, so a widening tile operand is copied into a
    buffer of that type first, and a symbol or scalar becomes a one-element buffer it reads as a broadcast.
    """

    sdfg: dace.SDFG
    in_edges: dict[str, MultiConnectorEdge]
    out_edge: MultiConnectorEdge
    ctype: str
    vlen: int
    backend: str
    pre: list[str] = field(default_factory=list)

    @classmethod
    def of(
        cls, node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG, out_conn: str, backend: str
    ) -> "IsaCall":
        out_edge = output_edge(state, node, out_conn)
        ctype = edge_ctype(sdfg, out_edge)
        return cls(sdfg, connected_edges(state, node), out_edge, ctype, require_k1(node), backend)

    @property
    def functions(self) -> bool:
        """Whether the backend lowers to tile functions; the others (``sve``, ``cuda``) call ``dace::tileops``
        templates, whose one-lane overloads take a value as well."""
        return self.backend in tile_functions.ISAS

    @property
    def element(self) -> str:
        return element_code(self.ctype)

    def output(self, out_conn: str) -> str:
        return pointer(out_conn, self.out_edge) if self.functions else out_conn

    def input(self, conn: str) -> str:
        return pointer(conn, self.in_edges[conn]) if self.functions else conn

    def mask(self, has_mask: bool) -> list[str]:
        """The mask argument of a tile function; a template takes ``nullptr`` for none."""
        if self.functions:
            return [self.input("_mask")] if has_mask else []
        return ["_mask" if has_mask else "nullptr"]

    def operand(self, operand: Operand, promote_tile: bool = True) -> tuple[str, str]:
        """``(broadcast flag, pointer)`` of an operand as the function takes it: ``v`` a tile, ``b`` one value."""
        if operand.kind == TILE:
            if not promote_tile or edge_ctype(self.sdfg, self.in_edges[operand.conn]) == self.ctype:
                return "v", self.input(operand.conn)
            buffer = f"_cast{operand.conn}"
            self.pre.append(f"{self.ctype} {buffer}[{self.vlen}];")
            self.pre.append(
                f"for (int _ci = 0; _ci < {self.vlen}; ++_ci) {buffer}[_ci] = ({self.ctype}){operand.conn}[_ci];"
            )
            return "v", buffer
        if operand.kind == SYMBOL:
            value = f"({self.ctype})({pyexpr2cpp(operand.expr)})"
        else:
            value = f"({self.ctype})({broadcast_ref(operand.conn, self.in_edges[operand.conn].data.subset)})"
        buffer = f"_bc{operand.conn}"
        self.pre.append(f"const {self.ctype} {buffer}[1] = {{ {value} }};")
        return "b", buffer

    def tasklet(
        self,
        node: nodes.LibraryNode,
        backend: str,
        operands: Sequence[Operand],
        out_conn: str,
        call: str,
        has_mask: bool,
    ) -> nodes.Tasklet:
        return nodes.Tasklet(
            label=f"{node.label}_{backend}",
            inputs=dict.fromkeys(input_connectors(operands, has_mask)),
            outputs={out_conn: None},
            code="\n".join([*self.pre, call]),
            language=dace.dtypes.Language.CPP,
        )
