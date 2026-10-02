# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What the ISA expansions of the tile nodes share: staging operands for a ``dace::tileops::tile_*`` call."""
from collections.abc import Sequence
from dataclasses import dataclass, field

import dace
from dace.codegen.cppunparse import pyexpr2cpp
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.subsets import Subset

from .kinds import SYMBOL, TILE
from .operands import Operand, connected_edges, edge_ctype, input_connectors, output_edge


def require_k1(node: nodes.LibraryNode) -> int:
    """The tile width of a node whose ISA backend takes one dim; a node of more dims lowers ``pure``."""
    if len(node.widths) != 1:
        raise NotImplementedError(f"{node.label}: ISA tile-op backend is K=1 only; K>=2 lowers to 'pure'")
    return node.widths[0]


def broadcast_ref(conn: str, subset: Subset) -> str:
    """The reference to the value of a connector that carries one element.

    Codegen passes a one-element access by value, whether it reads a ``Scalar``, one element of an array or a length-1
    array at ``[0]``; only a connector of several elements is a pointer. The element count of the access decides, not
    the shape of the descriptor.
    """
    return conn if subset.num_elements() == 1 else f"{conn}[0]"


@dataclass(slots=True)
class IsaCall:
    """The statements staging the operands of one ``dace::tileops::tile_*`` call, and the arguments they yield.

    The headers take one element type ``T`` for the operands and the output, so a widening tile operand is copied into
    a ``T`` buffer first, and a symbol or scalar becomes a one-element buffer read with ``Broadcast=true``.
    """
    sdfg: dace.SDFG
    in_edges: dict[str, MultiConnectorEdge]
    ctype: str
    vlen: int
    pre: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, node: nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG, out_conn: str) -> "IsaCall":
        return cls(sdfg, connected_edges(state, node), edge_ctype(sdfg, output_edge(state, node, out_conn)),
                   require_k1(node))

    def operand(self, operand: Operand, promote_tile: bool = True) -> tuple[str, str]:
        """``(broadcast flag, pointer)`` of an operand as the call takes it."""
        if operand.kind == TILE:
            if not promote_tile or edge_ctype(self.sdfg, self.in_edges[operand.conn]) == self.ctype:
                return "false", operand.conn
            buffer = f"_cast{operand.conn}"
            self.pre.append(f"{self.ctype} {buffer}[{self.vlen}];")
            self.pre.append(f"for (int _ci = 0; _ci < {self.vlen}; ++_ci) "
                            f"{buffer}[_ci] = ({self.ctype}){operand.conn}[_ci];")
            return "false", buffer
        if operand.kind == SYMBOL:
            value = f"({self.ctype})({pyexpr2cpp(operand.expr)})"
        else:
            value = f"({self.ctype})({broadcast_ref(operand.conn, self.in_edges[operand.conn].data.subset)})"
        buffer = f"_bc{operand.conn}"
        self.pre.append(f"const {self.ctype} {buffer}[1] = {{ {value} }};")
        return "true", buffer

    def tasklet(self, node: nodes.LibraryNode, backend: str, operands: Sequence[Operand], out_conn: str, call: str,
                has_mask: bool) -> nodes.Tasklet:
        return nodes.Tasklet(
            label=f"{node.label}_{backend}",
            inputs=dict.fromkeys(input_connectors(operands, has_mask)),
            outputs={out_conn: None},
            code="\n".join([*self.pre, call]),
            language=dace.dtypes.Language.CPP,
        )
