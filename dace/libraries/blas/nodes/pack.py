# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Pack`` library node: a matrix (or a batch of them) stored in blocks of columns (see
``dace.libraries.blas.blocked``)."""

from dace import SDFG, SDFGState, library, nodes, properties, symbolic
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas.blocked import blocked_shape
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandPackPure(ExpandTransformation):
    """A map over the input: every element is written to its place in the blocks."""

    environments = []

    @staticmethod
    def expansion(node: "Pack", parent_state: SDFGState, parent_sdfg: SDFG, **kwargs):
        edges = node.validate(parent_sdfg, parent_state)
        sdfg = SDFG(f"{node.label}_sdfg")
        for connector, e in edges.items():
            desc = parent_sdfg.arrays[e.data.data]
            sdfg.add_array(connector, e.data.subset.size(), desc.dtype, storage=desc.storage, strides=desc.strides)
        shape = sdfg.arrays["_inp"].shape
        *batch, row, column = [f"__i{d}" for d in range(len(shape))]
        b = node.block
        state = sdfg.add_state(node.label)
        state.add_mapped_tasklet(
            f"{node.label}_pack",
            {p: f"0:{s}" for p, s in zip([*batch, row, column], shape)},
            {"__in": mm.Memlet(f"_inp[{', '.join([*batch, row, column])}]")},
            "__out = __in",
            {"__out": mm.Memlet(f"_out[{', '.join([*batch, f'{column} // {b}', row, f'{column} % {b}'])}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class Pack(nodes.LibraryNode):
    """``_out = _inp`` stored in blocks of ``block`` columns: an input ``X[..., R, C]`` is the output ``S[..., C / block,
    R, block]``, with ``S[..., c // block, r, c % block] = X[..., r, c]`` (see ``dace.libraries.blas.blocked``), the
    layout in which ``dace.transformation.passes.BlockMatrixOperands`` stores matrix-product operands."""

    implementations = {"pure": ExpandPackPure}
    default_implementation = "pure"

    block = properties.Property(dtype=int, default=32, desc="The columns of a block")

    def __init__(self, name, *, block: int = 32, **kwargs):
        super().__init__(name, inputs={"_inp"}, outputs={"_out"}, **kwargs)
        self.block = block

    def validate(self, sdfg: SDFG, state: SDFGState) -> dict:
        """:returns: The edges of the connectors ``_inp`` and ``_out``.

        :raises ValueError: if a connector is missing, or the shapes do not match.
        """
        edges = {e.dst_conn: e for e in state.in_edges(self)}
        edges.update({e.src_conn: e for e in state.out_edges(self)})
        if set(edges) != {"_inp", "_out"}:
            raise ValueError(f'Pack "{self.label}" needs the connectors _inp and _out')
        inp, out = (edges[c].data.subset.size() for c in ("_inp", "_out"))
        if len(inp) < 2 or symbolic.equal(inp[-1] % self.block, 0) is False:
            raise ValueError(f'Pack "{self.label}": the input {inp} is not a matrix of whole blocks of {self.block}')
        expected = blocked_shape(inp, self.block)
        if len(out) != len(expected) or any(symbolic.equal(o, e) is False for o, e in zip(out, expected)):
            raise ValueError(f'Pack "{self.label}": the output has the shape {out}, expected {expected}')
        return edges


@oprepo.replaces("dace.libraries.blas.pack")
@oprepo.replaces("dace.libraries.blas.Pack")
def pack_libnode(pv: "ProgramVisitor", sdfg: SDFG, state: SDFGState, inp, out, *, block: int = 32):
    node = Pack("pack", block=block)
    state.add_node(node)
    state.add_edge(state.add_read(inp), None, node, "_inp", mm.Memlet(inp))
    state.add_edge(node, "_out", state.add_write(out), None, mm.Memlet(out))
    return []
