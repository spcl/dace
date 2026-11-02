# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Gather`` library node: an array indexed by an array of integers in one dimension (NumPy ``take``)."""

from dace import SDFG, SDFGState, dtypes, library, nodes, properties, symbolic
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandGatherPure(ExpandTransformation):
    """A map over the output: every element is read from the input at the position its index names."""

    environments = []

    @staticmethod
    def expansion(node: "Gather", parent_state: SDFGState, parent_sdfg: SDFG, **kwargs):
        edges = node.validate(parent_sdfg, parent_state)
        sdfg = SDFG(f"{node.label}_sdfg")
        for connector, e in edges.items():
            desc = parent_sdfg.arrays[e.data.data]
            sdfg.add_array(connector, e.data.subset.size(), desc.dtype, storage=desc.storage, strides=desc.strides)
        inp, index, out = (sdfg.arrays[c].shape for c in ("_inp", "_index", "_out"))
        params = [f"__o{d}" for d in range(len(out))]
        a = node.axis
        index_params = params[a : a + len(index)]
        inp_subset = params[:a] + [f"0:{inp[a]}"] + params[a + len(index) :]
        state = sdfg.add_state(node.label)
        state.add_mapped_tasklet(
            f"{node.label}_gather",
            {p: f"0:{s}" for p, s in zip(params, out)},
            {
                "__arr": mm.Memlet(f"_inp[{', '.join(inp_subset)}]"),
                "__idx": mm.Memlet(f"_index[{', '.join(index_params)}]"),
            },
            "__out = __arr[__idx]",
            {"__out": mm.Memlet(f"_out[{', '.join(params)}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class Gather(nodes.LibraryNode):
    """``_out = _inp[..., _index, ...]``: the input indexed by an array of integers in dimension ``axis``, as NumPy
    indexes an array (``numpy.take``). The output has the shape of the input, with dimension ``axis`` replaced by the
    dimensions of the index.

    A gather that feeds a matrix product can be fused into it (see ``dace.libraries.blas.indexed``).
    """

    implementations = {"pure": ExpandGatherPure}
    default_implementation = "pure"

    axis = properties.Property(dtype=int, default=0, desc="The dimension of the input that the index indexes")

    def __init__(self, name, *, axis: int = 0, **kwargs):
        super().__init__(name, inputs={"_inp", "_index"}, outputs={"_out"}, **kwargs)
        self.axis = axis

    def validate(self, sdfg: SDFG, state: SDFGState) -> dict:
        """:returns: The edges of the connectors ``_inp``, ``_index`` and ``_out``.

        :raises ValueError: if a connector is missing, the index is not an array of integers, or the shapes do not
            match.
        """
        edges = {e.dst_conn: e for e in state.in_edges(self)}
        edges.update({e.src_conn: e for e in state.out_edges(self)})
        if set(edges) != {"_inp", "_index", "_out"}:
            raise ValueError(f'Gather "{self.label}" needs the connectors _inp, _index and _out')
        dtype = sdfg.arrays[edges["_index"].data.data].dtype
        if dtype not in dtypes.INTEGER_TYPES or dtype in (dtypes.bool, dtypes.bool_):
            raise ValueError(f'The index of gather "{self.label}" must be an array of integers, not {dtype}')
        inp, index, out = (edges[c].data.subset.size() for c in ("_inp", "_index", "_out"))
        if not 0 <= self.axis < len(inp):
            raise ValueError(f'Gather "{self.label}": axis {self.axis} is out of range for a {len(inp)}-D input')
        expected = inp[: self.axis] + index + inp[self.axis + 1 :]
        if len(out) != len(expected) or any(symbolic.equal(o, e) is False for o, e in zip(out, expected)):
            raise ValueError(f'Gather "{self.label}": the output has the shape {out}, expected {expected}')
        return edges


@oprepo.replaces("dace.libraries.standard.gather")
@oprepo.replaces("dace.libraries.standard.Gather")
def gather_libnode(pv: "ProgramVisitor", sdfg: SDFG, state: SDFGState, inp, index, out, *, axis: int = 0):
    node = Gather("gather", axis=axis)
    state.add_node(node)
    state.add_edge(state.add_read(inp), None, node, "_inp", mm.Memlet(inp))
    state.add_edge(state.add_read(index), None, node, "_index", mm.Memlet(index))
    state.add_edge(node, "_out", state.add_write(out), None, mm.Memlet(out))
    return []
