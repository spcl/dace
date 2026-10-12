# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Norm2``: Fortran ``NORM2(X [, DIM])``, ``sqrt(sum(X**2))`` over the whole array or along ``DIM``."""

import dace
from dace import SDFG, SDFGState, library, properties
from dace import memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.libraries.standard.helper import connector_array, fortran_axis, map_ranges, reduced_coordinates
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandNorm2Pure(ExpandTransformation):
    """Sums the squares with a write-conflict resolution, then takes the square root."""

    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        sdfg = SDFG(f"{node.label}_sdfg")
        shape = connector_array(sdfg, "_x", sdutil.in_edge_with_name(node, parent_state, "_x").data, parent_sdfg)
        connector_array(sdfg, "_out", sdutil.out_edge_with_name(node, parent_state, "_out").data, parent_sdfg)
        axis = fortran_axis(node, node.dim, len(shape))
        out_ranges, out = reduced_coordinates(shape, axis)
        sdfg.add_transient("_sumsq", sdfg.arrays["_out"].shape, sdfg.arrays["_x"].dtype)

        init = sdfg.add_state(f"{node.label}_init")
        init.add_mapped_tasklet(
            "init", out_ranges, {}, "__out = 0", {"__out": dace.Memlet(f"_sumsq[{out}]")}, external_edges=True
        )
        squares = sdfg.add_state_after(init, f"{node.label}_squares")
        element = ", ".join(f"__i{d}" for d in range(len(shape)))
        squares.add_mapped_tasklet(
            "squares",
            map_ranges(shape),
            {"__in": dace.Memlet(f"_x[{element}]")},
            "__out = __in * __in",
            {"__out": dace.Memlet(f"_sumsq[{out}]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )
        root = sdfg.add_state_after(squares, f"{node.label}_sqrt")
        root.add_mapped_tasklet(
            "sqrt",
            out_ranges,
            {"__in": dace.Memlet(f"_sumsq[{out}]")},
            "__out = sqrt(__in)",
            {"__out": dace.Memlet(f"_out[{out}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class Norm2(nodes.LibraryNode):
    """Fortran ``NORM2(X [, DIM])``: the Euclidean norm."""

    implementations = {"pure": ExpandNorm2Pure}
    default_implementation = "pure"

    dim = properties.Property(
        dtype=int, default=None, allow_none=True, desc="Fortran 1-based axis; None reduces the whole array."
    )

    def __init__(self, name, *, dim=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_out"}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        sdutil.out_edge_with_name(self, state, "_out")
        fortran_axis(self, self.dim, sdutil.in_edge_with_name(self, state, "_x").data.subset.dims())


@oprepo.replaces("dace.libraries.standard.norm2")
@oprepo.replaces("dace.libraries.standard.Norm2")
def norm2_libnode(pv, sdfg: SDFG, state: SDFGState, x, out, *, dim=None):
    node = Norm2("norm2", dim=dim)
    state.add_edge(state.add_read(x), None, node, "_x", mm.Memlet(x))
    state.add_edge(node, "_out", state.add_write(out), None, mm.Memlet(out))
    return []
