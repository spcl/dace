# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CShift``: Fortran ``CSHIFT(ARRAY, SHIFT [, DIM])``, a circular shift by ``SHIFT`` positions along ``DIM``.

Element ``i`` along ``DIM`` of the result is element ``MODULO(i + SHIFT, n)`` of the input, so every section
perpendicular to ``DIM`` rotates on its own.
"""

import dace
from dace import SDFG, library, properties
from dace.libraries.standard.helper import connector_array, fortran_axis, map_ranges
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandCShiftPure(ExpandTransformation):
    """One map reading the input at the rotated index."""

    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        if node.shift is None:
            raise ValueError(f"{node.label}: the shift must be set before expansion")
        sdfg = SDFG(f"{node.label}_sdfg")
        shape = connector_array(sdfg, "_x", sdutil.in_edge_with_name(node, parent_state, "_x").data, parent_sdfg)
        connector_array(sdfg, "_out", sdutil.out_edge_with_name(node, parent_state, "_out").data, parent_sdfg)
        axis = fortran_axis(node, node.dim, len(shape))
        element = [f"__i{d}" for d in range(len(shape))]
        # ``Modulo`` is Fortran's MODULO: a negative shift wraps into range, where ``%`` would not
        rotated = [f"Modulo(__i{d} + ({node.shift}), {shape[d]})" if d == axis else i for d, i in enumerate(element)]
        sdfg.add_state().add_mapped_tasklet(
            "cshift",
            map_ranges(shape),
            {"__in": dace.Memlet(f"_x[{', '.join(rotated)}]")},
            "__out = __in",
            {"__out": dace.Memlet(f"_out[{', '.join(element)}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class CShift(nodes.LibraryNode):
    """Fortran ``CSHIFT(ARRAY, SHIFT [, DIM])``: rotates ``_x`` by ``shift`` positions along ``dim`` into ``_out``."""

    implementations = {"pure": ExpandCShiftPure}
    default_implementation = "pure"

    dim = properties.Property(dtype=int, default=1, desc="Fortran 1-based axis to rotate along.")
    shift = properties.SymbolicProperty(allow_none=True, default=None, desc="Positions to shift by, set by expansion.")

    def __init__(self, name, *, dim=1, shift=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_out"}, **kwargs)
        self.dim = dim
        self.shift = shift

    def validate(self, sdfg, state):
        x = sdutil.in_edge_with_name(self, state, "_x").data.subset
        out = sdutil.out_edge_with_name(self, state, "_out").data.subset
        fortran_axis(self, self.dim, x.dims())
        if x.size() != out.size():
            raise ValueError(f"{self.label}: _x has the shape {x.size()} and _out {out.size()}")
